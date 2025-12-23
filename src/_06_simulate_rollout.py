# src/_06_simulate_rollout.py
import os, argparse
import pandas as pd
import numpy as np
import torch
import random
import re

from src.gnn_utils import build_heterodata_from_dfs, load_checkpoint


# ---------- utils ----------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_list_field(x):
    if x is None:
        return []
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return []
    return [p.strip() for p in re.split(r"[;,|]+", s) if p.strip()]


def pick_col(df, candidates):
    cols = df.columns.astype(str).str.strip().tolist()
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def normalize_df_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def load_snapshot(snapshots_dir: str, project_id: str, time_ym: str):
    tdir = os.path.join(snapshots_dir, project_id, time_ym)
    if not os.path.isdir(tdir):
        raise FileNotFoundError(f"Snapshot not found: {tdir}")
    nodes = pd.read_csv(os.path.join(tdir, "nodes.csv"))
    edges = pd.read_csv(os.path.join(tdir, "edges.csv"))
    return normalize_df_cols(nodes), normalize_df_cols(edges)


def find_default_time(snapshots_dir: str, project_id: str):
    pdir = os.path.join(snapshots_dir, project_id)
    if not os.path.isdir(pdir):
        raise FileNotFoundError(f"Project snapshots not found: {pdir}")
    months = sorted([d for d in os.listdir(pdir) if os.path.isdir(os.path.join(pdir, d))])
    if not months:
        raise FileNotFoundError(f"No month folders under: {pdir}")
    return months[0]


def predict_cor(model, device, nodes_df, edges_df) -> float:
    data = build_heterodata_from_dfs(nodes_df, edges_df, cor_label=None).to(device)
    with torch.no_grad():
        pred = float(model(data).view(-1).cpu().item())
    return pred


# ---------- action application ----------
def apply_action(nodes_df, edges_df, action_row):
    nodes = nodes_df.copy()
    edges = edges_df.copy()

    if "node_type" not in nodes.columns or "node_id" not in nodes.columns:
        raise KeyError(f"nodes.csv missing node_type/node_id, got={list(nodes.columns)}")

    # edges src/dst alias
    src_col = pick_col(edges, ["src", "source", "from", "u", "head", "start", "node_u"])
    dst_col = pick_col(edges, ["dst", "target", "to", "v", "tail", "end", "node_v"])
    if src_col is None or dst_col is None:
        raise KeyError(f"edges.csv missing src/dst (or alias). got={list(edges.columns)}")

    # --- target risk nodes by taxonomy ---
    tax_list = [t.strip() for t in parse_list_field(action_row.get("target_risk_taxonomy", ""))]

    mask_risk = (nodes["node_type"].astype(str).str.strip() == "Risk")
    if tax_list and "taxonomy" in nodes.columns:
        mask_risk &= nodes["taxonomy"].fillna("").astype(str).str.strip().isin(tax_list)

    target_nodes = set(nodes.loc[mask_risk, "node_id"].astype(str).str.strip().tolist())

    # --- node changes ---
    p_mult = float(action_row.get("affect_node_prob_mult", 1.0))
    s_mult = float(action_row.get("affect_node_sev_mult", 1.0))
    c_mult = float(action_row.get("affect_node_conf_mult", 1.0))

    if "probability" in nodes.columns:
        nodes.loc[mask_risk, "probability"] = nodes.loc[mask_risk, "probability"].fillna(0.0) * p_mult
    if "severity_num" in nodes.columns:
        nodes.loc[mask_risk, "severity_num"] = nodes.loc[mask_risk, "severity_num"].fillna(0.0) * s_mult
    if "confidence" in nodes.columns:
        nodes.loc[mask_risk, "confidence"] = nodes.loc[mask_risk, "confidence"].fillna(0.0) * c_mult

    # --- edge changes ---
    edge_types = [e.strip() for e in parse_list_field(action_row.get("target_edge_types", ""))]

    if "edge_type" in edges.columns and edge_types:
        mtype = edges["edge_type"].fillna("").astype(str).str.strip().isin(edge_types)
    else:
        mtype = pd.Series([True] * len(edges), index=edges.index)

    src_s = edges[src_col].fillna("").astype(str).str.strip()
    dst_s = edges[dst_col].fillna("").astype(str).str.strip()
    mnode = src_s.isin(target_nodes) | dst_s.isin(target_nodes)

    mask_edge = mtype & mnode

    ew_mult = float(action_row.get("affect_edge_weight_mult", 1.0))
    ec_mult = float(action_row.get("affect_edge_conf_mult", 1.0))

    if "weight" in edges.columns:
        edges.loc[mask_edge, "weight"] = edges.loc[mask_edge, "weight"].fillna(1.0) * ew_mult
    if "confidence" in edges.columns:
        edges.loc[mask_edge, "confidence"] = edges.loc[mask_edge, "confidence"].fillna(0.0) * ec_mult

    hit_nodes = int(mask_risk.sum())
    hit_edges = int(mask_edge.sum())

    return nodes, edges, hit_nodes, hit_edges


# ---------- optimize: multi-actions within a step ----------
def optimize_bundle_one_step(
    cur_nodes, cur_edges, actions_df, remaining_budget, model, device,
    per_step_max_actions=5, min_gain=1e-6, require_edge_hit=True
):
    """
    Greedy bundle selection in ONE step:
      repeat: pick best action by (gain/cost) among remaining budget & not selected in this step,
      apply it, update cur graph, until no gain or reach K.
    """
    selected = []
    used_total = 0.0
    gain_total = 0.0
    hit_nodes_total = 0
    hit_edges_total = 0

    nodes = cur_nodes
    edges = cur_edges

    cor_before_step = predict_cor(model, device, nodes, edges)
    cor_current = cor_before_step

    # For this step, do not repeat same action
    chosen_set = set()

    # Keep iterating until no improvement
    while len(selected) < int(per_step_max_actions):
        best = None
        best_ratio = 0.0
        best_next = None

        for _, a in actions_df.iterrows():
            aid = str(a["action_id"])
            if aid in chosen_set:
                continue
            cost = float(a["cost"])
            if cost <= 0:
                continue
            if used_total + cost > float(remaining_budget) + 1e-9:
                continue

            n2, e2, hn, he = apply_action(nodes, edges, a)

            if require_edge_hit and he <= 0:
                continue

            cor2 = predict_cor(model, device, n2, e2)
            gain = cor_current - cor2  # want positive

            if gain <= float(min_gain):
                continue

            ratio = gain / (cost + 1e-9)
            if ratio > best_ratio:
                best_ratio = ratio
                best = (aid, cost, gain, hn, he, cor2)
                best_next = (n2, e2)

        if best is None:
            break

        aid, cost, gain, hn, he, cor2 = best
        nodes, edges = best_next

        chosen_set.add(aid)
        selected.append(aid)
        used_total += float(cost)
        gain_total += float(gain)
        hit_nodes_total += int(hn)
        hit_edges_total += int(he)
        cor_current = float(cor2)

    cor_after_step = cor_current
    return selected, nodes, edges, used_total, gain_total, hit_nodes_total, hit_edges_total, cor_before_step, cor_after_step


def policy_none(cur_nodes, cur_edges, remaining_budget, model, device):
    cor_before = predict_cor(model, device, cur_nodes, cur_edges)
    cor_after = cor_before
    return [], cur_nodes, cur_edges, 0.0, 0.0, 0, 0, cor_before, cor_after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default="data/snapshots")
    ap.add_argument("--actions", default="data/actions.csv")
    ap.add_argument("--ckpt", default="data/interim/gnn_ckpt.pt")
    ap.add_argument("--project_id", required=True)
    ap.add_argument("--time", dest="time_ym", default="", help="YYYY-MM; if empty, pick earliest available")
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--steps", type=int, default=10, help="number of steps")
    ap.add_argument("--policy", default="optimize", choices=["none", "optimize"])
    ap.add_argument("--out", default="", help="output csv path")
    ap.add_argument("--seed", type=int, default=42)

    # NEW
    ap.add_argument("--per_step_max_actions", type=int, default=5)
    ap.add_argument("--min_gain", type=float, default=1e-6)
    ap.add_argument("--require_edge_hit", action="store_true", help="only allow actions with hit_edges>0")
    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(args.ckpt, device)
    model.eval()

    actions_df = pd.read_csv(args.actions)
    actions_df = normalize_df_cols(actions_df)

    if "action_id" not in actions_df.columns:
        raise KeyError(f"actions file must contain action_id. got={list(actions_df.columns)}")
    if "cost" not in actions_df.columns:
        raise KeyError(f"actions file must contain cost. got={list(actions_df.columns)}")

    time_ym = args.time_ym.strip() or find_default_time(args.snapshots, args.project_id)
    nodes0, edges0 = load_snapshot(args.snapshots, args.project_id, time_ym)

    base_cor = predict_cor(model, device, nodes0, edges0)

    cur_nodes, cur_edges = nodes0, edges0
    remaining = float(args.budget)

    rows = []
    for step in range(1, int(args.steps) + 1):
        if args.policy == "none":
            selected, n2, e2, used_step, gain_step, hn, he, cor_before, cor_after = policy_none(
                cur_nodes, cur_edges, remaining, model, device
            )
        else:
            selected, n2, e2, used_step, gain_step, hn, he, cor_before, cor_after = optimize_bundle_one_step(
                cur_nodes, cur_edges, actions_df, remaining, model, device,
                per_step_max_actions=args.per_step_max_actions,
                min_gain=args.min_gain,
                require_edge_hit=args.require_edge_hit,
            )

        if used_step > remaining:
            # safety
            used_step = 0.0
            selected = []
            n2, e2 = cur_nodes, cur_edges
            cor_after = cor_before
            gain_step = 0.0
            hn, he = 0, 0

        remaining -= float(used_step)
        cur_nodes, cur_edges = n2, e2

        rows.append({
            "project_id": args.project_id,
            "time": time_ym,
            "step": step,
            "policy": args.policy,
            "budget_init": float(args.budget),
            "budget_remaining": float(remaining),
            "per_step_max_actions": int(args.per_step_max_actions),
            "min_gain": float(args.min_gain),
            "cor_pred_baseline": float(base_cor),
            "cor_pred_before": float(cor_before),
            "cor_pred_after": float(cor_after),
            "gain_step": float(gain_step),
            "delta_cor": float(cor_before - cor_after),
            "used_cost_step": float(used_step),
            "hit_nodes_step": int(hn),
            "hit_edges_step": int(he),
            "selected_actions": "|".join(selected),
        })

        if remaining <= 1e-9:
            break

        # 如果当前 step 没选到任何动作，就可以提前停（可选）
        # 这里我默认不提前停，让你看到“后续无动作”的事实
        # if args.policy == "optimize" and len(selected) == 0:
        #     break

    out_df = pd.DataFrame(rows)

    out_path = args.out.strip()
    if not out_path:
        out_path = f"results/rollout_{args.project_id}_{time_ym}_{args.policy}_steps{args.steps}_K{args.per_step_max_actions}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Saved rollout to {out_path}")
    print(out_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
