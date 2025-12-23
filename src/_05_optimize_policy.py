# src/_05_optimize_policy.py
import os, argparse, re, random
import pandas as pd
import numpy as np
import torch

from src.gnn_utils import build_heterodata_from_dfs, load_checkpoint


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_snapshot(snapshots_dir: str, project_id: str, time_ym: str):
    tdir = os.path.join(snapshots_dir, project_id, time_ym)
    if not os.path.isdir(tdir):
        raise FileNotFoundError(f"Snapshot not found: {tdir}")
    nodes = pd.read_csv(os.path.join(tdir, "nodes.csv"))
    edges = pd.read_csv(os.path.join(tdir, "edges.csv"))

    # 列名清洗（防止隐藏空格）
    nodes.columns = nodes.columns.astype(str).str.strip()
    edges.columns = edges.columns.astype(str).str.strip()
    return nodes, edges


def predict_cor(model, device, nodes_df, edges_df) -> float:
    data = build_heterodata_from_dfs(nodes_df, edges_df, cor_label=None).to(device)
    with torch.no_grad():
        pred = float(model(data).squeeze().detach().cpu().item())
    return float(pred)


def parse_list_field(x):
    if x is None:
        return []
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return []
    # 支持逗号/分号/竖线
    return [p.strip() for p in re.split(r"[;,|]+", s) if p.strip()]


def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def apply_action(nodes_df, edges_df, action_row):
    nodes = nodes_df.copy()
    edges = edges_df.copy()

    # ---- 列名清洗（更稳）----
    nodes.columns = nodes.columns.astype(str).str.strip()
    edges.columns = edges.columns.astype(str).str.strip()

    # ---- 必要列检查（避免静默错误）----
    need_node_cols = ["node_id", "node_type", "taxonomy", "probability", "severity_num", "confidence"]
    for c in need_node_cols:
        if c not in nodes.columns:
            raise KeyError(f"nodes.csv 缺少列 {c}. 当前列={list(nodes.columns)}")

    need_edge_cols = ["edge_type", "weight", "confidence"]
    for c in need_edge_cols:
        if c not in edges.columns:
            raise KeyError(f"edges.csv 缺少列 {c}. 当前列={list(edges.columns)}")

    # ---- 识别 src/dst 列（兼容别名）----
    src_col = pick_col(edges, ["src", "source", "from", "u", "head", "start", "node_u"])
    dst_col = pick_col(edges, ["dst", "target", "to", "v", "tail", "end", "node_v"])
    if src_col is None or dst_col is None:
        raise KeyError(f"edges.csv 缺少 src/dst 列（或别名）。当前列={list(edges.columns)}")

    # ---- 1) 目标 taxonomy 列表 ----
    tax_list = parse_list_field(action_row.get("target_risk_taxonomy", ""))
    tax_list = [t.strip() for t in tax_list]

    # ---- 2) 命中 Risk 节点（只改命中的那部分）----
    mask_risk = (nodes["node_type"].astype(str).str.strip() == "Risk")
    if tax_list:
        mask_risk &= nodes["taxonomy"].fillna("").astype(str).str.strip().isin(tax_list)

    # ✅ 关键：定义 target_nodes（你原来缺这句）
    target_nodes = set(nodes.loc[mask_risk, "node_id"].fillna("").astype(str).str.strip().tolist())

    # ---- 3) 节点修改（prob/sev/conf）----
    p_mult = float(action_row.get("affect_node_prob_mult", 1.0))
    s_mult = float(action_row.get("affect_node_sev_mult", 1.0))
    c_mult = float(action_row.get("affect_node_conf_mult", 1.0))

    nodes.loc[mask_risk, "probability"] = nodes.loc[mask_risk, "probability"].fillna(0.0).astype(float) * p_mult
    nodes.loc[mask_risk, "severity_num"] = nodes.loc[mask_risk, "severity_num"].fillna(0.0).astype(float) * s_mult
    nodes.loc[mask_risk, "confidence"] = nodes.loc[mask_risk, "confidence"].fillna(0.0).astype(float) * c_mult

    # ---- 4) edge_type 命中 ----
    edge_types = [e.strip() for e in parse_list_field(action_row.get("target_edge_types", ""))]
    if edge_types:
        mtype = edges["edge_type"].fillna("").astype(str).str.strip().isin(edge_types)
    else:
        mtype = pd.Series([True] * len(edges), index=edges.index)

    # ---- 5) 命中与目标节点相关的边（src/dst 任一在 target_nodes）----
    src_s = edges[src_col].fillna("").astype(str).str.strip()
    dst_s = edges[dst_col].fillna("").astype(str).str.strip()
    mnode = src_s.isin(target_nodes) | dst_s.isin(target_nodes)

    # ✅ 注意换行：mask_edge 单独一行
    mask_edge = mtype & mnode

    # ---- 6) 边修改（weight/confidence）----
    ew_mult = float(action_row.get("affect_edge_weight_mult", 1.0))
    ec_mult = float(action_row.get("affect_edge_conf_mult", 1.0))

    edges.loc[mask_edge, "weight"] = edges.loc[mask_edge, "weight"].fillna(1.0).astype(float) * ew_mult
    edges.loc[mask_edge, "confidence"] = edges.loc[mask_edge, "confidence"].fillna(0.0).astype(float) * ec_mult

    return nodes, edges


def greedy_optimize(model, device, base_nodes, base_edges, actions_df, budget: float):
    selected = []
    remaining = float(budget)

    cur_nodes, cur_edges = base_nodes.copy(), base_edges.copy()
    cur_cor = predict_cor(model, device, cur_nodes, cur_edges)

    while True:
        best = None
        best_ratio = 0.0
        best_next = None

        for _, a in actions_df.iterrows():
            aid = str(a.get("action_id", ""))
            if not aid or aid in selected:
                continue

            cost = float(a.get("cost", 0.0))
            if cost > remaining:
                continue

            n2, e2 = apply_action(cur_nodes, cur_edges, a)
            cor2 = predict_cor(model, device, n2, e2)

            gain = max(0.0, cur_cor - cor2)  # COR 越小越好
            ratio = gain / (cost + 1e-9)

            if ratio > best_ratio and gain > 0:
                best_ratio = ratio
                best = (aid, cost, gain, cor2)
                best_next = (n2, e2)

        if best is None:
            break

        aid, cost, gain, cor2 = best
        selected.append(aid)
        remaining -= cost
        cur_nodes, cur_edges = best_next
        cur_cor = cor2

    return selected, float(budget) - remaining, cur_cor


def main(cfg_path, snapshots_dir, actions_csv, ckpt_path, project_id, time_ym, budget):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(ckpt_path, device)
    model.eval()

    base_nodes, base_edges = load_snapshot(snapshots_dir, project_id, time_ym)
    base_cor = predict_cor(model, device, base_nodes, base_edges)

    actions_df = pd.read_csv(actions_csv)
    actions_df.columns = actions_df.columns.astype(str).str.strip()

    selected, used, new_cor = greedy_optimize(model, device, base_nodes, base_edges, actions_df, budget)

    print("====== OPT RESULT ======")
    print(f"project: {project_id} time: {time_ym}")
    print(f"baseline COR pred: {base_cor:.8f}")
    print(f"after-intervention COR pred: {new_cor:.8f}")
    print(f"ΔCOR: {(new_cor - base_cor):.8e}")
    print(f"budget: {budget:.1f} used: {used:.1f}")
    print(f"selected actions: {selected}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    ap.add_argument("--snapshots", default="data/snapshots")
    ap.add_argument("--actions", default="data/actions.csv")
    ap.add_argument("--ckpt", default="data/interim/gnn_ckpt.pt")
    ap.add_argument("--project_id", required=True)
    ap.add_argument("--time", dest="time_ym", required=True)   # YYYY-MM
    ap.add_argument("--budget", type=float, required=True)
    args = ap.parse_args()

    main(args.cfg, args.snapshots, args.actions, args.ckpt, args.project_id, args.time_ym, args.budget)
