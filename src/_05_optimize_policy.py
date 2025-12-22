# src/_05_optimize_policy.py
import os, argparse, yaml
import pandas as pd
import numpy as np
import torch

from src.gnn_utils import build_heterodata_from_dfs, load_checkpoint

def load_snapshot(snapshots_dir: str, project_id: str, time_ym: str):
    tdir = os.path.join(snapshots_dir, project_id, time_ym)
    if not os.path.isdir(tdir):
        raise FileNotFoundError(f"Snapshot not found: {tdir}")
    nodes = pd.read_csv(os.path.join(tdir, "nodes.csv"))
    edges = pd.read_csv(os.path.join(tdir, "edges.csv"))
    return nodes, edges

def predict_cor(model, device, nodes_df, edges_df) -> float:
    data = build_heterodata_from_dfs(nodes_df, edges_df, cor_label=None).to(device)
    with torch.no_grad():
        pred = model(data).detach().cpu().numpy()
    return float(pred[0])

def parse_list_field(s: str):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    return [x.strip() for x in str(s).split(";") if x.strip()]

def apply_action(nodes_df, edges_df, action_row):
    nodes = nodes_df.copy()
    edges = edges_df.copy()

    tax_list = parse_list_field(action_row.get("target_risk_taxonomy", ""))

    # 1) 命中的 Risk 节点
    mask_risk = (nodes["node_type"] == "Risk")
    if tax_list:
        mask_risk = mask_risk & (nodes["taxonomy"].fillna("").astype(str).isin(tax_list))
    target_nodes = set(nodes.loc[mask_risk, "node_id"].astype(str).tolist())

    # 2) 节点修改（prob/sev/conf 三者都动）
    p_mult = float(action_row.get("affect_node_prob_mult", 1.0))
    s_mult = float(action_row.get("affect_node_sev_mult", 1.0))
    c_mult = float(action_row.get("affect_node_conf_mult", 1.0))

    nodes.loc[mask_risk, "probability"] = nodes.loc[mask_risk, "probability"].fillna(0.0) * p_mult
    nodes.loc[mask_risk, "severity_num"] = nodes.loc[mask_risk, "severity_num"].fillna(0.0) * s_mult
    nodes.loc[mask_risk, "confidence"] = nodes.loc[mask_risk, "confidence"].fillna(0.0) * c_mult

    # 3) 边修改（命中风险节点相关边 + 指定边类型）
    edge_types = parse_list_field(action_row.get("target_edge_types", ""))
    if edge_types:
        mtype = edges["edge_type"].isin(edge_types)
    else:
        mtype = pd.Series([True] * len(edges))

    mnode = edges["src"].astype(str).isin(target_nodes) | edges["dst"].astype(str).isin(target_nodes)
    mask_edge = mtype & mnode

    ew_mult = float(action_row.get("affect_edge_weight_mult", 1.0))
    ec_mult = float(action_row.get("affect_edge_conf_mult", 1.0))
    edges.loc[mask_edge, "weight"] = edges.loc[mask_edge, "weight"].fillna(1.0) * ew_mult
    edges.loc[mask_edge, "confidence"] = edges.loc[mask_edge, "confidence"].fillna(0.0) * ec_mult

    return nodes, edges

def greedy_optimize(model, device, base_nodes, base_edges, actions_df, budget: float):
    selected = []
    remaining = budget

    cur_nodes, cur_edges = base_nodes.copy(), base_edges.copy()
    cur_cor = predict_cor(model, device, cur_nodes, cur_edges)

    while True:
        best = None
        best_ratio = 0.0
        best_next = None

        for _, a in actions_df.iterrows():
            if a["action_id"] in selected:
                continue
            cost = float(a["cost"])
            if cost > remaining:
                continue

            n2, e2 = apply_action(cur_nodes, cur_edges, a)
            cor2 = predict_cor(model, device, n2, e2)
            gain = max(0.0, cur_cor - cor2)  # ΔCOR
            ratio = gain / (cost + 1e-9)

            if ratio > best_ratio and gain > 0:
                best_ratio = ratio
                best = (a["action_id"], cost, gain, cor2)
                best_next = (n2, e2)

        if best is None:
            break

        aid, cost, gain, cor2 = best
        selected.append(aid)
        remaining -= cost
        cur_nodes, cur_edges = best_next
        cur_cor = cor2

    return selected, budget - remaining, cur_cor

def main(cfg_path, snapshots_dir, actions_csv, ckpt_path, project_id, time_ym, budget):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(ckpt_path, device)

    base_nodes, base_edges = load_snapshot(snapshots_dir, project_id, time_ym)
    base_cor = predict_cor(model, device, base_nodes, base_edges)

    actions_df = pd.read_csv(actions_csv)
    selected, used, new_cor = greedy_optimize(model, device, base_nodes, base_edges, actions_df, budget)

    print("====== OPT RESULT ======")
    print("project:", project_id, "time:", time_ym)
    print("baseline COR pred:", round(base_cor, 4))
    print("budget:", budget, "used:", used)
    print("selected actions:", selected)
    print("after-intervention COR pred:", round(new_cor, 4))
    print("ΔCOR:", round(base_cor - new_cor, 4))

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
