# src/_06_simulate_rollout.py
import os, argparse
import pandas as pd
import torch
from src.gnn_utils import load_checkpoint
from src._05_optimize_policy import load_snapshot, predict_cor, apply_action, parse_list_field

def list_times(snapshots_dir: str, project_id: str):
    pdir = os.path.join(snapshots_dir, project_id)
    ts = [d for d in os.listdir(pdir) if os.path.isdir(os.path.join(pdir, d))]
    return sorted(ts)

def load_actions(actions_csv: str):
    return pd.read_csv(actions_csv)

def policy_none():
    return []

def policy_fixed_topk(actions_df, k=2):
    # 演示：固定选前k个动作（你可以换成中心性Top-K等baseline）
    return actions_df["action_id"].tolist()[:k]

def policy_optimize(model, device, nodes, edges, actions_df, budget):
    # 复用05的贪心逻辑：逐步挑选并更新图
    selected = []
    remaining = budget
    cur_nodes, cur_edges = nodes.copy(), edges.copy()
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
            gain = max(0.0, cur_cor - cor2)
            ratio = gain / (cost + 1e-9)
            if ratio > best_ratio and gain > 0:
                best_ratio = ratio
                best = (a["action_id"], cost, cor2)
                best_next = (n2, e2)

        if best is None:
            break
        aid, cost, cor2 = best
        selected.append(aid)
        remaining -= cost
        cur_nodes, cur_edges = best_next
        cur_cor = cor2

    return selected, cur_cor

def main(snapshots_dir, actions_csv, ckpt, project_id, budget, out_csv, policy):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(ckpt, device)
    actions_df = load_actions(actions_csv)

    times = list_times(snapshots_dir, project_id)
    rows = []

    for t in times:
        nodes, edges = load_snapshot(snapshots_dir, project_id, t)
        base_cor = predict_cor(model, device, nodes, edges)

        if policy == "none":
            selected = []
            cor_after = base_cor
        elif policy == "fixed2":
            selected = policy_fixed_topk(actions_df, k=2)
            # apply fixed actions sequentially
            n2, e2 = nodes.copy(), edges.copy()
            for aid in selected:
                arow = actions_df[actions_df["action_id"] == aid].iloc[0]
                n2, e2 = apply_action(n2, e2, arow)
            cor_after = predict_cor(model, device, n2, e2)
        elif policy == "optimize":
            selected, cor_after = policy_optimize(model, device, nodes, edges, actions_df, budget)
        else:
            raise ValueError("policy must be none|fixed2|optimize")

        rows.append({
            "time": t,
            "cor_pred_baseline": base_cor,
            "cor_pred_after": cor_after,
            "delta_cor": base_cor - cor_after,
            "selected_actions": ";".join(selected)
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved rollout to {out_csv}")
    print(df.head())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default="data/snapshots")
    ap.add_argument("--actions", default="data/actions.csv")
    ap.add_argument("--ckpt", default="data/interim/gnn_ckpt.pt")
    ap.add_argument("--project_id", required=True)
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--policy", default="optimize")  # none|fixed2|optimize
    ap.add_argument("--out", default="data/interim/rollout.csv")
    args = ap.parse_args()

    main(args.snapshots, args.actions, args.ckpt, args.project_id, args.budget, args.out, args.policy)
