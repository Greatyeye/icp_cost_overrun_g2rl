# src/04_train_gnn.py
import os, argparse, yaml, random
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from src.gnn_utils import load_graph_from_snapshot_dir, HeteroGATModel, save_checkpoint
import copy


SEED = 42

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def stratified_split_by_cor(labels_df: pd.DataFrame, seed: int = 42,
                            train_per_bin: int = 8, val_per_bin: int = 1, test_per_bin: int = 1):
    """
    默认 q=3 个桶（每桶约 10 个项目），每桶 8/1/1 -> 总计 24/3/3
    """
    df = labels_df[["project_id", "COR"]].copy()
    df["project_id"] = df["project_id"].astype(str)
    df["COR"] = df["COR"].astype(float)

    # 分层桶：3 等分
    df["bin"] = pd.qcut(df["COR"], q=3, labels=False, duplicates="drop")

    n_bins = df["bin"].nunique()
    if n_bins < 3:
        # 极端情况：值重复太多导致桶数不足，退化为普通 shuffle
        pids = df["project_id"].tolist()
        rnd = random.Random(seed)
        rnd.shuffle(pids)
        return set(pids[:24]), set(pids[24:27]), set(pids[27:30])

    train, val, test = [], [], []
    for b, sub in df.groupby("bin"):
        ids = sub["project_id"].tolist()
        rnd = random.Random(seed + int(b))
        rnd.shuffle(ids)

        need = train_per_bin + val_per_bin + test_per_bin
        if len(ids) < need:
            raise RuntimeError(f"Bin {b} has only {len(ids)} projects, need {need}")

        train += ids[:train_per_bin]
        val   += ids[train_per_bin:train_per_bin + val_per_bin]
        test  += ids[train_per_bin + val_per_bin:train_per_bin + val_per_bin + test_per_bin]

    return set(train), set(val), set(test)

def eval_mae(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).view(-1)
            y = batch["Project"].y.to(device).view(-1)
            total += float(F.l1_loss(pred, y, reduction="sum").item())
            n += int(y.numel())
    return total / max(n, 1)

def main(cfg_path: str, snapshots_dir: str, labels_csv: str, out_ckpt: str):
    set_seed(SEED)
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

    labels_df = pd.read_csv(labels_csv)
    labels_df.columns = [c.strip() for c in labels_df.columns]
    cor_col = next((c for c in labels_df.columns if c.lower() == "cor"), None)
    if cor_col is None:
        raise KeyError(f"labels.csv must contain COR column. got={list(labels_df.columns)}")
    labels_df = labels_df.rename(columns={cor_col: "COR"})

    labels = dict(zip(labels_df["project_id"].astype(str), labels_df["COR"].astype(float)))

    # 只用：snapshots 里存在且 labels 里也有的项目
    snapshot_pids = sorted([d for d in os.listdir(snapshots_dir)
                            if os.path.isdir(os.path.join(snapshots_dir, d))])
    usable_pids = [pid for pid in snapshot_pids if pid in labels]

    if len(usable_pids) < 30:
        print(f"[warn] usable projects = {len(usable_pids)} (expect ~30). missing labels or snapshots?")

    # 分层划分（24/3/3）
    train_pids, val_pids, test_pids = stratified_split_by_cor(labels_df[labels_df["project_id"].astype(str).isin(usable_pids)],
                                                              seed=SEED)

    print(f"Split projects | train={len(train_pids)} val={len(val_pids)} test={len(test_pids)}")
    print("train:", sorted(train_pids))
    print("val  :", sorted(val_pids))
    print("test :", sorted(test_pids))

    train_graphs, val_graphs, test_graphs = [], [], []

    # snapshots_dir/<project_id>/<YYYY-MM>/
    for pid in sorted(os.listdir(snapshots_dir)):
        pdir = os.path.join(snapshots_dir, pid)
        if not os.path.isdir(pdir):
            continue
        if pid not in labels:
            continue

        target = None
        if pid in train_pids: target = train_graphs
        elif pid in val_pids: target = val_graphs
        elif pid in test_pids: target = test_graphs
        else:
            continue

        for t in sorted(os.listdir(pdir)):
            tdir = os.path.join(pdir, t)
            if not os.path.isdir(tdir):
                continue
            g = load_graph_from_snapshot_dir(tdir, cor_label=labels[pid])
            target.append(g)

    if not train_graphs or not val_graphs or not test_graphs:
        raise RuntimeError(f"Empty split graphs: train={len(train_graphs)} val={len(val_graphs)} test={len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=cfg["gnn"]["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_graphs, batch_size=1, shuffle=False)
    test_loader  = DataLoader(test_graphs, batch_size=1, shuffle=False)

    # 用所有图的 union metadata，防止某些关系只在 val/test 出现
    all_graphs = train_graphs + val_graphs + test_graphs
    node_types = sorted(set().union(*[set(g.node_types) for g in all_graphs]))
    edge_types = sorted(set().union(*[set(g.edge_types) for g in all_graphs]))
    metadata = (node_types, edge_types)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeteroGATModel(metadata, hidden_dim=cfg["gnn"]["hidden_dim"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["gnn"]["lr"])

    best_val = float("inf")
    best_state = None

    for epoch in range(cfg["gnn"]["epochs"]):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch).view(-1)
            y = batch["Project"].y.to(device).view(-1)
            loss = F.l1_loss(pred, y)  # MAE(mean)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())

        if (epoch + 1) % 10 == 0:
            train_mae = total / len(train_loader)
            val_mae = eval_mae(model, val_loader, device)
            print(f"epoch {epoch+1:03d} | train_MAE={train_mae:.4f} | val_MAE={val_mae:.4f}")

            if val_mae < best_val:
                best_val = val_mae
                save_checkpoint(out_ckpt, model, metadata, cfg)
                print(f"  ✅ saved best ckpt (val_MAE={best_val:.4f}) -> {out_ckpt}")

    # 最终 test
    test_mae = eval_mae(model, test_loader, device)
    print(f"Test MAE = {test_mae:.4f}")
    print(f"Best checkpoint saved at: {out_ckpt}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    ap.add_argument("--snapshots", default="data/snapshots")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--out_ckpt", default="data/interim/gnn_ckpt.pt")
    args = ap.parse_args()
    main(args.cfg, args.snapshots, args.labels, args.out_ckpt)
