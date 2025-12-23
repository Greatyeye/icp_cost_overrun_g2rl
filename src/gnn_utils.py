# src/gnn_utils.py
from __future__ import annotations
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear
from collections import defaultdict
from pandas.errors import EmptyDataError
from pandas.errors import EmptyDataError
import hashlib

def stable_hash(s: str, mod: int = 10**6) -> int:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]
    return int(h, 16) % mod

def taxonomy_hash(tax: str) -> float:
    tax = (tax or "").strip().lower()
    # 用稳定 hash，保证跨进程一致
    return stable_hash(tax, mod=1000) / 1000.0

def build_heterodata_from_dfs(nodes: pd.DataFrame, edges: pd.DataFrame, cor_label: float | None = None) -> HeteroData:
    data = HeteroData()

    # node_type map & index
    idx = {}
    for ntype, sub in nodes.groupby("node_type", sort=True):
        sub = sub.copy()
        sub["node_id"] = sub["node_id"].astype(str).str.strip()
        sub = sub.sort_values("node_id", kind="mergesort").reset_index(drop=True)
        idx[ntype] = {nid: i for i, nid in enumerate(sub["node_id"].tolist())}

        conf = torch.tensor(sub.get("confidence", 0.0).fillna(0.0).values, dtype=torch.float).view(-1, 1)
        prob = torch.tensor(sub.get("probability", 0.0).fillna(0.0).values, dtype=torch.float).view(-1, 1)
        sev  = torch.tensor(sub.get("severity_num", 0.0).fillna(0.0).values, dtype=torch.float).view(-1, 1)
        taxh = torch.tensor([taxonomy_hash(x) for x in sub.get("taxonomy", "").fillna("").astype(str).values],
                            dtype=torch.float).view(-1, 1)

        x = torch.cat([conf, prob, sev, taxh], dim=1)  # 4维最小特征
        data[ntype].x = x
        data[ntype].node_id = sub["node_id"].astype(str).tolist()

    for ntype in idx.keys():
        n = len(idx[ntype])
        if n <= 0:
            continue
        loop = torch.arange(n, dtype=torch.long)
        data[(ntype, "SELF_LOOP", ntype)].edge_index = torch.stack([loop, loop], dim=0)
        data[(ntype, "SELF_LOOP", ntype)].edge_attr = torch.zeros((n, 2), dtype=torch.float)  # edge_dim=2 对齐
    # edge types
    if edges is None or len(edges) == 0 or "edge_type" not in edges.columns:
        pass
    else:
        node_type_map = dict(zip(
            nodes["node_id"].astype(str).str.strip(),
            nodes["node_type"].astype(str).str.strip()
        ))

        buckets = defaultdict(list)  # (src_type, edge_type, dst_type) -> [(s_idx, d_idx, w, c), ...]

        for _, r in edges.iterrows():
            etype = str(r.get("edge_type", "")).strip()
            if not etype:
                continue

            s, d = str(r.get("src", "")).strip(), str(r.get("dst", "")).strip()
            st = node_type_map.get(s)
            dt = node_type_map.get(d)
            if st is None or dt is None:
                continue
            if s not in idx.get(st, {}) or d not in idx.get(dt, {}):
                continue

            w = float(r.get("weight", 1.0))
            c = float(r.get("confidence", 0.0))
            buckets[(st, etype, dt)].append((idx[st][s], idx[dt][d], w, c))

        for (st, etype, dt), triples in buckets.items():
            if not triples:
                continue

            src_i = torch.tensor([a for a, _, _, _ in triples], dtype=torch.long)
            dst_i = torch.tensor([b for _, b, _, _ in triples], dtype=torch.long)
            edge_index = torch.stack([src_i, dst_i], dim=0)
            edge_attr = torch.tensor([[w, c] for *_, w, c in triples], dtype=torch.float)

            data[(st, etype, dt)].edge_index = edge_index
            data[(st, etype, dt)].edge_attr = edge_attr

            # 可选：加反向边，消掉 “Project/Action 不作为 dst 不更新” 的 warning
            rev = f"rev_{etype}"
            data[(dt, rev, st)].edge_index = torch.stack([dst_i, src_i], dim=0)
            data[(dt, rev, st)].edge_attr = edge_attr

    if cor_label is not None and "Project" in data.node_types:
        data["Project"].y = torch.full((data["Project"].num_nodes,), float(cor_label), dtype=torch.float)

    return data

def load_graph_from_snapshot_dir(tdir: str, cor_label: float | None = None) -> HeteroData:
    nodes = pd.read_csv(os.path.join(tdir, "nodes.csv"), dtype={"node_id": str, "node_type": str})
    edges_path = os.path.join(tdir, "edges.csv")
    try:
        edges = pd.read_csv(edges_path, dtype={"src": str, "dst": str, "edge_type": str})
    except (FileNotFoundError, EmptyDataError):
        edges = pd.DataFrame(columns=["src", "dst", "edge_type", "weight", "confidence"])
    return build_heterodata_from_dfs(nodes, edges, cor_label=cor_label)

class HeteroGATModel(torch.nn.Module):
    def __init__(self, metadata, hidden_dim=64):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()
        for ntype in metadata[0]:
            self.lin_dict[ntype] = Linear(-1, hidden_dim)

        # 每种关系一个GATv2Conv，edge_dim=2 让边权/置信度参与注意力
        convs1 = {}
        convs2 = {}
        for et in metadata[1]:
            convs1[et] = GATv2Conv((-1, -1), hidden_dim, heads=2, concat=False, edge_dim=2, add_self_loops=False)
            convs2[et] = GATv2Conv((-1, -1), hidden_dim, heads=2, concat=False, edge_dim=2, add_self_loops=False)

        self.conv1 = HeteroConv(convs1, aggr="sum")
        self.conv2 = HeteroConv(convs2, aggr="sum")
        self.out = Linear(hidden_dim, 1)

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = {ntype: self.lin_dict[ntype](data[ntype].x) for ntype in data.node_types}

        edge_attr_dict = {}
        for et in data.edge_types:
            edge_attr_dict[et] = data[et].edge_attr

        x_dict = self.conv1(x_dict, data.edge_index_dict, edge_attr_dict=edge_attr_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, data.edge_index_dict, edge_attr_dict=edge_attr_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        # 每个图只有一个Project节点（我们按 project_id 建快照）
        yhat = self.out(x_dict["Project"]).view(-1)
        return yhat

def save_checkpoint(path: str, model: torch.nn.Module, metadata, cfg: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata, "cfg": cfg}, path)

def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model = HeteroGATModel(ckpt["metadata"], hidden_dim=ckpt["cfg"]["gnn"]["hidden_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
