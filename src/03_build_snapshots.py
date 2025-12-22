# src/03_build_snapshots.py
import os, json, argparse, yaml, re
from collections import defaultdict
import pandas as pd

NODE_COLS = [
    "node_id", "node_type", "name", "taxonomy",
    "confidence", "probability", "severity_num", "project_id"
]
EDGE_COLS = [
    "src", "dst", "edge_type", "weight", "confidence",
    "evidence_id", "project_id", "time"
]

def month_index(date_str: str) -> str:
    return str(date_str)[:7]  # YYYY-MM

def severity_to_num(sev: str) -> float:
    if not sev:
        return 0.0
    sev = str(sev).lower().strip()
    return {"low": 0.33, "medium": 0.66, "high": 1.0}.get(sev, 0.0)

def norm_type(t: str) -> str:
    """把 node_type 规范化成安全前缀，如 Risk -> risk, 'Cost Overrun' -> cost_overrun"""
    t = (t or "unknown").strip().lower()
    t = re.sub(r"\W+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t or "unknown"

def ensure_string_id(raw_id, node_type: str) -> str:
    """
    关键：确保 node_id/src/dst 不是纯数字。
    若 canonical_id = 334（或 "334"），会变成 "risk::334" 这类字符串，
    防止下游误把它当作“已经编码好的索引”直接用。
    """
    if raw_id is None:
        raw = "null"
    else:
        raw = str(raw_id).strip()

    # 如果已经带了命名空间（包含 ::），直接用（通常已是 'risk::xxxx'）
    if "::" in raw:
        return raw

    prefix = norm_type(node_type)
    return f"{prefix}::{raw}"

def main(cfg_path: str, in_jsonl: str, out_dir: str):
    _ = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))  # 目前不强依赖 cfg，但保留接口
    os.makedirs(out_dir, exist_ok=True)

    # (project_id, t) -> {node_id: node_row}
    nodes = defaultdict(dict)
    # (project_id, t) -> [edge_row...]
    edges = defaultdict(list)
    # evidence_id -> evidence record
    evidence = {}

    # (project_id, t) -> {canonical_id_raw(str): node_id(str)}
    # 用于 triples 里把 head_canonical_id / tail_canonical_id 映射成我们写入图里的 node_id
    cid2nid = defaultdict(dict)

    with open(in_jsonl, "r", encoding="utf-8") as r:
        for line in r:
            rec = json.loads(line)
            pid = str(rec.get("project_id"))
            t = month_index(rec.get("chunk_date"))
            key = (pid, t)

            # 1) Project 节点（固定非数字 id）
            proj_node = f"project::{pid}"
            nodes[key][proj_node] = {
                "node_id": proj_node,
                "node_type": "Project",
                "name": pid,
                "taxonomy": "",
                "confidence": 1.0,
                "probability": 0.0,
                "severity_num": 0.0,
                "project_id": pid
            }

            # 2) entities -> nodes
            for ent in rec.get("entities", []) or []:
                raw_cid = ent.get("canonical_id")
                if raw_cid is None:
                    continue

                etype = ent.get("type") or "Unknown"
                nid = ensure_string_id(raw_cid, etype)

                attrs = ent.get("attributes", {}) or {}
                prob = float(attrs.get("probability") or 0.0)
                sev_num = severity_to_num(attrs.get("severity"))
                conf = float(ent.get("confidence") or 0.0)

                nodes[key][nid] = {
                    "node_id": nid,
                    "node_type": etype,
                    "name": ent.get("name", "") or "",
                    "taxonomy": ent.get("taxonomy") or "",
                    "confidence": conf,
                    "probability": prob,
                    "severity_num": sev_num,
                    "project_id": pid
                }

                # 保存 canonical_id_raw -> node_id 映射（供 triples 用）
                cid2nid[key][str(raw_cid).strip()] = nid

                # 连接 Project -> entity（帮助池化/读图）
                edges[key].append({
                    "src": proj_node,
                    "dst": nid,
                    "edge_type": "ASSOCIATED_WITH",
                    "weight": 1.0,
                    "confidence": conf,
                    "evidence_id": None,
                    "project_id": pid,
                    "time": t
                })

            # 3) triples -> edges
            for tri in rec.get("triples", []) or []:
                h_raw = tri.get("head_canonical_id")
                t_raw = tri.get("tail_canonical_id")
                if h_raw is None or t_raw is None:
                    continue

                h_raw_s = str(h_raw).strip()
                t_raw_s = str(t_raw).strip()

                # 优先用 entities 阶段建立的映射；找不到则兜底：假设 tri 自己已带 namespace
                h = cid2nid[key].get(h_raw_s, h_raw_s)
                ta = cid2nid[key].get(t_raw_s, t_raw_s)

                # 依然要确保非纯数字（双保险）
                # 这里不知道 node_type，就用 unknown 前缀；但如果它本身含 :: 就不会变
                h = ensure_string_id(h, "Unknown")
                ta = ensure_string_id(ta, "Unknown")

                ev = tri.get("evidence", {}) or {}
                evid = f"ev::{rec.get('doc_id')}::{rec.get('chunk_id')}::{abs(hash(ev.get('quote','')))%10**9}"

                evidence[evid] = {
                    "evidence_id": evid,
                    "doc_id": rec.get("doc_id"),
                    "chunk_id": rec.get("chunk_id"),
                    "date": rec.get("chunk_date"),
                    "quote": ev.get("quote", ""),
                    "char_start": ev.get("char_start", None),
                    "char_end": ev.get("char_end", None),
                }

                wh = (tri.get("weight_hint") or "").lower().strip()
                w = {"low": 0.3, "medium": 0.6, "high": 1.0}.get(wh, 0.5)
                conf = float(tri.get("confidence") or 0.0)

                edges[key].append({
                    "src": h,
                    "dst": ta,
                    "edge_type": tri.get("relation") or "RELATED_TO",
                    "weight": float(w),
                    "confidence": conf,
                    "evidence_id": evid,
                    "project_id": pid,
                    "time": t
                })

    # 4) 写 snapshots：保证即使空边也写表头；节点稳定排序，便于复现
    for (pid, t), node_map in nodes.items():
        tdir = os.path.join(out_dir, pid, t)
        os.makedirs(tdir, exist_ok=True)

        node_rows = list(node_map.values())
        node_rows = sorted(node_rows, key=lambda r: (str(r.get("node_type","")), str(r.get("node_id",""))))

        edge_rows = edges.get((pid, t), [])
        # 也做稳定排序，便于调试复现
        edge_rows = sorted(edge_rows, key=lambda e: (str(e.get("edge_type","")), str(e.get("src","")), str(e.get("dst",""))))

        pd.DataFrame.from_records(node_rows, columns=NODE_COLS) \
            .to_csv(os.path.join(tdir, "nodes.csv"), index=False)

        pd.DataFrame.from_records(edge_rows, columns=EDGE_COLS) \
            .to_csv(os.path.join(tdir, "edges.csv"), index=False)

    with open(os.path.join(out_dir, "evidence.jsonl"), "w", encoding="utf-8") as w:
        for _, ev in evidence.items():
            w.write(json.dumps(ev, ensure_ascii=False) + "\n")

    print(f"Saved snapshots to: {out_dir}/<project_id>/<YYYY-MM>/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    ap.add_argument("--in", dest="inp", default="data/interim/canonical_extraction.jsonl")
    ap.add_argument("--out_dir", default="data/snapshots")
    args = ap.parse_args()
    main(args.cfg, args.inp, args.out_dir)
