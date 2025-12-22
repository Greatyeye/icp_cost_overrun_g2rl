# src/02_entity_link.py
import os, json, argparse, yaml
from typing import Dict, List, Tuple
from difflib import SequenceMatcher

def fuzzy_ratio(a: str, b: str) -> int:
    return int(100 * SequenceMatcher(None, a.lower(), b.lower()).ratio())

def try_load_sbert():
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        return SentenceTransformer, np
    except Exception:
        return None, None

def cosine_sim(a, b, np):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def main(cfg_path: str, in_jsonl: str, out_jsonl: str):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    th_hi = cfg["linking"]["cosine_merge_hi"]
    fuzzy_hi = cfg["linking"]["fuzzy_merge_hi"]

    SentenceTransformer, np = try_load_sbert()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") if SentenceTransformer else None

    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

    # global canonical pool per project per type
    canonical: Dict[Tuple[str,str], List[Dict]] = {}  # (project_id, type) -> list of {"cid","name","emb"}

    def get_cid(project_id: str, etype: str, name: str):
        key = (project_id, etype)
        pool = canonical.setdefault(key, [])
        # 1) fuzzy quick merge
        for item in pool:
            if fuzzy_ratio(item["name"], name) >= fuzzy_hi:
                return item["cid"]

        # 2) embedding merge
        if model is not None:
            emb = model.encode([name], normalize_embeddings=True)[0]
            best = (None, -1.0)
            for item in pool:
                sim = cosine_sim(emb, item["emb"], np)
                if sim > best[1]:
                    best = (item, sim)
            if best[0] is not None and best[1] >= th_hi:
                return best[0]["cid"]
        else:
            emb = None

        cid = f"{etype.lower()}::{len(pool)+1}::{name.strip().lower()[:30]}"
        pool.append({"cid": cid, "name": name, "emb": emb})
        return cid

    with open(in_jsonl, "r", encoding="utf-8") as r, open(out_jsonl, "w", encoding="utf-8") as w:
        for line in r:
            rec = json.loads(line)
            pid = rec["project_id"]

            # map temp_id -> canonical_id
            temp2cid = {}
            for ent in rec.get("entities", []):
                tid = ent["temp_id"]
                etype = ent["type"]
                name = ent["name"]
                cid = get_cid(pid, etype, name)
                ent["canonical_id"] = cid
                temp2cid[tid] = cid

            # rewrite triples
            for tri in rec.get("triples", []):
                tri["head_canonical_id"] = temp2cid.get(tri["head_temp_id"])
                tri["tail_canonical_id"] = temp2cid.get(tri["tail_temp_id"])

            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved canonical extraction to {out_jsonl}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    ap.add_argument("--in", dest="inp", default="data/interim/extraction.jsonl")
    ap.add_argument("--out", default="data/interim/canonical_extraction.jsonl")
    args = ap.parse_args()
    main(args.cfg, args.inp, args.out)
