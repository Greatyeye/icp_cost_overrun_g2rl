# src/01_llm_extract.py
import os, json, argparse, time, yaml
from typing import Set
from src.common import (
    LLMConfig, OpenAICompatibleClient,
    SYSTEM_PROMPT, ENTITIES_SCHEMA_DESC, TRIPLES_SCHEMA_DESC,
    build_user_prompt_entities, extract_first_json_obj
)

def load_done_ids(out_jsonl: str) -> Set[str]:
    done = set()
    if not os.path.exists(out_jsonl):
        return done
    with open(out_jsonl, "r", encoding="utf-8") as r:
        for line in r:
            try:
                obj = json.loads(line)
                done.add(obj["chunk_id"])
            except Exception:
                continue
    return done

def safe_chat(client, system, user, temperature=0.0, max_tries=8, base_sleep=2.0):
    """
    针对 429/5xx 做指数退避重试
    """
    last_err = None
    for i in range(max_tries):
        try:
            return client.chat(system, user, temperature=temperature)
        except Exception as e:
            last_err = e
            msg = str(e)
            # 常见限流/服务繁忙
            if ("429" in msg) or ("Too Many Requests" in msg) or ("503" in msg) or ("rate" in msg.lower()):
                sleep = base_sleep * (2 ** i)
                # 上限，避免无限等
                sleep = min(sleep, 120)
                print(f"[WARN] LLM rate-limited/busy. retry in {sleep:.1f}s | err={msg[:120]}")
                time.sleep(sleep)
                continue
            # 其它错误直接抛
            raise
    raise RuntimeError(f"LLM call failed after retries: {last_err}")

def main(cfg_path: str, chunks_jsonl: str, out_jsonl: str, min_interval_sec: float = 1.2):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    llm_cfg = LLMConfig(**cfg["llm"])
    client = OpenAICompatibleClient(llm_cfg)

    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    done_ids = load_done_ids(out_jsonl)
    print(f"Resume mode: already done chunks = {len(done_ids)}")

    last_call_ts = 0.0

    def throttle():
        nonlocal last_call_ts
        now = time.time()
        gap = now - last_call_ts
        if gap < min_interval_sec:
            time.sleep(min_interval_sec - gap)
        last_call_ts = time.time()

    with open(chunks_jsonl, "r", encoding="utf-8") as r, open(out_jsonl, "a", encoding="utf-8") as w:
        for line in r:
            rec = json.loads(line)
            if rec["chunk_id"] in done_ids:
                continue

            user_json = build_user_prompt_entities(
                rec["project_id"], rec["doc_id"], rec["chunk_id"], rec["chunk_date"], rec["text"]
            )

            # A) entities
            userA = user_json + "\n\n" + ENTITIES_SCHEMA_DESC
            throttle()
            outA = safe_chat(client, SYSTEM_PROMPT, userA, temperature=0.0)
            entities_obj = extract_first_json_obj(outA)
            entities = entities_obj.get("entities", [])

            # B) triples
            userB_payload = {
                "project_id": rec["project_id"],
                "doc_id": rec["doc_id"],
                "chunk_id": rec["chunk_id"],
                "chunk_date": rec["chunk_date"],
                "text": rec["text"],
                "entities": entities
            }
            userB = json.dumps(userB_payload, ensure_ascii=False) + "\n\n" + TRIPLES_SCHEMA_DESC
            throttle()
            outB = safe_chat(client, SYSTEM_PROMPT, userB, temperature=0.0)
            triples_obj = extract_first_json_obj(outB)
            triples = triples_obj.get("triples", [])

            wrec = {
                "project_id": rec["project_id"],
                "doc_id": rec["doc_id"],
                "chunk_id": rec["chunk_id"],
                "chunk_date": rec["chunk_date"],
                "entities": entities,
                "triples": triples
            }
            w.write(json.dumps(wrec, ensure_ascii=False) + "\n")
            w.flush()

    print(f"Saved extraction to {out_jsonl}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    ap.add_argument("--chunks", default="data/interim/chunks.jsonl")
    ap.add_argument("--out", default="data/interim/extraction.jsonl")
    ap.add_argument("--min_interval", type=float, default=1.2)
    args = ap.parse_args()
    main(args.cfg, args.chunks, args.out, min_interval_sec=args.min_interval)
