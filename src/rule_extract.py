# src/rule_extract.py
from __future__ import annotations

import os
import re
import json
import argparse
import hashlib
from typing import Dict, List, Tuple, Optional

# 复用你项目里 common 的工具（保持一致）
from src.common import extract_first_json_obj  # 若你没用到也不影响
# 注意：rule_extract 不需要 LLM

# -----------------------------
# 规则/词典（你后续可以自己扩展）
# -----------------------------
RISK_TAXONOMY: Dict[str, List[str]] = {
    "Price/Inflation/FX": [
        r"\binflation\b", r"\bprice(s)?\s+(increase|rise|hike)\b", r"\bcurrency\b",
        r"\bdepreciation\b", r"\bdevaluation\b", r"\bexchange\s+rate\b", r"\bfx\b",
        r"\bforeign\s+exchange\b", r"\bimport(ed)?\s+cost(s)?\b"
    ],
    "Procurement & Supply": [
        r"\bprocurement\b", r"\btender\b", r"\bbid(ding)?\b", r"\baward\b",
        r"\bsupplier(s)?\b", r"\bshipment(s)?\b", r"\bmaterials?\s+shortage\b",
        r"\bsupply\s+chain\b", r"\blogistics\b", r"\bdelivery\s+delay\b"
    ],
    "Design & Scope": [
        r"\bdesign\s+change(s)?\b", r"\bscope\s+change(s)?\b", r"\bvariation(s)?\b",
        r"\bchange\s+order(s)?\b", r"\badditional\s+works?\b", r"\bre-design\b"
    ],
    "Land & Resettlement": [
        r"\bland\s+acquisition\b", r"\bresettlement\b", r"\bright\s+of\s+way\b",
        r"\bexpropriation\b"
    ],
    "Contractor Performance": [
        r"\bcontractor\b", r"\bpoor\s+performance\b", r"\bcapacity\s+constraint(s)?\b",
        r"\bmobilization\b", r"\bunderperformance\b", r"\bimplementation\s+delay\b"
    ],
    "Funding & Finance": [
        r"\bcounterpart\s+funding\b", r"\bfunding\s+delay\b", r"\bbudget\s+shortfall\b",
        r"\bdisbursement\b", r"\bcontingency\b", r"\bfinancing\b"
    ],
    "Governance/Institutional": [
        r"\bimplementing\s+agency\b", r"\bpiu\b", r"\bcapacity\b",
        r"\bcoordination\b", r"\binstitutional\b"
    ],
    "Quality & HSE": [
        r"\bquality\b", r"\bdefect(s)?\b", r"\bsafety\b", r"\baccident(s)?\b",
        r"\benhancement\b", r"\benvironment(al)?\b"
    ],
    "Political/Social/External": [
        r"\bpolitical\b", r"\bunrest\b", r"\bconflict\b", r"\bstrike\b", r"\bprotest\b",
        r"\bborder\b", r"\bsanction(s)?\b", r"\bsecurity\b"
    ]
}

CLAIM_PATTERNS = [
    r"\bclaim(s)?\b", r"\bdispute(s)?\b", r"\barbitration\b", r"\blitigation\b",
    r"\badditional\s+payment\b", r"\bcompensation\b"
]

MITIGATION_PATTERNS = [
    r"\bmitigation\b", r"\bmeasure(s)?\b", r"\baction(s)?\b", r"\bwill\s+.*\b",
    r"\bplan(s)?\s+to\b", r"\bconsider(ing)?\b", r"\bpropos(e|ed)\b"
]

# 关系触发词（粗糙但好用）
CAUSE_CUES = [
    r"\bdue\s+to\b", r"\bbecause\s+of\b", r"\bresult(s|ed)?\s+in\b",
    r"\blead(s|ing)?\s+to\b", r"\bcaus(e|ed|ing)\b", r"\bas\s+a\s+result\b"
]

DELAY_CUES = [r"\bdelay(s|ed|ing)?\b", r"\bextension\b", r"\bslippage\b", r"\boverrun\b"]
COST_CUES = [r"\bcost\b", r"\bbudget\b", r"\boverrun\b", r"\bincrease\b", r"\bhigher\b"]
TIME_CUES = [r"\bschedule\b", r"\btime\b", r"\bdeadline\b"]

SEV_HIGH = [r"\bmajor\b", r"\bsignificant\b", r"\bserious\b", r"\bsubstantial\b", r"\bcritical\b"]
SEV_MED  = [r"\bmoderate\b", r"\bnotable\b", r"\bconsiderable\b"]

PROB_HIGH = [r"\blikely\b", r"\bexpected\b"]
PROB_MED  = [r"\bpossible\b", r"\bmay\b", r"\bcould\b", r"\bpotential\b"]


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def split_sentences(text: str) -> List[Tuple[str, int, int]]:
    """
    返回: [(sentence, start_idx, end_idx), ...]
    """
    # 简单句子切分：. ! ? 或换行分隔
    spans = []
    if not text:
        return spans
    # 先统一空白
    t = text
    # 用正则找分隔符位置
    parts = re.split(r"(?<=[\.\!\?])\s+|\n{2,}", t)
    cursor = 0
    for p in parts:
        p = p.strip()
        if not p:
            continue
        start = t.find(p, cursor)
        if start == -1:
            start = cursor
        end = start + len(p)
        spans.append((p, start, end))
        cursor = end
    return spans


def match_any(patterns: List[str], s: str) -> bool:
    for pat in patterns:
        if re.search(pat, s, flags=re.IGNORECASE):
            return True
    return False


def infer_severity(sentence: str) -> str:
    if match_any(SEV_HIGH, sentence):
        return "high"
    if match_any(SEV_MED, sentence):
        return "medium"
    return "low"


def infer_probability(sentence: str) -> float:
    if match_any(PROB_HIGH, sentence):
        return 0.75
    if match_any(PROB_MED, sentence):
        return 0.45
    return 0.25


def weight_hint_from_sentence(sentence: str) -> str:
    # 粗略把因果/成本/延误信号当作边权 hint
    if match_any(SEV_HIGH, sentence) or match_any(COST_CUES, sentence) or match_any(DELAY_CUES, sentence):
        return "high"
    if match_any(SEV_MED, sentence) or match_any(TIME_CUES, sentence):
        return "medium"
    return "low"


def extract_risks_from_sentence(sentence: str) -> List[Tuple[str, str]]:
    """
    返回 [(risk_name, taxonomy), ...]
    risk_name 用“命中的关键词/短语”近似表示
    """
    hits = []
    for tax, pats in RISK_TAXONOMY.items():
        for pat in pats:
            m = re.search(pat, sentence, flags=re.IGNORECASE)
            if m:
                name = m.group(0)
                name = re.sub(r"\s+", " ", name).strip()
                hits.append((name, tax))
                break
    # 去重（按name+tax）
    uniq = []
    seen = set()
    for n, t in hits:
        key = (n.lower(), t)
        if key not in seen:
            seen.add(key)
            uniq.append((n, t))
    return uniq


def extract_claims(sentence: str) -> List[str]:
    claims = []
    for pat in CLAIM_PATTERNS:
        m = re.search(pat, sentence, flags=re.IGNORECASE)
        if m:
            claims.append(m.group(0))
    # 简单归一
    uniq = []
    for c in claims:
        c = c.lower().strip()
        if c not in uniq:
            uniq.append(c)
    return uniq


def extract_mitigations(sentence: str) -> List[str]:
    if not match_any(MITIGATION_PATTERNS, sentence):
        return []
    # 简单抽取：把包含 will/plan/consider 的句子当作动作描述
    s = re.sub(r"\s+", " ", sentence).strip()
    # 限长，避免太长
    return [s[:200]]


def make_entity(temp_id: str, etype: str, name: str, taxonomy: str, conf: float,
                probability: float, severity: str) -> Dict:
    return {
        "temp_id": temp_id,
        "type": etype,
        "name": name,
        "taxonomy": taxonomy or "",
        "confidence": float(conf),
        "attributes": {
            "probability": float(probability),
            "severity": severity
        }
    }


def make_triple(head_temp_id: str, tail_temp_id: str, relation: str, conf: float,
                quote: str, char_start: Optional[int], char_end: Optional[int],
                weight_hint: str) -> Dict:
    return {
        "head_temp_id": head_temp_id,
        "tail_temp_id": tail_temp_id,
        "relation": relation,
        "confidence": float(conf),
        "weight_hint": weight_hint,
        "evidence": {
            "quote": quote[:300],
            "char_start": char_start,
            "char_end": char_end
        }
    }



def rule_extract_one_chunk(rec: Dict) -> Dict:
    text = rec.get("text", "")
    sents = split_sentences(text)

    entities: List[Dict] = []
    triples: List[Dict] = []

    # 用于去重与复用 temp_id
    ent_key_to_id: Dict[Tuple[str, str, str], str] = {}
    next_id = 1

    def get_or_create(etype: str, name: str, taxonomy: str, sentence: str) -> str:
        nonlocal next_id
        key = (etype, name.lower().strip(), taxonomy or "")
        if key in ent_key_to_id:
            return ent_key_to_id[key]
        tid = f"E{next_id}"
        next_id += 1
        sev = infer_severity(sentence)
        prob = infer_probability(sentence)
        conf = 0.55
        if sev == "high":
            conf = 0.65
        elif sev == "medium":
            conf = 0.60
        entities.append(make_entity(tid, etype, name, taxonomy, conf, prob, sev))
        ent_key_to_id[key] = tid
        return tid

    # 逐句抽取
    for sent, st, ed in sents:
        risks = extract_risks_from_sentence(sent)
        claims = extract_claims(sent)
        mits = extract_mitigations(sent)

        risk_ids = []
        for rname, tax in risks:
            rid = get_or_create("Risk", rname, tax, sent)
            risk_ids.append(rid)

        claim_ids = []
        for cname in claims:
            cid = get_or_create("Claim", cname, "Claims & Change", sent)
            claim_ids.append(cid)

        mit_ids = []
        for mdesc in mits:
            mid = get_or_create("Action", mdesc, "Mitigation", sent)
            mit_ids.append(mid)

        # 关系抽取（非常粗糙但能用）
        # 1) 风险-风险：共现 or 因果
        if len(risk_ids) >= 2:
            rel = "CAUSES" if match_any(CAUSE_CUES, sent) else "CO_OCCURS"
            wh = weight_hint_from_sentence(sent)
            for i in range(len(risk_ids)):
                for j in range(len(risk_ids)):
                    if i == j:
                        continue
                    triples.append(make_triple(
                        risk_ids[i], risk_ids[j], rel, 0.55,
                        quote=sent, char_start=st, char_end=ed, weight_hint=wh
                    ))

        # 2) 风险 -> 索赔
        if risk_ids and claim_ids:
            wh = weight_hint_from_sentence(sent)
            for rid in risk_ids:
                for cid in claim_ids:
                    triples.append(make_triple(
                        rid, cid, "LEADS_TO_CLAIM", 0.60,
                        quote=sent, char_start=st, char_end=ed, weight_hint=wh
                    ))

        # 3) 动作 -> 风险（缓解）
        if mit_ids and risk_ids:
            wh = "medium"
            for mid in mit_ids:
                for rid in risk_ids:
                    triples.append(make_triple(
                        mid, rid, "MITIGATES", 0.55,
                        quote=sent, char_start=st, char_end=ed, weight_hint=wh
                    ))

        # 4) 风险对成本/工期的“信号边”（不给 Project，后续由 03_build_snapshots 的 Project->entity 连接承接）
        if risk_ids and match_any(COST_CUES, sent):
            wh = weight_hint_from_sentence(sent)
            for rid in risk_ids:
                triples.append(make_triple(
                    rid, rid, "INCREASES_COST_SIGNAL", 0.50,
                    quote=sent, char_start=st, char_end=ed, weight_hint=wh
                ))
        if risk_ids and match_any(DELAY_CUES, sent):
            wh = weight_hint_from_sentence(sent)
            for rid in risk_ids:
                triples.append(make_triple(
                    rid, rid, "DELAYS_SIGNAL", 0.50,
                    quote=sent, char_start=st, char_end=ed, weight_hint=wh
                ))

    return {
        "project_id": rec["project_id"],
        "doc_id": rec["doc_id"],
        "chunk_id": rec["chunk_id"],
        "chunk_date": rec["chunk_date"],
        "entities": entities,
        "triples": triples
    }


def main(chunks_jsonl: str, out_jsonl: str, resume: bool = True):
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

    done = set()
    if resume and os.path.exists(out_jsonl):
        with open(out_jsonl, "r", encoding="utf-8") as r:
            for line in r:
                try:
                    obj = json.loads(line)
                    done.add(obj["chunk_id"])
                except Exception:
                    pass
        print(f"Resume mode: already done chunks = {len(done)}")

    written = 0
    with open(chunks_jsonl, "r", encoding="utf-8") as r, open(out_jsonl, "a", encoding="utf-8") as w:
        for line in r:
            rec = json.loads(line)
            if resume and rec["chunk_id"] in done:
                continue
            out = rule_extract_one_chunk(rec)
            w.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    print(f"Saved rule-based extraction to {out_jsonl}")
    print(f"Chunks processed this run: {written}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/interim/chunks.jsonl")
    ap.add_argument("--out", default="data/interim/extraction.jsonl")
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()
    main(args.chunks, args.out, resume=not args.no_resume)
