# src/00_prepare_corpus.py
from __future__ import annotations

import os
import re
import json
import argparse
from datetime import datetime, timezone
from typing import Optional

from dateutil import parser as dateparser

from src.common import read_pdf_text, read_txt, chunk_text, sha1


def clean_filename(name: str) -> str:
    """清理文件名里的换行/制表符/多空格（不改磁盘文件名，只用于记录source_file）。"""
    s = re.sub(r"[\r\n\t]+", " ", str(name))
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def infer_date_from_filename(fn: str) -> Optional[str]:
    """
    从文件名提取 YYYY-MM-DD 或 YYYY_MM_DD。
    找不到则返回 None。
    """
    fn = str(fn)
    m = re.search(r"(20\d{2})[-_](\d{2})[-_](\d{2})", fn)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def infer_date_from_text(text: str) -> Optional[str]:
    """
    从文本开头（前几千字符）提取“文档日期”（PAD/ICR/ISR常见）。
    注意：文本里可能出现多个日期（如汇率生效日），这里使用“优先更像发布日期”的策略：
      - 优先匹配 Month Day, Year / Day Month Year
      - 取首次出现的可解析日期
    """
    head = (text or "")[:6000]

    # 先找明确的 ISO 日期
    iso = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", head)
    if iso:
        return iso.group(0)

    # 常见英文月份
    month = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"

    patterns = [
        rf"\b{month}\s+\d{{1,2}},\s+\d{{4}}\b",      # March 25, 2014
        rf"\b\d{{1,2}}\s+{month}\s+\d{{4}}\b",      # 25 March 2014
    ]

    candidates = []
    for pat in patterns:
        candidates.extend(re.findall(pat, head, flags=re.IGNORECASE))

    # 过滤一些容易误抓的“非发布日期”上下文（可按需要扩展）
    # 例如："Exchange Rate Effective January 31, 2014" 不是文档发布日期，但也可能有用。
    # 这里不强删，只是尽量优先抓到封面/标题附近的日期。
    for c in candidates:
        try:
            dt = dateparser.parse(c, fuzzy=True, dayfirst=False)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue

    return None


def choose_chunk_date(filename: str, text: str) -> str:
    """
    日期优先级：
      1) 文件名 YYYY-MM-DD
      2) 文本开头的日期（封面/标题常见）
      3) 当前UTC日期
    """
    d1 = infer_date_from_filename(filename)
    if d1:
        return d1

    d2 = infer_date_from_text(text)
    if d2:
        return d2

    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def main(in_dir: str, out_jsonl: str, max_chars: int, overlap: int):
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

    written = 0
    project_count = 0
    doc_count = 0

    with open(out_jsonl, "w", encoding="utf-8") as w:
        # 遍历项目文件夹：data/raw_docs/P_001/...
        for project_id in sorted(os.listdir(in_dir)):
            pdir = os.path.join(in_dir, project_id)
            if not os.path.isdir(pdir):
                continue

            project_count += 1

            for fn in sorted(os.listdir(pdir)):
                path = os.path.join(pdir, fn)
                if not os.path.isfile(path):
                    continue

                ext = fn.lower().split(".")[-1]
                if ext not in ["pdf", "txt"]:
                    continue

                # read text
                text = read_pdf_text(path) if ext == "pdf" else read_txt(path)
                if not text or not text.strip():
                    # 空文档跳过
                    continue

                doc_count += 1

                doc_id = sha1(project_id + "|" + fn + "|" + str(os.path.getsize(path)))
                chunk_date = choose_chunk_date(fn, text)

                chunks = chunk_text(text, max_chars=max_chars, overlap_chars=overlap)
                if not chunks:
                    continue

                safe_fn = clean_filename(fn)
                for i, c in enumerate(chunks):
                    rec = {
                        "project_id": project_id,
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}_{i}",
                        "chunk_date": chunk_date,
                        "source_file": f"{project_id}/{safe_fn}",
                        "text": c
                    }
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1

    print(f"Saved chunks to {out_jsonl}")
    print(f"Projects scanned: {project_count} | Docs scanned: {doc_count} | Chunks written: {written}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/raw_docs")
    ap.add_argument("--out", default="data/interim/chunks.jsonl")
    ap.add_argument("--max_chars", type=int, default=1800)
    ap.add_argument("--overlap", type=int, default=200)
    args = ap.parse_args()
    main(args.in_dir, args.out, args.max_chars, args.overlap)
