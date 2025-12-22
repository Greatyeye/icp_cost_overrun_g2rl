# tools/clean_filenames.py
from pathlib import Path
import re

ROOT = Path("data/raw_docs")

def clean_name(name: str) -> str:
    # 把换行/制表符变空格，再把连续空格压成一个
    s = re.sub(r"[\r\n\t]+", " ", name)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

changed = 0
for p in ROOT.rglob("*.pdf"):
    new_name = clean_name(p.name)
    if new_name != p.name:
        new_path = p.with_name(new_name)
        print(f"RENAME:\n  {p.name}\n-> {new_name}\n")
        p.rename(new_path)
        changed += 1

print("done, renamed:", changed)
