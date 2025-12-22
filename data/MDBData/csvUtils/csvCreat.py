from pathlib import Path
import csv

SRC_DIR = Path("../downloaded_pdfs")   # 改成你的pdf目录
OUT_CSV = Path("data/project_map.csv")

pdfs = sorted([p for p in SRC_DIR.iterdir() if p.is_file() and p.suffix.lower()==".pdf"])
assert len(pdfs) >= 30, f"只找到{len(pdfs)}个PDF，不够30个"

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["project_id","source_file"])
    for i, p in enumerate(pdfs[:30], start=1):
        w.writerow([f"P_{i:03d}", p.name])

print("done:", OUT_CSV)
