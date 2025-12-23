# tools/eval_policy05.py
import os, re, glob, subprocess
import pandas as pd

BUDGETS = [0, 150, 300, 450, 600]
N = 20  # 先跑20个超支项目

# 更严格的科学计数法/浮点匹配：支持 e+00 / e-06 / +0.1 / -0.2
FLOAT_RE = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"

pat_base  = re.compile(r"baseline COR pred:\s*" + FLOAT_RE)
pat_after = re.compile(r"after-intervention COR pred:\s*" + FLOAT_RE)
pat_delta = re.compile(r"ΔCOR:\s*" + FLOAT_RE)
pat_used  = re.compile(r"used:\s*" + FLOAT_RE)
pat_sel   = re.compile(r"selected actions:\s*\[(.*)\]")

def _last_float(pat, s):
    m = pat.findall(s)
    if not m:
        return None
    # findall 返回 list[str]（因为只有一个捕获组）
    return float(m[-1])

def parse_out(s: str):
    base  = _last_float(pat_base, s)
    after = _last_float(pat_after, s)
    delta = _last_float(pat_delta, s)
    used  = _last_float(pat_used, s)
    selm = pat_sel.findall(s)
    sel = selm[-1].strip() if selm else ""
    return base, after, delta, used, sel

def pick_projects(n=20):
    lbl = pd.read_csv("data/labels.csv").sort_values("COR", ascending=False)
    out = []
    for _, r in lbl.iterrows():
        pid = str(r["project_id"])
        true_cor = float(r["COR"])
        months = sorted([os.path.basename(p) for p in glob.glob(f"data/snapshots/{pid}/????-??")])
        if months and true_cor > 0:
            out.append((pid, months[0], true_cor))
        if len(out) >= n:
            break
    return out

def run_one(pid, t, budget):
    cmd = [
        "python","-m","src._05_optimize_policy",
        "--snapshots","data/snapshots",
        "--actions","data/actions.csv",
        "--ckpt","data/interim/gnn_ckpt.pt",
        "--project_id",pid,
        "--time",t,
        "--budget",str(budget),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    base, after, delta, used, sel = parse_out(p.stdout)
    return p.returncode, base, after, delta, used, sel, p.stdout.strip(), p.stderr.strip()

def main():
    os.makedirs("results", exist_ok=True)
    targets = pick_projects(N)
    rows = []
    for pid, t, true_cor in targets:
        for b in BUDGETS:
            rc, base, after, delta, used, sel, out, err = run_one(pid, t, b)
            rows.append({
                "project_id": pid, "time": t, "true_COR": true_cor,
                "budget": b, "rc": rc,
                "pred_base": base, "pred_after": after, "delta": delta,
                "used": used, "selected": sel,
                "stderr": err
            })
            print("done", pid, t, "B=", b, "delta=", delta, "sel=", sel)

    df = pd.DataFrame(rows)
    df.to_csv("results/policy05_budget_sweep.csv", index=False)
    print("Saved results/policy05_budget_sweep.csv")

if __name__ == "__main__":
    main()
