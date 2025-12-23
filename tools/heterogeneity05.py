import pandas as pd
from pathlib import Path

df = pd.read_csv("results/policy05_budget_sweep.csv")
df["selected"] = df["selected"].fillna("").astype(str).str.replace("'", "", regex=False).str.strip()
df["delta"] = df["delta"].fillna(0.0)

def group_name(sel: str):
    if sel == "":
        return "None"
    if "A99" in sel:
        return "Strong(A99)"
    return "Weak(non-A99)"

df["group"] = df["selected"].apply(group_name)

# --- 组间 ΔCOR 统计（按预算也可以再细分） ---
tab = (df.groupby(["budget","group"])["delta"]
         .agg(mean="mean", median="median", p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75), n="count")
         .reset_index())
tab.to_csv("results/table_heterogeneity.csv", index=False)

print("Saved results/table_heterogeneity.csv")

# --- 图结构对比：读取每行对应 snapshot 统计边类型数量、risk taxonomy 覆盖等 ---
rows = []
for _, r in df.iterrows():
    pid, t = r["project_id"], r["time"]
    snap = Path(f"data/snapshots/{pid}/{t}")
    if not snap.exists():
        continue
    nodes = pd.read_csv(snap/"nodes.csv")
    edges = pd.read_csv(snap/"edges.csv")

    risk = nodes[nodes["node_type"]=="Risk"]
    risk_tax_n = risk["taxonomy"].fillna("").astype(str).str.strip().nunique()
    inc_cost_n = int((edges["edge_type"].astype(str).str.strip()=="INCREASES_COST_SIGNAL").sum())
    delays_n   = int((edges["edge_type"].astype(str).str.strip()=="DELAYS_SIGNAL").sum())

    rows.append({
        "project_id": pid,
        "time": t,
        "budget": r["budget"],
        "group": r["group"],
        "delta": r["delta"],
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "risk_nodes": len(risk),
        "risk_taxonomy_unique": risk_tax_n,
        "INCREASES_COST_SIGNAL": inc_cost_n,
        "DELAYS_SIGNAL": delays_n
    })

gs = pd.DataFrame(rows)
gs_tab = (gs.groupby(["budget","group"])
            .agg(
                mean_delta=("delta","mean"),
                mean_risk_tax=("risk_taxonomy_unique","mean"),
                mean_cost_edges=("INCREASES_COST_SIGNAL","mean"),
                mean_delay_edges=("DELAYS_SIGNAL","mean"),
                # 下面这一行修正了变量名
                mean_edges=("n_edges","mean"),
                n=("project_id","count")
            )
            .reset_index())
gs_tab.to_csv("results/table_group_graph_stats.csv", index=False)
print("Saved results/table_group_graph_stats.csv")
