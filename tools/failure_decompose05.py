import pandas as pd

df = pd.read_csv("results/policy05_budget_sweep.csv")
df["selected"] = df["selected"].fillna("").astype(str).str.strip()

# 只看预算>0 且 没选动作的行
fail = df[(df["budget"]>0) & (df["selected"]=="")].copy()

acts = pd.read_csv("data/actions.csv")

def parse_list(x):
    if pd.isna(x): return []
    s=str(x).strip()
    if not s or s.lower()=="nan": return []
    return [p.strip() for p in s.replace("|",";").replace(",",";").split(";") if p.strip()]

def diagnose(pid, t):
    nodes = pd.read_csv(f"data/snapshots/{pid}/{t}/nodes.csv")
    edges = pd.read_csv(f"data/snapshots/{pid}/{t}/edges.csv")

    reasons = {"no_taxonomy":0, "no_edge_hit":0, "both":0}
    # 如果存在任何 action 能命中，就不算 fail（理论上不会进来，但留保险）
    any_action_hit = False

    for _,a in acts.iterrows():
        tax = parse_list(a.get("target_risk_taxonomy",""))
        et  = parse_list(a.get("target_edge_types",""))

        m_risk = (nodes["node_type"]=="Risk")
        if tax:
            m_risk &= nodes["taxonomy"].fillna("").astype(str).str.strip().isin(tax)
        target=set(nodes.loc[m_risk,"node_id"].astype(str))
        hit_nodes=len(target)

        if et:
            mtype = edges["edge_type"].fillna("").astype(str).str.strip().isin(et)
        else:
            mtype = pd.Series([True]*len(edges))

        mnode = edges["src"].astype(str).isin(target) | edges["dst"].astype(str).isin(target)
        hit_edges=int((mtype & mnode).sum())

        if hit_nodes>0 and hit_edges>0:
            any_action_hit = True

        nt = (hit_nodes==0)
        ne = (hit_edges==0)
        if nt and ne:
            reasons["both"] += 1
        else:
            if nt: reasons["no_taxonomy"] += 1
            if ne: reasons["no_edge_hit"] += 1

    return any_action_hit, reasons

rows=[]
for (pid,t), sub in fail.groupby(["project_id","time"]):
    any_hit, reasons = diagnose(pid,t)
    rows.append({
        "project_id": pid,
        "time": t,
        "any_action_hit": any_hit,
        **reasons
    })

out = pd.DataFrame(rows)
out.to_csv("results/table_fail_reasons.csv", index=False)

cnt = out[["no_taxonomy","no_edge_hit","both"]].sum().to_frame("count")
cnt.to_csv("results/fail_reason_counts.csv")

print("Saved results/table_fail_reasons.csv and results/fail_reason_counts.csv")
