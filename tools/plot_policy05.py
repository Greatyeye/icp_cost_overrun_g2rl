import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/policy05_budget_sweep.csv")
df["delta"] = df["delta"].fillna(0.0)

# ---------- 图1：预算 vs ΔCOR（均值 + IQR误差条） ----------
g = df.groupby("budget")["delta"]
summary = g.agg(
    mean="mean",
    median="median",
    p25=lambda x: x.quantile(0.25),
    p75=lambda x: x.quantile(0.75),
    n="count"
).reset_index()

summary.to_csv("results/table_budget_summary.csv", index=False)

x = summary["budget"].values
y = summary["median"].values
yerr_low = (summary["median"] - summary["p25"]).values
yerr_high = (summary["p75"] - summary["median"]).values

plt.figure()
plt.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="o-")
plt.xlabel("Budget")
plt.ylabel("Median ΔCOR (IQR error bars)")
plt.title("Budget vs Mean ΔCOR (IQR error bars)")
plt.axhline(0, linewidth=1)
plt.tight_layout()
plt.savefig("results/fig_budget_delta.png", dpi=200)
plt.close()

# ---------- 图2：动作频次 ----------
df["selected"] = df["selected"].fillna("").astype(str)

acts = (df.loc[df["selected"].str.strip().ne(""), "selected"]
          .str.replace("'", "", regex=False)
          .str.split(",")
          .explode()
          .str.strip())

freq = acts.value_counts().sort_values(ascending=False)

plt.figure()
plt.bar(freq.index.astype(str), freq.values)
plt.xlabel("Action")
plt.ylabel("Count")
plt.title("Action Frequency (Policy05)")
plt.tight_layout()
plt.savefig("results/fig_action_freq.png", dpi=200)
plt.close()

print("Saved:")
print(" - results/table_budget_summary.csv")
print(" - results/fig_budget_delta.png")
print(" - results/fig_action_freq.png")
