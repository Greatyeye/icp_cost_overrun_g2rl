import pandas as pd

df = pd.read_csv("results/policy05_budget_sweep.csv")

# 选到动作=有效（或者 delta<0 也行，你可以两者都算）
df["has_action"] = df["selected"].fillna("").astype(str).str.strip().ne("")
df["improved"] = df["delta"].fillna(0.0) < 0

summary = (df.groupby("budget")
           .agg(
               n=("project_id", "count"),
               action_rate=("has_action", "mean"),
               improve_rate=("improved", "mean"),
               delta_mean=("delta", "mean"),
               delta_median=("delta", "median"),
               delta_p25=("delta", lambda x: x.quantile(0.25)),
               delta_p75=("delta", lambda x: x.quantile(0.75)),
           )
           .reset_index())

print("\n=== Budget-level summary ===")
print(summary.to_string(index=False))

# 动作频次（整体）
acts = (df.loc[df["has_action"], "selected"]
          .astype(str)
          .str.replace("'", "", regex=False)
          .str.split(",")
          .explode()
          .str.strip())
act_freq = acts.value_counts()

print("\n=== Action frequency (overall) ===")
print(act_freq.to_string())
