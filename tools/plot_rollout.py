#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def pick_col(df: pd.DataFrame, candidates):
    cols = {c.strip(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    low = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def normalize_time(s: pd.Series) -> pd.Series:
    ss = s.astype(str).str.strip()
    dt = pd.to_datetime(ss, errors="coerce", utc=False)
    if dt.notna().mean() >= 0.6:
        return dt
    return ss


def load_rollout(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    time_col = pick_col(df, ["time", "month", "date", "t", "step"])
    base_col = pick_col(df, ["cor_pred_baseline", "baseline", "base_cor", "cor_before", "pred_baseline"])
    after_col = pick_col(df, ["cor_pred_after", "after", "new_cor", "cor_after", "pred_after"])
    delta_col = pick_col(df, ["delta_cor", "delta", "deltaCOR"])
    act_col = pick_col(df, ["selected_actions", "actions", "selected", "action_ids"])

    if time_col is None:
        raise KeyError(f"[{label}] 缺少时间列（time/step 等）。当前列={list(df.columns)}")
    if base_col is None and after_col is None:
        raise KeyError(f"[{label}] 至少应包含 baseline 或 after。当前列={list(df.columns)}")

    out = pd.DataFrame()
    out["time_raw"] = df[time_col].astype(str).str.strip()
    out["time_norm"] = normalize_time(df[time_col])

    out["baseline"] = df[base_col] if base_col is not None else pd.NA
    out["after"] = df[after_col] if after_col is not None else pd.NA

    if delta_col is not None:
        out["delta"] = df[delta_col]
    else:
        out["delta"] = pd.to_numeric(out["after"], errors="coerce") - pd.to_numeric(out["baseline"], errors="coerce")

    out["actions"] = df[act_col].astype(str) if act_col is not None else ""
    out["policy"] = label

    for c in ["baseline", "after", "delta"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.sort_values("time_norm").reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--none", required=True, help="rollout csv for policy=none (baseline)")
    ap.add_argument("--opt", required=True, help="rollout csv for policy=optimize (your method)")
    ap.add_argument("--out_dir", default="results", help="output directory")
    ap.add_argument("--project", default="", help="project id for naming")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_none = load_rollout(args.none, "none")
    df_opt = load_rollout(args.opt, "optimize")

    # ✅ 只保留 merge 必需列，避免 time_norm/policy 重名
    left = df_none[["time_raw", "time_norm", "baseline", "after", "delta", "actions"]].rename(
        columns={
            "baseline": "baseline_none",
            "after": "after_none",
            "delta": "delta_none",
            "actions": "actions_none",
        }
    )
    right = df_opt[["time_raw", "time_norm", "baseline", "after", "delta", "actions"]].rename(
        columns={
            "baseline": "baseline_opt",
            "after": "after_opt",
            "delta": "delta_opt",
            "actions": "actions_opt",
        }
    )

    merged = pd.merge(left, right, on="time_raw", how="outer", suffixes=("_none", "_opt"))

    # ✅ 用可用的 time_norm 排序（优先 none 的，其次 opt 的）
    tn_none = "time_norm_none" if "time_norm_none" in merged.columns else None
    tn_opt = "time_norm_opt" if "time_norm_opt" in merged.columns else None

    if tn_none and merged[tn_none].notna().any():
        merged["time_norm"] = merged[tn_none]
    elif tn_opt and merged[tn_opt].notna().any():
        merged["time_norm"] = merged[tn_opt]
    else:
        merged["time_norm"] = merged["time_raw"]

    merged = merged.sort_values("time_norm").reset_index(drop=True)

    project = args.project.strip() or "project"

    out_table = os.path.join(args.out_dir, f"table_rollout_compare_{project}.csv")
    merged.to_csv(out_table, index=False)

    # -------- plot --------
    x = merged["time_raw"].astype(str).tolist()

    plt.figure()
    if "baseline_none" in merged.columns:
        plt.plot(x, merged["baseline_none"], marker="o", label="None: baseline")
    if "after_none" in merged.columns and merged["after_none"].notna().any():
        plt.plot(x, merged["after_none"], marker="o", label="None: after")

    if "baseline_opt" in merged.columns and merged["baseline_opt"].notna().any():
        plt.plot(x, merged["baseline_opt"], marker="o", label="Optimize: baseline")
    if "after_opt" in merged.columns:
        plt.plot(x, merged["after_opt"], marker="o", label="Optimize: after")

    plt.xlabel("time")
    plt.ylabel("COR prediction")
    plt.title(f"Rollout compare: {project}")
    plt.xticks(rotation=45, ha="right")
    plt.axhline(0, linewidth=1)
    plt.legend()
    plt.tight_layout()

    out_fig = os.path.join(args.out_dir, f"fig_rollout_compare_{project}.png")
    plt.savefig(out_fig, dpi=200)
    plt.close()

    print("Saved:")
    print(" -", out_table)
    print(" -", out_fig)


if __name__ == "__main__":
    main()
