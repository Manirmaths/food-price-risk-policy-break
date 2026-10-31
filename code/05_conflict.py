"""Descriptive state-month evidence on conflict exposure and future food-price growth."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
RESULTS = ROOT / "results"
HORIZON = 3
BREAK_TARGET_START = pd.Timestamp("2023-06-01")
CONFLICT = "log1p_violent_fatalities_s3"


def hac_mean(values: np.ndarray, lag: int = 2) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    centered = values - values.mean()
    long_run = centered @ centered / n
    for k in range(1, min(lag, n - 1) + 1):
        long_run += 2 * (1 - k / (lag + 1)) * (centered[k:] @ centered[:-k]) / n
    se = np.sqrt(max(long_run, 0) / n)
    critical = stats.t.ppf(0.975, n - 1)
    statistic = values.mean() / se if se > 0 else np.nan
    return {
        "months": n,
        "mean_rank_correlation": values.mean(),
        "se_hac": se,
        "ci_lo": values.mean() - critical * se,
        "ci_hi": values.mean() + critical * se,
        "p_value": 2 * stats.t.sf(abs(statistic), n - 1) if se > 0 else np.nan,
    }


def add_target(panel: pd.DataFrame) -> pd.DataFrame:
    future = panel[["state", "item", "date", "logp"]].copy()
    future["date"] = future["date"] - pd.DateOffset(months=HORIZON)
    future = future.rename(columns={"logp": "logp_fwd"})
    panel = panel.merge(future, on=["state", "item", "date"], how="left")
    panel["y"] = 100 * (panel["logp_fwd"] - panel["logp"])
    panel["target_date"] = panel["date"] + pd.DateOffset(months=HORIZON)
    return panel[panel["y"].notna()]


def main() -> None:
    panel = pd.read_csv(BUILD / "panel.csv.gz", parse_dates=["date"])
    data = add_target(panel)
    data = data[data["date"] >= pd.Timestamp("2021-01-01")]
    state_month = (
        data.groupby(["date", "target_date", "state"], as_index=False)
        .agg(outcome_q90=("y", lambda values: values.quantile(0.90)),
             conflict=(CONFLICT, "first"), items=("item", "nunique"))
    )
    state_month["period"] = np.where(
        state_month["target_date"] >= BREAK_TARGET_START, "break", "stable"
    )
    state_month["conflict_quintile"] = state_month.groupby("date")["conflict"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 5, labels=False) + 1
    )
    state_month.to_csv(RESULTS / "conflict_state_month.csv", index=False)

    monthly_rows = []
    for date, group in state_month.groupby("date", sort=True):
        monthly_rows.append(
            {"date": date, "target_date": group["target_date"].iloc[0],
             "period": group["period"].iloc[0], "states": len(group),
             "rank_correlation": stats.spearmanr(group["conflict"], group["outcome_q90"]).statistic}
        )
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(RESULTS / "conflict_monthly.csv", index=False)

    rows = []
    for period in ["stable", "break", "all"]:
        subset = monthly if period == "all" else monthly[monthly["period"].eq(period)]
        rows.append({"period": period, **hac_mean(subset["rank_correlation"].to_numpy())})
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "conflict_summary.csv", index=False)

    quintiles = (
        state_month.groupby(["period", "conflict_quintile"], as_index=False)
        .agg(mean_state_q90=("outcome_q90", "mean"),
             median_state_q90=("outcome_q90", "median"),
             state_months=("outcome_q90", "size"))
    )
    quintiles.to_csv(RESULTS / "conflict_quintiles.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n", quintiles.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
