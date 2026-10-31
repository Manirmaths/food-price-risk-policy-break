"""Ex-post diagnostics of the primary forecast errors (alpha = 0.90, h = 3).

1. Oracle decomposition.  For each base forecast and origin month we compare
   raw pinball loss with loss after the hindsight-optimal common shift and after
   hindsight-optimal commodity-specific shifts.  The common share S is the
   fraction of the commodity-oracle gain delivered by one common shift.
   Because S rises mechanically with the size of a common bias, it is reported
   for frozen, expanding and rolling bases (and corrected bases), so the break
   share can be read after most of the level error has been removed.
2. Rolling 11-month pre-break placebo windows for every base.
3. A squared-error decomposition of residuals into a common month component,
   a commodity-within-month component and an idiosyncratic remainder.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
RESULTS = ROOT / "results"
ALPHA = 0.90
RNG = np.random.default_rng(20260814)
BASES = [
    "static", "linear_qr", "hist_gb",
    "linear_qr_expand", "hist_gb_expand", "linear_qr_roll", "hist_gb_roll",
    "linear_qr_expand_adapt", "hist_gb_expand_adapt",
]


def pinball(y, q):
    r = y - q
    return np.where(r >= 0, ALPHA * r, (ALPHA - 1) * r)


def month_rows(group, model):
    y, q = group["y"].to_numpy(), group[model].to_numpy()
    r = y - q
    common = np.quantile(r, ALPHA)
    item_shift = pd.Series(r).groupby(group["item"].to_numpy()).quantile(ALPHA)
    q_item = q + group["item"].map(item_shift).to_numpy()
    # squared-error decomposition
    month_mean = r.mean()
    item_mean = pd.Series(r).groupby(group["item"].to_numpy()).transform("mean").to_numpy()
    return {
        "date": group["date"].iloc[0], "target_date": group["target_date"].iloc[0],
        "period": group["period"].iloc[0], "model": model, "n": len(group),
        "common_shift": common, "coverage": np.mean(y <= q),
        "loss_raw": pinball(y, q).mean(), "loss_common_oracle": pinball(y, q + common).mean(),
        "loss_item_oracle": pinball(y, q_item).mean(),
        "msq_common": month_mean ** 2,
        "msq_commodity": np.mean((item_mean - month_mean) ** 2),
        "msq_idiosyncratic": np.mean((r - item_mean) ** 2),
        "resid_sd": r.std(),
    }


def block_indices(n, block, draws):
    starts = RNG.integers(0, n, size=(draws, int(np.ceil(n / block))))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]


def share(g):
    gc = (g["loss_raw"] - g["loss_common_oracle"]).to_numpy()
    gi = (g["loss_raw"] - g["loss_item_oracle"]).to_numpy()
    return gc, gi


def summarise(g, draws=20_000):
    g = g.sort_values("date")
    gc, gi = share(g)
    idx = block_indices(len(g), 3, draws)
    boot = gc[idx].sum(1) / gi[idx].sum(1)
    tot = g[["msq_common", "msq_commodity", "msq_idiosyncratic"]].mean()
    return {
        "months": len(g), "loss_raw": g["loss_raw"].mean(),
        "loss_common_oracle": g["loss_common_oracle"].mean(),
        "loss_item_oracle": g["loss_item_oracle"].mean(),
        "common_share": gc.sum() / gi.sum(),
        "common_share_lo": np.quantile(boot, 0.025), "common_share_hi": np.quantile(boot, 0.975),
        "mean_common_shift": g["common_shift"].mean(),
        "msq_total": tot.sum(),
        "share_msq_common": tot["msq_common"] / tot.sum(),
        "share_msq_commodity": tot["msq_commodity"] / tot.sum(),
        "share_msq_idiosyncratic": tot["msq_idiosyncratic"] / tot.sum(),
        "mean_resid_sd": g["resid_sd"].mean(),
    }


def placebo(monthly, months=11):
    rows, summ = [], []
    for model, md in monthly.groupby("model"):
        md = md.sort_values("date")
        stable = md[md["period"].eq("stable")].reset_index(drop=True)
        actual = md[md["period"].eq("break")]
        shares = []
        for start in range(len(stable) - months + 1):
            w = stable.iloc[start:start + months]
            gc, gi = share(w)
            s = gc.sum() / gi.sum()
            shares.append(s)
            rows.append({"model": model, "window": "placebo", "end_target": w["target_date"].iloc[-1], "common_share": s})
        gc, gi = share(actual)
        a = gc.sum() / gi.sum()
        rows.append({"model": model, "window": "break", "end_target": actual["target_date"].iloc[-1], "common_share": a})
        arr = np.array(shares)
        summ.append({"model": model, "actual_share": a, "placebo_min": arr.min(), "placebo_median": np.median(arr),
                     "placebo_max": arr.max(), "placebo_windows": len(arr),
                     "rank_above_all": bool(a > arr.max()),
                     "upper_rank_p": (1 + np.sum(arr >= a)) / (len(arr) + 1)})
    return pd.DataFrame(rows), pd.DataFrame(summ)


def main():
    f = pd.read_csv(BUILD / "forecasts_a90_h3.csv.gz", parse_dates=["date", "target_date"])
    rows = [month_rows(g, m) for m in BASES for _, g in f.groupby("date", sort=True)]
    monthly = pd.DataFrame(rows)
    monthly.to_csv(RESULTS / "decomposition_monthly.csv", index=False)
    summ = []
    for m in BASES:
        md = monthly[monthly["model"].eq(m)]
        for period in ["stable", "break"]:
            summ.append({"model": m, "period": period, **summarise(md[md["period"].eq(period)])})
    summ = pd.DataFrame(summ)
    summ.to_csv(RESULTS / "decomposition_summary.csv", index=False)
    windows, psumm = placebo(monthly)
    windows.to_csv(RESULTS / "placebo_windows.csv", index=False)
    psumm.to_csv(RESULTS / "placebo_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(summ.round(3).to_string(index=False))
    print(psumm.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
