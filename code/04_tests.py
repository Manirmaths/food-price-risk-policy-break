"""Origin-month inference on loss differentials and coverage.

All forecasts issued in an origin month are first averaged, so the month is
the sampling unit.  Three statistics are reported for each comparison:

* Newey-West HAC t-statistic with h-1 lags and Student-t(T-1) reference;
* the Diebold-Mariano statistic with the Harvey, Leybourne and Newbold (1997)
  small-sample correction and Student-t(T-1) reference;
* a fixed-smoothing test based on the equal-weighted cosine (EWC) long-run
  variance with B = floor(sqrt(T)) basis functions and Student-t(B) reference
  (Lazarus et al. 2018; Coroneo and Iacone 2020).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
RESULTS = ROOT / "results"


def pinball(y, q, a):
    r = y - q
    return np.where(r >= 0, a * r, (a - 1) * r)


def nw(d, lag):
    d = np.asarray(d, float)
    n = len(d)
    c = d - d.mean()
    lr = c @ c / n
    for k in range(1, min(lag, n - 1) + 1):
        lr += 2 * (1 - k / (lag + 1)) * (c[k:] @ c[:-k]) / n
    se = np.sqrt(max(lr, 0) / n)
    t = d.mean() / se if se > 0 else np.nan
    crit = stats.t.ppf(0.975, n - 1)
    return {"estimate": d.mean(), "se_nw": se, "ci_lo": d.mean() - crit * se, "ci_hi": d.mean() + crit * se,
            "t_nw": t, "p_nw": 2 * stats.t.sf(abs(t), n - 1) if se > 0 else np.nan, "months": n}


def hln(d, h):
    d = np.asarray(d, float)
    n = len(d)
    c = d - d.mean()
    lr = c @ c / n
    for k in range(1, min(h - 1, n - 1) + 1):
        lr += 2 * (c[k:] @ c[:-k]) / n
    if lr <= 0:  # rectangular kernel can be negative; fall back to Bartlett weights
        out = nw(d, h - 1)
        k = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        return out["t_nw"] * k, 2 * stats.t.sf(abs(out["t_nw"] * k), n - 1)
    dm = d.mean() / np.sqrt(lr / n)
    k = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    t = dm * k
    return t, 2 * stats.t.sf(abs(t), n - 1)


def ewc(d):
    d = np.asarray(d, float)
    n = len(d)
    b = max(1, int(np.floor(np.sqrt(n))))
    c = d - d.mean()
    tt = np.arange(1, n + 1)
    lam = [np.sqrt(2 / n) * np.sum(c * np.cos(np.pi * j * (tt - 0.5) / n)) for j in range(1, b + 1)]
    omega = np.mean(np.square(lam))
    t = np.sqrt(n) * d.mean() / np.sqrt(omega)
    return t, 2 * stats.t.sf(abs(t), b), b


def infer(d, h, null=0.0):
    d = np.asarray(d, float) - null
    out = nw(d, h - 1)
    out["estimate"] += null
    out["ci_lo"] += null
    out["ci_hi"] += null
    out["t_hln"], out["p_hln"] = hln(d, h)
    out["t_ewc"], out["p_ewc"], out["ewc_df"] = ewc(d)
    return out


def monthly(f, models, alpha):
    rows = []
    for (date, target, period), g in f.groupby(["date", "target_date", "period"], sort=True):
        y = g["y"].to_numpy()
        for m in models:
            q = g[m].to_numpy()
            rows.append({"date": date, "target_date": target, "period": period, "model": m,
                         "loss": pinball(y, q, alpha).mean(), "coverage": np.mean(y <= q)})
    return pd.DataFrame(rows)


PRIMARY = [
    # frozen conditional versus frozen unconditional
    ("linear_qr", "static", "Frozen linear QR - frozen empirical"),
    ("hist_gb", "static", "Frozen boosting - frozen empirical"),
    # re-estimation
    ("linear_qr_expand", "linear_qr", "Expanding - frozen (linear QR)"),
    ("hist_gb_expand", "hist_gb", "Expanding - frozen (boosting)"),
    ("linear_qr_roll", "linear_qr", "Rolling - frozen (linear QR)"),
    ("hist_gb_roll", "hist_gb", "Rolling - frozen (boosting)"),
    ("linear_qr_roll", "linear_qr_expand", "Rolling - expanding (linear QR)"),
    ("hist_gb_roll", "hist_gb_expand", "Rolling - expanding (boosting)"),
    # correction on top of each scheme
    ("static_adapt", "static", "RQC gain (empirical)"),
    ("linear_qr_adapt", "linear_qr", "RQC gain, frozen (linear QR)"),
    ("hist_gb_adapt", "hist_gb", "RQC gain, frozen (boosting)"),
    ("linear_qr_expand_adapt", "linear_qr_expand", "RQC gain, expanding (linear QR)"),
    ("hist_gb_expand_adapt", "hist_gb_expand", "RQC gain, expanding (boosting)"),
    ("linear_qr_roll_adapt", "linear_qr_roll", "RQC gain, rolling (linear QR)"),
    ("hist_gb_roll_adapt", "hist_gb_roll", "RQC gain, rolling (boosting)"),
    # ACI versus RQC
    ("linear_qr_expand_aci", "linear_qr_expand", "ACI gain, expanding (linear QR)"),
    ("hist_gb_expand_aci", "hist_gb_expand", "ACI gain, expanding (boosting)"),
    ("linear_qr_expand_adapt", "linear_qr_expand_aci", "RQC - ACI, expanding (linear QR)"),
    ("hist_gb_expand_adapt", "hist_gb_expand_aci", "RQC - ACI, expanding (boosting)"),
    ("linear_qr_adapt", "linear_qr_aci", "RQC - ACI, frozen (linear QR)"),
    ("hist_gb_adapt", "hist_gb_aci", "RQC - ACI, frozen (boosting)"),
    # combined versus benchmarks
    ("linear_qr_expand_adapt", "linear_qr_adapt", "Expanding+RQC - frozen+RQC (linear QR)"),
    ("hist_gb_expand_adapt", "hist_gb_adapt", "Expanding+RQC - frozen+RQC (boosting)"),
    ("linear_qr_expand_adapt", "rolling_q", "Expanding+RQC linear QR - rolling empirical"),
    ("hist_gb_expand_adapt", "rolling_q", "Expanding+RQC boosting - rolling empirical"),
    ("hist_gb_expand_adapt", "linear_qr_expand_adapt", "Expanding+RQC boosting - linear QR"),
    ("hist_gb_expand", "linear_qr_expand", "Expanding boosting - expanding linear QR"),
]
ROBUST = [
    ("linear_qr_expand", "linear_qr", "Expanding - frozen (linear QR)"),
    ("hist_gb_expand", "hist_gb", "Expanding - frozen (boosting)"),
    ("linear_qr_expand_adapt", "linear_qr_expand", "RQC gain, expanding (linear QR)"),
    ("hist_gb_expand_adapt", "hist_gb_expand", "RQC gain, expanding (boosting)"),
    ("linear_qr_adapt", "linear_qr", "RQC gain, frozen (linear QR)"),
    ("hist_gb_adapt", "hist_gb", "RQC gain, frozen (boosting)"),
]


def main():
    loss_rows, cov_rows = [], []
    for path in sorted(glob.glob(str(BUILD / "forecasts_a*_h*.csv.gz"))):
        f = pd.read_csv(path, parse_dates=["date", "target_date"])
        alpha, h = float(f["alpha"].iloc[0]), int(f["horizon"].iloc[0])
        primary = (round(alpha, 2), h) == (0.90, 3)
        comps = PRIMARY if primary else ROBUST
        models = sorted({m for c in comps for m in c[:2]} | {"static", "static_adapt", "rolling_q"})
        models = [m for m in models if m in f.columns]
        mm = monthly(f, models, alpha)
        if primary:
            mm.to_csv(RESULTS / "monthly_metrics.csv", index=False)
        for period in ["stable", "break", "all"]:
            sub = mm if period == "all" else mm[mm["period"].eq(period)]
            wide = sub.pivot(index="date", columns="model", values="loss")
            for a, b, label in comps:
                if a in wide and b in wide:
                    loss_rows.append({"alpha": alpha, "horizon": h, "period": period, "comparison": label,
                                      "model_1": a, "model_2": b, **infer((wide[a] - wide[b]).to_numpy(), h)})
            covw = sub.pivot(index="date", columns="model", values="coverage")
            for m in models:
                cov_rows.append({"alpha": alpha, "horizon": h, "period": period, "model": m,
                                 **infer(covw[m].to_numpy(), h, null=alpha)})
    loss = pd.DataFrame(loss_rows)
    cov = pd.DataFrame(cov_rows)
    loss.to_csv(RESULTS / "loss_inference.csv", index=False)
    cov.to_csv(RESULTS / "coverage_inference.csv", index=False)
    pd.set_option("display.width", 250)
    p = loss[(loss.alpha.round(2) == 0.90) & (loss.horizon == 3)]
    print(p[["period", "comparison", "estimate", "ci_lo", "ci_hi", "p_nw", "p_hln", "p_ewc"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
