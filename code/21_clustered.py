"""Country-level inference and power for the pre-specified evaluation (exploratory, not in the protocol).

The 18 country and base-forecast combinations share shocks within a country, so the
pooled sign test over combinations overstates the evidence. Here each country contributes
one number, the mean over base forecasts of log(loss of first / loss of second) in the
relevant window. We report a sign test over countries, a t interval with (countries - 1)
degrees of freedom and a percentile bootstrap that resamples whole countries.

Power: for each combination, the smallest relative loss difference the fixed-smoothing
test (B = floor(sqrt(T)) degrees of freedom) detects with 80% power at the Holm threshold
for the smallest p-value, 0.05 / 18, given the observed standard error.

Writes results/external/clustered_tests.csv and results/external/power.csv.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("t18", ROOT / "code" / "18_external_tests.py")
t18 = importlib.util.module_from_spec(spec)
sys.modules["t18"] = t18
spec.loader.exec_module(t18)
EXT = t18.EXT
RNG = np.random.default_rng(20261108)
NBOOT = 20000

DESIGNS = [(0.90, 3), (0.75, 3), (0.95, 3), (0.90, 1), (0.90, 6)]
COMPARISONS = ([("H1", m, "none", ["break"]) for m in
                ["rqc", "rqc3", "ewrqc", "lcs", "aci", "tspc", "tspc_ar", "tspc_u", "tspc_aru"]]
               + [("H2", "tspc_u", "rqc", ["break"]), ("H3", "tspc_aru", "rqc", ["stable"])]
               + [("H4", m, "rqc", ["stable", "break", "post"]) for m in ["rqc3", "ewrqc", "lcs", "aci", "tspc", "tspc_u"]])


def log_ratio(m, base, m1, m2, periods, col):
    x = m[(m.base == base) & m.period.isin(periods)].pivot_table(index="s", columns="method", values=col).dropna()
    return float(np.log(x[m1].mean() / x[m2].mean()))


def country_values(a, h, m1, m2, periods, col, countries):
    vals = {}
    for c in countries:
        m = t18.load(c, a, h)
        if m is None:
            continue
        vals[c] = float(np.mean([log_ratio(m, b, m1, m2, periods, col) for b in t18.BASES]))
    return pd.Series(vals)


def summarise(v):
    n = len(v)
    mean = v.mean()
    se = v.std(ddof=1) / np.sqrt(n)
    tq = stats.t.ppf(0.95, n - 1)
    idx = RNG.integers(0, n, size=(NBOOT, n))
    boot = v.to_numpy()[idx].mean(axis=1)
    neg = int((v < 0).sum())
    return {"countries": n, "favour": f"{neg}/{n}", "sign_p": stats.binomtest(neg, n, 0.5).pvalue,
            "ratio": np.exp(mean), "t90_lo": np.exp(mean - tq * se), "t90_hi": np.exp(mean + tq * se),
            "boot90_lo": np.exp(np.quantile(boot, 0.05)), "boot90_hi": np.exp(np.quantile(boot, 0.95)),
            "t_p": 2 * stats.t.sf(abs(mean / se), n - 1) if se > 0 else np.nan}


def power_table(a=0.90, h=3):
    rows = []
    for hyp, m1, m2, periods in COMPARISONS[:11]:
        for c in t18.COUNTRIES:
            m = t18.load(c, a, h)
            for b in t18.BASES:
                d, ref = t18.diff(m, b, m1, m2, periods)
                mean, se, df = t18.ewc(d)
                crit = stats.t.ppf(1 - 0.05 / 18 / 2, df)
                # smallest effect with 80% power: solve noncentral t power = 0.8
                lo, hi = 0.0, 50.0
                for _ in range(60):
                    mid = (lo + hi) / 2
                    pw = stats.nct.sf(crit, df, mid) + stats.nct.cdf(-crit, df, mid)
                    lo, hi = (mid, hi) if pw < 0.8 else (lo, mid)
                rows.append({"hypothesis": hyp, "comparison": f"{m1}-{m2}", "country": c, "base": b, "months": len(d),
                             "df": df, "rel_se": se / ref, "mde_rel": hi * se / ref, "obs_rel": mean / ref})
    return pd.DataFrame(rows)


def main():
    rows = []
    for a, h in DESIGNS:
        for col in ["loss", "loss_rel"]:
            for hyp, m1, m2, periods in COMPARISONS:
                countries = t18.COUNTRIES + (["NGA_POST"] if hyp == "H4" else [])
                v = country_values(a, h, m1, m2, periods, col, countries)
                if len(v) < 3:
                    continue
                rows.append({"alpha": a, "h": h, "targets": col, "hypothesis": hyp, "comparison": f"{m1}-{m2}",
                             **summarise(v), **{f"c_{k}": np.exp(x) for k, x in v.items()}})
    out = pd.DataFrame(rows)
    out.to_csv(EXT / "clustered_tests.csv", index=False)
    pw = power_table()
    pw.to_csv(EXT / "power.csv", index=False)
    show = out[(out.alpha == 0.90) & (out.h == 3) & (out.targets == "loss")]
    print(show[["hypothesis", "comparison", "favour", "sign_p", "ratio", "t90_lo", "t90_hi", "boot90_lo", "boot90_hi", "t_p"]]
          .round(3).to_string(index=False))
    print(pw.groupby("comparison")[["rel_se", "mde_rel", "obs_rel"]].median().round(3))


if __name__ == "__main__":
    main()
