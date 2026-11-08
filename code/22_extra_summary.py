"""Summaries of the exploratory analyses in 20_extra.py (alpha = 0.90, h = 3).

Writes results/external/extra_summary.csv (loss ratios by country, base, period and target
set) and results/external/extra_country.csv (country-level geometric means with t intervals),
and prints the main numbers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "results" / "external"
COUNTRIES = ["NGA", "LBN", "SDN", "HTI", "MMR", "AFG", "NGA_POST"]
BREAK = ["NGA", "LBN", "SDN", "HTI", "MMR", "AFG"]
PERIODS = {"break": ["break"], "stable": ["stable"], "whole": ["stable", "break", "post"]}


def load():
    fr = [pd.read_csv(EXT / f"{c}_a90_h3_extra.csv") for c in COUNTRIES if (EXT / f"{c}_a90_h3_extra.csv").exists()]
    return pd.concat(fr, ignore_index=True)


def ratios(d):
    rows = []
    for (c, b), g in d.groupby(["country", "base"]):
        for pname, per in PERIODS.items():
            x = g[g.period.isin(per)]
            if x.empty:
                continue
            for col in ["loss", "loss_obs"]:
                t = x.pivot_table(index="s", columns="method", values=col).dropna()
                if len(t) < 4:
                    continue
                mean = t.mean()
                for m in mean.index:
                    rows.append({"country": c, "base": b, "period": pname, "targets": col, "method": m,
                                 "months": len(t), "loss": mean[m], "vs_rqc": mean[m] / mean["rqc"],
                                 "vs_none": mean[m] / mean["none"]})
            cov = x.pivot_table(index="s", columns="method", values="cov").mean()
            for m in cov.index:
                rows.append({"country": c, "base": b, "period": pname, "targets": "coverage", "method": m, "loss": cov[m]})
    return pd.DataFrame(rows)


def country_level(r, period, targets, ref, countries):
    out = []
    x = r[(r.period == period) & (r.targets == targets) & r.country.isin(countries)]
    for m, g in x.groupby("method"):
        v = g.groupby("country")[f"vs_{ref}"].apply(lambda s: np.log(s).mean())
        n = len(v)
        if n < 3:
            continue
        se = v.std(ddof=1) / np.sqrt(n)
        tq = stats.t.ppf(0.95, n - 1)
        neg = int((v < 0).sum())
        out.append({"period": period, "targets": targets, "ref": ref, "method": m, "countries": n,
                    "favour": f"{neg}/{n}", "sign_p": stats.binomtest(neg, n, 0.5).pvalue,
                    "ratio": np.exp(v.mean()), "lo": np.exp(v.mean() - tq * se), "hi": np.exp(v.mean() + tq * se),
                    **{f"c_{k}": np.exp(val) for k, val in v.items()}})
    return out


def main():
    d = load()
    r = ratios(d)
    r.to_csv(EXT / "extra_summary.csv", index=False)
    rows = []
    for targets in ["loss", "loss_obs"]:
        for ref in ["rqc", "none"]:
            rows += country_level(r, "break", targets, ref, BREAK)
            rows += country_level(r, "stable", targets, ref, BREAK)
            rows += country_level(r, "whole", targets, ref, COUNTRIES)
    c = pd.DataFrame(rows)
    c.to_csv(EXT / "extra_country.csv", index=False)
    pd.set_option("display.width", 250)
    for targets in ["loss", "loss_obs"]:
        print("==", targets)
        print(c[(c.targets == targets) & (c.ref == "rqc")][["period", "method", "favour", "ratio", "lo", "hi"]]
              .round(3).to_string(index=False))
        print(c[(c.targets == targets) & (c.ref == "none") & (c.period == "break")][["method", "favour", "ratio", "lo", "hi"]]
              .round(3).to_string(index=False))
    cov = r[r.targets == "coverage"].groupby(["period", "method"])["loss"].median().unstack("period").round(3)
    print("median coverage\n", cov)
    al = d.groupby(["country", "base"])["alarms4"].first()
    print(al.to_string())
    obs = d[d.method == "none"].groupby("country").apply(lambda g: (g.n_obs.fillna(0).sum() / g.n.sum()))
    print("observed share of evaluated targets\n", obs.round(3))


if __name__ == "__main__":
    main()
