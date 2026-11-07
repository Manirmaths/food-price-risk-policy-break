"""Hypothesis tests of PROTOCOL.md section 7 on the external RTFP results."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "results" / "external"
COUNTRIES = ["NGA", "LBN", "SDN", "HTI", "MMR", "AFG"]
BASES = ["empirical", "linear_qr", "hist_gb"]
MARGIN = 0.05


def ewc(d):
    """Mean, standard error and degrees of freedom of the equal-weighted cosine statistic."""
    d = np.asarray(d, float)
    n = len(d)
    b = max(1, int(np.floor(np.sqrt(n))))
    c = d - d.mean()
    tt = np.arange(1, n + 1)
    lam = [np.sqrt(2 / n) * np.sum(c * np.cos(np.pi * j * (tt - 0.5) / n)) for j in range(1, b + 1)]
    se = np.sqrt(np.mean(np.square(lam)) / n)
    return d.mean(), se, b


def pval(d):
    m, se, b = ewc(d)
    t = m / se if se > 0 else np.nan
    return 2 * stats.t.sf(abs(t), b) if se > 0 else np.nan


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def load(country, a, h):
    f = EXT / f"{country}_a{int(round(a * 100))}_h{h}_monthly.csv"
    return pd.read_csv(f) if f.exists() else None


def diff(m, base, m1, m2, periods, col="loss"):
    x = m[(m.base == base) & m.period.isin(periods)].pivot_table(index="s", columns="method", values=col)
    return (x[m1] - x[m2]).dropna().to_numpy(), x[m2].mean()


def directional(a, h, m1, m2, periods, col="loss"):
    rows = []
    for c in COUNTRIES:
        m = load(c, a, h)
        if m is None:
            continue
        for b in BASES:
            d, ref = diff(m, b, m1, m2, periods, col)
            if len(d) < 4:
                continue
            rows.append({"country": c, "base": b, "comparison": f"{m1}-{m2}", "months": len(d),
                         "diff": d.mean(), "rel_diff": d.mean() / ref, "p": pval(d)})
    r = pd.DataFrame(rows)
    if len(r):
        r["p_holm"] = holm(r["p"].to_numpy())
        r["sig_better"] = (r["p_holm"] < 0.05) & (r["diff"] < 0)
        r["sig_worse"] = (r["p_holm"] < 0.05) & (r["diff"] > 0)
    return r


def sign_test(r):
    neg = int((r["diff"] < 0).sum())
    n = int((r["diff"] != 0).sum())
    return neg, n, stats.binomtest(neg, n, 0.5).pvalue if n else np.nan


def equivalence(a, h, countries, col="loss"):
    rows = []
    for c in countries:
        m = load(c, a, h)
        if m is None:
            continue
        for b in BASES:
            x = m[m.base == b].pivot_table(index="s", columns="method", values=col)
            ref = x["rqc"].mean()
            for k in ["rqc3", "ewrqc", "lcs", "aci", "tspc", "tspc_u"]:
                d = (x[k] - x["rqc"]).dropna().to_numpy()
                mean, se, df = ewc(d)
                tcrit = stats.t.ppf(0.95, df)
                lo, hi = (mean - tcrit * se) / ref, (mean + tcrit * se) / ref
                rows.append({"country": c, "base": b, "method": k, "rel_diff": mean / ref, "ci90_lo": lo, "ci90_hi": hi,
                             "equivalent": (lo > -MARGIN) and (hi < MARGIN)})
    return pd.DataFrame(rows)


def persistence_rule(a, h, countries):
    rows = []
    for c in countries:
        m = load(c, a, h)
        if m is None:
            continue
        for b in BASES:
            x = m[m.base == b]
            rho = x["rho_pre"].iloc[0]
            p = x.pivot_table(index="s", columns="method", values="loss")
            rows.append({"country": c, "base": b, "rho_pre": rho, "tspc_wins": bool(p["tspc"].mean() < p["rqc"].mean())})
    r = pd.DataFrame(rows)
    if r.empty:
        return None
    r["predicted"] = r["rho_pre"] > 0.5
    tab = pd.crosstab(r["predicted"], r["tspc_wins"]).reindex(index=[False, True], columns=[False, True], fill_value=0)
    p = stats.fisher_exact(tab.to_numpy())[1]
    return r, tab, p


def main():
    designs = [(0.90, 3), (0.75, 3), (0.95, 3), (0.90, 1), (0.90, 6)]
    allrows, summary = [], []
    for a, h in designs:
        for col in ["loss", "loss_rel"]:
            label = "primary" if (a, h) == (0.90, 3) and col == "loss" else f"a{a}_h{h}_{col}"
            for k in ["rqc", "rqc3", "ewrqc", "lcs", "aci", "tspc", "tspc_ar", "tspc_u", "tspc_aru"]:
                r = directional(a, h, k, "none", ["break"], col)
                if len(r):
                    r["hypothesis"], r["design"] = "H1", label
                    allrows.append(r)
                    neg, n, ps = sign_test(r)
                    summary.append({"design": label, "hypothesis": "H1", "comparison": f"{k}-none", "favour": f"{neg}/{n}",
                                    "sign_p": ps, "holm_better": int(r.sig_better.sum()), "holm_worse": int(r.sig_worse.sum()),
                                    "median_rel_diff": r.rel_diff.median()})
            for hyp, m1, m2, per in [("H2", "tspc_u", "rqc", ["break"]), ("H3", "tspc_aru", "rqc", ["stable"])]:
                r = directional(a, h, m1, m2, per, col)
                if len(r):
                    r["hypothesis"], r["design"] = hyp, label
                    allrows.append(r)
                    neg, n, ps = sign_test(r)
                    summary.append({"design": label, "hypothesis": hyp, "comparison": f"{m1}-{m2}", "favour": f"{neg}/{n}",
                                    "sign_p": ps, "holm_better": int(r.sig_better.sum()), "holm_worse": int(r.sig_worse.sum()),
                                    "median_rel_diff": r.rel_diff.median()})
            eq = equivalence(a, h, COUNTRIES + ["NGA_POST"], col)
            if len(eq):
                eq["design"] = label
                eq.to_csv(EXT / f"equivalence_{label}.csv", index=False)
                for k, g in eq.groupby("method"):
                    summary.append({"design": label, "hypothesis": "H4", "comparison": f"{k}-rqc",
                                    "favour": f"{int(g.equivalent.sum())}/{len(g)} equivalent",
                                    "median_rel_diff": g.rel_diff.median()})
            if col == "loss":
                res = persistence_rule(a, h, COUNTRIES + ["NGA_POST"])
                if res is None:
                    continue
                pr, tab, pf = res
                pr["design"] = label
                pr.to_csv(EXT / f"persistence_{label}.csv", index=False)
                summary.append({"design": label, "hypothesis": "H5", "comparison": "rho_pre>0.5 predicts TSPC<RQC",
                                "favour": f"pred&win {tab.loc[True, True]}, pred&lose {tab.loc[True, False]}, "
                                          f"nopred&win {tab.loc[False, True]}, nopred&lose {tab.loc[False, False]}",
                                "sign_p": pf})
    pd.concat(allrows).to_csv(EXT / "directional_tests.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(EXT / "hypothesis_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(s[s.design == "primary"].to_string(index=False))


if __name__ == "__main__":
    main()
