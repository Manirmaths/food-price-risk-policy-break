"""Tables and figure for the pre-specified external evaluation (PROTOCOL.md)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "results" / "external"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tables"
FIG = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

COUNTRIES = ["NGA", "LBN", "SDN", "HTI", "MMR", "AFG", "NGA_POST"]
CNAME = {"NGA": "Nigeria", "LBN": "Lebanon", "SDN": "Sudan", "HTI": "Haiti", "MMR": "Myanmar",
         "AFG": "Afghanistan", "NGA_POST": "Nigeria, 2024--26"}
EVENT = {"NGA": ("Subsidy removal", "Jun 2023"),
         "LBN": ("Financial crisis", "Nov 2019"),
         "SDN": ("Outbreak of war", "May 2023"),
         "HTI": ("Fuel price increase", "Oct 2022"),
         "MMR": ("Military coup", "Mar 2021"),
         "AFG": ("Change of government", "Sep 2021"),
         "NGA_POST": ("None designated", "--")}
FOODS = {"NGA": 16, "LBN": 4, "SDN": 10, "HTI": 9, "MMR": 5, "AFG": 13, "NGA_POST": 16}
MARKETS = {"NGA": 74, "LBN": 28, "SDN": 20, "HTI": 50, "MMR": 301, "AFG": 41, "NGA_POST": 74}
LABEL = {"none": "No correction", "rqc": "RQC, 12 months", "rqc3": "RQC, 3 months",
         "ewrqc": "RQC, exp.\\ weighted", "lcs": "Latest cross-section", "aci": "ACI",
         "tspc": "TSPC", "tspc_ar": "TSPC-AR", "tspc_u": "TSPC-U", "tspc_aru": "TSPC-ARU"}
BASES = ["empirical", "linear_qr", "hist_gb"]


def load(c, a=0.90, h=3):
    f = EXT / f"{c}_a{int(round(a * 100))}_h{h}_monthly.csv"
    return pd.read_csv(f) if f.exists() else None


def num(v, d=2):
    return f"{v:.{d}f}".replace("-", "$-$")


def fmt_p(p):
    return "--" if pd.isna(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


def panels():
    rows = []
    for c in COUNTRIES:
        m = load(c)
        x = m[(m.method == "rqc") & (m.base == "empirical")]
        rho = m.groupby("base")["rho_pre"].first()
        ev, bt = EVENT[c]
        s0, s1 = int(x.s.min()), int(x.s.max())
        span = f"{(s0 - 1) % 12 + 1}/{(s0 - 1) // 12 % 100:02d}--{(s1 - 1) % 12 + 1}/{(s1 - 1) // 12 % 100:02d}"
        rows.append(f"{CNAME[c]} & {ev} & {bt} & {MARKETS[c]} & {FOODS[c]} & {int(x.n.median()):,} & "
                    f"{span} & {num(rho.min())}, {num(rho.max())} \\\\")
    tex = r"""\begin{table*}[t]
\caption{Panels of the pre-specified evaluation (World Bank RTFP, version of 2 November 2026).\label{tab:extpanels}}
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lllrrrlr@{}}
\toprule
\textbf{Panel} & \textbf{Event} & \textbf{Break} & \textbf{Markets} & \textbf{Foods} & \textbf{Cells} & \textbf{Origins} & \textbf{$\widehat\rho_{3,\mathrm{pre}}$} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\begin{tablenotes}
\item \textit{Note:} Cells: median number of market--food forecasts per origin, $\alpha=0.90$, $h=3$. Origins: first and last evaluation origin (month/year); Break: first break target month; each break panel has 24 stable and 24 break origins. $\widehat\rho_{3,\mathrm{pre}}$: smallest and largest, over the three base forecasts, of the lag-3 autocorrelation of the monthly median error, estimated from residuals matured before the first evaluation origin. The last row is the period of the Nigerian panel after the end of the data used to develop the corrections.
\end{tablenotes}
\end{table*}
"""
    (OUT / "tab_extpanels.tex").write_text(tex)


def clustered(comp):
    c = pd.read_csv(EXT / "clustered_tests.csv")
    r = c[(c.alpha == 0.90) & (c.h == 3) & (c.targets == "loss") & (c.comparison == comp)].iloc[0]
    return (f"{r.favour} & {fmt_p(r.sign_p)} & "
            f"{r.ratio:.2f} [{r.t90_lo:.2f}, {r.t90_hi:.2f}]")


def hypotheses(label="primary"):
    s = pd.read_csv(EXT / "hypothesis_summary.csv")
    s = s[s.design == label]
    lines = []
    h1 = s[s.hypothesis == "H1"]
    for _, r in h1.iterrows():
        k = r.comparison.split("-")[0]
        lines.append(f"H1 & {LABEL[k]} vs none & {r.favour} & {fmt_p(r.sign_p)} & "
                     f"{int(r.holm_better)}/{int(r.holm_worse)} & {100 * r.median_rel_diff:+.1f} & {clustered(r.comparison)} \\\\")
    lines.append(r"\addlinespace")
    for hyp, txt in [("H2", "TSPC-U vs RQC, break"), ("H3", "TSPC-ARU vs RQC, stable")]:
        r = s[s.hypothesis == hyp].iloc[0]
        lines.append(f"{hyp} & {txt} & {r.favour} & {fmt_p(r.sign_p)} & "
                     f"{int(r.holm_better)}/{int(r.holm_worse)} & {100 * r.median_rel_diff:+.1f} & {clustered(r.comparison)} \\\\")
    lines.append(r"\addlinespace")
    for k in ["rqc3", "ewrqc", "lcs", "aci", "tspc", "tspc_u"]:
        r = s[(s.hypothesis == "H4") & (s.comparison == f"{k}-rqc")].iloc[0]
        n = r.favour.split()[0]
        lines.append(f"H4 & {LABEL[k]} vs RQC & {n} & -- & -- & {100 * r.median_rel_diff:+.1f} & {clustered(r.comparison)} \\\\")
    h5 = s[s.hypothesis == "H5"].iloc[0]
    t5 = pd.read_csv(EXT / "persistence_primary.csv")
    hi = t5[t5.rho_pre > 0.5]
    lo = t5[t5.rho_pre <= 0.5]
    h5txt = (f"{len(hi)} of {len(t5)} combinations {'has' if len(hi) == 1 else 'have'} $\\widehat\\rho_{{3,\\mathrm{{pre}}}}>1/2$, and TSPC beats RQC in "
             f"{int(hi.tspc_wins.sum())} of them; of the other {len(lo)}, TSPC beats RQC in {int(lo.tspc_wins.sum())}")
    tex = r"""\begin{table}[t]
\caption{Pre-specified hypotheses, $\alpha=0.90$, $h=3$.\label{tab:exthyp}}
\centering\footnotesize\setlength{\tabcolsep}{2pt}
\begin{tabular*}{\columnwidth}{@{\extracolsep\fill}llrrrrrrr@{}}
\toprule
 & & \multicolumn{4}{c}{\textbf{Combinations}} & \multicolumn{3}{c}{\textbf{Countries}} \\
\cmidrule(lr){3-6}\cmidrule(l){7-9}
 & \textbf{Comparison} & \textbf{Fav.} & \textbf{Sign $p$} & \textbf{Holm} & \textbf{\%} & \textbf{Fav.} & \textbf{Sign $p$} & \textbf{Ratio [90\% CI]} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\begin{tablenotes}
\item \textit{Note:} H1--H3: country--base combinations, of 18, in which the first forecast has the lower mean loss; exact two-sided sign test; combinations significantly better/worse at 5\% under the fixed-smoothing test after Holm's correction across the 18; median over combinations of the mean loss difference in per cent of the second forecast's loss. H4: combinations, of 21 (including Nigeria 2024--26), in which the 90\% confidence interval of the relative difference over the whole window lies inside $\pm5\%$, and the median relative difference. Countries: each country contributes the mean over base forecasts of the log loss ratio (6 countries for H1--H3, 7 for H4, counting Nigeria 2024--26 separately); countries favourable to the first forecast, exact sign test, and the geometric-mean loss ratio with a Student-$t$ interval with countries $-1$ degrees of freedom. H5: """ + \
        f"{h5txt}; Fisher's exact $p={h5.sign_p:.2f}$." + r"""
\end{tablenotes}
\end{table}
"""
    (OUT / "tab_exthyp.tex").write_text(tex)


def by_country():
    methods = ["none", "rqc3", "ewrqc", "lcs", "aci", "tspc", "tspc_ar", "tspc_u", "tspc_aru"]
    cols, per = {}, {"stable": [], "break": []}
    for c in COUNTRIES:
        m = load(c)
        ratios = {}
        for b in BASES:
            x = m[m.base == b].pivot_table(index=["s", "period"], columns="method", values="loss").reset_index()
            for k in methods:
                ratios.setdefault(k, []).append(x[k].mean() / x["rqc"].mean())
            for pp in ["stable", "break"]:
                y = x[x.period == pp]
                if len(y):
                    per[pp].append({k: y[k].mean() / y["rqc"].mean() for k in methods})
        cols[c] = {k: np.exp(np.mean(np.log(v))) for k, v in ratios.items()}
    tab = pd.DataFrame(cols)
    for pp in ["stable", "break"]:
        tab[pp] = pd.DataFrame(per[pp]).apply(lambda v: np.exp(np.mean(np.log(v))))
    lines = []
    for k in methods:
        vals = tab.loc[k]
        row = " & ".join(f"{v:.2f}" for v in vals)
        lines.append(f"{LABEL[k]} & {row} \\\\")
        if k in ("none", "aci"):
            lines.append(r"\addlinespace")
    head = " & ".join(r"\textbf{" + c.replace("NGA_POST", "NGA$^\\ast$") + "}" for c in COUNTRIES)
    tex = r"""\begin{table*}[t]
\caption{Loss relative to the 12-month RQC in the pre-specified evaluation, $\alpha=0.90$, $h=3$.\label{tab:extcountry}}
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}l rrrrrrr rr@{}}
\toprule
 & \multicolumn{7}{c}{\textbf{Whole window}} & \multicolumn{2}{c}{\textbf{Six break panels}} \\
\cmidrule(lr){2-8}\cmidrule(l){9-10}
\textbf{Correction} & """ + head + r""" & \textbf{Stable} & \textbf{Break} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\begin{tablenotes}
\item \textit{Note:} Ratio of mean pinball loss to that of RQC, geometric mean over the empirical, linear QR and boosting base forecasts (whole window), or over the 18 country--base combinations (last two columns). NGA$^\ast$: Nigeria, origins from May 2024, after the data used to develop the corrections. Values below one favour the correction.
\end{tablenotes}
\end{table*}
"""
    (OUT / "tab_extcountry.tex").write_text(tex)
    return tab


def persistence_fig():
    p = pd.read_csv(EXT / "persistence_primary.csv")
    ratio = []
    for _, r in p.iterrows():
        m = load(r.country)
        x = m[m.base == r.base].pivot_table(index="s", columns="method", values="loss")
        ratio.append(x["tspc"].mean() / x["rqc"].mean())
    p["ratio"] = ratio
    p.to_csv(EXT / "persistence_primary_ratio.csv", index=False)
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    marks = {"empirical": ("o", "#1f4e79"), "linear_qr": ("s", "#b0413e"), "hist_gb": ("^", "#4f7f3a")}
    names = {"empirical": "Empirical", "linear_qr": "Linear QR", "hist_gb": "Boosting"}
    for b, (mk, col) in marks.items():
        q = p[p.base == b]
        ax.scatter(q.rho_pre, q.ratio, marker=mk, color=col, s=22, label=names[b], zorder=3)
        for _, r in q.iterrows():
            ax.annotate(r.country.replace("NGA_POST", "NGA*"), (r.rho_pre, r.ratio), fontsize=5.5,
                        xytext=(3, 2), textcoords="offset points", color="0.3")
    ax.axvline(0.5, ls="--", lw=0.8, color="0.4")
    ax.axhline(1.0, ls=":", lw=0.8, color="0.4")
    ax.set_xlabel(r"Lag-3 autocorrelation of median error before evaluation, $\hat\rho_{3,\mathrm{pre}}$")
    ax.set_ylabel("Loss of TSPC / loss of RQC")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_persistence.pdf")
    return p


if __name__ == "__main__":
    panels()
    hypotheses()
    t = by_country()
    print(t.round(3))
    print(persistence_fig()[["country", "base", "rho_pre", "ratio"]].round(3))
