"""Write every manuscript table as a LaTeX fragment from the results files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUILD, RESULTS = ROOT / "build", ROOT / "results"
OUT = ROOT / "tables"
OUT.mkdir(exist_ok=True)

LEARN = {"static": "Empirical", "linear_qr": "Linear QR", "hist_gb": "Gradient boosting"}


def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def f2(x):
    return "--" if pd.isna(x) else f"{x:.2f}"


def pval(p):
    if pd.isna(p):
        return "--"
    return "$<$.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def summary(tag="a90_h3"):
    s = pd.read_csv(RESULTS / f"summary_{tag}.csv").drop_duplicates(["period", "model"])
    return s.set_index(["model", "period"])


def table_descriptive():
    f = pd.read_csv(BUILD / "forecasts_a90_h3.csv.gz", parse_dates=["date", "target_date"])
    rows = []
    for period, label in [("stable", "Stable targets (Apr 2021--May 2023)"), ("break", "Break targets (Jun 2023--Apr 2024)")]:
        g = f[f["period"].eq(period)]
        y = g["y"]
        rows.append([label, f"{len(g):,}", f2(y.mean()), f2(y.std()), f2(y.quantile(0.5)), f2(y.quantile(0.9)),
                     f2(100 * np.mean(y > 20)), f2(g.groupby("date")["y"].quantile(0.9).std())])
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    tex = r"""\begin{table*}[t]
\caption{Three-month log price growth by evaluation period.\label{tab:desc}}
\centering\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrrrrrr@{}}
\toprule
 & \textbf{Forecasts} & \textbf{Mean} & \textbf{SD} & \textbf{Median} & \textbf{90th pct.} & \textbf{Share $>$20\% (\%)} & \textbf{SD of monthly 90th pct.} \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular*}
\begin{tablenotes}
\item \textit{Note:} Statistics pool all state--commodity outcomes $y^{(3)}_{sit}$ (percentage points) whose target month falls in each period. The last column is the standard deviation across origin months of the cross-sectional 90th percentile.
\end{tablenotes}
\end{table*}
"""
    (OUT / "tab_desc.tex").write_text(tex)


def table_main():
    s = summary()
    rows = []
    rows.append(r"\multicolumn{6}{@{}l}{\textit{Unconditional benchmarks}} \\")
    for m, lab in [("static", "Empirical q90, frozen"), ("static_item", "Commodity q90, frozen"),
                   ("rolling_q", "Empirical q90, rolling 12 months")]:
        rows.append(f"{lab} & & " + " & ".join(f3(s.loc[(m, p), k]) for p in ["stable", "break"] for k in ["loss", "coverage"]) + r" \\")
    for learner in ["linear_qr", "hist_gb"]:
        rows.append(r"\addlinespace" + f"\n\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{LEARN[learner]}}}}} \\\\")
        for scheme, suf in [("Frozen", ""), ("Expanding", "_expand"), ("Rolling (24 months)", "_roll")]:
            for corr, csuf in [("None", ""), ("RQC", "_adapt"), ("ACI", "_aci")]:
                m = learner + suf + csuf
                first = scheme if corr == "None" else ""
                cells = []
                for p in ["stable", "break"]:
                    for k in ["loss", "coverage"]:
                        v = f3(s.loc[(m, p), k])
                        cells.append(v)
                rows.append(f"{first} & {corr} & " + " & ".join(cells) + r" \\")
    # bold best break loss
    best = min(s.loc[(m, "break"), "loss"] for m in s.index.get_level_values(0).unique() if not m.split("_")[-1].startswith(("w", "g")))
    body = "\n".join(rows).replace(f3(best), r"\textbf{" + f3(best) + "}", 1)
    tex = r"""\begin{table*}[t]
\caption{Out-of-sample performance of 90th-percentile forecasts of three-month price growth.\label{tab:main}}
\centering\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}llrrrr@{}}
\toprule
 & & \multicolumn{2}{c}{\textbf{Stable targets}} & \multicolumn{2}{c}{\textbf{Break targets}} \\
\cmidrule(lr){3-4}\cmidrule(l){5-6}
\textbf{Estimation} & \textbf{Correction} & \textbf{Loss} & \textbf{Coverage} & \textbf{Loss} & \textbf{Coverage} \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular*}
\begin{tablenotes}
\item \textit{Note:} Loss is the mean pinball loss at $\alpha=0.90$; coverage is the share of outcomes at or below the forecast (nominal 0.90). There are 41,363 stable and 17,501 break forecasts. RQC is the residual-quantile correction with a 12-month window; ACI is adaptive conformal inference on the same window with $\gamma=0.05$. Every refit and correction uses only targets that had matured by the forecast origin. The lowest break-period loss is in bold.
\end{tablenotes}
\end{table*}
"""
    (OUT / "tab_main.tex").write_text(tex)


KEY = [
    ("Frozen boosting - frozen empirical", "Frozen boosting $-$ frozen empirical"),
    ("Expanding - frozen (linear QR)", "Expanding $-$ frozen, linear QR"),
    ("Expanding - frozen (boosting)", "Expanding $-$ frozen, boosting"),
    ("Rolling - expanding (linear QR)", "Rolling $-$ expanding, linear QR"),
    ("Rolling - expanding (boosting)", "Rolling $-$ expanding, boosting"),
    ("RQC gain, expanding (linear QR)", "Expanding + RQC $-$ expanding, linear QR"),
    ("RQC gain, expanding (boosting)", "Expanding + RQC $-$ expanding, boosting"),
    ("RQC gain, rolling (boosting)", "Rolling + RQC $-$ rolling, boosting"),
    ("ACI gain, expanding (boosting)", "Expanding + ACI $-$ expanding, boosting"),
    ("RQC - ACI, expanding (boosting)", "RQC $-$ ACI, expanding boosting"),
    ("Expanding+RQC linear QR - rolling empirical", "Expanding + RQC linear QR $-$ rolling empirical"),
    ("Expanding+RQC boosting - rolling empirical", "Expanding + RQC boosting $-$ rolling empirical"),
    ("Expanding boosting - expanding linear QR", "Expanding boosting $-$ expanding linear QR"),
]


def table_tests():
    t = pd.read_csv(RESULTS / "loss_inference.csv")
    t = t[(t["alpha"].round(2) == 0.90) & (t["horizon"] == 3)].set_index(["period", "comparison"])
    rows = []
    for key, lab in KEY:
        cells = []
        for p in ["stable", "break"]:
            r = t.loc[(p, key)]
            cells += [f"{r['estimate']:.3f}", f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]"] if p == "stable" else \
                     [f"{r['estimate']:.3f}", f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]", pval(r["p_nw"]), pval(r["p_hln"]), pval(r["p_ewc"])]
        rows.append(lab + " & " + " & ".join(cells) + r" \\")
    tex = r"""\begin{table*}[t]
\caption{Month-level tests of equal predictive accuracy.\label{tab:tests}}
\centering\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrcrcccc@{}}
\toprule
 & \multicolumn{2}{c}{\textbf{Stable (26 months)}} & \multicolumn{5}{c}{\textbf{Break (11 months)}} \\
\cmidrule(lr){2-3}\cmidrule(l){4-8}
\textbf{Loss difference} & \textbf{Mean} & \textbf{95\% CI} & \textbf{Mean} & \textbf{95\% CI} & \textbf{$p_{\mathrm{NW}}$} & \textbf{$p_{\mathrm{HLN}}$} & \textbf{$p_{\mathrm{EWC}}$} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\begin{tablenotes}
\item \textit{Note:} Each row is the mean over origin months of the first model's mean pinball loss minus the second model's; negative values favour the first model. Intervals use Newey--West standard errors with two lags and Student-$t$ critical values. $p_{\mathrm{NW}}$: Newey--West; $p_{\mathrm{HLN}}$: Diebold--Mariano with the \citet{harvey1997} correction; $p_{\mathrm{EWC}}$: fixed-smoothing test with the equal-weighted cosine variance estimator and $B=\lfloor T^{1/2}\rfloor$ \citep{lazarus2018,coroneo2020}.
\end{tablenotes}
\end{table*}
"""
    (OUT / "tab_tests.tex").write_text(tex)


def table_decomp():
    d = pd.read_csv(RESULTS / "decomposition_summary.csv").set_index(["model", "period"])
    p = pd.read_csv(RESULTS / "placebo_summary.csv").set_index("model")
    rows = []
    for m, lab in [("static", "Empirical, frozen"), ("linear_qr", "Linear QR, frozen"), ("linear_qr_expand", "Linear QR, expanding"),
                   ("linear_qr_roll", "Linear QR, rolling"), ("hist_gb", "Boosting, frozen"), ("hist_gb_expand", "Boosting, expanding"),
                   ("hist_gb_roll", "Boosting, rolling"), ("hist_gb_expand_adapt", "Boosting, expanding + RQC")]:
        st, br, pl = d.loc[(m, "stable")], d.loc[(m, "break")], p.loc[m]
        rows.append(f"{lab} & {st['common_share']:.2f} & {br['common_share']:.2f} & [{br['common_share_lo']:.2f}, {br['common_share_hi']:.2f}] & "
                    f"{pl['placebo_min']:.2f}--{pl['placebo_max']:.2f} & {'Yes' if pl['rank_above_all'] else 'No'} & "
                    f"{br['mean_common_shift']:.1f} & {st['mean_resid_sd']:.1f} & {br['mean_resid_sd']:.1f}" + r" \\")
    tex = r"""\begin{table*}[t]
\caption{Ex-post decomposition of forecast errors.\label{tab:decomp}}
\centering\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrcccrrr@{}}
\toprule
 & \multicolumn{5}{c}{\textbf{Common share $S$}} & \textbf{Oracle shift} & \multicolumn{2}{c}{\textbf{Residual SD}} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-7}\cmidrule(l){8-9}
\textbf{Base forecast} & \textbf{Stable} & \textbf{Break} & \textbf{95\% CI} & \textbf{Placebo range} & \textbf{Above all} & \textbf{Break} & \textbf{Stable} & \textbf{Break} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\begin{tablenotes}
\item \textit{Note:} $S$ is the share of the loss reduction available from commodity-specific hindsight shifts that one common hindsight shift delivers (equation~\ref{eq:share}). Intervals use 20,000 circular block-bootstrap draws over origin months (block length three). The placebo range covers the 16 rolling 11-month windows of stable months. The oracle shift is the mean common hindsight shift in percentage points. Residual SD is the mean within-month cross-sectional standard deviation of $y-q$.
\end{tablenotes}
\end{table*}
"""
    (OUT / "tab_decomp.tex").write_text(tex)


def table_windows():
    s = summary()
    rows = []
    for b, lab in [("linear_qr", "Linear QR, frozen"), ("linear_qr_expand", "Linear QR, expanding"),
                   ("hist_gb", "Boosting, frozen"), ("hist_gb_expand", "Boosting, expanding")]:
        cells = [f3(s.loc[(b, "stable"), "loss"]), f3(s.loc[(b, "break"), "loss"])]
        for w in [6, 12, 18, 24]:
            m = f"{b}_adapt_w{w}"
            cells += [f3(s.loc[(m, "stable"), "loss"]), f3(s.loc[(m, "break"), "loss"])]
        rows.append(lab + " & " + " & ".join(cells) + r" \\")
    tex = r"""\begin{table*}[t]
\caption{Sensitivity of the residual-quantile correction to the residual window.\label{tab:windows}}
\centering\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrrrrrrrrr@{}}
\toprule
 & \multicolumn{2}{c}{\textbf{No correction}} & \multicolumn{2}{c}{\textbf{$W=6$}} & \multicolumn{2}{c}{\textbf{$W=12$}} & \multicolumn{2}{c}{\textbf{$W=18$}} & \multicolumn{2}{c}{\textbf{$W=24$}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(l){10-11}
\textbf{Base forecast} & \textbf{Stable} & \textbf{Break} & \textbf{Stable} & \textbf{Break} & \textbf{Stable} & \textbf{Break} & \textbf{Stable} & \textbf{Break} & \textbf{Stable} & \textbf{Break} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\begin{tablenotes}
\item \textit{Note:} Mean pinball loss ($\alpha=0.90$, $h=3$). $W$ is the number of months of matured residuals used by the residual-quantile correction. Only the look-back length changes; base forecasts, the maturity rule and the evaluation sample are fixed.
\end{tablenotes}
\end{table*}
"""
    (OUT / "tab_windows.tex").write_text(tex)


def table_robust():
    t = pd.read_csv(RESULTS / "loss_inference.csv")
    t = t[t["period"].eq("break")]
    rows = []
    for a in [0.75, 0.90, 0.95]:
        for h in [1, 3, 6]:
            try:
                s = summary(f"a{int(round(a * 100))}_h{h}")
            except FileNotFoundError:
                continue
            tt = t[(t["alpha"].round(2) == a) & (t["horizon"] == h)].set_index("comparison")
            cells = [f"{a:.2f}", str(h)]
            for learner, key_e, key_r in [("linear_qr", "Expanding - frozen (linear QR)", "RQC gain, expanding (linear QR)"),
                                          ("hist_gb", "Expanding - frozen (boosting)", "RQC gain, expanding (boosting)")]:
                fr = s.loc[(learner, "break"), "loss"]
                ex = s.loc[(learner + "_expand", "break"), "loss"]
                er = s.loc[(learner + "_expand_adapt", "break"), "loss"]
                cells += [f3(fr), f3(ex), f3(er), f"{100 * (1 - er / ex):.1f}", pval(tt.loc[key_r, "p_ewc"])]
            rows.append(" & ".join(cells) + r" \\")
    tex = r"""\begin{table*}[t]
\caption{Break-period results across quantile levels and horizons.\label{tab:robust}}
\centering\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}ccrrrrcrrrrc@{}}
\toprule
 & & \multicolumn{5}{c}{\textbf{Linear QR}} & \multicolumn{5}{c}{\textbf{Gradient boosting}} \\
\cmidrule(lr){3-7}\cmidrule(l){8-12}
$\alpha$ & $h$ & \textbf{Frozen} & \textbf{Exp.} & \textbf{Exp.+RQC} & \textbf{Gain (\%)} & $p_{\mathrm{EWC}}$ & \textbf{Frozen} & \textbf{Exp.} & \textbf{Exp.+RQC} & \textbf{Gain (\%)} & $p_{\mathrm{EWC}}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\begin{tablenotes}
\item \textit{Note:} Break-period mean pinball loss at quantile level $\alpha$ and horizon $h$ months. Gain is the percentage reduction in loss from adding the 12-month residual-quantile correction to the expanding-window forecast; $p_{\mathrm{EWC}}$ is the fixed-smoothing $p$-value for that difference. Hyperparameters are those selected for $\alpha=0.90$, $h=3$.
\end{tablenotes}
\end{table*}
"""
    (OUT / "tab_robust.tex").write_text(tex)


def table_aci():
    s = summary()
    rows = []
    for b, lab in [("linear_qr", "Linear QR, frozen"), ("linear_qr_expand", "Linear QR, expanding"),
                   ("hist_gb", "Boosting, frozen"), ("hist_gb_expand", "Boosting, expanding")]:
        cells = [f3(s.loc[(b + "_adapt", "break"), "loss"])]
        for g in ["0.01", "0.025", "0.05", "0.1", "0.2"]:
            cells.append(f3(s.loc[(f"{b}_aci_g{g}", "break"), "loss"]))
        cells.append(f3(s.loc[(b + "_adapt", "stable"), "loss"]))
        for g in ["0.01", "0.05", "0.2"]:
            cells.append(f3(s.loc[(f"{b}_aci_g{g}", "stable"), "loss"]))
        rows.append(lab + " & " + " & ".join(cells) + r" \\")
    tex = r"""\begin{table*}[t]
\caption{Adaptive conformal inference: sensitivity to the step size $\gamma$.\label{tab:aci}}
\centering\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrrrrrrrrr@{}}
\toprule
 & \multicolumn{6}{c}{\textbf{Break targets}} & \multicolumn{4}{c}{\textbf{Stable targets}} \\
\cmidrule(lr){2-7}\cmidrule(l){8-11}
\textbf{Base forecast} & \textbf{RQC} & \textbf{0.01} & \textbf{0.025} & \textbf{0.05} & \textbf{0.10} & \textbf{0.20} & \textbf{RQC} & \textbf{0.01} & \textbf{0.05} & \textbf{0.20} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\begin{tablenotes}
\item \textit{Note:} Mean pinball loss ($\alpha=0.90$, $h=3$). Columns headed by a number report ACI with that step size $\gamma$; RQC is the fixed-level residual-quantile correction. Both use a 12-month residual window.
\end{tablenotes}
\end{table*}
"""
    (OUT / "tab_aci.tex").write_text(tex)


def table_coverage():
    c = pd.read_csv(RESULTS / "coverage_inference.csv")
    c = c[(c["alpha"].round(2) == 0.90) & (c["horizon"] == 3) & c["period"].eq("break")].set_index("model")
    rows = []
    for m, lab in [("rolling_q", "Empirical q90, rolling"), ("linear_qr_expand", "Linear QR, expanding"),
                   ("linear_qr_expand_adapt", "Linear QR, expanding + RQC"), ("linear_qr_expand_aci", "Linear QR, expanding + ACI"),
                   ("hist_gb_expand", "Boosting, expanding"), ("hist_gb_expand_adapt", "Boosting, expanding + RQC"),
                   ("hist_gb_expand_aci", "Boosting, expanding + ACI")]:
        r = c.loc[m]
        rows.append(f"{lab} & {r['estimate']:.3f} & [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] & {pval(r['p_nw'])} & {pval(r['p_hln'])} & {pval(r['p_ewc'])}" + r" \\")
    tex = r"""\begin{table}[t]
\caption{Break-period coverage tests against the nominal 0.90.\label{tab:coverage}}
\centering\footnotesize
\begin{tabular*}{\columnwidth}{@{\extracolsep\fill}lrcccc@{}}
\toprule
\textbf{Forecast} & \textbf{Cov.} & \textbf{95\% CI} & $p_{\mathrm{NW}}$ & $p_{\mathrm{HLN}}$ & $p_{\mathrm{EWC}}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\begin{tablenotes}
\item \textit{Note:} Mean of monthly coverage over the 11 break origin months, with tests of the null that it equals 0.90.
\end{tablenotes}
\end{table}
"""
    (OUT / "tab_coverage.tex").write_text(tex)


if __name__ == "__main__":
    table_descriptive()
    table_main()
    table_tests()
    table_decomp()
    table_windows()
    table_aci()
    table_coverage()
    table_robust()
