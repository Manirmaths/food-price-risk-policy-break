"""LaTeX tables for the two-scale panel calibration and the simulation study."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS, TABLES = ROOT / "results", ROOT / "tables"
TABLES.mkdir(exist_ok=True)

BASE_LABELS = [("static", "Empirical quantile, frozen"),
               ("linear_qr", "Linear QR, frozen"), ("linear_qr_expand", "Linear QR, expanding"),
               ("linear_qr_roll", "Linear QR, rolling"),
               ("hist_gb", "Boosting, frozen"), ("hist_gb_expand", "Boosting, expanding"),
               ("hist_gb_roll", "Boosting, rolling")]
METHODS = [("none", "None"), ("rqc", "RQC"), ("aci", "ACI"), ("tspc", "TSPC"), ("tspc_ls", "TSPC-LS")]


def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def table_primary(s):
    s = s[(s.alpha == 0.9) & (s.horizon == 3)].set_index(["base", "period", "method"])["loss"]
    rows = []
    for b, label in BASE_LABELS:
        cells = []
        for period in ["stable", "break"]:
            vals = {m: s.get((b, period, m), np.nan) for m, _ in METHODS}
            corrected = {m: v for m, v in vals.items() if m != "none" and np.isfinite(v)}
            best = min(corrected, key=corrected.get)
            cells += [("\\textbf{" + f3(v) + "}") if m == best else f3(v) for m, v in vals.items()]
        rows.append(label + " & " + " & ".join(cells) + r" \\")
    head = " & ".join(l for _, l in METHODS)
    tex = r"""\begin{table*}[t]
\caption{Delayed calibration of each base forecast, $\alpha=0.90$, $h=3$.\label{tab:tspc}}
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrrrrrrrrr@{}}
\toprule
 & \multicolumn{5}{c}{\textbf{Stable targets}} & \multicolumn{5}{c}{\textbf{Break targets}} \\
\cmidrule(lr){2-6}\cmidrule(l){7-11}
\textbf{Base forecast} & """ + head + " & " + head + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\begin{tablenotes}
\item \textit{Note:} Mean pinball loss. RQC: pooled residual-quantile correction; ACI: adaptive conformal inference ($\gamma=0.05$); TSPC: two-scale panel calibration; TSPC-LS: TSPC with a cross-sectional scale. All corrections use the same 12 months of matured residuals. The lowest loss among the corrections is in bold. The empirical quantile has no ACI variant because its forecast is constant.
\end{tablenotes}
\end{table*}
"""
    (TABLES / "tab_tspc.tex").write_text(tex)


def table_designs(s, pers):
    rows = []
    allp = []
    for a in [0.75, 0.90, 0.95]:
        for h in [1, 3, 6]:
            x = s[(s.alpha == a) & (s.horizon == h)].pivot_table(index=["base", "period"], columns="method", values="loss")
            al = x.xs("all", level="period")
            st = x.xs("stable", level="period")
            br = x.xs("break", level="period")
            wins = int((al.tspc < al.rqc).sum())
            p = pers[(pers.a == a) & (pers.h == h)]
            r = [f"{a:.2f}", str(h), f"{wins}/{len(al)}",
                 f"{(st.tspc / st.rqc).mean():.2f}", f"{(br.tspc / br.rqc).mean():.2f}", f"{(al.tspc / al.rqc).mean():.2f}",
                 f"{(st.aci / st.rqc).mean():.2f}", f"{(br.aci / br.rqc).mean():.2f}", f"{(al.aci / al.rqc).mean():.2f}",
                 f"{p.rho.mean():.2f}"]
            rows.append(" & ".join(r) + r" \\")
            allp.append(x)
    x = pd.concat(allp)
    al, st, br = (x.xs(k, level="period") for k in ["all", "stable", "break"])
    tot = ["All", "", f"{int((al.tspc < al.rqc).sum())}/{len(al)}",
           f"{(st.tspc / st.rqc).mean():.2f}", f"{(br.tspc / br.rqc).mean():.2f}", f"{(al.tspc / al.rqc).mean():.2f}",
           f"{(st.aci / st.rqc).mean():.2f}", f"{(br.aci / br.rqc).mean():.2f}", f"{(al.aci / al.rqc).mean():.2f}",
           f"{pers.rho.mean():.2f}"]
    tex = r"""\begin{table}[t]
\caption{TSPC and ACI relative to RQC across quantile levels and horizons.\label{tab:designs}}
\centering\footnotesize\setlength{\tabcolsep}{2.6pt}
\begin{tabular*}{\columnwidth}{@{\extracolsep\fill}ccc rrr rrr r@{}}
\toprule
 & & & \multicolumn{3}{c}{\textbf{TSPC / RQC}} & \multicolumn{3}{c}{\textbf{ACI / RQC}} & \\
\cmidrule(lr){4-6}\cmidrule(lr){7-9}
$\alpha$ & $h$ & \textbf{Wins} & \textbf{Stable} & \textbf{Break} & \textbf{All} & \textbf{Stable} & \textbf{Break} & \textbf{All} & $\widehat\rho_h$ \\
\midrule
""" + "\n".join(rows) + "\n\\midrule\n" + " & ".join(tot) + r""" \\
\bottomrule
\end{tabular*}
\begin{tablenotes}
\item \textit{Note:} Ratios are mean pinball loss relative to RQC, averaged over base forecasts (seven for the primary design, which includes rolling refits, five otherwise); ACI ratios exclude the empirical quantile. Wins counts base forecasts for which TSPC has lower whole-sample loss than RQC. $\widehat\rho_h$ is the mean lag-$h$ autocorrelation of the monthly median residual, computed ex post over the full residual history.
\end{tablenotes}
\end{table}
"""
    (TABLES / "tab_designs.tex").write_text(tex)


def table_simulation(sim):
    labels = {"stable": "Stationary level", "level": "Level break", "scale": "Scale break",
              "contam": "Forecaster change", "idio": "No common shocks"}
    rows = []
    for sc in ["stable", "level", "scale", "contam", "idio"]:
        n = 40 if sc == "idio" else 1600
        for i, h in enumerate([1, 3, 6]):
            x = sim[(sim.scenario == sc) & (sim.cells == n) & (sim.horizon == h)].set_index("method")
            name = f"{labels[sc]} ($N={n:,}$)" if i == 0 else ""
            rows.append(" & ".join([name, str(h), f"{x.loc['rqc', 'loss']:.2f}",
                                    *[f"{x.loc[m, 'ratio_rqc']:.2f}" for m in ["aci", "tspc", "tspc_ls", "tspc_ar"]],
                                    f"{x.loc['rqc', 'coverage']:.2f}", f"{x.loc['tspc', 'coverage']:.2f}"]) + r" \\")
        rows.append(r"\addlinespace")
    tex = r"""\begin{table}[t]
\caption{Simulation: pinball loss relative to RQC.\label{tab:sim}}
\centering\footnotesize\setlength{\tabcolsep}{2.4pt}
\begin{tabular*}{\columnwidth}{@{\extracolsep\fill}lc r rrrr rr@{}}
\toprule
 & & \textbf{RQC} & \multicolumn{4}{c}{\textbf{Loss relative to RQC}} & \multicolumn{2}{c}{\textbf{Coverage}} \\
\cmidrule(lr){4-7}\cmidrule(l){8-9}
\textbf{Scenario} & $h$ & \textbf{loss} & \textbf{ACI} & \textbf{TSPC} & \textbf{-LS} & \textbf{-AR} & \textbf{RQC} & \textbf{TSPC} \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular*}
\begin{tablenotes}
\item \textit{Note:} Means over 500 replications of the mean pinball loss ($\alpha=0.90$) in the evaluation months of each scenario, expressed relative to RQC within each replication. -LS and -AR are TSPC with a cross-sectional scale and with an autoregressive level forecast. Nominal coverage is 0.90.
\end{tablenotes}
\end{table}
"""
    (TABLES / "tab_sim.tex").write_text(tex)


def pv(p):
    return "--" if pd.isna(p) else ("$<$.001" if p < 0.001 else f"{p:.3f}".lstrip("0"))


def table_tests():
    li = pd.read_csv(RESULTS / "loss_inference.csv")
    li = li[(li.alpha.round(2) == 0.9) & (li.horizon == 3)]
    tt = pd.read_csv(RESULTS / "tspc_tests.csv")
    tt = tt[(tt.alpha.round(2) == 0.9) & (tt.horizon == 3)].rename(columns={"diff": "estimate"})
    old = [("Expanding - frozen (linear QR)", r"Expanding $-$ frozen, linear QR"),
           ("Expanding - frozen (boosting)", r"Expanding $-$ frozen, boosting"),
           ("RQC gain, expanding (linear QR)", r"Expanding + RQC $-$ expanding, linear QR"),
           ("RQC gain, expanding (boosting)", r"Expanding + RQC $-$ expanding, boosting"),
           ("RQC - ACI, expanding (boosting)", r"RQC $-$ ACI, expanding boosting"),
           ("Expanding boosting - expanding linear QR", r"Expanding boosting $-$ expanding linear QR")]
    new = [("static", "rqc", r"TSPC $-$ RQC, empirical quantile, frozen"),
           ("linear_qr", "rqc", r"TSPC $-$ RQC, linear QR, frozen"),
           ("linear_qr", "aci", r"TSPC $-$ ACI, linear QR, frozen"),
           ("hist_gb", "rqc", r"TSPC $-$ RQC, boosting, frozen"),
           ("linear_qr_expand", "rqc", r"TSPC $-$ RQC, linear QR, expanding"),
           ("hist_gb_expand", "rqc", r"TSPC $-$ RQC, boosting, expanding"),
           ("hist_gb_expand", "aci", r"TSPC $-$ ACI, boosting, expanding")]

    def cells(st, br):
        return [f"{st.estimate:.3f}", pv(st.p_ewc), f"{br.estimate:.3f}", pv(br.p_nw), pv(br.p_hln), pv(br.p_ewc)]

    rows = [r"\multicolumn{7}{@{}l}{\textit{Re-estimation and pooled corrections}} \\"]
    for c, label in old:
        x = li[li.comparison == c].set_index("period")
        rows.append(label + " & " + " & ".join(cells(x.loc["stable"], x.loc["break"])) + r" \\")
    rows.append(r"\addlinespace")
    rows.append(r"\multicolumn{7}{@{}l}{\textit{Two-scale panel calibration}} \\")
    for b, ref, label in new:
        x = tt[(tt.base == b) & (tt.comparison == f"tspc-{ref}")].set_index("period")
        rows.append(label + " & " + " & ".join(cells(x.loc["stable"], x.loc["break"])) + r" \\")
    tex = r"""\begin{table*}[t]
\caption{Month-level tests of equal predictive accuracy, $\alpha=0.90$, $h=3$.\label{tab:tests}}
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrcrccc@{}}
\toprule
 & \multicolumn{2}{c}{\textbf{Stable (26 months)}} & \multicolumn{4}{c}{\textbf{Break (11 months)}} \\
\cmidrule(lr){2-3}\cmidrule(l){4-7}
\textbf{Loss difference} & \textbf{Mean} & \textbf{$p_{\mathrm{EWC}}$} & \textbf{Mean} & \textbf{$p_{\mathrm{NW}}$} & \textbf{$p_{\mathrm{HLN}}$} & \textbf{$p_{\mathrm{EWC}}$} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\begin{tablenotes}
\item \textit{Note:} Each row is the mean over origin months of the first forecast's mean pinball loss minus the second's; negative values favour the first. $p_{\mathrm{NW}}$: Newey--West with $h-1$ lags; $p_{\mathrm{HLN}}$: Diebold--Mariano with the Harvey--Leybourne--Newbold correction; $p_{\mathrm{EWC}}$: fixed-smoothing test with the equal-weighted cosine variance and $B=\lfloor T^{1/2}\rfloor$. All use Student-$t$ critical values.
\end{tablenotes}
\end{table*}
"""
    (TABLES / "tab_tests2.tex").write_text(tex)


def main():
    table_tests()
    s = pd.read_csv(RESULTS / "tspc_summary.csv")
    table_primary(s)
    pers = pd.read_csv(RESULTS / "tspc_persistence.csv")
    table_designs(s, pers)
    if (RESULTS / "simulation.csv").exists():
        table_simulation(pd.read_csv(RESULTS / "simulation.csv"))


if __name__ == "__main__":
    main()
