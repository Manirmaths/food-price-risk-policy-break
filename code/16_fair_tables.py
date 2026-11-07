"""LaTeX tables for the fair evaluation (14_fair_evaluation.py) and the simulation (15_simulation.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS, TABLES = ROOT / "results", ROOT / "tables"
LAB = [("none", "No correction"), ("rqc", "RQC, 12 months"), ("rqc3", "RQC, 3 months"), ("ewrqc", "RQC, exp.\\ weighted"),
       ("lcs", "Latest cross-section"), ("aci", "ACI"), ("tspc", "TSPC"), ("tspc_ar", "TSPC-AR"),
       ("tspc_u", "TSPC-U"), ("tspc_aru", "TSPC-ARU")]


def f(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def primary():
    s = pd.read_csv(RESULTS / "fair_summary.csv")
    s = s[(s.alpha.round(2) == 0.9) & (s.horizon == 3) & (s.design == "restart")]
    x = s.set_index(["base", "period", "method"])["loss"]
    bases = [("linear_qr_expand", "Linear QR, exp."), ("hist_gb_expand", "Boosting, exp."), ("linear_qr", "Linear QR, frozen")]
    rows = []
    for m, lab in LAB:
        cells = []
        for b, _ in bases:
            for p in ["stable", "break"]:
                col = [x[(b, p, k)] for k, _ in LAB if k != "none"]
                v = x[(b, p, m)]
                cells.append(("\\textbf{" + f(v) + "}") if m != "none" and np.isclose(v, min(col)) else f(v))
        rows.append(lab + " & " + " & ".join(cells) + r" \\")
        if m in ("none", "aci"):
            rows.append(r"\addlinespace")
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{\textbf{{{l}}}}}" for _, l in bases)
    tex = r"""\begin{table*}[t]
\caption{Corrections after a calibration restart, $\alpha=0.90$, $h=3$.\label{tab:primary}}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrrrrr@{}}
\toprule
 & """ + head + r""" \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}
\textbf{Correction} & \textbf{Stable} & \textbf{Break} & \textbf{Stable} & \textbf{Break} & \textbf{Stable} & \textbf{Break} \\
\midrule
""" + "\n".join(rows[:-1] if rows[-1] == r"\addlinespace" else rows) + r"""
\bottomrule
\end{tabular*}
\begin{tablenotes}
\item \textit{Note:} Mean pinball loss. Calibration restarts in January 2021, when the evaluation forecasters are installed, so only residuals of forecasts issued from then on are used; no correction is applied before the first of them matures. Expanding (exp.) models are re-estimated monthly. The lowest loss among the corrections is in bold.
\end{tablenotes}
\end{table*}
"""
    (TABLES / "tab_primary.tex").write_text(tex)


def across():
    s = pd.read_csv(RESULTS / "fair_summary.csv")
    t = pd.read_csv(RESULTS / "fair_tests.csv")
    cols = [("full", "all"), ("restart", "all"), ("restart", "stable"), ("restart", "break"), ("restart", "clean")]
    ratio = {}
    for d, p in cols:
        x = s[(s.design == d) & (s.period == p)].pivot_table(index=["alpha", "horizon", "base"], columns="method", values="loss")
        ratio[(d, p)] = x.div(x["rqc"], axis=0).mean()
    x = s[(s.design == "restart") & (s.period == "all")].pivot_table(index=["alpha", "horizon", "base"], columns="method", values="loss")
    beats_none = x.lt(x["none"], axis=0).sum()
    tr = t[(t.design == "restart") & (t.period == "all")]
    rows = []
    for m, lab in LAB:
        sig = tr[tr.comparison == f"{m}-rqc"]
        sb = int(((sig.p_ewc < 0.05) & (sig["diff"] < 0)).sum()) if len(sig) else None
        sw = int(((sig.p_ewc < 0.05) & (sig["diff"] > 0)).sum()) if len(sig) else None
        cells = [f(ratio[c][m], 2) for c in cols]
        cells.append("--" if m == "none" else str(int(beats_none[m])))
        cells.append("--" if sb is None else f"{sb}/{sw}")
        rows.append(lab + " & " + " & ".join(cells) + r" \\")
    n = len(x)
    tex = r"""\begin{table}[t]
\caption{Loss relative to the 12-month RQC, averaged over 47 combinations of base forecast and design.\label{tab:across}}
\centering\footnotesize\setlength{\tabcolsep}{2.5pt}
\begin{tabular*}{\columnwidth}{@{\extracolsep\fill}l rrrrr rr@{}}
\toprule
 & \textbf{Full} & \multicolumn{4}{c}{\textbf{Restart}} & & \\
\cmidrule(lr){2-2}\cmidrule(lr){3-6}
\textbf{Correction} & \textbf{All} & \textbf{All} & \textbf{Stable} & \textbf{Break} & \textbf{Clean} & \textbf{$>$None} & \textbf{Sig.} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\begin{tablenotes}
\item \textit{Note:} Full: residual history includes the pre-evaluation forecasts; Restart: only residuals of forecasts issued from January 2021. Clean: origins from March 2022 (for $h=3$), whose 12-month window contains no pre-evaluation residual. $>$None: combinations, of """ + str(n) + r""", in which the correction has lower whole-sample loss than no correction (restart). Sig.: combinations in which the correction is significantly better/worse than RQC under the fixed-smoothing test at 5\% (restart, whole sample).
\end{tablenotes}
\end{table}
"""
    (TABLES / "tab_across.tex").write_text(tex)


def simulation():
    p = RESULTS / "simulation2.csv"
    if not p.exists():
        return
    sim = pd.read_csv(p)
    lab = {"stationary": "Stationary level", "persistent": "Persistent shift", "transient": "Transient shock",
           "scale": "Scale increase", "tail": "Upper tail only", "dependent": "Shift, dependent cells",
           "replacement": "Replacement, full", "replacement_restart": "Replacement, restart"}
    meth = ["rqc3", "ewrqc", "lcs", "aci", "tspc", "tspc_ar", "tspc_u", "tspc_aru"]
    rows = []
    for sc in lab:
        for i, h in enumerate([1, 3, 6]):
            x = sim[(sim.scenario == sc) & (sim.horizon == h)].set_index("method")
            if x.empty:
                continue
            vals = {m: x.loc[m, "ratio_rqc"] for m in meth}
            best = min(vals, key=vals.get)
            cells = [("\\textbf{" + f(v, 2) + "}") if m == best else f(v, 2) for m, v in vals.items()]
            rows.append(" & ".join([lab[sc] if i == 0 else "", str(h), f(x.loc["rqc", "loss"], 2), f(x.loc["none", "ratio_rqc"], 2)] + cells) + r" \\")
        rows.append(r"\addlinespace")
    se = sim[~sim.method.isin(["rqc", "none"])].ratio_se.max()
    se_none = sim[sim.method == "none"].ratio_se.max()
    tex = r"""\begin{table*}[t]
\caption{Simulation: mean pinball loss relative to the 12-month RQC.\label{tab:sim}}
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lc rr rrrrrrrr@{}}
\toprule
 & & \textbf{RQC} & & \multicolumn{8}{c}{\textbf{Loss relative to RQC}} \\
\cmidrule(l){5-12}
\textbf{Scenario} & $h$ & \textbf{loss} & \textbf{None} & \textbf{RQC-3} & \textbf{EW} & \textbf{LCS} & \textbf{ACI} & \textbf{TSPC} & \textbf{-AR} & \textbf{-U} & \textbf{-ARU} \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular*}
\begin{tablenotes}
\item \textit{Note:} Means over 400 replications of a $37\times43$ panel; ratios are computed within each replication. The largest Monte Carlo standard error of a ratio is """ + f(se, 3) + r""" for the corrections and """ + f(se_none, 3) + r""" for no correction. RQC-3: three-month RQC; EW: exponentially weighted RQC; LCS: latest cross-section. The lowest ratio among the corrections is in bold.
\end{tablenotes}
\end{table*}
"""
    (TABLES / "tab_sim2.tex").write_text(tex)


if __name__ == "__main__":
    primary()
    across()
    simulation()
