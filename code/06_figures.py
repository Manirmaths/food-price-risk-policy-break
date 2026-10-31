"""Manuscript figures, saved as PDF with a PNG preview.

Figure 1  data and macroeconomic drivers
Figure 2  loss and coverage by forecast origin
Figure 3  level of re-estimated forecasts against outcomes (also writes extrapolation.csv)
Figure 4  common-error share in pre-break placebo windows
Figure 5  level shift applied by the residual-quantile correction
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUILD, RESULTS, FIG = ROOT / "build", ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
BREAK_TARGET = pd.Timestamp("2023-06-01")
BREAK_ORIGIN = pd.Timestamp("2023-03-01")
HORIZON = 3

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "lines.linewidth": 1.4, "figure.dpi": 150,
})
INK, GREY = "#222222", "#8c8c8c"
COLOURS = {"frozen": "#b2182b", "expanding": "#2166ac", "rolling": "#5aae61", "expanding+RQC": "#053061"}
LEARNERS = [("linear_qr", "Linear quantile regression"), ("hist_gb", "Gradient boosting")]


def date_axis(ax, interval: int = 6) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1, 13, interval)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.tick_params(axis="x", which="minor", length=2)


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig1_data() -> None:
    panel = pd.read_csv(BUILD / "panel.csv.gz", parse_dates=["date"])
    growth = panel.dropna(subset=["dlog3"]).groupby("date")["dlog3"]
    q = pd.DataFrame({"q10": growth.quantile(0.10), "q50": growth.median(), "q90": growth.quantile(0.90)})
    # Macro features are stored with a one-month lag; shift back to plot contemporaneous values.
    macro = np.exp(panel.groupby("date")[["log_fx_l1", "log_pms_l1"]].median().shift(-1))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    ax.fill_between(q.index, q["q10"], q["q90"], color="#9ecae1", alpha=0.6, lw=0, label="10th–90th percentile")
    ax.plot(q.index, q["q50"], color=INK, label="Median")
    ax.plot(q.index, q["q90"], color="#08519c", lw=1.0, label="90th percentile")
    ax.axvline(BREAK_TARGET, color=INK, ls="--", lw=0.8)
    ax.set_ylabel("Three-month log change (%)")
    ax.set_title("(a) Cross-section of state–commodity price growth")
    date_axis(ax, 12)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    ax.plot(macro.index, macro["log_fx_l1"], color="#b2182b", label="Naira per US dollar")
    ax.set_ylabel("NGN per USD")
    ax2 = ax.twinx()
    ax2.plot(macro.index, macro["log_pms_l1"], color="#2166ac", label="Petrol (PMS), NGN per litre")
    ax2.set_ylabel("NGN per litre")
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax.axvline(BREAK_TARGET, color=INK, ls="--", lw=0.8)
    ax.set_title("(b) Exchange rate and petrol price")
    date_axis(ax, 12)
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig1_data")


def fig2_loss_coverage() -> None:
    monthly = pd.read_csv(RESULTS / "monthly_metrics.csv", parse_dates=["date", "target_date"])
    series = [("frozen", "{}"), ("expanding", "{}_expand"), ("rolling", "{}_roll"), ("expanding+RQC", "{}_expand_adapt")]
    benchmark = monthly[monthly["model"].eq("rolling_q")].sort_values("date")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True)
    for col, (learner, title) in enumerate(LEARNERS):
        for row, metric in enumerate(["loss", "coverage"]):
            ax = axes[row, col]
            for label, pattern in series:
                d = monthly[monthly["model"].eq(pattern.format(learner))].sort_values("date")
                ax.plot(d["date"], d[metric], color=COLOURS[label], label=label,
                        lw=1.6 if label == "expanding+RQC" else 1.1)
            ax.plot(benchmark["date"], benchmark[metric], color=GREY, ls=":", lw=1.1, label="rolling empirical q90")
            ax.axvline(BREAK_ORIGIN, color=INK, ls="--", lw=0.8)
            if metric == "coverage":
                ax.axhline(0.9, color=INK, lw=0.6, ls=":")
                ax.set_ylim(0, 1.02)
                ax.set_ylabel("Coverage")
                ax.set_xlabel("Forecast origin")
            else:
                ax.set_ylabel("Mean pinball loss")
                ax.set_title(title)
            date_axis(ax)
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig2_loss_coverage")


def fig3_extrapolation() -> None:
    forecasts = pd.read_csv(BUILD / "forecasts_a90_h3.csv.gz", parse_dates=["date", "target_date"])
    panel = pd.read_csv(BUILD / "panel.csv.gz", parse_dates=["date"])
    future = panel[["state", "item", "date", "logp"]].copy()
    future["date"] = future["date"] - pd.DateOffset(months=HORIZON)
    outcomes = panel[["state", "item", "date", "logp"]].merge(
        future.rename(columns={"logp": "logp_fwd"}), on=["state", "item", "date"])
    outcomes["y"] = 100 * (outcomes["logp_fwd"] - outcomes["logp"])
    outcomes["target_date"] = outcomes["date"] + pd.DateOffset(months=HORIZON)

    rows = []
    for origin, g in forecasts.groupby("date"):
        observed = outcomes.loc[outcomes["target_date"] <= origin, "y"]
        q99 = observed.quantile(0.99)
        rows.append({
            "origin": origin, "period": g["period"].iloc[0], "outcome_q90": g["y"].quantile(0.9),
            "train_q99": q99, "share_above_train_q99": np.mean(g["y"] > q99),
            "linear_qr_expand": g["linear_qr_expand"].mean(), "hist_gb_expand": g["hist_gb_expand"].mean(),
            "hist_gb_expand_adapt": g["hist_gb_expand_adapt"].mean(),
        })
    r = pd.DataFrame(rows)
    r.to_csv(RESULTS / "extrapolation.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.plot(r["origin"], r["outcome_q90"], color=INK, lw=1.8, label="Realised cross-sectional 90th percentile")
    ax.plot(r["origin"], r["linear_qr_expand"], color=COLOURS["expanding"], label="Expanding linear QR (mean forecast)")
    ax.plot(r["origin"], r["hist_gb_expand"], color=COLOURS["frozen"], label="Expanding boosting (mean forecast)")
    ax.plot(r["origin"], r["hist_gb_expand_adapt"], color=COLOURS["frozen"], ls="--", label="Expanding boosting + RQC")
    ax.plot(r["origin"], r["train_q99"], color=GREY, ls=":", label="99th percentile of training targets")
    ax.axvline(BREAK_ORIGIN, color=INK, ls="--", lw=0.8)
    ax.set_ylabel("Three-month log change (%)")
    ax.set_xlabel("Forecast origin")
    date_axis(ax)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, "fig3_extrapolation")


def fig4_placebo() -> None:
    windows = pd.read_csv(RESULTS / "placebo_windows.csv", parse_dates=["end_target"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    for ax, (learner, title) in zip(axes, LEARNERS):
        for base, label in [(learner, "frozen"), (learner + "_expand", "expanding")]:
            model = windows[windows["model"].eq(base)]
            placebo = model[model["window"].eq("placebo")].sort_values("end_target")
            actual = model.loc[model["window"].eq("break"), "common_share"].iloc[0]
            ax.plot(placebo["end_target"], placebo["common_share"], marker="o", ms=2.5,
                    color=COLOURS[label], label=f"{label}: pre-break windows")
            ax.axhline(actual, color=COLOURS[label], ls="--", lw=1.0, label=f"{label}: break window")
        ax.set_title(title)
        ax.set_xlabel("Last target month of 11-month window")
        date_axis(ax)
    axes[0].set_ylabel("Common share $S$")
    axes[0].legend(frameon=False, loc="lower left", fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig4_placebo")


def fig5_correction_path() -> None:
    path = pd.read_csv(RESULTS / "correction_path_a90_h3.csv", parse_dates=["origin"])
    path = path[path["window"].eq(12) & path["method"].eq("rqc")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, (learner, title) in zip(axes, LEARNERS):
        for base, label in [(learner, "frozen"), (learner + "_expand", "expanding"), (learner + "_roll", "rolling")]:
            d = path[path["base"].eq(base)].sort_values("origin")
            ax.plot(d["origin"], d["shift"], color=COLOURS[label], label=f"{label} base")
        ax.axhline(0, color=INK, lw=0.6)
        ax.axvline(BREAK_ORIGIN, color=INK, ls="--", lw=0.8)
        ax.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2022-01-01"), color="#f2f2f2", lw=0, zorder=0)
        ax.set_title(title)
        ax.set_xlabel("Forecast origin")
        date_axis(ax)
    axes[0].set_ylabel("Level shift (percentage points)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig5_correction_path")


if __name__ == "__main__":
    fig1_data()
    fig2_loss_coverage()
    fig3_extrapolation()
    fig4_placebo()
    fig5_correction_path()
