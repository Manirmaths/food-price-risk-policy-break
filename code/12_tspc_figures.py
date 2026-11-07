"""Figure for the two-scale panel calibration (primary design, alpha = 0.90, h = 3)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("figs", ROOT / "code" / "06_figures.py")
figs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figs)
BUILD, RESULTS, INK, GREY = figs.BUILD, figs.RESULTS, figs.INK, figs.GREY

COL = {"rqc": "#ef8a62", "aci": "#5aae61", "tspc": "#053061", "oracle": GREY}


def main():
    ev = pd.read_csv(BUILD / "tspc_a90_h3.csv.gz", parse_dates=["date"])
    path = pd.read_csv(RESULTS / "tspc_path.csv")
    path = path[(path.alpha == 0.9) & (path.horizon == 3)]
    aci = pd.read_csv(RESULTS / "correction_path_a90_h3.csv", parse_dates=["origin"])
    aci = aci[(aci.method == "aci") & (aci.window == 12) & (aci.gamma.round(3) == 0.05)]
    aci = aci.drop_duplicates(["base", "origin"]).sort_values("origin")
    dates = sorted(ev["date"].unique())
    month = {d.year * 12 + d.month: d for d in pd.to_datetime(dates)}

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for ax, base, title in [(axes[0], "linear_qr", "(a) Linear QR, frozen"),
                            (axes[1], "hist_gb_expand", "(b) Boosting, expanding")]:
        r = ev["y"] - ev[base]
        oracle = r.groupby(ev["date"]).quantile(0.90)
        p = path[path.base == base].assign(date=lambda d: d.t.map(month)).set_index("date")
        a = aci[aci.base == base].set_index("origin")["shift"]
        ax.plot(oracle.index, oracle, color=COL["oracle"], lw=1.0, label="Ex-post optimal shift")
        ax.plot(p.index, p["rqc"], color=COL["rqc"], label="RQC")
        ax.plot(a.index, a, color=COL["aci"], lw=1.0, label="ACI")
        ax.plot(p.index, p["tspc"], color=COL["tspc"], label="TSPC")
        ax.axvline(figs.BREAK_ORIGIN, color=INK, ls="--", lw=0.8)
        ax.set_title(title, loc="left")
        ax.set_ylabel("Shift (percentage points)")
        figs.months_axis(ax, 12)
    axes[0].legend(frameon=False, loc="center", bbox_to_anchor=(0.66, 0.56))

    ax = axes[2]
    base = "hist_gb_expand"
    r = ev["y"] - ev[base]
    med = r.groupby(ev["date"]).median()
    centred = (r - r.groupby(ev["date"]).transform("median")).groupby(ev["date"]).quantile(0.90)
    p = path[path.base == base].assign(date=lambda d: d.t.map(month)).set_index("date")
    ax.plot(med.index, med, color="#b2182b", lw=1.0, label="Level, realised")
    ax.plot(p.index, p["level"], color="#b2182b", ls="--", lw=1.0, label="Level, TSPC")
    ax.plot(centred.index, centred, color="#2166ac", lw=1.0, label="Shape, realised")
    ax.plot(p.index, p["shape"], color="#2166ac", ls="--", lw=1.0, label="Shape, TSPC")
    ax.axhline(0, color=INK, lw=0.5)
    ax.axvline(figs.BREAK_ORIGIN, color=INK, ls="--", lw=0.8)
    ax.set_title("(c) Level and shape, boosting", loc="left")
    ax.set_ylabel("Percentage points")
    figs.months_axis(ax, 12)
    ax.set_ylim(-14, 46)
    ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.6)
    fig.tight_layout(w_pad=1.0)
    figs.save(fig, "fig3_tspc")


if __name__ == "__main__":
    main()
