"""Checks of the panel, the forecast timing and the reported results."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUILD, RESULTS = ROOT / "build", ROOT / "results"
failures = []


def check(ok, label):
    print(("PASS " if ok else "FAIL ") + label)
    if not ok:
        failures.append(label)


def main():
    panel = pd.read_csv(BUILD / "panel.csv.gz", parse_dates=["date"])
    check(panel["state"].nunique() == 37 and panel["item"].nunique() == 43, "37 states and 43 commodities")
    check(len(panel) == 139_917, "139,917 observed prices")

    f = pd.read_csv(BUILD / "forecasts_a90_h3.csv.gz", parse_dates=["date", "target_date"])
    check(f["date"].nunique() == 37 and len(f) == 58_864, "37 origins and 58,864 primary forecasts")
    check((f["target_date"] == f["date"] + pd.DateOffset(months=3)).all(), "three-month target dates")
    check((f["period"].eq("stable").sum(), f["period"].eq("break").sum()) == (41_363, 17_501), "stable/break counts")
    check(f.loc[f["period"].eq("break"), "target_date"].min() == pd.Timestamp("2023-06-01"), "break starts June 2023")
    cols = [c for c in f.columns if c not in {"alpha", "horizon", "date", "target_date", "state", "item", "y", "period"}]
    check(np.isfinite(f[cols].to_numpy(dtype=float)).all(), "all primary forecasts finite")

    for path in sorted(glob.glob(str(RESULTS / "training_audit_*.csv"))):
        a = pd.read_csv(path, parse_dates=["origin", "first_training_target", "last_training_target"])
        check((a["last_training_target"] <= a["origin"]).all(), f"no training target after origin ({Path(path).name})")
        roll = a[a["scheme"].eq("rolling")]
        if len(roll):
            check((roll["first_training_target"] > roll["origin"] - pd.DateOffset(months=24)).all(),
                  f"rolling window is 24 months ({Path(path).name})")

    for path in sorted(glob.glob(str(RESULTS / "correction_path_*.csv"))):
        p = pd.read_csv(path)
        check((p["n_residuals"] > 0).all(), f"every correction has a residual history ({Path(path).name})")

    s = pd.read_csv(RESULTS / "summary_a90_h3.csv").drop_duplicates(["period", "model"]).set_index(["model", "period"])
    check(abs(s.loc[("static", "break"), "loss"] - 8.014) < 5e-4, "frozen empirical break loss 8.014")
    check(abs(s.loc[("static_adapt", "break"), "loss"] - s.loc[("rolling_q", "break"), "loss"]) < 1e-9,
          "RQC on a constant equals rolling empirical quantile in the break")
    check(len(glob.glob(str(RESULTS / "summary_a*_h*.csv"))) == 9, "nine quantile-horizon designs")
    if failures:
        sys.exit(f"{len(failures)} checks failed")
    print("All checks passed.")


if __name__ == "__main__":
    main()
