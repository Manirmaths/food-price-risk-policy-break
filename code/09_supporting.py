"""Bundle all result tables into one Supporting Information workbook."""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
README = [
    ("Contents", "Supporting Information for 'Forecasting Food Price Risk Through a Policy Break'."),
    ("forecast_summary", "Mean pinball loss and coverage for every forecast, period, quantile level (alpha) and horizon (months). "
                         "Model suffixes: _expand expanding-window refit; _roll 24-month rolling refit; _adapt residual-quantile "
                         "correction (12-month window); _adapt_wW window of W months; _aci adaptive conformal inference (gamma=0.05); "
                         "_aci_gG step size G."),
    ("loss_inference", "Month-level tests of equal pinball loss. estimate = mean of model_1 minus model_2. p_nw Newey-West; "
                       "p_hln Harvey-Leybourne-Newbold corrected Diebold-Mariano; p_ewc fixed-smoothing EWC test."),
    ("coverage_inference", "Month-level tests of mean coverage against the nominal quantile level."),
    ("monthly_metrics", "Primary design (alpha=0.90, h=3): loss and coverage for every origin month."),
    ("decomposition", "Common- and commodity-oracle decomposition by base forecast and period."),
    ("placebo_windows", "Common share for each rolling 11-month pre-break window and for the break."),
    ("correction_paths", "Level shift (and ACI quantile level tau) applied at each origin."),
    ("training_audit", "First and last training target and sample size of every refit."),
    ("extrapolation", "Realised 90th percentile, average re-estimated forecasts and training-sample 99th percentile by origin."),
]


def main():
    with pd.ExcelWriter(RESULTS / "Supporting_Information.xlsx") as xw:
        pd.DataFrame(README, columns=["sheet", "description"]).to_excel(xw, sheet_name="README", index=False)
        summ = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RESULTS / "summary_a*_h*.csv")))])
        summ.drop_duplicates(["alpha", "horizon", "period", "model"]).to_excel(xw, sheet_name="forecast_summary", index=False)
        summ.drop_duplicates(["alpha", "horizon", "period", "model"]).to_csv(RESULTS / "forecast_summary.csv", index=False)
        pd.read_csv(RESULTS / "loss_inference.csv").to_excel(xw, sheet_name="loss_inference", index=False)
        pd.read_csv(RESULTS / "coverage_inference.csv").to_excel(xw, sheet_name="coverage_inference", index=False)
        pd.read_csv(RESULTS / "monthly_metrics.csv").to_excel(xw, sheet_name="monthly_metrics", index=False)
        pd.read_csv(RESULTS / "decomposition_summary.csv").to_excel(xw, sheet_name="decomposition", index=False)
        pd.read_csv(RESULTS / "placebo_windows.csv").to_excel(xw, sheet_name="placebo_windows", index=False)
        pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RESULTS / "correction_path_*.csv")))]).to_excel(
            xw, sheet_name="correction_paths", index=False)
        pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RESULTS / "training_audit_*.csv")))]).to_excel(
            xw, sheet_name="training_audit", index=False)
        pd.read_csv(RESULTS / "extrapolation.csv").to_excel(xw, sheet_name="extrapolation", index=False)


if __name__ == "__main__":
    main()
