# Two-Scale Calibration of Panel Quantile Forecasts

Code and results for the paper

> Isa, G. A., Ibrahim, M., Alhassan, B., Danjuma, S. M. and Shehu, I. A. *Two-scale calibration of panel quantile forecasts with delayed feedback under structural change.* Submitted to the *Journal of Applied Statistics*.

The paper proposes two-scale panel calibration (TSPC) for quantile forecasts whose errors are observed only after the forecast horizon. TSPC shifts every forecast by the median error of the newest matured cross-section (the level) plus a pooled quantile of median-centred errors (the shape). It is compared with a pooled residual-quantile correction (RQC) and adaptive conformal inference (ACI) in a simulation study and in an application that forecasts upper quantiles of retail price growth for 43 foods in Nigeria's 36 states and the Federal Capital Territory, using the National Bureau of Statistics (NBS) Selected Food Price Watch from January 2017 to April 2024. The evaluation covers the 2023 removal of the petrol subsidy and the unification of the exchange rate.

## Design

| Element | Setting |
| --- | --- |
| Panel | 37 states x 43 commodities x 88 months; 139,917 observed prices (91 missing, not interpolated) |
| Target | 100 x (log P[t+h] - log P[t]); primary design alpha = 0.90, h = 3 |
| Models | Empirical quantile; linear quantile regression (smoothed pinball loss, L2 penalty); histogram gradient boosting (quantile loss) |
| Estimation | Frozen (targets up to Dec 2020); expanding window refitted every month; 24-month rolling window refitted every month |
| Corrections | RQC and ACI (step size 0.05) on 12 months of matured residuals; TSPC, TSPC-LS (adaptive scale) and TSPC-AR (autoregressive level) on the same residuals |
| Timing | At origin t, every training sample and residual history contains only targets dated t or earlier |
| Evaluation | 37 origins, Jan 2021 to Jan 2024; 26 stable and 11 break target months (break from Jun 2023) |
| Inference | Origin-month series with Newey-West, Harvey-Leybourne-Newbold and fixed-smoothing (EWC) p-values |
| Robustness | alpha in {0.75, 0.90, 0.95} and h in {1, 3, 6}; residual windows of 6 to 24 months; ACI step sizes 0.01 to 0.20 |

## Main results

Break-period mean pinball loss, alpha = 0.90, h = 3:

| Model | Frozen | Expanding | Expanding + RQC | Expanding + ACI | Expanding + TSPC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear quantile regression | 11.360 | 3.936 | 3.734 | 3.945 | 4.408 |
| Gradient boosting | 12.392 | 6.088 | 4.335 | 4.256 | 4.270 |

Across nine quantile-horizon designs, TSPC has lower whole-sample loss than RQC in 33 of 47 combinations of base forecast and design, in all 15 at h = 6, and in 20 of the 22 whose median error has lag-h autocorrelation above one half. Full results are in `results/tspc_summary.csv`, `results/tspc_tests.csv` and `results/simulation.csv`.

## Data

The source files are public but are not redistributed here. Place them in `data/source/`, or set `FOOD_RISK_SOURCE_DIR` to their folder:

| File | Content |
| --- | --- |
| `food_prices_combined_available_2017_Apr2026.csv` | NBS Selected Food Price Watch (state level to April 2024) |
| `pms_monthly_long.csv` | NBS petrol (PMS) prices by state |
| `diesel_data_combined_Jun2015_Apr2026.csv` | NBS diesel prices by state |
| `transport_costs_complete_Jan2017_Apr2026.csv` | NBS transport fares by state |
| `exchange_rate_monthly_2017_2026.csv` | Naira per US dollar, monthly |
| `cpi_complete_1995_2026.csv` | NBS consumer price index, chain-linked |
| `ACLED Data_2026-08-05_event_date_from_2017-03-01_event_date_to_2024-04-30.csv` | ACLED weekly state aggregates (appendix only) |

NBS stopped publishing state-level food price tables after April 2024; later releases report national and zonal averages only.

## Running the analysis

```bash
pip install -r requirements.txt
python code/00_make_raw.py        # assemble the macro and conflict inputs in data/raw
python code/01_build_panel.py     # build the panel and predictors
python code/02_forecast.py        # all nine designs; pass "primary" or e.g. "0.75:1,0.95:6" to run a subset
python code/03_decomposition.py   # oracle decomposition and placebo windows
python code/04_tests.py           # origin-month tests
python code/05_conflict.py        # conflict appendix
python code/06_figures.py         # figures
python code/08_tables.py          # LaTeX tables
python code/09_supporting.py      # Supporting Information workbook
python code/07_verify.py          # integrity checks
python code/10_panel_calibration.py  # TSPC, TSPC-LS and TSPC-AR for all designs, with tests
python code/11_simulation.py 500     # simulation study (pass "500 k 2" to run half k of the grid)
python code/12_tspc_figures.py       # calibration figure
python code/13_tspc_tables.py        # calibration, design and simulation tables
```

The primary design refits both models at 52 origins under two schemes and takes about one hour on two CPU cores. Each of the other eight designs takes about five minutes. The calibration step takes about one minute, and the simulation about one hour on two cores. `requirements-lock.txt` lists the package versions used for the reported results.

## Repository layout

```
code/      analysis scripts, run in numerical order
results/   CSV outputs and the Supporting Information workbook
figures/   manuscript figures (PDF and PNG)
tables/    manuscript tables (LaTeX)
```

## Licence

Code is released under the MIT Licence. The source data remain subject to the terms of NBS and ACLED.
