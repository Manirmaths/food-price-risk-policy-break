# Forecasting Food Price Risk Through a Policy Break

Code and results for the paper

> Isa, G. A., Ibrahim, M., Alhassan, B., Danjuma, S. M. and Shehu, I. A. *Forecasting Food Price Risk Through a Policy Break: Re-Estimation, Delayed Level Correction and the Limits of Tree Extrapolation.* Submitted to the *Journal of Forecasting*.

The paper forecasts the 90th percentile of three-month retail price growth for 43 foods in Nigeria's 36 states and the Federal Capital Territory, using the National Bureau of Statistics (NBS) Selected Food Price Watch from January 2017 to April 2024. The evaluation covers the 2023 removal of the petrol subsidy and the unification of the exchange rate.

## Design

| Element | Setting |
| --- | --- |
| Panel | 37 states x 43 commodities x 88 months; 139,917 observed prices (91 missing, not interpolated) |
| Target | 100 x (log P[t+h] - log P[t]); primary design alpha = 0.90, h = 3 |
| Models | Empirical quantile; linear quantile regression (smoothed pinball loss, L2 penalty); histogram gradient boosting (quantile loss) |
| Estimation | Frozen (targets up to Dec 2020); expanding window refitted every month; 24-month rolling window refitted every month |
| Corrections | Residual-quantile correction (RQC, 12-month window); adaptive conformal inference (ACI, step size 0.05) |
| Timing | At origin t, every training sample and residual history contains only targets dated t or earlier |
| Evaluation | 37 origins, Jan 2021 to Jan 2024; 26 stable and 11 break target months (break from Jun 2023) |
| Inference | Origin-month series with Newey-West, Harvey-Leybourne-Newbold and fixed-smoothing (EWC) p-values |
| Robustness | alpha in {0.75, 0.90, 0.95} and h in {1, 3, 6}; residual windows of 6 to 24 months; ACI step sizes 0.01 to 0.20 |

## Main results

Break-period mean pinball loss, alpha = 0.90, h = 3:

| Model | Frozen | Expanding | Expanding + RQC |
| --- | ---: | ---: | ---: |
| Linear quantile regression | 11.360 | 3.936 | 3.734 |
| Gradient boosting | 12.392 | 6.088 | 4.335 |

The rolling 12-month empirical quantile has a break-period loss of 5.349. Full results for every design, test and diagnostic are in `results/Supporting_Information.xlsx`.

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
```

The primary design refits both models at 52 origins under two schemes and takes about one hour on two CPU cores. Each of the other eight designs takes about five minutes. `requirements-lock.txt` lists the package versions used for the reported results.

## Repository layout

```
code/      analysis scripts, run in numerical order
results/   CSV outputs and the Supporting Information workbook
figures/   manuscript figures (PDF and PNG)
tables/    manuscript tables (LaTeX)
```

## Licence

Code is released under the MIT Licence. The source data remain subject to the terms of NBS and ACLED.
