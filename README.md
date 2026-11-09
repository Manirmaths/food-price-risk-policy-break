# Delayed Calibration of Panel Quantile Forecasts

Code and results for the paper

> Shehu, I. A., Isa, G. A. and Ibrahim, M. *Delayed calibration of panel quantile forecasts after structural breaks: a pre-specified evaluation on food prices in six countries.* Submitted to the *Journal of Applied Statistics*.

The paper studies corrections that keep panel quantile forecasts calibrated when forecast errors are observed only after the forecast horizon. It compares pooled, short-window, exponentially weighted, latest-cross-section and adaptive conformal corrections with two-scale panel calibration (TSPC), which takes the common level of the errors from the newest matured cross-section and their shape from a pooled window, and with TSPC variants that forecast the level and allow for its uncertainty. The comparison uses a simulation study and an application that forecasts upper quantiles of retail price growth for 43 foods in Nigeria's 36 states and the Federal Capital Territory, using the National Bureau of Statistics (NBS) Selected Food Price Watch from January 2017 to April 2024, across the 2023 removal of the petrol subsidy and unification of the exchange rate. Because that sample shaped the corrections, they are then tested, under the protocol in `PROTOCOL.md` (fixed before the data were read), on World Bank Real-Time Food Prices market panels for Nigeria, Lebanon, Sudan, Haiti, Myanmar and Afghanistan.

## Design

| Element | Setting |
| --- | --- |
| Panel | 37 states x 43 commodities x 88 months; 139,917 observed prices (91 missing, not interpolated) |
| Target | 100 x (log P[t+h] - log P[t]); primary design alpha = 0.90, h = 3 |
| Models | Empirical quantile; linear quantile regression (smoothed pinball loss, L2 penalty); histogram gradient boosting (quantile loss) |
| Estimation | Frozen (targets up to Dec 2020); expanding window refitted every month; 24-month rolling window refitted every month |
| Corrections | RQC (12 and 3 months, exponentially weighted), latest cross-section, ACI (step size 0.05), TSPC, TSPC-AR, TSPC-U and TSPC-ARU, all on matured residuals only; full and restarted residual histories |
| Timing | At origin t, every training sample and residual history contains only targets dated t or earlier |
| Evaluation | 37 origins, Jan 2021 to Jan 2024; 26 stable and 11 break target months (break from Jun 2023) |
| Inference | Origin-month series with Newey-West, Harvey-Leybourne-Newbold and fixed-smoothing (EWC) p-values |
| Robustness | alpha in {0.75, 0.90, 0.95} and h in {1, 3, 6}; residual windows of 6 to 24 months; ACI step sizes 0.01 to 0.20 |

## Main results

- Re-estimating the base model matters most: expanding-window refits lower the break-period loss of frozen forecasts by 51% to 65% (alpha = 0.90, h = 3).
- After a calibration restart, any of the delayed corrections lowers break-period loss by about 30% relative to no correction, averaged over 47 combinations of base forecast and design.
- With the full residual history, TSPC lowers loss by 10% relative to the 12-month RQC, but so does the latest-cross-section correction; after a restart, all corrections are within 3% of one another and rarely significantly different.
- Pre-specified evaluation (six countries, alpha = 0.90, h = 3): every correction has lower break-period loss than no correction in most of 18 country and base-forecast combinations (median reductions 10% to 18%). TSPC-U does not beat RQC after the breaks (10 of 18). Corrections that carry the latest common level forward are 7% to 9% worse than RQC, mainly before the breaks, where the common error was negatively autocorrelated in 17 of 21 cases. Exponentially weighted RQC is the most reliable correction. At the country level, RQC, exponentially weighted RQC, TSPC-AR, TSPC-U and TSPC-ARU lowered break-period loss in all six countries (loss ratios 0.80 to 0.85). The four secondary designs (alpha = 0.75 and 0.95 at h = 3; h = 1 and 6 at alpha = 0.90) give the same conclusions.
- Exploratory analyses, added after the results were seen and outside the protocol: the conclusions hold on prices observed in the field (14% to 54% of evaluated targets), and a correction that shortens its window after a CUSUM alarm is level with RQC over the whole window (country-level ratio 1.01). Results: `results/external/clustered_tests.csv`, `power.csv`, `extra_country.csv` and `<COUNTRY>_a90_h3_extra.csv`.
- Full results: `results/external/hypothesis_summary.csv`, `results/external/directional_tests.csv`, `results/fair_summary.csv`, `results/fair_tests.csv` and `results/simulation2.csv`.

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

For the pre-specified evaluation, unzip the RTFP market files (`<ISO>_RTFP_mkt_*/*.csv`, version 2026-11-02, World Bank Microdata Library) into `data/rtfp/`, or set `RTFP_DIR`.

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
python code/10_panel_calibration.py  # TSPC, TSPC-LS and TSPC-AR for all designs
python code/12_tspc_figures.py       # calibration figure
python code/13_tspc_tables.py        # earlier calibration tables
python code/14_fair_evaluation.py    # fair comparison: full and restart residual histories, all corrections
python code/15_simulation.py 400     # simulation study (pass "400 k 2" to run half k of the grid)
python code/16_fair_tables.py        # tables for the fair comparison and the simulation
python code/17_external.py NGA 0.90 3   # pre-specified evaluation; countries NGA, NGA_POST, LBN, SDN, HTI, MMR, AFG
python code/18_external_tests.py         # hypotheses H1-H5 of PROTOCOL.md
python code/19_external_tables.py        # tables and figure for the evaluation
python code/20_extra.py NGA 0.90 3       # exploratory (outside the protocol): observed-price subset and break-detection corrections
python code/21_clustered.py              # exploratory: country-level tests and power of the per-combination tests
python code/22_extra_summary.py          # exploratory: summaries of 20_extra.py
python code/11_simulation.py 500     # first simulation design (online supplement of an earlier version)
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
