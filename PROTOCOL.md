# Pre-specified protocol for the external evaluation

Version 1, written on 7 November 2026 before any of the data described below were read.
At the time of writing, the six data files had been downloaded but not opened; no prices,
growth rates or forecast errors from them had been examined. The tagged commit
`protocol-v1` fixes this document and the correction code in `code/14_fair_evaluation.py`.
Any later change will be recorded as a dated amendment below, with its reason, and results
under the original protocol will still be reported.

## 1. Purpose

The application in the paper used one panel (NBS state food prices, 2017 to April 2024) whose
2021-2024 period informed the development of the corrections. This protocol specifies a
confirmatory evaluation of the same corrections on data not used in their development.

## 2. Data

World Bank Real-Time Food Prices (RTFP), "Monthly food price estimates by product and market",
dataset version 2026-11-02, for six countries: Nigeria (NGA), Lebanon (LBN), Sudan (SDN),
Haiti (HTI), Myanmar (MMR) and Afghanistan (AFG).

- A cell is a market and food commodity. Non-food series (exchange rates, fuel, price indices)
  are excluded. The price used is the dataset's monthly price estimate for the commodity
  (the closing estimate where open, high, low and close are given).
- A cell enters a month's cross-section only if its price is positive at both ends of the
  target window. No further interpolation is applied.
- If the file provides a reliability or observed/imputed indicator, a secondary analysis keeps
  only targets whose two endpoint prices are rated at or above the median reliability of that
  country's file.
- Target: y = 100 [log P(t+h) - log P(t)], in log points.

## 3. Evaluation windows and break dates

Break dates come from documented events, not from the data. The first break target month is
the first calendar month after the event.

| Country | Event | First break target month |
| --- | --- | --- |
| NGA | Petrol subsidy removal (29 May 2023) and exchange-rate unification (June 2023) | June 2023 |
| LBN | Financial crisis and protests from 17 October 2019; currency collapse | November 2019 |
| SDN | Outbreak of war, 15 April 2023 | May 2023 |
| HTI | Fuel price increase announced 11 September 2022 | October 2022 |
| MMR | Military coup, 1 February 2021 | March 2021 |
| AFG | Change of government, 15 August 2021 | September 2021 |

For each break, the evaluation covers forecast origins whose targets fall in the 24 months
before the first break target month (the stable window) and the 24 months from it (the break
window), truncated at the end of the data.

Nigeria additionally has an untouched period: forecast origins from May 2024 to the last
origin whose target is observed. It has no designated break.

## 4. Base forecasts

Three base forecasts of the alpha-quantile, all re-estimated every three months on an
expanding window of targets matured by the estimation origin (forecasts between re-estimations
use the latest fit):

1. Empirical quantile of training outcomes by commodity.
2. Linear quantile regression (smoothed pinball loss, L2 penalty 1e-3, as in the paper).
3. Histogram gradient boosting with quantile loss (220 iterations, learning rate 0.05,
   31 leaves, minimum leaf size 100, L2 regularisation 5, as in the paper).

Features: own-price log changes over 1, 3 and 12 months; six-month rolling mean, standard
deviation and maximum of monthly changes; twelve-month rolling standard deviation;
cross-sectional median, 90th percentile and standard deviation of monthly changes in the
country; commodity and market identifiers; seasonal harmonics. No macroeconomic predictors.
Missing feature values are replaced by training medians. No tuning on the evaluation data.

Pre-evaluation forecasts are issued from W + h months before the first evaluation origin so
that every correction starts with a full residual history from the same procedure.

## 5. Corrections

Exactly the ten in `code/14_fair_evaluation.py`, with the same constants: no correction;
RQC with 12- and 3-month windows; exponentially weighted RQC (24 months, half-life 3);
latest cross-section (LCS); ACI (gamma 0.05, level bounded to [0.5, 0.995]); TSPC; TSPC-AR;
TSPC-U; TSPC-ARU. Only residuals whose targets have matured by the origin are used.

## 6. Designs

Primary: alpha = 0.90, h = 3. Secondary: alpha in {0.75, 0.95} with h = 3, and h in {1, 6}
with alpha = 0.90.

## 7. Hypotheses and tests (primary design)

Loss is mean pinball loss. For each comparison, d(t) is the monthly mean loss difference over
cells; tests use the fixed-smoothing (EWC) statistic of the paper with Student-t(B) critical
values, B = floor(sqrt(T)).

- H1. In break windows, each correction has lower loss than no correction.
- H2. In break windows, TSPC-U has lower loss than RQC (the simulation's prediction for
  persistent shifts).
- H3. In stable windows, TSPC-ARU has loss no greater than RQC (the prediction for a
  stationary common error).
- H4. Over each full evaluation window, RQC-3, EW-RQC, LCS, ACI, TSPC and TSPC-U are
  equivalent to RQC within a margin of 5% of RQC's mean loss: equivalence is concluded when
  the 90% EWC confidence interval of mean d(t) / mean RQC loss lies inside (-0.05, 0.05).
- H5. The sign of the TSPC-RQC difference over the full window is predicted by whether the
  lag-h autocorrelation of the monthly median error, estimated only from residuals matured
  before the first evaluation origin, exceeds 1/2.

For H1-H3, each country and base forecast gives one test; Holm's correction is applied across
the 18 country-base combinations of each hypothesis, and an exact sign test across the 18
mean differences gives the pooled result. H4 is reported per combination and as a count.
H5 is reported as a 2 x 2 table with Fisher's exact test.

The Nigerian untouched period is evaluated for H4 and H5 only.

## 8. Reporting

All results are reported, whatever their direction, in the paper or its supplement, together
with any deviation from this protocol. Results from the earlier NBS analysis are kept and
labelled as exploratory.

## Amendments

None.
