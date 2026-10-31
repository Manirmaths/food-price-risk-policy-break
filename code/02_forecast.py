"""Frozen, expanding and rolling quantile forecasts with delayed level corrections.

For every design (alpha in {0.75, 0.90, 0.95}, h in {1, 3, 6}) this script
produces three estimation schemes for linear quantile regression and histogram
gradient boosting:

* frozen    - fitted once on targets that matured by December 2020;
* expanding - refitted at every origin on every target matured by that origin;
* rolling   - refitted at every origin on targets matured in the previous
              ROLL_MONTHS months.

Each base forecast then receives two feasible delayed corrections:

* the residual-quantile level correction (RQC): the alpha-quantile of residuals
  whose targets matured in the last W months (W = 12 in the primary design);
* adaptive conformal inference (ACI, Gibbs and Candes 2021) applied to the same
  residual window: the quantile level tau_t is updated with the miss rate of
  forecasts that matured at t, so feedback arrives with an h-month delay.

Only residuals whose target date is no later than the forecast origin are ever
used.  Pre-evaluation forecasts issued October 2019 to December 2020 (for h=3)
provide an honest out-of-sample residual history for each scheme.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("FOOD_RISK_BUILD_DIR", ROOT / "build"))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

EVAL_START = pd.Timestamp("2021-01-01")
TRAIN_TARGET_END = pd.Timestamp("2020-12-01")
CAL_TRAIN_END = pd.Timestamp("2019-12-01")
CAL_TARGET_START = pd.Timestamp("2020-01-01")
VALID_TRAIN_END = pd.Timestamp("2018-12-01")
VALID_START = pd.Timestamp("2019-01-01")
VALID_END = pd.Timestamp("2019-12-01")
BREAK_TARGET_START = pd.Timestamp("2023-06-01")
CAL_WINDOW_MONTHS = 12
SENSITIVITY_WINDOWS = [6, 12, 18, 24]
ROLL_MONTHS = 24
ACI_GAMMA = 0.05
ACI_GAMMAS = [0.01, 0.025, 0.05, 0.10, 0.20]
ACI_TAU_BOUNDS = (0.50, 0.995)
SEED = 20260814

NUMERIC_FEATURES = [
    "dlog1", "dlog3", "dlog12", "roll_mean6", "roll_sd6", "roll_sd12", "roll_max6",
    "log_pms_l1", "d_pms_l1", "d3_pms_l1",
    "log_diesel_l1", "d_diesel_l1", "d3_diesel_l1",
    "log_transport_l1", "d_transport_l1", "d3_transport_l1",
    "log_fx_l1", "d_fx_l1", "d3_fx_l1",
    "log_cpi_l1", "d_cpi_l1", "d3_cpi_l1",
    "xs_med", "xs_q90", "xs_sd", "xs3_med", "xs3_q90", "st_med", "st_q90",
    "sin1", "cos1", "sin2", "cos2",
]
LEARNERS = ["linear_qr", "hist_gb"]
SCHEMES = {"frozen": "", "expanding": "_expand", "rolling": "_roll"}


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> np.ndarray:
    r = y - q
    return np.where(r >= 0, alpha * r, (alpha - 1) * r)


def add_target(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    future = panel[["state", "item", "date", "logp"]].copy()
    future["date"] = future["date"] - pd.DateOffset(months=horizon)
    future = future.rename(columns={"logp": "logp_fwd"})
    data = panel.merge(future, on=["state", "item", "date"], how="left")
    data["y"] = 100 * (data["logp_fwd"] - data["logp"])
    data["target_date"] = data["date"] + pd.DateOffset(months=horizon)
    return data[data["y"].notna()].copy()


# ----------------------------------------------------------------- learners
@dataclass
class DenseDesign:
    medians: pd.Series
    means: pd.Series
    scales: pd.Series
    states: list
    items: list

    @classmethod
    def fit(cls, data):
        numeric = data[NUMERIC_FEATURES].astype(float)
        medians = numeric.median()
        filled = numeric.fillna(medians)
        return cls(medians, filled.mean(), filled.std().replace(0, 1),
                   sorted(data["state"].unique()), sorted(data["item"].unique()))

    def transform(self, data):
        numeric = data[NUMERIC_FEATURES].astype(float).fillna(self.medians)
        numeric = ((numeric - self.means) / self.scales).to_numpy(dtype=np.float64)
        s = pd.Categorical(data["state"], categories=self.states).codes
        i = pd.Categorical(data["item"], categories=self.items).codes
        if (s < 0).any() or (i < 0).any():
            raise ValueError("unseen state or commodity")
        state = np.eye(len(self.states))[s][:, 1:]
        item = np.eye(len(self.items))[i][:, 1:]
        return np.column_stack([np.ones(len(data)), numeric, state, item])


@dataclass
class LinearQR:
    design: DenseDesign
    coef: np.ndarray

    def predict(self, data):
        return self.design.transform(data) @ self.coef


def fit_linear_qr(data, alpha, l2, smooth=0.20, maxiter=600):
    """Linear QR with a smoothed pinball objective; evaluation uses exact pinball."""
    design = DenseDesign.fit(data)
    x = design.transform(data)
    y = data["y"].to_numpy(dtype=np.float64)
    beta0 = np.zeros(x.shape[1])
    beta0[0] = np.quantile(y, alpha)

    def objective(beta):
        r = y - x @ beta
        loss = alpha * r + smooth * np.logaddexp(0.0, -r / smooth)
        grad = -(x.T @ (alpha - expit(-r / smooth))) / len(y)
        grad[1:] += l2 * beta[1:]
        return float(loss.mean() + 0.5 * l2 * beta[1:] @ beta[1:]), grad

    res = minimize(objective, beta0, method="L-BFGS-B", jac=True,
                   options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6, "maxls": 30})
    return LinearQR(design, res.x)


@dataclass
class HistQR:
    model: HistGradientBoostingRegressor
    medians: pd.Series
    states: list
    items: list

    def predict(self, data):
        num = data[NUMERIC_FEATURES].astype(float).fillna(self.medians).to_numpy()
        s = pd.Categorical(data["state"], categories=self.states).codes
        i = pd.Categorical(data["item"], categories=self.items).codes
        return self.model.predict(np.column_stack([num, s, i]))


def fit_hist_qr(data, alpha, params):
    medians = data[NUMERIC_FEATURES].astype(float).median()
    states, items = sorted(data["state"].unique()), sorted(data["item"].unique())
    num = data[NUMERIC_FEATURES].astype(float).fillna(medians).to_numpy()
    s = pd.Categorical(data["state"], categories=states).codes
    i = pd.Categorical(data["item"], categories=items).codes
    x = np.column_stack([num, s, i])
    cat = np.zeros(x.shape[1], dtype=bool)
    cat[-2:] = True
    model = HistGradientBoostingRegressor(
        loss="quantile", quantile=alpha, learning_rate=0.05,
        max_iter=params["max_iter"], max_leaf_nodes=params["max_leaf_nodes"],
        min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"],
        categorical_features=cat, early_stopping=False, random_state=SEED)
    model.fit(x, data["y"].to_numpy())
    return HistQR(model, medians, states, items)


def fit_both(train, alpha, l2, hp):
    return {"linear_qr": fit_linear_qr(train, alpha, l2), "hist_gb": fit_hist_qr(train, alpha, hp)}


def select_parameters(data):
    train = data[data["target_date"] <= VALID_TRAIN_END]
    valid = data[data["target_date"].between(VALID_START, VALID_END)]
    alpha, rows = 0.90, []
    yv = valid["y"].to_numpy()
    static = np.repeat(train["y"].quantile(alpha), len(valid))
    rows.append({"model": "static_reference", "parameters": "none",
                 "loss": pinball(yv, static, alpha).mean(), "coverage": np.mean(yv <= static)})
    for l2 in [1e-5, 1e-4, 1e-3]:
        p = fit_linear_qr(train, alpha, l2).predict(valid)
        rows.append({"model": "linear_qr", "parameters": f"l2={l2:g}",
                     "loss": pinball(yv, p, alpha).mean(), "coverage": np.mean(yv <= p)})
    candidates = [
        {"max_iter": 180, "max_leaf_nodes": 15, "min_samples_leaf": 150, "l2_regularization": 1.0},
        {"max_iter": 220, "max_leaf_nodes": 31, "min_samples_leaf": 250, "l2_regularization": 2.0},
        {"max_iter": 220, "max_leaf_nodes": 31, "min_samples_leaf": 100, "l2_regularization": 5.0},
    ]
    for c in candidates:
        p = fit_hist_qr(train, alpha, c).predict(valid)
        rows.append({"model": "hist_gb", "parameters": repr(c),
                     "loss": pinball(yv, p, alpha).mean(), "coverage": np.mean(yv <= p)})
    table = pd.DataFrame(rows)
    best_l2 = float(table[table.model == "linear_qr"].sort_values("loss").iloc[0]["parameters"].split("=")[1])
    best_text = table[table.model == "hist_gb"].sort_values("loss").iloc[0]["parameters"]
    best_hp = next(c for c in candidates if repr(c) == best_text)
    return best_l2, best_hp, table


# ------------------------------------------------------------ base forecasts
def refit_predictions(data, rows, alpha, l2, hp, scheme):
    """Refit at each issuance origin (expanding or rolling) and predict that origin."""
    out = rows.sort_values(["date", "state", "item"]).copy()
    suffix = SCHEMES[scheme]
    for m in LEARNERS:
        out[m + suffix] = np.nan
    audit = []
    for origin in sorted(out["date"].unique()):
        origin = pd.Timestamp(origin)
        print(f"  {scheme} refit at {origin:%Y-%m}", flush=True)
        mask = data["target_date"] <= origin
        if scheme == "rolling":
            mask &= data["target_date"] > origin - pd.DateOffset(months=ROLL_MONTHS)
        train = data[mask]
        models = fit_both(train, alpha, l2, hp)
        cur = out["date"].eq(origin)
        for m in LEARNERS:
            out.loc[cur, m + suffix] = models[m].predict(out.loc[cur])
        audit.append({"scheme": scheme, "origin": origin, "first_training_target": train["target_date"].min(),
                      "last_training_target": train["target_date"].max(), "training_rows": len(train)})
    return out, pd.DataFrame(audit)


def base_forecasts(panel, alpha, horizon, l2, hp, schemes):
    data = add_target(panel, horizon)
    ids = ["date", "target_date", "state", "item"]
    evaluation = data[data["date"] >= EVAL_START].copy()
    calibration = data[(data["target_date"] >= CAL_TARGET_START) & (data["date"] < EVAL_START)].copy()

    # frozen evaluation forecasts from targets that mature by Dec 2020
    train = data[data["target_date"] <= TRAIN_TARGET_END]
    frozen = fit_both(train, alpha, l2, hp)
    evaluation["static"] = train["y"].quantile(alpha)
    item_q = train.groupby("item")["y"].quantile(alpha)
    evaluation["static_item"] = evaluation["item"].map(item_q).fillna(evaluation["static"])
    for m in LEARNERS:
        evaluation[m] = frozen[m].predict(evaluation)

    # pre-evaluation forecasts from a model fitted to Dec 2019 (out of sample)
    cal_train = data[data["target_date"] <= CAL_TRAIN_END]
    cal_models = fit_both(cal_train, alpha, l2, hp)
    calibration["static"] = cal_train["y"].quantile(alpha)
    cal_item_q = cal_train.groupby("item")["y"].quantile(alpha)
    calibration["static_item"] = calibration["item"].map(cal_item_q).fillna(calibration["static"])
    for m in LEARNERS:
        calibration[m] = cal_models[m].predict(calibration)

    audits = []
    for scheme in schemes:
        if scheme == "frozen":
            continue
        cols = [m + SCHEMES[scheme] for m in LEARNERS]
        ev, a1 = refit_predictions(data, evaluation[ids + NUMERIC_FEATURES], alpha, l2, hp, scheme)
        ca, a2 = refit_predictions(data, calibration[ids + NUMERIC_FEATURES], alpha, l2, hp, scheme)
        evaluation = evaluation.merge(ev[ids + cols], on=ids, how="left", validate="one_to_one")
        calibration = calibration.merge(ca[ids + cols], on=ids, how="left", validate="one_to_one")
        a1.insert(0, "sample", "evaluation")
        a2.insert(0, "sample", "pre_evaluation_calibration")
        audits += [a2, a1]
    audit = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    evaluation["period"] = np.where(evaluation["target_date"] >= BREAK_TARGET_START, "break", "stable")
    return evaluation, calibration, audit


# --------------------------------------------------------------- corrections
class ResidualBank:
    """Residuals of one base forecast, indexed by target month (calibration + evaluation)."""

    def __init__(self, evaluation, calibration, base):
        frames = [calibration[["target_date", "y", base]], evaluation[["target_date", "y", base]]]
        both = pd.concat(frames, ignore_index=True).dropna()
        both["r"] = both["y"] - both[base]
        self.by_month = {pd.Timestamp(k): g["r"].to_numpy() for k, g in both.groupby("target_date")}

    def window(self, origin, months):
        lower = origin - pd.DateOffset(months=months - 1)
        parts = [v for k, v in self.by_month.items() if lower <= k <= origin]
        return np.concatenate(parts) if parts else np.array([])


def correct(evaluation, calibration, base, alpha, window=CAL_WINDOW_MONTHS, method="rqc", gamma=ACI_GAMMA):
    """Return corrected forecasts plus the per-origin shift (and ACI level)."""
    bank = ResidualBank(evaluation, calibration, base)
    ev = evaluation
    out = np.full(len(ev), np.nan)
    origins = sorted(ev["date"].unique())
    tau = alpha
    path = []
    date_index = {pd.Timestamp(o): np.flatnonzero(ev["date"].to_numpy() == o) for o in origins}
    target = ev["target_date"].to_numpy()
    y = ev["y"].to_numpy()
    for origin in origins:
        origin = pd.Timestamp(origin)
        if method == "aci":
            matured = np.flatnonzero(target == np.datetime64(origin))
            matured = matured[np.isfinite(out[matured])]
            if len(matured):
                miss = np.mean(y[matured] > out[matured])
                tau = float(np.clip(tau + gamma * (miss - (1 - alpha)), *ACI_TAU_BOUNDS))
        res = bank.window(origin, window)
        level = tau if method == "aci" else alpha
        shift = float(np.quantile(res, level)) if len(res) else 0.0
        idx = date_index[origin]
        out[idx] = ev[base].to_numpy()[idx] + shift
        path.append({"origin": origin, "base": base, "method": method, "window": window,
                     "gamma": gamma if method == "aci" else np.nan, "tau": level, "shift": shift,
                     "n_residuals": len(res)})
    return out, pd.DataFrame(path)


def rolling_outcome_quantile(evaluation, calibration, alpha, window=CAL_WINDOW_MONTHS):
    both = pd.concat([calibration[["target_date", "y"]], evaluation[["target_date", "y"]]])
    by = {pd.Timestamp(k): g["y"].to_numpy() for k, g in both.groupby("target_date")}
    out = np.full(len(evaluation), np.nan)
    for origin in sorted(evaluation["date"].unique()):
        origin = pd.Timestamp(origin)
        lower = origin - pd.DateOffset(months=window - 1)
        vals = np.concatenate([v for k, v in by.items() if lower <= k <= origin])
        out[evaluation["date"].to_numpy() == origin] = np.quantile(vals, alpha)
    return out


def summarise(ev, models, alpha, horizon):
    rows = []
    for period in ["all", "stable", "break"]:
        sub = ev if period == "all" else ev[ev["period"].eq(period)]
        y = sub["y"].to_numpy()
        for m in models:
            q = sub[m].to_numpy()
            rows.append({"alpha": alpha, "horizon": horizon, "period": period, "model": m,
                         "origins": sub["date"].nunique(), "forecasts": len(sub),
                         "loss": pinball(y, q, alpha).mean(), "coverage": np.mean(y <= q)})
    return pd.DataFrame(rows)


def run_design(panel, alpha, horizon, l2, hp, primary):
    t0 = time.time()
    # rolling refits are reported for the primary design only
    schemes = ["frozen", "expanding", "rolling"] if primary else ["frozen", "expanding"]
    ev, cal, audit = base_forecasts(panel, alpha, horizon, l2, hp, schemes)
    bases = ["static"] + [m + SCHEMES[s] for s in schemes for m in LEARNERS]
    paths, sens = [], []
    for b in bases:
        ev[b + "_adapt"], p = correct(ev, cal, b, alpha)
        paths.append(p)
        if b != "static":
            ev[b + "_aci"], p = correct(ev, cal, b, alpha, method="aci")
            paths.append(p)
        for w in SENSITIVITY_WINDOWS:
            q, _ = correct(ev, cal, b, alpha, window=w)
            col = f"{b}_adapt_w{w}"
            ev[col] = q
            sens.append(col)
        if primary and b != "static":
            for g in ACI_GAMMAS:
                q, p = correct(ev, cal, b, alpha, method="aci", gamma=g)
                ev[f"{b}_aci_g{g:g}"] = q
                sens.append(f"{b}_aci_g{g:g}")
                paths.append(p)
    ev["rolling_q"] = rolling_outcome_quantile(ev, cal, alpha)
    models = ["static", "static_item", "rolling_q"] + bases + [b + "_adapt" for b in bases] + \
             [b + "_aci" for b in bases if b != "static"] + sens
    summary = summarise(ev, models, alpha, horizon)
    ev["alpha"], ev["horizon"] = alpha, horizon
    keep = ["alpha", "horizon", "date", "target_date", "state", "item", "y", "period"] + models
    tag = f"a{int(round(alpha * 100))}_h{horizon}"
    ev[keep].to_csv(BUILD / f"forecasts_{tag}.csv.gz", index=False, compression="gzip")
    cal_keep = ["date", "target_date", "state", "item", "y", "static", "static_item"] + bases[1:]
    cal[cal_keep].to_csv(BUILD / f"calibration_{tag}.csv.gz", index=False, compression="gzip")
    path = pd.concat(paths, ignore_index=True)
    path.insert(0, "horizon", horizon)
    path.insert(0, "alpha", alpha)
    if len(audit):
        audit.insert(0, "horizon", horizon)
        audit.insert(0, "alpha", alpha)
    print(f"design alpha={alpha} h={horizon} done in {time.time() - t0:.0f}s", flush=True)
    return summary, path, audit


def main():
    designs = [(0.90, 3)] + [(a, h) for a in [0.75, 0.90, 0.95] for h in [1, 3, 6] if (a, h) != (0.90, 3)]
    if len(sys.argv) > 1 and sys.argv[1] == "primary":
        designs = designs[:1]
    elif len(sys.argv) > 1 and sys.argv[1] == "robustness":
        designs = designs[1:]
    elif len(sys.argv) > 1:  # e.g. "0.75:1,0.95:6"
        designs = [(float(a), int(h)) for a, h in (x.split(":") for x in sys.argv[1].split(","))]
    panel = pd.read_csv(BUILD / "panel.csv.gz", parse_dates=["date"])
    l2, hp, selection = select_parameters(add_target(panel, 3))
    selection.to_csv(RESULTS / "model_selection.csv", index=False)
    print("selected l2", l2, "boosting", hp, flush=True)
    for alpha, horizon in designs:
        summary, path, audit = run_design(panel, alpha, horizon, l2, hp, primary=(alpha, horizon) == (0.90, 3))
        tag = f"a{int(round(alpha * 100))}_h{horizon}"
        summary.to_csv(RESULTS / f"summary_{tag}.csv", index=False)
        path.to_csv(RESULTS / f"correction_path_{tag}.csv", index=False)
        if len(audit):
            audit.to_csv(RESULTS / f"training_audit_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
