"""Simulation study of delayed panel calibration (second design).

The corrections are the same code as in the application (14_fair_evaluation.py).

Panel: 37 states x 43 commodities (1,591 cells), 120 months.  Monthly growth
    g(j, t) = kappa(t) + lambda(j) + sigma(t) e(j, t) [+ state and commodity effects],
lambda(j) ~ N(0, 1), e standardised Student-t(5), kappa(t) = 1 + v(t),
v(t) = 0.6 v(t-1) + 0.5 eta(t).  The target issued at s sums g over s+1..s+h,
and the base forecast is the stable-state conditional mean h(1 + lambda(j)).

Scenarios (break or episode at month 84 unless stated)
  stationary  no change;
  persistent  kappa rises by 2 per month from month 85 onward;
  transient   kappa rises by 4 per month in months 85-87 only;
  scale       sigma doubles from month 85;
  tail        from month 85 positive shocks are tripled, so the median is unchanged
              and the upper tail lengthens;
  dependent   as persistent, plus independent N(0, 0.7^2) state and commodity
              effects in every month;
  replacement the forecasts issued in months 70-75 are 15h too low (a replaced
              forecaster); evaluated with the full residual history and with a
              calibration restart that discards residuals issued before month 76.

Outputs per scenario, horizon and method: mean pinball loss, mean ratio to RQC
and its Monte Carlo standard error, coverage, loss in the delay window (issue
months whose targets include the change but before any changed residual can
have matured) and afterwards, and recovery time.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fair", ROOT / "code" / "14_fair_evaluation.py")
fair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fair)
RESULTS = ROOT / "results"

ALPHA, W, GAMMA = 0.90, 12, 0.05
S_, I_ = 37, 43
N = S_ * I_
T, TB, TC = 120, 84, 75
EVAL = 24
REPS = 200
SEED = 20261107
METHODS = ["none", "rqc", "rqc3", "ewrqc", "lcs", "aci", "tspc", "tspc_ar", "tspc_u", "tspc_aru"]
SCENARIOS = ["stationary", "persistent", "transient", "scale", "tail", "dependent", "replacement"]


def pinball(y, q, a=ALPHA):
    r = y - q
    return np.where(r >= 0, a * r, (a - 1) * r)


def simulate(rng, scenario, h):
    n_t = T + h + 1
    lam = rng.normal(0, 1, N)
    v = np.zeros(n_t)
    eta = rng.normal(0, 1, n_t)
    for t in range(1, n_t):
        v[t] = 0.6 * v[t - 1] + 0.5 * eta[t]
    kappa = 1 + v
    sigma = np.ones(n_t)
    if scenario in ("persistent", "dependent"):
        kappa[TB + 1:] += 2.0
    if scenario == "transient":
        kappa[TB + 1:TB + 4] += 4.0
    if scenario == "scale":
        sigma[TB + 1:] = 2.0
    e = rng.standard_t(5, size=(n_t, N)) / np.sqrt(5 / 3)
    if scenario == "tail":
        e[TB + 1:] = e[TB + 1:] + 2 * np.maximum(e[TB + 1:], 0)
    g = kappa[:, None] + lam[None, :] + sigma[:, None] * e
    if scenario == "dependent":
        st = np.repeat(np.arange(S_), I_)
        it = np.tile(np.arange(I_), S_)
        g += rng.normal(0, 0.7, (n_t, S_))[:, st] + rng.normal(0, 0.7, (n_t, I_))[:, it]
    c = np.vstack([np.zeros(N), np.cumsum(g[1:], axis=0)])
    y = np.array([c[s + h] - c[s] for s in range(T)])
    q = np.tile(h * (1 + lam), (T, 1))
    if scenario == "replacement":
        q[TC - 5:TC + 1] -= 15 * h
    return y, q


def window(scenario, h):
    if scenario == "stationary":
        return list(range(W + h + 12, T - h)), 0
    if scenario == "replacement":
        return list(range(TC + 1, TC + 1 + EVAL)), 0
    return list(range(TB - h + 1, TB - h + 1 + EVAL)), h      # first h issues: delay window


def calibrate(y, q, h, months, first=0):
    r = y - q
    f = fair.Fair.__new__(fair.Fair)
    f.a, f.h = ALPHA, h
    f.r = {s: r[s] for s in range(first, T)}
    f.m = {s: float(np.median(v)) for s, v in f.r.items()}
    want = set(months)
    tau, aci, out = ALPHA, {}, {}
    for t in range(min(first + h, min(months)), T):
        s0 = t - h
        if s0 in aci and s0 in f.r:
            tau = float(np.clip(tau + GAMMA * (np.mean(r[s0] > aci[s0]) - (1 - ALPHA)), 0.5, 0.995))
        pool = f.pool(t, W)
        if t in want:
            out[t] = f.shifts(t, tau)                  # all shifts are zero before any residual matures
            aci[t] = out[t]["aci"]
        else:
            aci[t] = float(np.quantile(np.concatenate([f.r[s] for s in pool]), tau)) if pool else 0.0
    return out


def evaluate(scenario, h, reps, seed):
    rng = np.random.default_rng(seed)
    months, delay = window(scenario, h)
    designs = ["full", "restart"] if scenario == "replacement" else ["full"]
    rec = {d: {m: [] for m in METHODS + ["oracle"]} for d in designs}
    cov = {d: {m: [] for m in METHODS} for d in designs}
    for _ in range(reps):
        y, q = simulate(rng, scenario, h)
        oracle = np.array([pinball(y[t], q[t] + np.quantile(y[t] - q[t], ALPHA)).mean() for t in months])
        for d in designs:
            sh = calibrate(y, q, h, months, first=TC + 1 if d == "restart" else 0)
            for m in METHODS:
                per_month = np.array([pinball(y[t], q[t] + sh[t][m]).mean() for t in months])
                rec[d][m].append(per_month)
                cov[d][m].append(np.mean([np.mean(y[t] <= q[t] + sh[t][m]) for t in months]))
            rec[d]["oracle"].append(oracle)
    rows = []
    for d in designs:
        L = {m: np.array(v) for m, v in rec[d].items()}            # reps x months
        base = L["rqc"].mean(axis=1)
        orc = L["oracle"].mean(axis=0)
        for m in METHODS:
            tot = L[m].mean(axis=1)
            ratio = tot / base
            mm = L[m].mean(axis=0)
            post = mm[delay:]
            ok = np.flatnonzero(post <= 1.2 * orc[delay:])
            rows.append({"scenario": scenario if d == "full" else scenario + "_restart", "horizon": h, "method": m,
                         "loss": tot.mean(), "ratio_rqc": ratio.mean(), "ratio_se": ratio.std(ddof=1) / np.sqrt(reps),
                         "coverage": np.mean(cov[d][m]),
                         "loss_delay": mm[:delay].mean() if delay else np.nan, "loss_post": post.mean(),
                         "recovery": int(ok[0]) if len(ok) else np.nan, "oracle_loss": orc.mean()})
    return rows


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else REPS
    part, parts = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    grid = [(s, h) for s in SCENARIOS for h in [1, 3, 6]]
    rows = []
    for k, (s, h) in enumerate(grid):
        if k % parts != part:
            continue
        rows += evaluate(s, h, reps, [SEED, k])
        print(s, h, flush=True)
    out = pd.DataFrame(rows)
    name = "simulation2.csv" if parts == 1 else f"simulation2_part{part}.csv"
    out.to_csv(RESULTS / name, index=False)
    files = [RESULTS / f"simulation2_part{k}.csv" for k in range(parts)]
    if parts > 1 and all(f.exists() for f in files):
        pd.concat([pd.read_csv(f) for f in files]).to_csv(RESULTS / "simulation2.csv", index=False)
        for f in files:
            f.unlink()


if __name__ == "__main__":
    main()
