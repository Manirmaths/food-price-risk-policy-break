"""Monte Carlo study of delayed calibration in a panel.

Monthly growth of cell i in month t is

    g(i, t) = mu(t) + lambda(i) + sigma(t) e(i, t),

with lambda(i) ~ N(0, 1), e(i, t) standardised Student-t(5), and a common
level mu(t) = 1 + u(t), u(t) = 0.6 u(t-1) + 0.5 eta(t).  The target issued at s
is y(i, s) = g(i, s+1) + ... + g(i, s+h), observed h months after issuance.
The base forecast is the stable-period conditional mean h(1 + lambda(i)), so
every correction has to learn both the level and the upper-quantile offset.

Scenarios
  stable  no break;
  level   mu rises by 2 per month from the break month onward;
  scale   sigma doubles from the break month onward;
  contam  the base forecasts issued in the six months before evaluation are
          15h too low, as with a poorly initialised model; this affects the
          residual history only;
  idio    no common variation (u = 0) and only 40 cells.

Corrections (window W = 12 issuance months of matured residuals):
  RQC, ACI (gamma = 0.05, delayed miss rates), TSPC, TSPC-LS and TSPC-AR,
  which forecasts the common level h months ahead with a lag-h regression
  fitted to the matured medians.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ALPHA, W, GAMMA = 0.90, 12, 0.05
T, T_BREAK, T_CONTAM = 120, 84, 75
EVAL_MONTHS = 24
REPS = 500
SEED = 20261107


def pinball(y, q, a):
    r = y - q
    return np.where(r >= 0, a * r, (a - 1) * r)


def simulate(rng, n, h, scenario):
    lam = rng.normal(0, 1, n)
    eta = rng.normal(0, 1, T + h + 1)
    u = np.zeros(T + h + 1)
    if scenario != "idio":
        for t in range(1, T + h + 1):
            u[t] = 0.6 * u[t - 1] + 0.5 * eta[t]
    mu = 1 + u
    sigma = np.ones(T + h + 1)
    if scenario == "level":
        mu[T_BREAK + 1:] += 2.0
    if scenario == "scale":
        sigma[T_BREAK + 1:] = 2.0
    e = rng.standard_t(5, size=(T + h + 1, n)) / np.sqrt(5 / 3)
    g = mu[:, None] + lam[None, :] + sigma[:, None] * e
    csum = np.vstack([np.zeros(n), np.cumsum(g[1:], axis=0)])        # csum[t] = g(1..t)
    y = np.array([csum[s + h] - csum[s] for s in range(T)])          # y[s] issued at s
    q = np.tile(h * (1 + lam), (T, 1))
    if scenario == "contam":
        q[T_CONTAM - 5:T_CONTAM + 1] -= 15 * h
    return y, q


def ar_level(med, s0, h, min_pairs=6):
    """Forecast of the common level for issuance s0 + h from medians dated s0 or earlier."""
    x, z = med[:s0 - h + 1], med[h:s0 + 1]
    if len(z) < min_pairs:
        return float(np.mean(med[:s0 + 1]))
    vx = np.var(x)
    b = 0.0 if vx == 0 else float(np.clip(np.cov(x, z, bias=True)[0, 1] / vx, 0.0, 1.0))
    return float(z.mean() - b * x.mean() + b * med[s0])


def corrections(y, q, h, a=ALPHA):
    r = y - q
    med = np.median(r, axis=1)
    iqr = np.subtract(*np.quantile(r, [0.75, 0.25], axis=1)) / 1.349
    out = {k: np.full(y.shape[0], np.nan) for k in ["rqc", "aci", "tspc", "tspc_ls", "tspc_ar"]}
    tau = a
    for t in range(W + h, y.shape[0]):
        win = slice(t - h - W + 1, t - h + 1)
        s0 = t - h
        pooled = r[win].ravel()
        # ACI: update with the miss rate of forecasts that matured at t (issued at t - h)
        if np.isfinite(out["aci"][s0]):
            miss = np.mean(y[s0] > q[s0] + out["aci"][s0])
            tau = float(np.clip(tau + GAMMA * (miss - (1 - a)), 0.5, 0.995))
        out["rqc"][t] = np.quantile(pooled, a)
        out["aci"][t] = np.quantile(pooled, tau)
        centred = (r[win] - med[win, None]).ravel()
        shape = np.quantile(centred, a)
        out["tspc"][t] = med[s0] + shape
        out["tspc_ar"][t] = ar_level(med, s0, h) + shape
        std = ((r[win] - med[win, None]) / iqr[win, None]).ravel()
        out["tspc_ls"][t] = med[s0] + iqr[s0] * np.quantile(std, a)
    return out


def evaluate(scenario, n, h, reps, rng):
    if scenario in ("level", "scale"):
        months = range(T_BREAK - h + 1, T_BREAK - h + 1 + EVAL_MONTHS)     # targets after the break
    elif scenario == "contam":
        months = range(T_CONTAM + 1, T_CONTAM + 1 + EVAL_MONTHS)          # first origins after the episode
    else:
        months = range(W + h + 12, T - h)
    months = [t for t in months if t < T]
    acc = {k: [] for k in ["none", "rqc", "aci", "tspc", "tspc_ls", "tspc_ar"]}
    cov = {k: [] for k in acc}
    for _ in range(reps):
        y, q = simulate(rng, n, h, scenario)
        sh = corrections(y, q, h)
        for k in acc:
            shift = 0.0 if k == "none" else sh[k][months][:, None]
            qq = q[months] + shift
            acc[k].append(pinball(y[months], qq, ALPHA).mean())
            cov[k].append(np.mean(y[months] <= qq))
    rows = []
    for k in acc:
        rows.append({"scenario": scenario, "cells": n, "horizon": h, "method": k,
                     "loss": np.mean(acc[k]), "loss_se": np.std(acc[k], ddof=1) / np.sqrt(reps),
                     "coverage": np.mean(cov[k]), "ratio_rqc": np.mean(np.array(acc[k]) / np.array(acc["rqc"]))})
    return rows


def main():
    """Usage: 11_simulation.py [reps] [part k of n]; parts are merged when all are present."""
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else REPS
    part, parts = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    grid = [(s, n, h) for s in ["stable", "level", "scale", "contam"] for n in [40, 400, 1600] for h in [1, 3, 6]]
    grid += [("idio", 40, h) for h in [1, 3, 6]]
    rows = []
    for k, (s, n, h) in enumerate(grid):
        if k % parts != part:
            continue
        rng = np.random.default_rng([SEED, k])          # one stream per design cell
        rows += evaluate(s, n, h, reps, rng)
        print(s, n, h, flush=True)
    out = pd.DataFrame(rows)
    if parts == 1:
        out.to_csv(RESULTS / "simulation.csv", index=False)
        return
    out.to_csv(RESULTS / f"simulation_part{part}.csv", index=False)
    files = [RESULTS / f"simulation_part{k}.csv" for k in range(parts)]
    if all(f.exists() for f in files):
        pd.concat([pd.read_csv(f) for f in files]).to_csv(RESULTS / "simulation.csv", index=False)
        for f in files:
            f.unlink()


if __name__ == "__main__":
    main()
