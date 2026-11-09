"""Comparison of the delayed corrections on the same base forecasts, including the
latest-cross-section correction and the TSPC variants with level uncertainty.

Every correction below is applied to the same base forecasts at the same
origins and uses only residuals whose targets have matured by the origin.

Residual-history designs
  full     residuals from the pre-evaluation forecasts (Oct 2019 - Dec 2020 for
           h = 3) and from the evaluation forecasts, as in the main analysis;
  restart  calibration restarts when the evaluation forecaster is installed in
           January 2021: only residuals of forecasts issued from January 2021
           are used, and no correction is applied until the first one matures.
A further period, "clean", keeps only origins whose 12-month window contains
no pre-evaluation residual.

Corrections (shift added to every base forecast issued at t; s0 = t - h)
  none      no correction
  rqc       alpha-quantile of residuals pooled over the last 12 issue months
  rqc3/6    the same with 3 or 6 months
  ewrqc     exponentially weighted quantile over 24 months, half-life 3 months
  lcs       alpha-quantile of the latest matured cross-section (issue month s0)
  aci       adaptive conformal inference on the 12-month pool (gamma = 0.05),
            updated with the miss rate of the corrected forecasts issued at s0
  tspc      m(s0) + pooled quantile of median-centred residuals
  tspc_ar   lag-h regression forecast of the level + the same shape
  tspc_u    alpha-quantile of m(s0) + D + c, where D ranges over past h-month
            changes of the monthly median and c over the centred residuals:
            the random-walk level forecast with its own forecast uncertainty
  tspc_aru  alpha-quantile of L + eta + c, where L is the lag-h regression
            forecast and eta its in-sample residuals
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tspc", ROOT / "code" / "10_panel_calibration.py")
tspc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tspc)
sys.path.insert(0, str(ROOT / "code"))
tests = __import__("04_tests")

BUILD, RESULTS = tspc.BUILD, tspc.RESULTS
W, GAMMA, TAU_BOUNDS = 12, 0.05, (0.5, 0.995)
EVAL_START = 2021 * 12 + 1
GRID = (np.arange(1000) + 0.5) / 1000
METHODS = ["none", "rqc", "rqc3", "rqc6", "ewrqc", "lcs", "aci", "tspc", "tspc_ar", "tspc_u", "tspc_aru"]
BASES = tspc.BASES


def weighted_quantile(values, weights, a):
    o = np.argsort(values)
    v, w = values[o], weights[o]
    c = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(c, a)])


class Fair:
    def __init__(self, frame, a, h):
        self.a, self.h = a, h
        self.r = {s: g["r"].to_numpy() for s, g in frame.groupby("s")}
        self.m = {s: float(np.median(v)) for s, v in self.r.items()}

    def pool(self, t, months):
        return [s for s in range(t - self.h - months + 1, t - self.h + 1) if s in self.r]

    def level_changes(self, s0):
        return np.array([self.m[u] - self.m[u - self.h] for u in sorted(self.m) if u <= s0 and u - self.h in self.m])

    def ar_fit(self, s0, min_pairs=6):
        pairs = [(self.m[u - self.h], self.m[u]) for u in sorted(self.m) if u <= s0 and u - self.h in self.m]
        if len(pairs) < min_pairs:
            return None
        x, z = np.array(pairs).T
        vx = np.var(x)
        b = 0.0 if vx == 0 else float(np.clip(np.cov(x, z, bias=True)[0, 1] / vx, 0.0, 1.0))
        a0 = z.mean() - b * x.mean()
        return a0 + b * self.m[s0], z - a0 - b * x

    def shifts(self, t, tau):
        s0 = t - self.h
        out = dict.fromkeys(METHODS, 0.0)
        win = self.pool(t, W)
        if not win or s0 not in self.r:
            return out
        pooled = np.concatenate([self.r[s] for s in win])
        centred = np.concatenate([self.r[s] - self.m[s] for s in win])
        cq = np.quantile(centred, GRID)
        shape = float(np.quantile(centred, self.a))
        out["rqc"] = float(np.quantile(pooled, self.a))
        out["aci"] = float(np.quantile(pooled, tau))
        for k, n in [("rqc3", 3), ("rqc6", 6)]:
            out[k] = float(np.quantile(np.concatenate([self.r[s] for s in self.pool(t, n)]), self.a))
        ew = self.pool(t, 24)
        vals = np.concatenate([self.r[s] for s in ew])
        wts = np.concatenate([np.full(len(self.r[s]), 0.5 ** ((s0 - s) / 3) / len(self.r[s])) for s in ew])
        out["ewrqc"] = weighted_quantile(vals, wts, self.a)
        out["lcs"] = float(np.quantile(self.r[s0], self.a))
        out["tspc"] = self.m[s0] + shape
        d = self.level_changes(s0)
        out["tspc_u"] = float(np.quantile((self.m[s0] + d[:, None] + cq[None, :]).ravel(), self.a)) if len(d) >= 3 else out["tspc"]
        fit = self.ar_fit(s0)
        if fit is None:
            out["tspc_ar"] = out["tspc_aru"] = out["tspc"]
        else:
            lev, eta = fit
            out["tspc_ar"] = lev + shape
            out["tspc_aru"] = float(np.quantile((lev + eta[:, None] + cq[None, :]).ravel(), self.a))
        return out


def run(a, h):
    tag = f"a{int(round(a * 100))}_h{h}"
    ev = pd.read_csv(BUILD / f"forecasts_{tag}.csv.gz", parse_dates=["date", "target_date"])
    cal = pd.read_csv(BUILD / f"calibration_{tag}.csv.gz", parse_dates=["date", "target_date"])
    ev["t"] = tspc.month_index(ev["date"])
    origins = sorted(ev["t"].unique())
    clean_start = EVAL_START + h + W - 1
    bases = [b for b in BASES if b in cal.columns]
    y = ev["y"].to_numpy()
    periods = {"all": np.ones(len(ev), bool), "stable": ev["period"].eq("stable").to_numpy(),
               "break": ev["period"].eq("break").to_numpy(), "clean": ev["t"].to_numpy() >= clean_start,
               "clean_stable": (ev["t"].to_numpy() >= clean_start) & ev["period"].eq("stable").to_numpy()}
    rows, trows, paths = [], [], []
    for design in ["full", "restart"]:
        for b in bases:
            cols = ["date", "y", b]
            frame = pd.concat([cal[cols], ev[cols]], ignore_index=True).dropna().rename(columns={b: "q"})
            frame["s"] = tspc.month_index(frame["date"])
            if design == "restart":
                frame = frame[frame["s"] >= EVAL_START]
            frame["r"] = frame["y"] - frame["q"]
            f = Fair(frame, a, h)
            tau, sh = a, {}
            for t in origins:
                s0 = t - h
                if s0 in sh and s0 in f.r:          # ACI feedback from forecasts issued at s0
                    miss = np.mean(f.r[s0] > sh[s0]["aci"])
                    tau = float(np.clip(tau + GAMMA * (miss - (1 - a)), *TAU_BOUNDS))
                sh[t] = f.shifts(t, tau)
            sh = pd.DataFrame(sh).T
            p = sh.copy()
            p.index.name = "t"
            p = p.reset_index()
            p.insert(0, "base", b)
            p.insert(0, "design", design)
            paths.append(p)
            q = {m: ev[b].to_numpy() + ev["t"].map(sh[m]).to_numpy() for m in METHODS}
            loss = {m: tspc.pinball(y, q[m], a) for m in METHODS}
            for per, mask in periods.items():
                for m in METHODS:
                    rows.append({"alpha": a, "horizon": h, "design": design, "period": per, "base": b, "method": m,
                                 "loss": loss[m][mask].mean(), "coverage": float(np.mean(y[mask] <= q[m][mask]))})
            pairs = [(m, "rqc") for m in ["tspc", "tspc_u", "tspc_ar", "tspc_aru", "lcs", "ewrqc", "rqc3", "aci"]] + \
                    [(m, "none") for m in ["rqc", "aci", "tspc", "tspc_u", "tspc_aru", "ewrqc"]] + \
                    [("tspc_u", "tspc"), ("tspc_aru", "tspc_ar"), ("tspc", "lcs"), ("tspc", "ewrqc"), ("tspc_u", "aci")]
            for per in ["all", "stable", "break", "clean"]:
                mask = periods[per]
                for m1, m2 in pairs:
                    d = pd.Series(loss[m1][mask] - loss[m2][mask]).groupby(ev["date"].to_numpy()[mask]).mean().to_numpy()
                    nw = tests.nw(d, max(h - 1, 0))
                    _, p_hln = tests.hln(d, h)
                    _, p_ewc, _ = tests.ewc(d)
                    trows.append({"alpha": a, "horizon": h, "design": design, "period": per, "base": b,
                                  "comparison": f"{m1}-{m2}", "months": len(d), "diff": d.mean(),
                                  "p_nw": nw["p_nw"], "p_hln": p_hln, "p_ewc": p_ewc})
    path = pd.concat(paths, ignore_index=True)
    path.insert(0, "horizon", h)
    path.insert(0, "alpha", a)
    return pd.DataFrame(rows), pd.DataFrame(trows), path


def main():
    designs = [(a, h) for a in [0.75, 0.90, 0.95] for h in [1, 3, 6]]
    if len(sys.argv) > 1 and sys.argv[1] == "primary":
        designs = [(0.90, 3)]
    out = [run(a, h) for a, h in designs]
    pd.concat([o[0] for o in out]).to_csv(RESULTS / "fair_summary.csv", index=False)
    pd.concat([o[1] for o in out]).to_csv(RESULTS / "fair_tests.csv", index=False)
    pd.concat([o[2] for o in out]).to_csv(RESULTS / "fair_path.csv", index=False)


if __name__ == "__main__":
    main()
