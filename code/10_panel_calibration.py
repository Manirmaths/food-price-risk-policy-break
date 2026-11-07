"""Two-scale panel calibration (TSPC) of quantile forecasts with delayed feedback.

The residual-quantile correction (RQC) shifts every forecast issued at origin t
by the alpha-quantile of the residuals that matured in the last W months.  In a
panel, each month's matured residuals form a large cross-section, so their
level can be estimated from that month alone.  TSPC therefore separates the
residual distribution into a fast common level and a slow shape:

    shift(t) = m(t - h) + Q_alpha{ r_i(s) - m(s) : t - h - W < s <= t - h },

where r_i(s) = y_i(s) - q_i(s) is the matured residual of cell i for a forecast
issued at s, m(s) is the cross-sectional median of the residuals issued at s,
and the newest issuance whose targets have matured at t is s = t - h.  The
level reacts after h months; the pooled RQC quantile reacts after roughly
h + (W - 1)/2 months.  Because the shape term is computed on
median-centred residuals, it does not depend on how the level moved within
the window.

Two extensions are reported as sensitivity checks:

* TSPC-LS rescales the shape by the latest cross-sectional scale,
  shift = m(t-h) + S(t-h) Q_alpha{(r - m(s)) / S(s)}, with S = IQR / 1.349;
* TSPC-AR forecasts the level h months ahead with a lag-h regression of the
  matured medians on their own h-month lag (slope restricted to [0, 1]),
  so a transient level is shrunk towards its historical mean.

Every quantity used at origin t is dated t or earlier.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from importlib import import_module

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("FOOD_RISK_BUILD_DIR", ROOT / "build"))
RESULTS = ROOT / "results"
WINDOW = 12
BASES = ["static", "linear_qr", "hist_gb", "linear_qr_expand", "hist_gb_expand", "linear_qr_roll", "hist_gb_roll"]
METHODS = ["rqc", "tspc", "tspc_ls", "tspc_ar"]
tests = None


def pinball(y, q, alpha):
    r = y - q
    return np.where(r >= 0, alpha * r, (alpha - 1) * r)


def month_index(d):
    d = pd.to_datetime(d)
    return (d.dt.year * 12 + d.dt.month).to_numpy()


def robust_scale(x):
    lo, hi = np.quantile(x, [0.25, 0.75])
    return (hi - lo) / 1.349


class Calibrator:
    def __init__(self, frame, alpha, horizon, window=WINDOW):
        self.a, self.h, self.W = alpha, horizon, window
        self.r = {s: g["r"].to_numpy() for s, g in frame.groupby("s")}
        self.m = {s: float(np.median(v)) for s, v in self.r.items()}
        self.S = {s: robust_scale(v) for s, v in self.r.items()}

    def persistence(self):
        """Lag-h autocorrelation of the monthly median residual over the whole matured history."""
        months = sorted(self.m)
        pairs = np.array([(self.m[k - self.h], self.m[k]) for k in months if k - self.h in self.m])
        return float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1])

    def _window(self, t):
        return [s for s in range(t - self.h - self.W + 1, t - self.h + 1) if s in self.r]

    def ar_level(self, s0, min_pairs=6):
        """Lag-h regression forecast of the level for issuance s0 + h, from medians dated s0 or earlier."""
        months = sorted(k for k in self.m if k <= s0)
        pairs = [(self.m[k - self.h], self.m[k]) for k in months if k - self.h in self.m]
        if len(pairs) < min_pairs:
            return float(np.mean([self.m[k] for k in months]))
        x, z = np.array(pairs).T
        vx = np.var(x)
        b = 0.0 if vx == 0 else float(np.clip(np.cov(x, z, bias=True)[0, 1] / vx, 0.0, 1.0))
        return float(z.mean() - b * x.mean() + b * self.m[s0])

    def shift(self, t):
        win = self._window(t)
        s0 = t - self.h
        pooled = np.concatenate([self.r[s] for s in win])
        centred = np.concatenate([self.r[s] - self.m[s] for s in win])
        standard = np.concatenate([(self.r[s] - self.m[s]) / self.S[s] for s in win])
        shape = float(np.quantile(centred, self.a))
        out = {"rqc": float(np.quantile(pooled, self.a)),
               "tspc": self.m[s0] + shape,
               "tspc_ls": self.m[s0] + self.S[s0] * float(np.quantile(standard, self.a)),
               "level": self.m[s0], "shape": shape}
        out["tspc_ar"] = self.ar_level(s0) + shape
        return out


def origin_series(ev, col, base_col, alpha):
    """Origin-month mean loss difference col - base_col."""
    d = pinball(ev["y"].to_numpy(), ev[col].to_numpy(), alpha) - pinball(ev["y"].to_numpy(), ev[base_col].to_numpy(), alpha)
    return pd.Series(d, index=ev.index).groupby(ev["date"]).mean()


def run(alpha, horizon):
    tag = f"a{int(round(alpha * 100))}_h{horizon}"
    ev = pd.read_csv(BUILD / f"forecasts_{tag}.csv.gz", parse_dates=["date", "target_date"])
    cal = pd.read_csv(BUILD / f"calibration_{tag}.csv.gz", parse_dates=["date", "target_date"])
    ev["t"] = month_index(ev["date"])
    origins = sorted(ev["t"].unique())
    bases = [b for b in BASES if b in cal.columns]
    new_cols, paths, pers = {}, [], []
    for b in bases:
        cols = ["date", "state", "item", "y", b]
        frame = pd.concat([cal[cols], ev[cols]], ignore_index=True).dropna().rename(columns={b: "q"})
        frame["s"] = month_index(frame["date"])
        frame["r"] = frame["y"] - frame["q"]
        cb = Calibrator(frame, alpha, horizon)
        pers.append({"a": alpha, "h": horizon, "base": b, "rho": cb.persistence()})
        sh = pd.DataFrame({t: cb.shift(t) for t in origins}).T
        sh.index.name = "t"
        for m in METHODS:
            new_cols[f"{b}__{m}"] = ev[b].to_numpy() + ev["t"].map(sh[m]).to_numpy()
        p = sh.reset_index()
        p.insert(0, "base", b)
        paths.append(p)
    ev = pd.concat([ev, pd.DataFrame(new_cols, index=ev.index)], axis=1)

    rows = []
    for period in ["all", "stable", "break"]:
        sub = ev if period == "all" else ev[ev["period"].eq(period)]
        y = sub["y"].to_numpy()
        for b in bases:
            cands = {"none": b, "aci": b + "_aci"} | {m: f"{b}__{m}" for m in METHODS}
            for m, c in cands.items():
                if c not in sub:
                    continue
                q = sub[c].to_numpy()
                rows.append({"alpha": alpha, "horizon": horizon, "period": period, "base": b, "method": m,
                             "loss": pinball(y, q, alpha).mean(), "coverage": float(np.mean(y <= q))})
    summary = pd.DataFrame(rows)

    trows = []
    for period in ["all", "stable", "break"]:
        sub = ev if period == "all" else ev[ev["period"].eq(period)]
        for b in bases:
            for ref in ["rqc", "aci", "none"]:
                ref_col = {"rqc": f"{b}__rqc", "aci": b + "_aci", "none": b}[ref]
                if ref_col not in sub:
                    continue
                d = origin_series(sub, f"{b}__tspc", ref_col, alpha).to_numpy()
                nw = tests.nw(d, horizon - 1 if horizon > 1 else 0)
                t_hln, p_hln = tests.hln(d, horizon)
                t_ewc, p_ewc, B = tests.ewc(d)
                trows.append({"alpha": alpha, "horizon": horizon, "period": period, "base": b,
                              "comparison": f"tspc-{ref}", "months": len(d), "diff": d.mean(),
                              "p_nw": nw["p_nw"], "p_hln": p_hln, "p_ewc": p_ewc, "B": B})
    path = pd.concat(paths, ignore_index=True)
    path.insert(0, "horizon", horizon)
    path.insert(0, "alpha", alpha)
    keep = ["date", "target_date", "state", "item", "y", "period"] + bases + list(new_cols) + \
           [b + "_aci" for b in bases if b + "_aci" in ev]
    ev[keep].to_csv(BUILD / f"tspc_{tag}.csv.gz", index=False, compression="gzip")
    pers = pd.DataFrame(pers)
    al = summary[summary.period.eq("all")].pivot_table(index="base", columns="method", values="loss")
    pers["ratio"] = pers["base"].map(al["tspc"] / al["rqc"])
    return summary, pd.DataFrame(trows), path, pers


def main():
    designs = [(a, h) for a in [0.75, 0.90, 0.95] for h in [1, 3, 6]]
    if len(sys.argv) > 1 and sys.argv[1] == "primary":
        designs = [(0.90, 3)]
    out = [run(a, h) for a, h in designs]
    pd.concat([o[0] for o in out]).to_csv(RESULTS / "tspc_summary.csv", index=False)
    pd.concat([o[1] for o in out]).to_csv(RESULTS / "tspc_tests.csv", index=False)
    pd.concat([o[2] for o in out]).to_csv(RESULTS / "tspc_path.csv", index=False)
    pd.concat([o[3] for o in out]).to_csv(RESULTS / "tspc_persistence.csv", index=False)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "code"))
    tests = import_module("04_tests")
    main()
