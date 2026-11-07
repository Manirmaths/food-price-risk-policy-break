"""External evaluation on World Bank RTFP market panels, following PROTOCOL.md.

Usage: python code/17_external.py COUNTRY [alpha horizon]
COUNTRY in {NGA, NGA_POST, LBN, SDN, HTI, MMR, AFG}; NGA_POST is the untouched
Nigerian period.  Writes results/external/<COUNTRY>_a<alpha>_h<h>_*.csv.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RTFP = Path(os.environ.get("RTFP_DIR", ROOT / "data" / "rtfp"))
OUT = ROOT / "results" / "external"
OUT.mkdir(parents=True, exist_ok=True)


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fc = load_module("fc", "02_forecast.py")
fair = load_module("fair", "14_fair_evaluation.py")

FEATURES = ["dlog1", "dlog3", "dlog12", "roll_mean6", "roll_sd6", "roll_sd12", "roll_max6",
            "xs_med", "xs_q90", "xs_sd", "sin1", "cos1", "sin2", "cos2"]
fc.NUMERIC_FEATURES = FEATURES            # the learners read this module-level list
L2 = 1e-3
HP = {"max_iter": 220, "max_leaf_nodes": 31, "min_samples_leaf": 100, "l2_regularization": 5.0}
W, REFIT = 12, 3
BREAKS = {"NGA": "2023-06-01", "LBN": "2019-11-01", "SDN": "2023-05-01",
          "HTI": "2022-10-01", "MMR": "2021-03-01", "AFG": "2021-09-01"}
METHODS = fair.METHODS
BASES = ["empirical", "linear_qr", "hist_gb"]


def month_index(d):
    d = pd.to_datetime(d)
    return (d.dt.year * 12 + d.dt.month).to_numpy() if hasattr(d, "dt") else d.year * 12 + d.month


def load_panel(iso):
    f = next(RTFP.glob(f"{iso}_RTFP_mkt_*/*.csv"))
    raw = pd.read_csv(f)
    items = [c[2:] for c in raw.columns if c.startswith("c_") and c != "c_food_price_index"]
    raw["state"] = raw["adm1_name"].astype(str) + "|" + raw["adm2_name"].astype(str) + "|" + raw["mkt_name"].astype(str)
    frames = []
    for it in items:
        x = raw[["state", "price_date", f"c_{it}", f"trust_{it}"]].copy()
        x.columns = ["state", "date", "price", "trust"]
        x["item"] = it
        frames.append(x)
    p = pd.concat(frames, ignore_index=True)
    p["date"] = pd.to_datetime(p["date"])
    p = p[p["price"] > 0].copy()
    p = p.sort_values(["state", "item", "date"])
    p["logp"] = np.log(p["price"])
    g = p.groupby(["state", "item"], sort=False)
    for k in (1, 3, 12):                                    # calendar lags on a monthly grid
        lag = p[["state", "item", "date", "logp"]].copy()
        lag["date"] = lag["date"] + pd.DateOffset(months=k)
        p = p.merge(lag.rename(columns={"logp": f"lag{k}"}), on=["state", "item", "date"], how="left")
        p[f"dlog{k}"] = 100 * (p["logp"] - p[f"lag{k}"])
    p = p.sort_values(["state", "item", "date"])
    g = p.groupby(["state", "item"], sort=False)["dlog1"]
    p["roll_mean6"] = g.transform(lambda s: s.rolling(6, min_periods=3).mean())
    p["roll_sd6"] = g.transform(lambda s: s.rolling(6, min_periods=3).std())
    p["roll_sd12"] = g.transform(lambda s: s.rolling(12, min_periods=6).std())
    p["roll_max6"] = g.transform(lambda s: s.rolling(6, min_periods=3).max())
    xs = p.groupby("date")["dlog1"].agg(xs_med="median", xs_q90=lambda s: s.quantile(0.9), xs_sd="std")
    p = p.merge(xs, left_on="date", right_index=True, how="left")
    m = p["date"].dt.month
    p["sin1"], p["cos1"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    p["sin2"], p["cos2"] = np.sin(4 * np.pi * m / 12), np.cos(4 * np.pi * m / 12)
    return p.drop(columns=[c for c in p.columns if c.startswith("lag")])


def add_target(p, h):
    fut = p[["state", "item", "date", "logp", "trust"]].copy()
    fut["date"] = fut["date"] - pd.DateOffset(months=h)
    d = p.merge(fut.rename(columns={"logp": "logp_fwd", "trust": "trust_fwd"}), on=["state", "item", "date"], how="left")
    d["y"] = 100 * (d["logp_fwd"] - d["logp"])
    d["target_date"] = d["date"] + pd.DateOffset(months=h)
    return d[d["y"].notna()].copy()


def windows(country, h, last_target):
    if country == "NGA_POST":
        e0 = pd.Timestamp("2024-05-01")
        e1 = last_target - pd.DateOffset(months=h)
        return e0, e1, None
    b = pd.Timestamp(BREAKS[country])
    e0 = b - pd.DateOffset(months=24 + h)
    e1 = min(b + pd.DateOffset(months=23), last_target) - pd.DateOffset(months=h)
    return e0, e1, b


GB_REGION = False   # set when a country has more than 255 markets (protocol amendment 2)


def gb_view(d):
    """Boosting's native categorical splits allow at most 255 levels; use the region instead of the market if needed."""
    return d.assign(state=d["state"].str.split("|").str[0]) if GB_REGION else d


def base_forecasts(data, origins, alpha):
    out = []
    fits = None
    for k, o in enumerate(origins):
        if k % REFIT == 0:
            train = data[data["target_date"] <= o]
            fits = {"empirical": train.groupby("item")["y"].quantile(alpha),
                    "linear_qr": fc.fit_linear_qr(train, alpha, L2),
                    "hist_gb": fc.fit_hist_qr(gb_view(train), alpha, HP)}
            known_states, known_items = set(train["state"]), set(train["item"])
            print(f"  refit at {o:%Y-%m} on {len(train):,} targets", flush=True)
        cur = data[(data["date"] == o) & data["state"].isin(known_states) & data["item"].isin(known_items)].copy()
        cur["empirical"] = cur["item"].map(fits["empirical"])
        cur["linear_qr"] = fits["linear_qr"].predict(cur)
        cur["hist_gb"] = fits["hist_gb"].predict(gb_view(cur))
        out.append(cur)
    return pd.concat(out, ignore_index=True)


def run(country, alpha, h):
    iso = "NGA" if country == "NGA_POST" else country
    p = load_panel(iso)
    data = add_target(p, h)
    global GB_REGION
    GB_REGION = p["state"].nunique() > 255
    e0, e1, b = windows(country, h, data["target_date"].max())
    first = e0 - pd.DateOffset(months=W + h)
    origins = list(pd.date_range(first, e1, freq="MS"))
    print(country, alpha, h, "origins", origins[0].date(), "to", origins[-1].date(), flush=True)
    fcst = base_forecasts(data, origins, alpha)
    fcst["s"] = month_index(fcst["date"])
    eval_months = [t for t in sorted(fcst["s"].unique()) if t >= e0.year * 12 + e0.month]
    if b is not None:
        bi = b.year * 12 + b.month
        fcst["period"] = np.where(month_index(fcst["target_date"]) >= bi, "break", "stable")
    else:
        fcst["period"] = "post"
    ev = fcst[fcst["s"].isin(eval_months)].copy()
    tmed = p["trust"].median()
    ev["reliable"] = (ev["trust"] >= tmed) & (ev["trust_fwd"] >= tmed)
    shifts, paths = {}, []
    for base in BASES:
        r = fcst["y"] - fcst[base]
        f = fair.Fair.__new__(fair.Fair)
        f.a, f.h = alpha, h
        f.r = {s: v.to_numpy() for s, v in r.groupby(fcst["s"])}
        f.m = {s: float(np.median(v)) for s, v in f.r.items()}
        tau, aci, sh = alpha, {}, {}
        for t in sorted(fcst["s"].unique()):
            s0 = t - h
            if s0 in aci and s0 in f.r:
                tau = float(np.clip(tau + fair.GAMMA * (np.mean(f.r[s0] > aci[s0]) - (1 - alpha)), *fair.TAU_BOUNDS))
            if t in eval_months:
                sh[t] = f.shifts(t, tau)
                aci[t] = sh[t]["aci"]
            else:
                pool = f.pool(t, W)
                aci[t] = float(np.quantile(np.concatenate([f.r[x] for x in pool]), tau)) if pool and s0 in f.r else 0.0
        pre = sorted(m for m in f.m if m < eval_months[0] - h + 1)
        pairs = np.array([(f.m[u - h], f.m[u]) for u in pre if u - h in f.m])
        rho_pre = float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]) if len(pairs) >= 3 else np.nan
        shifts[base] = (pd.DataFrame(sh).T, rho_pre)
    rows, monthly = [], []
    y = ev["y"].to_numpy()
    for base in BASES:
        sh, rho_pre = shifts[base]
        for m in METHODS:
            q = ev[base].to_numpy() + ev["s"].map(sh[m]).to_numpy()
            loss = fc.pinball(y, q, alpha)
            cov = (y <= q).astype(float)
            mm = pd.DataFrame({"s": ev["s"].to_numpy(), "period": ev["period"].to_numpy(), "loss": loss, "cov": cov,
                               "rel": ev["reliable"].to_numpy()})
            g = mm.groupby(["s", "period"]).agg(loss=("loss", "mean"), cov=("cov", "mean"), n=("loss", "size")).reset_index()
            gr = mm[mm.rel].groupby(["s", "period"]).agg(loss_rel=("loss", "mean")).reset_index()
            g = g.merge(gr, on=["s", "period"], how="left")
            g["base"], g["method"], g["rho_pre"] = base, m, rho_pre
            monthly.append(g)
    monthly = pd.concat(monthly, ignore_index=True)
    monthly.insert(0, "horizon", h)
    monthly.insert(0, "alpha", alpha)
    monthly.insert(0, "country", country)
    tag = f"{country}_a{int(round(alpha * 100))}_h{h}"
    monthly.to_csv(OUT / f"{tag}_monthly.csv", index=False)
    print("done", tag, "cells per month", int(ev.groupby("s").size().median()), flush=True)


if __name__ == "__main__":
    c = sys.argv[1]
    a = float(sys.argv[2]) if len(sys.argv) > 2 else 0.90
    hh = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    run(c, a, hh)
