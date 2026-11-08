"""Additional analyses of the pre-specified evaluation (outside the protocol).

1. Loss on targets whose two prices were both observed in the field, not estimated by the
   RTFP model (the product column without prefix is the observed price).
2. Exploratory break-detection corrections: a self-starting two-sided CUSUM on the monthly
   median matured error; after an alarm the correction pools only the errors matured since
   the alarm (at least the three newest months) until the window is full again.

Usage: python code/20_extra.py COUNTRY [alpha horizon]
Writes results/external/<COUNTRY>_a<alpha>_h<h>_extra.csv.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ext", ROOT / "code" / "17_external.py")
ext = importlib.util.module_from_spec(spec)
sys.modules["ext"] = ext
spec.loader.exec_module(ext)
fc, fair = ext.fc, ext.fair

K_REF = 0.5                      # CUSUM reference value, in standard deviations
H_LEVELS = (3.0, 4.0, 5.0)       # alarm thresholds; 4 is the main setting
W = ext.W
_orig_load = ext.load_panel


def load_panel(iso):
    """As in 17_external, plus a flag for prices observed rather than model-estimated."""
    p = _orig_load(iso)
    f = next(ext.RTFP.glob(f"{iso}_RTFP_mkt_*/*.csv"))
    raw = pd.read_csv(f)
    raw["state"] = raw["adm1_name"].astype(str) + "|" + raw["adm2_name"].astype(str) + "|" + raw["mkt_name"].astype(str)
    items = p["item"].unique()
    obs = pd.concat([raw.loc[raw[it].notna(), ["state", "price_date"]].assign(item=it) for it in items if it in raw])
    obs["date"] = pd.to_datetime(obs.pop("price_date"))
    obs["observed"] = True
    p = p.merge(obs, on=["state", "item", "date"], how="left")
    p["observed"] = p["observed"].fillna(False).astype(bool)
    return p


def add_target(p, h):
    fut = p[["state", "item", "date", "logp", "trust", "observed"]].copy()
    fut["date"] = fut["date"] - pd.DateOffset(months=h)
    d = p.merge(fut.rename(columns={"logp": "logp_fwd", "trust": "trust_fwd", "observed": "observed_fwd"}),
                on=["state", "item", "date"], how="left")
    d["y"] = 100 * (d["logp_fwd"] - d["logp"])
    d["target_date"] = d["date"] + pd.DateOffset(months=h)
    return d[d["y"].notna()].copy()


def cusum_alarms(m, H):
    """Months (of matured errors) at which a self-starting two-sided CUSUM alarms.

    z_u standardises m_u by the mean and standard deviation of the W previous monthly
    medians; both sums restart after an alarm.  Uses only m up to u, so it is causal."""
    months = sorted(m)
    sp = sn = 0.0
    alarms = []
    for u in months:
        ref = [m[v] for v in range(u - W, u) if v in m]
        if len(ref) < 6:
            continue
        sd = float(np.std(ref, ddof=1))
        if sd <= 0:
            continue
        z = (m[u] - float(np.mean(ref))) / sd
        sp, sn = max(0.0, sp + z - K_REF), max(0.0, sn - z - K_REF)
        if sp > H or sn > H:
            alarms.append(u)
            sp = sn = 0.0
    return alarms


def detect_shift(f, t, alarms, ew=False):
    """Correction at origin t: pooled (or exponentially weighted) residual quantile, with the
    window cut back to the errors matured since the last alarm while it is not yet full."""
    s0 = t - f.h
    past = [u for u in alarms if u <= s0]
    if past and s0 - past[-1] + 1 < W:
        n = max(3, s0 - past[-1] + 1)
        return float(np.quantile(np.concatenate([f.r[s] for s in f.pool(t, n)]), f.a))
    sh = f.shifts(t, f.a)
    return sh["ewrqc"] if ew else sh["rqc"]


def run(country, alpha, h):
    ext.load_panel, ext.add_target = load_panel, add_target
    iso = "NGA" if country == "NGA_POST" else country
    p = load_panel(iso)
    data = add_target(p, h)
    ext.GB_REGION = p["state"].nunique() > 255
    e0, e1, b = ext.windows(country, h, data["target_date"].max())
    origins = list(pd.date_range(e0 - pd.DateOffset(months=W + h), e1, freq="MS"))
    fcst = ext.base_forecasts(data, origins, alpha)
    fcst["s"] = ext.month_index(fcst["date"])
    e0i = e0.year * 12 + e0.month
    eval_months = [t for t in sorted(fcst["s"].unique()) if t >= e0i]
    if b is not None:
        fcst["period"] = np.where(ext.month_index(fcst["target_date"]) >= b.year * 12 + b.month, "break", "stable")
    else:
        fcst["period"] = "post"
    ev = fcst[fcst["s"].isin(eval_months)].copy()
    ev["obs"] = ev["observed"] & ev["observed_fwd"]
    y = ev["y"].to_numpy()
    out = []
    for base in ext.BASES:
        r = fcst["y"] - fcst[base]
        f = fair.Fair.__new__(fair.Fair)
        f.a, f.h = alpha, h
        f.r = {s: v.to_numpy() for s, v in r.groupby(fcst["s"])}
        f.m = {s: float(np.median(v)) for s, v in f.r.items()}
        alarms = {H: cusum_alarms(f.m, H) for H in H_LEVELS}
        sh = {}
        for t in eval_months:
            base_sh = f.shifts(t, alpha)
            row = {m: base_sh[m] for m in ["none", "rqc", "ewrqc", "rqc3", "lcs", "tspc"]}
            for H in H_LEVELS:
                row[f"cusum{int(H)}"] = detect_shift(f, t, alarms[H])
                row[f"cusum{int(H)}_ew"] = detect_shift(f, t, alarms[H], ew=True)
            sh[t] = row
        sh = pd.DataFrame(sh).T
        first_alarm = {H: [u for u in alarms[H] if u >= e0i - h] for H in H_LEVELS}
        for m in sh.columns:
            q = ev[base].to_numpy() + ev["s"].map(sh[m]).to_numpy()
            mm = pd.DataFrame({"s": ev["s"].to_numpy(), "period": ev["period"].to_numpy(),
                               "loss": fc.pinball(y, q, alpha), "cov": (y <= q).astype(float), "obs": ev["obs"].to_numpy()})
            g = mm.groupby(["s", "period"]).agg(loss=("loss", "mean"), cov=("cov", "mean"), n=("loss", "size")).reset_index()
            go = mm[mm.obs].groupby(["s", "period"]).agg(loss_obs=("loss", "mean"), n_obs=("loss", "size")).reset_index()
            g = g.merge(go, on=["s", "period"], how="left")
            g["base"], g["method"] = base, m
            g["alarms4"] = ";".join(str(u) for u in first_alarm[4.0])
            out.append(g)
    out = pd.concat(out, ignore_index=True)
    out.insert(0, "horizon", h)
    out.insert(0, "alpha", alpha)
    out.insert(0, "country", country)
    tag = f"{country}_a{int(round(alpha * 100))}_h{h}"
    out.to_csv(ext.OUT / f"{tag}_extra.csv", index=False)
    print("done", tag, "obs share", round(float(ev["obs"].mean()), 3), flush=True)


if __name__ == "__main__":
    c = sys.argv[1]
    a = float(sys.argv[2]) if len(sys.argv) > 2 else 0.90
    hh = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    run(c, a, hh)
