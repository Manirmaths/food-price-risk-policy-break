"""Build the state-commodity monthly panel and its predictors.

Lags are computed on the complete calendar grid before missing prices are
dropped, and macroeconomic features use values dated t-1 or earlier.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("FOOD_RISK_RAW_DIR", ROOT / "data" / "raw"))
BUILD = Path(os.environ.get("FOOD_RISK_BUILD_DIR", ROOT / "build"))
BUILD.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2017-01-01")
END = pd.Timestamp("2024-04-01")


def norm_state(value: object) -> str:
    value = str(value).strip().title()
    return {
        "Abuja": "FCT",
        "Federal Capital Territory": "FCT",
        "Fct": "FCT",
        "Nassarawa": "Nasarawa",
    }.get(value, value)


def month_start(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.to_period("M").dt.to_timestamp()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input not found: {path}\n"
            "Set FOOD_RISK_RAW_DIR to the directory containing the raw files."
        )
    return path


def load_prices() -> pd.DataFrame:
    path = require(RAW / "food_prices_combined_available_2017_Apr2026.csv")
    food = pd.read_csv(path)
    food["date"] = month_start(food["DATE"])
    food["state"] = food["STATE"].map(norm_state)
    food["item"] = food["ITEMS"].astype(str).str.strip()
    food["price"] = pd.to_numeric(food["PRICE"], errors="coerce")
    food = food[
        food["date"].between(START, END)
        & food["state"].ne("National")
        & food["price"].gt(0)
    ]
    food = (
        food.groupby(["state", "item", "date"], as_index=False)["price"]
        .mean()
        .sort_values(["state", "item", "date"])
    )

    states = sorted(food["state"].unique())
    items = sorted(food["item"].unique())
    dates = pd.date_range(START, END, freq="MS")
    if (len(states), len(items), len(dates)) != (37, 43, 88):
        raise ValueError(
            "Unexpected NBS dimensions: "
            f"{len(states)} states x {len(items)} items x {len(dates)} months"
        )

    # Use the full grid while constructing lags.  This prevents a missing month
    # from silently turning a row lag into a non-calendar lag.
    grid = pd.MultiIndex.from_product(
        [states, items, dates], names=["state", "item", "date"]
    ).to_frame(index=False)
    return grid.merge(food, on=["state", "item", "date"], how="left")


def load_macro() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = require(
        RAW / "csv_files" / "food_policy_core_panel_with_exposures_Mar2017_Apr2024.csv"
    )
    use = [
        "date",
        "state",
        "pms_price",
        "diesel_price",
        "intracity_fare",
        "exchange_rate",
        "cpi_index_chainlinked",
    ]
    raw = pd.read_csv(path, usecols=use)
    raw["date"] = month_start(raw["date"])
    raw["state"] = raw["state"].map(norm_state)
    raw = raw[raw["date"].between(START, END)]

    state = (
        raw.groupby(["state", "date"], as_index=False)[
            ["pms_price", "diesel_price", "intracity_fare"]
        ]
        .first()
        .rename(
            columns={
                "pms_price": "pms",
                "diesel_price": "diesel",
                "intracity_fare": "transport",
            }
        )
        .sort_values(["state", "date"])
    )
    national = (
        raw.groupby("date", as_index=False)[["exchange_rate", "cpi_index_chainlinked"]]
        .first()
        .rename(columns={"exchange_rate": "fx", "cpi_index_chainlinked": "cpi"})
        .sort_values("date")
    )
    return state, national


def lag_macro(state: pd.DataFrame, national: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use t-1 macro vintages to avoid assuming same-month report availability."""
    state = state.copy()
    for col in ["pms", "diesel", "transport"]:
        logv = np.log(pd.to_numeric(state[col], errors="coerce").where(state[col] > 0))
        by_state = logv.groupby(state["state"])
        state[f"log_{col}_l1"] = by_state.shift(1)
        state[f"d_{col}_l1"] = 100 * (by_state.shift(1) - by_state.shift(2))
        state[f"d3_{col}_l1"] = 100 * (by_state.shift(1) - by_state.shift(4))

    national = national.copy()
    for col in ["fx", "cpi"]:
        logv = np.log(pd.to_numeric(national[col], errors="coerce").where(national[col] > 0))
        national[f"log_{col}_l1"] = logv.shift(1)
        national[f"d_{col}_l1"] = 100 * (logv.shift(1) - logv.shift(2))
        national[f"d3_{col}_l1"] = 100 * (logv.shift(1) - logv.shift(4))
    return state, national


def load_conflict() -> pd.DataFrame:
    path = require(
        RAW / "acled_nigeria_state_month_conflict_midpoint_2017-03_to_2024-04.csv"
    )
    raw = pd.read_csv(path)
    raw["date"] = month_start(raw["date"])
    raw["state"] = raw["state"].map(norm_state)
    event_cols = [
        "eventtype_battles_events",
        "eventtype_explosions_remote_violence_events",
        "eventtype_violence_against_civilians_events",
    ]
    fatality_cols = [c.replace("_events", "_fatalities") for c in event_cols]
    raw["conflict_events"] = raw["total_events_all"].fillna(0)
    raw["conflict_fatalities"] = raw["total_fatalities_all"].fillna(0)
    raw["violent_events"] = raw[event_cols].fillna(0).sum(axis=1)
    raw["violent_fatalities"] = raw[fatality_cols].fillna(0).sum(axis=1)
    return raw[
        [
            "state",
            "date",
            "conflict_events",
            "conflict_fatalities",
            "violent_events",
            "violent_fatalities",
        ]
    ]


def add_price_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["state", "item", "date"]).copy()
    panel["logp"] = np.log(panel["price"])
    group = panel.groupby(["state", "item"], sort=False)
    panel["dlog1"] = 100 * group["logp"].diff(1)
    panel["dlog3"] = 100 * group["logp"].diff(3)
    panel["dlog12"] = 100 * group["logp"].diff(12)
    panel["roll_mean6"] = group["dlog1"].transform(
        lambda s: s.rolling(6, min_periods=3).mean()
    )
    panel["roll_sd6"] = group["dlog1"].transform(
        lambda s: s.rolling(6, min_periods=3).std()
    )
    panel["roll_sd12"] = group["dlog1"].transform(
        lambda s: s.rolling(12, min_periods=6).std()
    )
    panel["roll_max6"] = group["dlog1"].transform(
        lambda s: s.rolling(6, min_periods=3).max()
    )
    return panel


def add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    xs = (
        panel.groupby("date")["dlog1"]
        .agg(xs_med="median", xs_q90=lambda s: s.quantile(0.90), xs_sd="std")
        .reset_index()
    )
    xs3 = (
        panel.groupby("date")["dlog3"]
        .agg(xs3_med="median", xs3_q90=lambda s: s.quantile(0.90))
        .reset_index()
    )
    state = (
        panel.groupby(["state", "date"])["dlog1"]
        .agg(st_med="median", st_q90=lambda s: s.quantile(0.90))
        .reset_index()
    )
    return panel.merge(xs, on="date").merge(xs3, on="date").merge(
        state, on=["state", "date"]
    )


def add_conflict_features(panel: pd.DataFrame) -> pd.DataFrame:
    cols = ["conflict_events", "conflict_fatalities", "violent_events", "violent_fatalities"]
    panel[cols] = panel[cols].fillna(0)
    state_month = panel[["state", "date", *cols]].drop_duplicates(["state", "date"])
    state_month = state_month.sort_values(["state", "date"])
    group = state_month.groupby("state", sort=False)
    for col in cols:
        for window in [3, 12]:
            state_month[f"{col}_s{window}"] = group[col].transform(
                lambda s, w=window: s.rolling(w, min_periods=1).sum()
            )
            state_month[f"log1p_{col}_s{window}"] = np.log1p(
                state_month[f"{col}_s{window}"]
            )
    keep = ["state", "date", *[c for c in state_month if c.startswith("log1p_")]]
    panel = panel.drop(columns=cols)
    return panel.merge(state_month[keep], on=["state", "date"], how="left")


def build() -> pd.DataFrame:
    panel = load_prices()
    expected = len(panel)
    observed = int(panel["price"].notna().sum())

    state_macro, national_macro = lag_macro(*load_macro())
    panel = add_price_features(panel)
    panel = panel.merge(state_macro, on=["state", "date"], how="left")
    panel = panel.merge(national_macro, on="date", how="left")
    panel = panel.merge(load_conflict(), on=["state", "date"], how="left")
    panel = add_conflict_features(panel)
    panel = add_cross_sectional_features(panel)

    month = panel["date"].dt.month
    panel["sin1"] = np.sin(2 * np.pi * month / 12)
    panel["cos1"] = np.cos(2 * np.pi * month / 12)
    panel["sin2"] = np.sin(4 * np.pi * month / 12)
    panel["cos2"] = np.cos(4 * np.pi * month / 12)

    missing = panel.loc[panel["price"].isna(), ["state", "item", "date"]].copy()
    missing.to_csv(BUILD / "missing_price_cells.csv", index=False)
    panel = panel[panel["price"].notna()].copy()
    panel.to_csv(BUILD / "panel.csv.gz", index=False, compression="gzip")

    audit = pd.DataFrame(
        {
            "measure": [
                "states",
                "items",
                "months",
                "expected_balanced_cells",
                "observed_price_cells",
                "missing_price_cells",
                "first_month",
                "last_month",
            ],
            "value": [
                panel["state"].nunique(),
                panel["item"].nunique(),
                panel["date"].nunique(),
                expected,
                observed,
                expected - observed,
                panel["date"].min().strftime("%Y-%m"),
                panel["date"].max().strftime("%Y-%m"),
            ],
        }
    )
    audit.to_csv(BUILD / "panel_audit.csv", index=False)
    print(audit.to_string(index=False))
    return panel


if __name__ == "__main__":
    build()
