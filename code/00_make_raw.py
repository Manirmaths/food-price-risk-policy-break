"""Assemble the pipeline inputs from the public source files.

Reads the downloaded NBS, exchange-rate, CPI and ACLED files from
FOOD_RISK_SOURCE_DIR (default data/source) and writes three inputs to
FOOD_RISK_RAW_DIR (default data/raw):

* the state food-price file (copied unchanged);
* a state-month macro panel with petrol, diesel and intra-city bus fares,
  plus the national exchange rate and chain-linked CPI;
* an ACLED state-month conflict file, with each week assigned to the month
  that contains its midpoint.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

SOURCE = Path(os.environ.get("FOOD_RISK_SOURCE_DIR", "data/source"))
RAW = Path(os.environ.get("FOOD_RISK_RAW_DIR", "data/raw"))
START, END = "2017-01-01", "2024-04-01"
FOOD_FILE = "food_prices_combined_available_2017_Apr2026.csv"
ACLED_FILE = "ACLED Data_2026-08-05_event_date_from_2017-03-01_event_date_to_2024-04-30.csv"
EVENT_TYPES = {
    "Battles": "eventtype_battles",
    "Explosions/Remote violence": "eventtype_explosions_remote_violence",
    "Violence against civilians": "eventtype_violence_against_civilians",
}


def month_start(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.to_period("M").dt.to_timestamp()


def norm_state(value: object) -> str:
    value = str(value).strip().replace("_", " ").title()
    return {
        "Abuja": "FCT",
        "Federal Capital Territory": "FCT",
        "Fct": "FCT",
        "Nassarawa": "Nasarawa",
        "Akwa-Ibom": "Akwa Ibom",
        "Cross-River": "Cross River",
    }.get(value, value)


def state_series(path: Path, name: str, item_filter: str | None = None) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig")
    data.columns = [c.strip().lower() for c in data.columns]
    if item_filter is not None:
        data = data[data["item"].str.contains(item_filter, case=False, na=False)]
    data["date"] = month_start(data["date"])
    data["state"] = data["state"].map(norm_state)
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    return data.groupby(["state", "date"], as_index=False)["price"].mean().rename(columns={"price": name})


def macro_panel() -> pd.DataFrame:
    pms = state_series(SOURCE / "pms_monthly_long.csv", "pms_price")
    diesel = state_series(SOURCE / "diesel_data_combined_Jun2015_Apr2026.csv", "diesel_price")
    fare = state_series(SOURCE / "transport_costs_complete_Jan2017_Apr2026.csv", "intracity_fare", "within")

    fx = pd.read_csv(SOURCE / "exchange_rate_monthly_2017_2026.csv")
    fx = pd.DataFrame({"date": month_start(fx["Date"]), "exchange_rate": fx["Exchange_Rate_Price"]})

    cpi = pd.read_csv(SOURCE / "cpi_complete_1995_2026.csv")
    cpi = cpi[cpi["items"].str.contains("chain-linked")]
    cpi = pd.DataFrame({"date": month_start(cpi["date"]), "cpi_index_chainlinked": cpi["price"]})

    panel = pms.merge(diesel, on=["state", "date"], how="outer").merge(fare, on=["state", "date"], how="outer")
    panel = panel.merge(fx, on="date", how="left").merge(cpi, on="date", how="left")
    keep = panel["date"].between(START, END) & panel["state"].ne("National")
    return panel[keep]


def conflict_panel() -> pd.DataFrame:
    acled = pd.read_csv(SOURCE / ACLED_FILE, encoding="utf-8-sig")
    acled["date"] = month_start(pd.to_datetime(acled["week"]) + pd.Timedelta(days=3))
    acled["state"] = acled["admin1"].map(norm_state)
    acled["fatalities"] = pd.to_numeric(acled["fatalities"], errors="coerce").fillna(0)
    out = acled.groupby(["state", "date"]).agg(
        total_events_all=("events", "sum"), total_fatalities_all=("fatalities", "sum")
    ).reset_index()
    for event_type, prefix in EVENT_TYPES.items():
        subset = acled[acled["event_type"].eq(event_type)].groupby(["state", "date"]).agg(
            **{f"{prefix}_events": ("events", "sum"), f"{prefix}_fatalities": ("fatalities", "sum")}
        ).reset_index()
        out = out.merge(subset, on=["state", "date"], how="left")
    return out[out["date"].between("2017-03-01", END)]


def main() -> None:
    (RAW / "csv_files").mkdir(parents=True, exist_ok=True)
    shutil.copy(SOURCE / FOOD_FILE, RAW / FOOD_FILE)
    macro = macro_panel()
    macro.to_csv(RAW / "csv_files" / "food_policy_core_panel_with_exposures_Mar2017_Apr2024.csv", index=False)
    conflict = conflict_panel()
    conflict.to_csv(RAW / "acled_nigeria_state_month_conflict_midpoint_2017-03_to_2024-04.csv", index=False)
    print(f"macro panel: {macro['state'].nunique()} states, {len(macro)} rows")
    print(f"conflict panel: {conflict['state'].nunique()} states, {len(conflict)} rows")


if __name__ == "__main__":
    main()
