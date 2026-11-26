"""Zone x month (or week) panels built from the hourly aggregates."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .zones import BORDER, INSIDE, UNKNOWN

AIRPORTS = frozenset({1, 132, 138})  # EWR, JFK, LaGuardia: demand driven by flights, not the city


def load_panel(agg: Path, service: str = "fhvhv", direction: str = "pickup", freq: str = "month",
               hours: str | None = None, start: str = "2023-01-01", end: str = "2026-10-01") -> pd.DataFrame:
    """Trips per zone per period. `hours` is an optional SQL filter on the timestamp column `hour`."""
    where = f" and ({hours})" if hours else ""
    q = f"""
    select date_trunc('{freq}', hour) as period, zone, sum(trips) as trips
    from '{agg}/hourly_zone.parquet'
    where service = '{service}' and direction = '{direction}'
      and hour >= '{start}' and hour < '{end}' {where}
    group by all
    """
    df = duckdb.connect().execute(q).df()
    df["period"] = pd.to_datetime(df["period"])
    return df


def label_units(df: pd.DataFrame, min_monthly_trips: float = 1000, ref_year: int = 2024) -> pd.DataFrame:
    """Keep a balanced panel of zones and label each as inside / border / outside.

    Dropped: unknown zones, airports, and outside zones too small to give stable log counts.
    """
    df = df[~df["zone"].isin(UNKNOWN | AIRPORTS)].copy()
    periods = df["period"].nunique()
    full = df.groupby("zone")["period"].nunique() == periods
    df = df[df["zone"].isin(full[full].index)]
    months_in_ref = df.loc[df["period"].dt.year == ref_year, "period"].dt.to_period("M").nunique()
    ref = df[df["period"].dt.year == ref_year].groupby("zone")["trips"].sum() / max(months_in_ref, 1)
    big = ref[ref >= min_monthly_trips].index
    df = df[df["zone"].isin(big) | df["zone"].isin(INSIDE)]
    df["group"] = np.select([df["zone"].isin(INSIDE), df["zone"].isin(BORDER)], ["inside", "border"], "outside")
    df["log_trips"] = np.log(df["trips"])
    return df.reset_index(drop=True)
