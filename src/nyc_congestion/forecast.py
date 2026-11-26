"""Day-ahead hourly demand forecasting per taxi zone.

Setting: every evening we forecast each hour of the next day. So every feature for hour t may only use trips up
to t - 24 hours (the "lag >= 24" rule). Breaking that rule leaks the future and makes backtests look too good.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

from .zones import INSIDE, POLICY_START

MIN_LAG = 24
FEATURES = [
    "zone", "hour_of_day", "day_of_week", "month", "is_holiday", "inside", "post_policy",
    "lag_24", "lag_48", "lag_168", "lag_336", "mean_same_hour_7d", "mean_24h", "trend_ratio",
    "temperature_c", "precipitation_mm", "snowfall_cm",
]
CATEGORICAL = ["zone"]


def top_zones(agg: Path, n: int = 80, service: str = "fhvhv") -> list[int]:
    q = f"""select zone from '{agg}/hourly_zone.parquet'
            where service='{service}' and direction='pickup' and year(hour)=2024 and zone between 1 and 263
            group by zone order by sum(trips) desc limit {n}"""
    return [int(r[0]) for r in duckdb.connect().execute(q).fetchall()]


def load_hourly(agg: Path, zones: list[int], service: str = "fhvhv") -> pd.DataFrame:
    """Complete zone x hour grid of pickups (hours with no trips are 0, not missing)."""
    q = f"""select zone, hour, sum(trips) y from '{agg}/hourly_zone.parquet'
            where service='{service}' and direction='pickup' and zone in ({",".join(map(str, zones))})
            group by all"""
    df = duckdb.connect().execute(q).df()
    hours = pd.date_range(df["hour"].min(), df["hour"].max(), freq="h")
    grid = pd.MultiIndex.from_product([zones, hours], names=["zone", "hour"]).to_frame(index=False)
    out = grid.merge(df, on=["zone", "hour"], how="left").fillna({"y": 0})
    return out.sort_values(["zone", "hour"]).reset_index(drop=True)


def add_features(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["zone", "hour"]).reset_index(drop=True)
    g = df.groupby("zone")["y"]
    for lag in (24, 48, 168, 336):
        df[f"lag_{lag}"] = g.shift(lag)
    # Same hour on each of the previous 7 days: lags 24, 48, ..., 168.
    df["mean_same_hour_7d"] = sum(g.shift(24 * k) for k in range(1, 8)) / 7
    df["mean_24h"] = g.shift(MIN_LAG).rolling(24).mean().reset_index(level=0, drop=True)
    last_week = g.shift(MIN_LAG).rolling(168).mean().reset_index(level=0, drop=True)
    four_weeks = g.shift(MIN_LAG).rolling(672).mean().reset_index(level=0, drop=True)
    df["trend_ratio"] = last_week / (four_weeks + 1)
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.day_of_week
    df["month"] = df["hour"].dt.month
    hol = holidays.country_holidays("US", subdiv="NY", years=range(2023, 2028))
    df["is_holiday"] = df["hour"].dt.date.map(lambda d: d in hol).astype(int)
    df["inside"] = df["zone"].isin(INSIDE).astype(int)
    df["post_policy"] = (df["hour"] >= pd.Timestamp(POLICY_START)).astype(int)
    df = df.merge(weather, on="hour", how="left")
    df["zone"] = df["zone"].astype("category")
    return df


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Weighted absolute percentage error: total absolute error / total actual. Robust to zero-demand hours."""
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def baselines(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {"naive_yesterday": df["lag_24"], "naive_last_week": df["lag_168"],
            "mean_same_hour_7d": df["mean_same_hour_7d"]}


def train(df: pd.DataFrame, objective: str = "poisson", alpha: float | None = None, rounds: int = 500,
          seed: int = 42) -> lgb.Booster:
    params = {"objective": objective, "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "num_threads": 4,
              "seed": seed, "verbose": -1}
    if alpha is not None:
        params["alpha"] = alpha
    data = lgb.Dataset(df[FEATURES], df["y"], categorical_feature=CATEGORICAL, free_raw_data=False)
    return lgb.train(params, data, num_boost_round=rounds)
