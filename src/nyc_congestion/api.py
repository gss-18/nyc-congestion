"""Forecast API: day-ahead hourly Uber/Lyft pickups for a taxi zone.

  GET /health                        model info
  GET /zones                         zones the model covers
  GET /forecast?zone=161&date=2026-10-01   24 hourly forecasts with p10/p90 bands

Day-ahead only: the requested date must be at most one day after the last hour of data, because every feature
needs trips from at least 24 hours earlier. When the requested day has no recorded weather (a real future day),
the API uses the average weather for that month and hour, and says so in the response.

Run locally: uv run uvicorn nyc_congestion.api:app --reload
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from .forecast import FEATURES, add_features, load_hourly

ROOT = Path(os.environ.get("NYC_CONGESTION_ROOT", Path(__file__).resolve().parents[2]))
MODEL_DIR = ROOT / "runs" / "forecast"

app = FastAPI(title="NYC zone demand forecast", version="0.1.0")


@lru_cache(maxsize=1)
def resources() -> dict:
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    models = {"p50": lgb.Booster(model_file=str(MODEL_DIR / "model.txt"))}
    for q in ("p10", "p90"):
        path = MODEL_DIR / f"model_{q}.txt"
        if path.exists():
            models[q] = lgb.Booster(model_file=str(path))
    weather = pd.read_parquet(ROOT / "data" / "raw" / "weather.parquet")
    return {"meta": meta, "models": models, "weather": weather}


def climatology(weather: pd.DataFrame, day: date) -> pd.DataFrame:
    """Average weather for each hour of `day`'s calendar month (a stand-in when no forecast is available).

    Falls back to the average over all months if the history has no data for that month.
    """
    cols = ["temperature_c", "precipitation_mm", "snowfall_cm"]
    w = weather.assign(m=weather["hour"].dt.month, h=weather["hour"].dt.hour)
    same_month = w[w["m"] == day.month]
    avg = (same_month if len(same_month) else w).groupby("h")[cols].mean()  # no such month yet: use all months
    hours = pd.date_range(pd.Timestamp(day), periods=24, freq="h")
    return avg.reindex(hours.hour).set_index(hours).rename_axis("hour").reset_index()


def feature_rows(history: pd.DataFrame, weather: pd.DataFrame, day: date) -> tuple[pd.DataFrame, bool]:
    """Features for the 24 hours of `day`, from one zone's hourly history (zone, hour, y)."""
    hours = pd.date_range(pd.Timestamp(day), periods=24, freq="h")
    future = pd.DataFrame({"zone": history["zone"].iloc[0], "hour": hours, "y": float("nan")})
    frame = pd.concat([history[history["hour"] < hours[0]], future], ignore_index=True)
    used_climatology = not weather["hour"].isin(hours).all()
    if used_climatology:
        weather = pd.concat([weather[~weather["hour"].isin(hours)], climatology(weather, day)])
    feats = add_features(frame, weather)
    return feats[feats["hour"].isin(hours)], used_climatology


@app.get("/health")
def health() -> dict:
    r = resources()
    return {"status": "ok", "trained_through": r["meta"]["trained_through"], "models": sorted(r["models"])}


@app.get("/zones")
def zones() -> dict:
    return {"zones": resources()["meta"]["zones"]}


@app.get("/forecast")
def forecast(zone: Annotated[int, Query(description="TLC taxi zone id")],
             date_: Annotated[date, Query(alias="date", description="day to forecast, YYYY-MM-DD")]) -> dict:
    r = resources()
    meta = r["meta"]
    if zone not in meta["zones"]:
        raise HTTPException(404, f"zone {zone} is not covered; see /zones")
    agg = ROOT / "data" / "agg"
    history = load_hourly(agg, [zone])
    last = history["hour"].max().date()
    if date_ > last + timedelta(days=1):
        raise HTTPException(422, f"day-ahead only: latest forecastable date is {last + timedelta(days=1)}")
    start = pd.Timestamp(date_) - pd.Timedelta(days=30)  # enough history for every lag/rolling feature
    feats, clim = feature_rows(history[history["hour"] >= start], r["weather"], date_)
    preds = {name: m.predict(feats[FEATURES]) for name, m in r["models"].items()}
    return {
        "zone": zone, "date": str(date_), "weather_source": "monthly average (no forecast)" if clim else "recorded",
        "hours": [{"hour": int(h.hour), **{k: round(float(v[i]), 1) for k, v in preds.items()}}
                  for i, h in enumerate(feats["hour"])],
    }
