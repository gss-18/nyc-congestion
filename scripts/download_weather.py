"""Hourly weather for Manhattan from the Open-Meteo historical archive (free, no key).

https://open-meteo.com/en/docs/historical-weather-api
Writes data/raw/weather.parquet: hour (local NYC time), temperature_c, precipitation_mm, snowfall_cm.

Usage: uv run python scripts/download_weather.py
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "weather.parquet"
URL = "https://archive-api.open-meteo.com/v1/archive"
PARAMS = {
    "latitude": 40.758, "longitude": -73.985,  # Midtown Manhattan
    "start_date": "2023-01-01", "end_date": "2026-09-30",
    "hourly": "temperature_2m,precipitation,snowfall",
    "timezone": "America/New_York",  # TLC timestamps are local time
}

if __name__ == "__main__":
    r = httpx.get(URL, params=PARAMS, timeout=120, headers={"User-Agent": "nyc-congestion-study/0.1"})
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({"hour": pd.to_datetime(h["time"]), "temperature_c": h["temperature_2m"],
                       "precipitation_mm": h["precipitation"], "snowfall_cm": h["snowfall"]})
    df = df.drop_duplicates("hour")  # the DST fall-back hour appears twice in local time
    df.to_parquet(OUT, index=False)
    print(f"[weather] {len(df):,} hours, {df['hour'].min()} to {df['hour'].max()}, "
          f"missing values: {int(df.isna().sum().sum())}")
