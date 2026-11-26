from datetime import date

import numpy as np
import pandas as pd

from nyc_congestion.api import climatology, feature_rows
from nyc_congestion.forecast import FEATURES


def _weather(days=40):
    hours = pd.date_range("2026-08-20", periods=24 * days, freq="h")
    return pd.DataFrame({"hour": hours, "temperature_c": hours.hour.astype(float), "precipitation_mm": 0.0,
                         "snowfall_cm": 0.0})


def test_climatology_gives_24_hours_of_monthly_averages():
    c = climatology(_weather(), date(2026, 9, 5))
    assert len(c) == 24 and c["temperature_c"].tolist() == list(map(float, range(24)))


def test_feature_rows_for_the_day_after_the_data():
    hours = pd.date_range("2026-08-20", "2026-09-30 23:00", freq="h")
    history = pd.DataFrame({"zone": 161, "hour": hours, "y": np.arange(len(hours), dtype=float)})
    weather = _weather(days=42)  # ends 2026-09-30, so Oct 1 needs climatology
    feats, clim = feature_rows(history, weather, date(2026, 10, 1))
    assert clim
    assert len(feats) == 24
    assert feats[FEATURES].drop(columns=["zone"]).notna().all().all()
    assert feats["lag_24"].iloc[0] == history["y"].iloc[-24]  # Oct 1 00:00 <- Sep 30 00:00


def test_climatology_falls_back_when_month_is_missing():
    c = climatology(_weather(days=10), date(2026, 12, 1))  # history only covers August
    assert c["temperature_c"].notna().all()
