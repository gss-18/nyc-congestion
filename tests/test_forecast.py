import numpy as np
import pandas as pd

from nyc_congestion.forecast import add_features, wape


def _df():
    hours = pd.date_range("2024-01-01", periods=24 * 30, freq="h")
    rows = [pd.DataFrame({"zone": z, "hour": hours, "y": np.arange(len(hours)) + 1000 * z}) for z in (1, 2)]
    weather = pd.DataFrame({"hour": hours, "temperature_c": 0.0, "precipitation_mm": 0.0, "snowfall_cm": 0.0})
    return add_features(pd.concat(rows), weather)


def test_lags_stay_within_zone_and_use_only_the_past():
    df = _df()
    z2 = df[df["zone"] == 2].reset_index(drop=True)
    assert np.isnan(z2.loc[23, "lag_24"])  # no value from zone 1 leaks into zone 2's first day
    assert z2.loc[100, "lag_24"] == z2.loc[76, "y"]


def test_rolling_features_never_see_the_last_24_hours():
    df = _df()
    z1 = df[df["zone"] == 1].reset_index(drop=True)
    t = 500
    assert z1.loc[t, "mean_24h"] == z1.loc[t - 47:t - 24, "y"].mean()


def test_wape():
    assert wape(np.array([10, 0, 10]), np.array([8, 1, 12])) == 0.25
