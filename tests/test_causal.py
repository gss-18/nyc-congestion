"""Check the estimators on fake data where the true effect is known."""

import numpy as np
import pandas as pd
import pytest

from nyc_congestion.causal import event_study_twfe, imputation_did, rmspe_ratio, synthetic_control

PERIODS = pd.date_range("2023-01-01", "2026-09-01", freq="MS")
POLICY = "2025-01-01"


def fake_panel(effect=-0.10, treated_trend=-0.05, n_treated=10, n_control=30, seed=0):
    """log trips = zone level + common shock + treated trend (per year) + seasonality + effect after policy."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.05, len(PERIODS))
    years = (PERIODS - PERIODS[0]).days / 365.25
    rows = []
    for z in range(n_treated + n_control):
        treated = z < n_treated
        level = rng.normal(8, 1)
        season = 0.08 * np.sin(2 * np.pi * PERIODS.month / 12) * (1.5 if treated else 1.0)
        y = level + common + season + rng.normal(0, 0.01, len(PERIODS))
        if treated:
            y = y + treated_trend * years + effect * (PERIODS >= POLICY)
        rows.append(pd.DataFrame({"zone": z, "period": PERIODS, "log_trips": y,
                                  "group": "inside" if treated else "outside"}))
    df = pd.concat(rows, ignore_index=True)
    df["trips"] = np.exp(df["log_trips"])
    return df


def test_imputation_recovers_effect_despite_pretrend_and_seasonality():
    res = imputation_did(fake_panel(effect=-0.10), POLICY)
    assert res.effect_pct_mean_zone == pytest.approx((np.exp(-0.10) - 1) * 100, abs=1.0)


def test_ignoring_the_pretrend_gives_a_biased_answer():
    naive = imputation_did(fake_panel(effect=-0.10), POLICY, trend=False, seasonality=False)
    assert naive.effect_pct_mean_zone < -13  # the old downward trend gets counted as policy effect


def test_no_effect_gives_zero():
    res = imputation_did(fake_panel(effect=0.0), POLICY)
    assert abs(res.effect_pct_mean_zone) < 1.0


def test_event_study_shows_pretrend():
    es = event_study_twfe(fake_panel(effect=0.0, treated_trend=-0.05), POLICY)
    early = es.loc[es["offset"] == -24, "coef"].item()
    assert early == pytest.approx(0.10, abs=0.03)  # 2 years before: 2 x 5% higher than at the baseline month


def test_synthetic_control_matches_a_mix_of_donors():
    idx = pd.RangeIndex(60)
    rng = np.random.default_rng(1)
    donors = pd.DataFrame({f"d{i}": 100 + rng.normal(0, 5, 60).cumsum() for i in range(5)}, index=idx)
    norm = donors / donors[:40].mean()
    treated = (0.7 * norm["d0"] + 0.3 * norm["d3"]) * 50
    pre = np.arange(60) < 40
    w, synth = synthetic_control(treated, donors, pre)
    assert w[0] == pytest.approx(0.7, abs=0.05) and w[3] == pytest.approx(0.3, abs=0.05)
    assert rmspe_ratio(treated, synth, pre) < 5


def test_bootstrap_is_reproducible_regardless_of_row_order():
    from nyc_congestion.causal import bootstrap

    df = fake_panel(effect=-0.05)
    a = bootstrap(df, POLICY, reps=30)
    b = bootstrap(df.sample(frac=1, random_state=3), POLICY, reps=30)
    assert a.ci_aggregate == b.ci_aggregate
