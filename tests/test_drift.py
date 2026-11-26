import numpy as np

from nyc_congestion.drift import psi, status


def test_same_distribution_is_stable():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(0, 1, 50_000), rng.normal(0, 1, 50_000)) < 0.01


def test_shifted_distribution_alerts():
    rng = np.random.default_rng(0)
    value = psi(rng.normal(0, 1, 50_000), rng.normal(1, 1, 50_000))
    assert status(value) == "alert"


def test_constant_reference_returns_zero():
    assert psi(np.ones(100), np.arange(100.0)) == 0.0
