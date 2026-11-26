import numpy as np
import pytest

from nyc_congestion.switchback import exposure, simulate


def _series(weeks=8, seed=0):
    rng = np.random.default_rng(seed)
    how = np.tile(np.arange(168), weeks)
    pattern = np.sin(2 * np.pi * how / 24)
    return 5 + pattern + rng.normal(0, 0.05, len(how)), how


def test_exposure_without_carryover_is_the_treatment():
    t = np.array([0, 1, 1, 0.0])
    assert exposure(t, 0).tolist() == t.tolist()
    assert 0 < exposure(t, 1)[1] < 1  # reacts gradually


def test_no_carryover_is_unbiased_with_good_coverage():
    y, how = _series()
    r = simulate(y, how, true_effect=-0.03, block_len=2, carryover_half_life=0, sims=200)
    assert abs(r["bias"]) < 0.002
    assert r["ci95_coverage"] == pytest.approx(0.95, abs=0.05)


def test_carryover_biases_short_blocks_and_burn_in_helps():
    y, how = _series()
    short = simulate(y, how, -0.03, block_len=1, carryover_half_life=1, sims=200)
    burn = simulate(y, how, -0.03, block_len=4, carryover_half_life=1, sims=200, burn_in=2)
    assert short["bias"] > 0.005  # pulled toward zero
    assert abs(burn["bias"]) < abs(short["bias"]) / 3
