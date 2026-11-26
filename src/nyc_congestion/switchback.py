"""Simulate switchback experiments on real hourly demand.

Why switchbacks: in a two-sided marketplace, randomizing *riders* into a price test is broken because treated and
control riders share the same drivers (interference). A switchback randomizes *time blocks* instead: the whole
area gets treatment or control for L hours, then a coin flip decides the next block.

Trade-off this module measures:
  - short blocks -> many blocks -> low variance, but effects spill over from one block into the next (carryover),
    which biases the estimate toward zero;
  - long blocks -> little carryover bias, but few blocks -> high variance;
  - dropping a "burn-in" period after each switch removes most carryover bias at the cost of some data;
  - CUPED (using last week's demand at the same hour as a covariate) cuts variance for free.

We inject a known effect into real demand, so bias, error, coverage and power can be measured exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Estimate:
    effect: float
    se: float


def assign_blocks(n_hours: int, block_len: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (treatment 0/1 per hour, block id per hour)."""
    block = np.arange(n_hours) // block_len
    coin = rng.integers(0, 2, block.max() + 1)
    return coin[block].astype(float), block


def exposure(treat: np.ndarray, carryover_half_life: float) -> np.ndarray:
    """Effective exposure: demand reacts gradually to a switch (exponential smoothing of the 0/1 treatment)."""
    if carryover_half_life <= 0:
        return treat
    a = 0.5 ** (1 / carryover_half_life)
    e = np.empty_like(treat)
    e[0] = treat[0]
    for t in range(1, len(treat)):
        e[t] = a * e[t - 1] + (1 - a) * treat[t]
    return e


def _demean_by(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    sums = np.bincount(groups, weights=x)
    counts = np.bincount(groups)
    return x - (sums / np.maximum(counts, 1))[groups]


def estimate(log_y: np.ndarray, treat: np.ndarray, block: np.ndarray, hour_of_week: np.ndarray,
             burn_in: int = 0, covariate: np.ndarray | None = None, fixed_effects: bool = True) -> Estimate:
    """OLS of log demand on treatment (+ hour-of-week fixed effects, + optional CUPED covariate).

    Standard error is clustered by block, because hours inside one block share the same coin flip.
    burn_in drops the first hours of each block.
    """
    pos_in_block = np.arange(len(block)) - np.searchsorted(block, block)
    keep = pos_in_block >= burn_in
    y, d, b, how = log_y[keep], treat[keep], block[keep], hour_of_week[keep]
    cols = [d]
    if covariate is not None:
        cols.append(covariate[keep])
    if fixed_effects:
        y = _demean_by(y, how)
        cols = [_demean_by(c, how) for c in cols]
    else:
        y = y - y.mean()
        cols = [c - c.mean() for c in cols]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    bread = np.linalg.pinv(X.T @ X)
    scores = np.zeros((b.max() + 1, X.shape[1]))
    np.add.at(scores, b, X * resid[:, None])
    V = bread @ (scores.T @ scores) @ bread
    return Estimate(float(beta[0]), float(np.sqrt(V[0, 0])))


def simulate(log_base: np.ndarray, hour_of_week: np.ndarray, true_effect: float, block_len: int,
             carryover_half_life: float, sims: int = 500, burn_in: int = 0, covariate: np.ndarray | None = None,
             fixed_effects: bool = True, seed: int = 42) -> dict[str, float]:
    """Monte Carlo over random block assignments. Returns bias, RMSE, 95% CI coverage and power."""
    rng = np.random.default_rng(seed)
    est, covered, rejected = [], 0, 0
    for _ in range(sims):
        treat, block = assign_blocks(len(log_base), block_len, rng)
        y = log_base + true_effect * exposure(treat, carryover_half_life)
        e = estimate(y, treat, block, hour_of_week, burn_in, covariate, fixed_effects)
        est.append(e.effect)
        covered += abs(e.effect - true_effect) <= 1.96 * e.se
        rejected += abs(e.effect) > 1.96 * e.se
    est = np.array(est)
    return {"bias": float(est.mean() - true_effect), "rmse": float(np.sqrt(((est - true_effect) ** 2).mean())),
            "ci95_coverage": covered / sims, "power": rejected / sims}
