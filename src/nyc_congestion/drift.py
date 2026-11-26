"""Data drift monitoring with the Population Stability Index (PSI).

PSI compares the distribution of a variable now against a reference period:
    PSI = sum over bins of (now% - ref%) * ln(now% / ref%)
Bins are deciles of the reference. Common rule of thumb: < 0.1 stable, 0.1-0.2 watch, > 0.2 investigate.

Demand is seasonal, so each month is compared with the SAME month of the reference year. Otherwise every summer
would "drift" from every winter.
"""

from __future__ import annotations

import numpy as np

WATCH, ALERT = 0.1, 0.2


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (nearly) constant reference: nothing meaningful to compare
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference) + eps
    cur = np.histogram(current, edges)[0] / len(current) + eps
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def status(value: float) -> str:
    return "alert" if value > ALERT else "watch" if value > WATCH else "ok"
