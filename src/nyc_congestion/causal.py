"""Causal estimators for a policy that hits a group of units at one date.

Main estimator: imputation difference-in-differences (Borusyak, Jaravel & Spiess 2024), extended with a
treated-group trend and treated-group seasonality, both learned from the pre-policy period only.

  1. Fit  log y_it = a_i + g_t + treated_i * (seasonality_moy(t) + trend * t)
     on every untreated observation (all control rows + treated rows before the policy).
  2. Predict what treated units would have done after the policy without it (the counterfactual).
  3. Effect = actual - counterfactual. Uncertainty: bootstrap over units (zones), so correlation within a zone
     over time is respected.

Also here: a classic two-way fixed-effects event study (to show the raw pre-trend) and synthetic control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _design(df: pd.DataFrame, treated_col: str, trend: bool, seasonality: bool) -> np.ndarray:
    """Dummy-coded design matrix: unit FE, period FE (first dropped), treated x month-of-year, treated x trend."""
    units = pd.get_dummies(df["zone"].astype("category"), dtype=float)
    periods = pd.get_dummies(df["period"].astype("category"), dtype=float).iloc[:, 1:]
    parts = [units.to_numpy(), periods.to_numpy()]
    tr = df[treated_col].to_numpy(dtype=float)[:, None]
    if seasonality:
        moy = pd.get_dummies(df["period"].dt.month.astype(pd.CategoricalDtype(range(1, 13))), dtype=float)
        parts.append(moy.iloc[:, 1:].to_numpy() * tr)
    if trend:
        t = ((df["period"] - df["period"].min()).dt.days / 365.25).to_numpy()[:, None]
        parts.append(t * tr)
    return np.hstack(parts)


@dataclass
class ImputationResult:
    effect_pct_aggregate: float          # change in total treated trips vs counterfactual, %
    effect_pct_mean_zone: float          # average zone-level effect (exp of mean log gap - 1), %
    by_period: pd.DataFrame               # period, actual, counterfactual, effect_pct (treated aggregate)
    ci_aggregate: tuple[float, float] | None = None
    ci_mean_zone: tuple[float, float] | None = None


def imputation_did(df: pd.DataFrame, policy_start: str, treated_group: str = "inside",
                   control_group: str = "outside", trend: bool = True, seasonality: bool = True) -> ImputationResult:
    d = df[df["group"].isin([treated_group, control_group])].copy()
    d["treated"] = (d["group"] == treated_group).astype(int)
    d["post"] = (d["period"] >= pd.Timestamp(policy_start)).astype(int)
    X = _design(d, "treated", trend, seasonality)
    y = d["log_trips"].to_numpy()
    fit_rows = ~((d["treated"] == 1) & (d["post"] == 1)).to_numpy()
    beta, *_ = np.linalg.lstsq(X[fit_rows], y[fit_rows], rcond=None)
    target = ~fit_rows
    d.loc[target, "cf_log"] = X[target] @ beta
    t = d[target]
    gap = t["log_trips"] - t["cf_log"]
    t = t.assign(cf=np.exp(t["cf_log"]))
    by = t.groupby("period").agg(actual=("trips", "sum"), counterfactual=("cf", "sum")).reset_index()
    by["effect_pct"] = (by["actual"] / by["counterfactual"] - 1) * 100
    return ImputationResult(
        effect_pct_aggregate=float((t["trips"].sum() / t["cf"].sum() - 1) * 100),
        effect_pct_mean_zone=float((np.exp(gap.mean()) - 1) * 100),
        by_period=by,
    )


def bootstrap(df: pd.DataFrame, policy_start: str, reps: int = 1000, seed: int = 42, **kw) -> ImputationResult:
    """Point estimate plus 95% intervals from resampling zones (within each group) with replacement."""
    res = imputation_did(df, policy_start, **kw)
    rng = np.random.default_rng(seed)
    groups = {g: np.sort(z["zone"].unique()) for g, z in df.groupby("group")}  # sorted: reproducible draws
    agg, mean = [], []
    for _ in range(reps):
        parts = []
        for g, zones in groups.items():
            draw = rng.choice(zones, size=len(zones), replace=True)
            for k, z in enumerate(draw):  # relabel so duplicated zones get their own fixed effect
                parts.append(df[df["zone"] == z].assign(zone=f"{g}-{k}"))
        b = imputation_did(pd.concat(parts, ignore_index=True), policy_start, **kw)
        agg.append(b.effect_pct_aggregate)
        mean.append(b.effect_pct_mean_zone)
    res.ci_aggregate = tuple(np.percentile(agg, [2.5, 97.5]).round(2))
    res.ci_mean_zone = tuple(np.percentile(mean, [2.5, 97.5]).round(2))
    return res


def event_study_twfe(df: pd.DataFrame, policy_start: str, treated_group: str = "inside",
                     control_group: str = "outside", ref_offset: int = -1) -> pd.DataFrame:
    """Classic event study: log y = a_i + g_t + sum_k b_k * treated * 1[period offset = k].

    No trend, no group seasonality: this shows the raw pattern, including any pre-trend.
    Offsets are in months relative to the policy month; ref_offset is the omitted baseline.
    """
    d = df[df["group"].isin([treated_group, control_group])].copy()
    d["treated"] = (d["group"] == treated_group).astype(int)
    p0 = pd.Timestamp(policy_start).to_period("M")
    d["offset"] = (d["period"].dt.to_period("M") - p0).apply(lambda x: x.n)
    units = pd.get_dummies(d["zone"].astype("category"), dtype=float).to_numpy()
    periods = pd.get_dummies(d["period"].astype("category"), dtype=float).iloc[:, 1:].to_numpy()
    offsets = sorted(o for o in d["offset"].unique() if o != ref_offset)
    ev = np.column_stack([((d["offset"] == o) & (d["treated"] == 1)).to_numpy(float) for o in offsets])
    X = np.hstack([units, periods, ev])
    y = d["log_trips"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    # Cluster-robust (by zone) standard errors for the event coefficients.
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for z in d["zone"].unique():
        idx = (d["zone"] == z).to_numpy()
        s = X[idx].T @ resid[idx]
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv
    k = len(offsets)
    coef, se = beta[-k:], np.sqrt(np.diag(V)[-k:])
    return pd.DataFrame({"offset": offsets, "coef": coef, "se": se,
                         "effect_pct": (np.exp(coef) - 1) * 100})


def synthetic_control(treated: pd.Series, donors: pd.DataFrame, pre_mask: np.ndarray) -> tuple[np.ndarray, pd.Series]:
    """Weights w >= 0, sum(w) = 1 so the weighted donors match the treated series before the policy.

    Series are indexed to their own pre-period mean, so big and small zones are comparable.
    Returns (weights, synthetic series on the treated scale).
    """
    tr = treated / treated[pre_mask].mean()
    dn = donors / donors[pre_mask].mean()
    A, b = dn[pre_mask].to_numpy(), tr[pre_mask].to_numpy()
    n = A.shape[1]
    res = minimize(lambda w: np.sum((A @ w - b) ** 2), np.full(n, 1 / n), method="SLSQP",
                   jac=lambda w: 2 * A.T @ (A @ w - b), bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 500})
    w = np.clip(res.x, 0, None)
    w /= w.sum()
    synth = pd.Series(dn.to_numpy() @ w, index=treated.index) * treated[pre_mask].mean()
    return w, synth


def rmspe_ratio(actual: pd.Series, synth: pd.Series, pre_mask: np.ndarray) -> float:
    """Post-period fit error divided by pre-period fit error. Large = the series broke away after the policy."""
    gap = np.log(actual) - np.log(synth)
    return float(np.sqrt((gap[~pre_mask] ** 2).mean()) / np.sqrt((gap[pre_mask] ** 2).mean()))
