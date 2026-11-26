"""Causal effect of congestion pricing on Uber/Lyft and taxi trips.

Analyses (all monthly zone panels unless noted):
  main         Uber/Lyft pickups, inside vs outside zones, imputation DiD with trend + seasonality, bootstrap CI
  sensitivity  same with no adjustment / seasonality only, to show how much the pre-trend matters
  placebo      fake policy in July 2024 using only 2023-2024 data: should find ~0
  outcomes     dropoffs; peak (weekday 5am-9pm) vs off-peak pickups; border-zone spillover; yellow taxi
  event study  classic TWFE by month (raw pattern incl. pre-trend)
  synthetic    weekly synthetic control for total inside pickups, with placebo-in-space p-value

Usage: uv run python scripts/causal_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nyc_congestion.causal import (  # noqa: E402
    bootstrap,
    event_study_twfe,
    imputation_did,
    rmspe_ratio,
    synthetic_control,
)
from nyc_congestion.panel import label_units, load_panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "data" / "agg"
OUT = ROOT / "runs" / "causal"
POLICY = "2025-01-01"  # tolling began Sunday 2025-01-05; January is counted as treated (27 of 31 days)
PEAK = "dayofweek(hour) between 1 and 5 and hour(hour) between 5 and 20"


def summarize(res) -> dict:
    out = {"effect_pct_aggregate": round(res.effect_pct_aggregate, 2),
           "effect_pct_mean_zone": round(res.effect_pct_mean_zone, 2)}
    if res.ci_aggregate:
        out["ci95_aggregate"] = [float(x) for x in res.ci_aggregate]
        out["ci95_mean_zone"] = [float(x) for x in res.ci_mean_zone]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    fhv = label_units(load_panel(AGG, "fhvhv"))
    counts = fhv.groupby("group")["zone"].nunique().to_dict()
    results["units"] = counts
    print(f"zones: {counts}")

    main_res = bootstrap(fhv, POLICY, reps=200)
    results["main_fhvhv_pickups"] = summarize(main_res)
    print("main:", results["main_fhvhv_pickups"])

    results["sensitivity"] = {
        "no_adjustment": summarize(imputation_did(fhv, POLICY, trend=False, seasonality=False)),
        "seasonality_only": summarize(imputation_did(fhv, POLICY, trend=False, seasonality=True)),
        "trend_and_seasonality": results["main_fhvhv_pickups"],
    }
    print("sensitivity:", results["sensitivity"])

    pre_only = fhv[fhv["period"] < "2025-01-01"]
    results["placebo_jul_2024"] = summarize(bootstrap(pre_only, "2024-07-01", reps=200))
    print("placebo:", results["placebo_jul_2024"])

    outcomes = {
        "fhvhv_dropoffs": label_units(load_panel(AGG, "fhvhv", "dropoff")),
        "fhvhv_pickups_peak": label_units(load_panel(AGG, "fhvhv", hours=PEAK)),
        "fhvhv_pickups_offpeak": label_units(load_panel(AGG, "fhvhv", hours=f"not ({PEAK})")),
        "yellow_pickups": label_units(load_panel(AGG, "yellow"), min_monthly_trips=300),
    }
    results["outcomes"] = {}
    for name, panel in outcomes.items():
        results["outcomes"][name] = summarize(bootstrap(panel, POLICY, reps=100))
        print(name, results["outcomes"][name])
    results["outcomes"]["fhvhv_border_spillover"] = summarize(
        bootstrap(fhv, POLICY, reps=100, treated_group="border"))
    print("border spillover:", results["outcomes"]["fhvhv_border_spillover"])

    es = event_study_twfe(fhv, POLICY)
    pre = es[es["offset"] < 0]
    slope = np.polyfit(pre["offset"], pre["coef"], 1)[0] * 12 * 100
    results["event_study_pre_trend_pct_per_year"] = round(float(slope), 2)
    print(f"raw pre-trend: {slope:.2f}% per year (inside vs outside)")

    results["synthetic_control"] = synth(load_panel(AGG, "fhvhv", freq="week"))
    print("synthetic control:", results["synthetic_control"])

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    plot_event_and_gap(es, main_res)


def synth(weekly: pd.DataFrame) -> dict:
    """Weekly synthetic control for total inside pickups; donors = sizeable outside zones."""
    lab = label_units(weekly, min_monthly_trips=1000)
    lab = lab[(lab["period"] >= "2023-01-02") & (lab["period"] <= "2026-09-21")]
    lab = lab[lab["period"] != "2024-12-30"]  # week containing Jan 5 is partly treated
    wide = lab.pivot_table(index="period", columns="zone", values="trips")
    inside = lab.loc[lab["group"] == "inside", "zone"].unique()
    outside = lab.loc[lab["group"] == "outside", "zone"].unique()
    treated = wide[inside].sum(axis=1)
    donors = wide[outside]
    pre = (wide.index < "2025-01-01")
    w, s = synthetic_control(treated, donors, pre)
    effect = (treated[~pre].sum() / s[~pre].sum() - 1) * 100
    ratio = rmspe_ratio(treated, s, pre)
    # Placebo in space: pretend each of the 40 largest donors was treated; how unusual is the real ratio?
    big = donors[pre].mean().sort_values(ascending=False).index[:40]
    placebo = []
    for z in big:
        _, ps = synthetic_control(donors[z], donors.drop(columns=z), pre)
        placebo.append(rmspe_ratio(donors[z], ps, pre))
    p = (1 + sum(r >= ratio for r in placebo)) / (1 + len(placebo))
    top = pd.Series(w, index=outside).sort_values(ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(treated.index, treated / 1e6, label="actual (inside zone)", lw=1.3)
    ax.plot(s.index, s / 1e6, label="synthetic control", lw=1.3)
    ax.axvline(pd.Timestamp("2025-01-05"), color="k", ls="--", lw=1)
    ax.set(ylabel="weekly Uber/Lyft pickups (millions)", title=f"Synthetic control: effect {effect:.1f}%, "
           f"placebo p = {p:.3f}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "synthetic_control.png", dpi=130)
    return {"effect_pct": round(float(effect), 2), "rmspe_ratio": round(ratio, 2), "placebo_p_value": round(p, 3),
            "placebo_units": len(placebo), "top_donor_weights": {int(k): round(float(v), 3) for k, v in top.items()}}


def plot_event_and_gap(es: pd.DataFrame, res) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.errorbar(es["offset"], es["effect_pct"], yerr=1.96 * es["se"] * 100, fmt="o", ms=3, capsize=2)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(-0.5, color="k", ls="--", lw=1)
    ax.set(title="Raw event study (no trend adjustment)", xlabel="months since tolling began",
           ylabel="inside vs outside, % (Dec 2024 = 0)")
    ax.grid(alpha=0.3)
    ax = axes[1]
    by = res.by_period
    ax.bar(by["period"], by["effect_pct"], width=20)
    ax.axhline(res.effect_pct_aggregate, color="r", lw=1.2, label=f"average {res.effect_pct_aggregate:.1f}% "
               f"(95% CI {res.ci_aggregate[0]} to {res.ci_aggregate[1]})")
    ax.set(title="Effect by month vs trend- and season-adjusted counterfactual", ylabel="% change in inside pickups")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "event_study.png", dpi=130)


if __name__ == "__main__":
    main()
