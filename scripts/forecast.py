"""Backtest day-ahead hourly forecasts for the 80 busiest zones (Uber/Lyft pickups) and train the final model.

  1. Rolling-origin backtest: for each month Apr-Sep 2026, train on everything before it, forecast that month.
     Compare LightGBM with three baselines (yesterday, same hour last week, 7-day same-hour mean).
  2. Prediction intervals: p10 / p90 quantile models on the last month; check that ~80% of actuals fall inside.
  3. Structural break: train only on data before the toll, forecast Jan-Mar 2025. Do forecasts go wrong
     inside the zone when its world changes?
  4. Final models trained on all data -> runs/forecast/model.txt (+ model_p10.txt, model_p90.txt) and
     metadata.json, used by the API. `--final-only` runs just this step.
Every run is logged to MLflow (sqlite:///mlruns.db); browse with: uv run mlflow ui --backend-store-uri sqlite:///mlruns.db

Usage: uv run python scripts/forecast.py [--final-only]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nyc_congestion.forecast import FEATURES, add_features, baselines, load_hourly, top_zones, train, wape  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "data" / "agg"
OUT = ROOT / "runs" / "forecast"
TEST_MONTHS = pd.date_range("2026-04-01", "2026-09-01", freq="MS")


def scores(y: np.ndarray, preds: dict[str, np.ndarray], inside: np.ndarray) -> dict:
    out = {}
    for name, p in preds.items():
        out[name] = {"wape": round(wape(y, p), 4), "wape_inside": round(wape(y[inside], p[inside]), 4),
                     "wape_outside": round(wape(y[~inside], p[~inside]), 4),
                     "bias_pct": round(float((p.sum() / y.sum() - 1) * 100), 2)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlruns.db'}")
    mlflow.set_experiment("day-ahead-zone-demand")
    weather = pd.read_parquet(ROOT / "data" / "raw" / "weather.parquet")
    zones = top_zones(AGG)
    df = add_features(load_hourly(AGG, zones), weather).dropna(subset=["lag_336", "trend_ratio"])
    print(f"[data] {len(zones)} zones, {len(df):,} zone-hours, {df['hour'].min()} to {df['hour'].max()}")
    if args.final_only:
        train_final(df, zones)
        return
    results: dict = {"zones": len(zones), "rows": len(df), "backtest": {}}

    with mlflow.start_run(run_name="backtest"):
        mlflow.log_params({"zones": len(zones), "objective": "poisson", "rounds": 500, "num_leaves": 63,
                           "features": ",".join(FEATURES)})
        all_y, all_p = [], {}
        for month in TEST_MONTHS:
            end = month + pd.offsets.MonthBegin(1)
            tr, te = df[df["hour"] < month], df[(df["hour"] >= month) & (df["hour"] < end)]
            model = train(tr)
            preds = {"lightgbm": model.predict(te[FEATURES]),
                     **{k: v.to_numpy() for k, v in baselines(te).items()}}
            s = scores(te["y"].to_numpy(), preds, te["inside"].to_numpy(bool))
            results["backtest"][str(month.date())] = s
            for name, m in s.items():
                mlflow.log_metric(f"wape_{name}", m["wape"], step=month.month)
            all_y.append(te["y"].to_numpy())
            for k, v in preds.items():
                all_p.setdefault(k, []).append(v)
            print(f"[backtest] {month:%Y-%m}  lightgbm {s['lightgbm']['wape']:.3f}  "
                  f"last-week {s['naive_last_week']['wape']:.3f}  7d-mean {s['mean_same_hour_7d']['wape']:.3f}")
        y = np.concatenate(all_y)
        inside = np.concatenate([df[(df["hour"] >= m) & (df["hour"] < m + pd.offsets.MonthBegin(1))]["inside"]
                                 .to_numpy(bool) for m in TEST_MONTHS])
        results["backtest_overall"] = scores(y, {k: np.concatenate(v) for k, v in all_p.items()}, inside)
        for name, m in results["backtest_overall"].items():
            mlflow.log_metric(f"overall_wape_{name}", m["wape"])

        # Prediction intervals on the last month.
        last = TEST_MONTHS[-1]
        tr, te = df[df["hour"] < last], df[df["hour"] >= last]
        lo = train(tr, objective="quantile", alpha=0.1).predict(te[FEATURES])
        hi = train(tr, objective="quantile", alpha=0.9).predict(te[FEATURES])
        cover = float(((te["y"] >= lo) & (te["y"] <= hi)).mean())
        results["interval_p10_p90"] = {"month": str(last.date()), "coverage": round(cover, 3),
                                       "target": 0.8, "mean_width": round(float((hi - lo).mean()), 1)}
        mlflow.log_metric("interval_coverage_80", cover)
        print(f"[interval] p10-p90 coverage {cover:.3f} (target 0.80)")

    with mlflow.start_run(run_name="structural-break-2025"):
        tr = df[df["hour"] < "2025-01-01"]
        te = df[(df["hour"] >= "2025-01-05") & (df["hour"] < "2025-04-01")]
        model = train(tr)
        preds = {"lightgbm": model.predict(te[FEATURES]), **{k: v.to_numpy() for k, v in baselines(te).items()}}
        results["structural_break_jan_mar_2025"] = scores(te["y"].to_numpy(), preds, te["inside"].to_numpy(bool))
        mlflow.log_metric("wape_lightgbm", results["structural_break_jan_mar_2025"]["lightgbm"]["wape"])
        print("[break]", results["structural_break_jan_mar_2025"]["lightgbm"])

    results["top_features"] = train_final(df, zones)

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["backtest_overall"], indent=1))


def train_final(df: pd.DataFrame, zones: list[int]) -> dict:
    """Train p50 (Poisson) and p10/p90 (quantile) models on all data, save them for the API."""
    with mlflow.start_run(run_name="final-model"):
        model = train(df)
        model.save_model(str(OUT / "model.txt"))
        for name, alpha in [("p10", 0.1), ("p90", 0.9)]:
            train(df, objective="quantile", alpha=alpha).save_model(str(OUT / f"model_{name}.txt"))
        meta = {"zones": zones, "features": FEATURES, "trained_through": str(df["hour"].max()),
                "service": "fhvhv", "target": "pickups per zone-hour", "min_lag_hours": 24}
        (OUT / "metadata.json").write_text(json.dumps(meta, indent=2))
        for f in ["model.txt", "model_p10.txt", "model_p90.txt", "metadata.json"]:
            mlflow.log_artifact(str(OUT / f))
        imp = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(7, 5))
        (imp / imp.sum() * 100).plot.barh(ax=ax)
        ax.set(xlabel="share of total gain, %", title="Feature importance (final model)")
        fig.tight_layout()
        fig.savefig(OUT / "feature_importance.png", dpi=130)
        mlflow.log_artifact(str(OUT / "feature_importance.png"))
        return (imp / imp.sum() * 100).round(1).sort_values(ascending=False).head(6).to_dict()


if __name__ == "__main__":
    main()
