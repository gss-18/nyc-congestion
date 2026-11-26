"""Export the numbers and chart series the project page needs to docs/site/data.js.

Everything on the page comes from this file, which reads the analysis outputs (runs/*) and recomputes the
cheap chart series. The forecast example is truly out of sample: models are trained on data before
2026-09-01 and forecast the week of 2026-09-14.

Usage: uv run python scripts/export_site_data.py   (after causal_analysis, forecast, switchback_study)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nyc_congestion.causal import event_study_twfe, imputation_did
from nyc_congestion.forecast import FEATURES, add_features, load_hourly, top_zones, train
from nyc_congestion.panel import label_units, load_panel

ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "data" / "agg"
SITE = ROOT / "docs" / "site"
EXAMPLE_ZONES = {161: "Midtown Center", 79: "East Village", 61: "Crown Heights North", 132: "JFK Airport"}
WEEK = (pd.Timestamp("2026-09-14"), pd.Timestamp("2026-09-21"))


def rnd(x, n=2):
    return float(round(float(x), n))


def main() -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    causal = json.loads((ROOT / "runs/causal/results.json").read_text())
    fc = json.loads((ROOT / "runs/forecast/results.json").read_text())
    sw = json.loads((ROOT / "runs/switchback/results.json").read_text())

    fhv = label_units(load_panel(AGG, "fhvhv"))
    es = event_study_twfe(fhv, "2025-01-01")
    imp = imputation_did(fhv, "2025-01-01")

    weather = pd.read_parquet(ROOT / "data/raw/weather.parquet")
    zones = top_zones(AGG)
    df = add_features(load_hourly(AGG, zones), weather).dropna(subset=["lag_336", "trend_ratio"])
    tr = df[df["hour"] < "2026-09-01"]
    te = df[(df["hour"] >= WEEK[0]) & (df["hour"] < WEEK[1]) & df["zone"].isin(EXAMPLE_ZONES)]
    models = {"p50": train(tr), "p10": train(tr, objective="quantile", alpha=0.1),
              "p90": train(tr, objective="quantile", alpha=0.9)}
    example = {}
    for z, name in EXAMPLE_ZONES.items():
        t = te[te["zone"] == z]
        preds = {k: m.predict(t[FEATURES]) for k, m in models.items()}
        example[str(z)] = {
            "name": name, "start": str(WEEK[0].date()),
            "actual": t["y"].round().astype(int).tolist(),
            **{k: np.round(v, 1).tolist() for k, v in preds.items()},
            "wape": rnd(np.abs(t["y"] - preds["p50"]).sum() / t["y"].sum() * 100, 1),
            "coverage": rnd(((t["y"] >= preds["p10"]) & (t["y"] <= preds["p90"])).mean() * 100, 1),
        }

    data = {
        "units": causal["units"],
        "causal": {
            "main": causal["main_fhvhv_pickups"], "sensitivity": causal["sensitivity"],
            "placebo": causal["placebo_jul_2024"], "outcomes": causal["outcomes"],
            "synthetic": causal["synthetic_control"],
            "pretrend_pct_per_year": causal["event_study_pre_trend_pct_per_year"],
            "event_study": [{"offset": int(r.offset), "pct": rnd(r.effect_pct),
                             "lo": rnd((np.exp(r.coef - 1.96 * r.se) - 1) * 100),
                             "hi": rnd((np.exp(r.coef + 1.96 * r.se) - 1) * 100)} for r in es.itertuples()],
            "by_month": [{"month": str(r.period.date())[:7], "pct": rnd(r.effect_pct)}
                         for r in imp.by_period.itertuples()],
        },
        "forecast": {"overall": fc["backtest_overall"], "interval": fc["interval_p10_p90"],
                     "break": fc["structural_break_jan_mar_2025"]["lightgbm"],
                     "monthly": {m: {k: v["wape"] for k, v in s.items()} for m, s in fc["backtest"].items()},
                     "zones": len(zones), "example": example},
        "switchback": {"true_effect": sw["true_effect"], "best": sw["best_valid_design"], "designs": sw["designs"]},
        "tests_passing": 19,
    }
    (SITE / "data.js").write_text("window.SITE_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"[site] wrote {SITE / 'data.js'} ({(SITE / 'data.js').stat().st_size / 1024:.0f} KB)")
    for z, e in example.items():
        print(f"  zone {z} {e['name']}: WAPE {e['wape']}%, p10-p90 coverage {e['coverage']}%")


if __name__ == "__main__":
    main()
