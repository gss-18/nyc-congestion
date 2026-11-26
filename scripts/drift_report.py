"""Monthly drift report for the forecasting features and the target.

Reference = 2024 (before tolling). Each month of 2025-2026 is compared with the same calendar month of 2024,
separately for inside and outside zones. Output: runs/drift/psi.csv and a heatmap.

Usage: uv run python scripts/drift_report.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from nyc_congestion.drift import psi, status  # noqa: E402
from nyc_congestion.forecast import add_features, load_hourly, top_zones  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "drift"
VARIABLES = ["y", "lag_168", "trend_ratio", "temperature_c", "precipitation_mm"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    weather = pd.read_parquet(ROOT / "data" / "raw" / "weather.parquet")
    agg = ROOT / "data" / "agg"
    df = add_features(load_hourly(agg, top_zones(agg)), weather)
    df["ym"] = df["hour"].dt.to_period("M")
    ref = df[df["hour"].dt.year == 2024]
    rows = []
    for ym, cur in df[df["hour"] >= "2025-01-01"].groupby("ym"):
        r = ref[ref["hour"].dt.month == ym.month]
        for grp, flag in [("inside", 1), ("outside", 0)]:
            for v in VARIABLES:
                value = psi(r.loc[r["inside"] == flag, v].to_numpy(float), cur.loc[cur["inside"] == flag, v]
                            .to_numpy(float))
                rows.append({"month": str(ym), "group": grp, "variable": v, "psi": round(value, 4),
                             "status": status(value)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "psi.csv", index=False)
    print(out.groupby(["group", "variable"])["psi"].agg(["mean", "max"]).round(3).to_string())
    print("\nalerts:", out[out["status"] == "alert"][["month", "group", "variable", "psi"]].to_string(index=False))

    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    for ax, grp in zip(axes, ["inside", "outside"], strict=True):
        m = out[out["group"] == grp].pivot(index="variable", columns="month", values="psi").loc[VARIABLES]
        im = ax.imshow(m.to_numpy(), aspect="auto", cmap="Reds", vmin=0, vmax=0.3)
        ax.set_yticks(range(len(VARIABLES)), VARIABLES)
        ax.set_title(f"PSI vs same month of 2024 - {grp} zones (>0.2 = alert)")
        ax.set_xticks(range(m.shape[1]), m.columns, rotation=60, fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(OUT / "psi_heatmap.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
