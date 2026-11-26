"""Design a switchback pricing experiment on real demand: which block length and estimator should we use?

Base series: hourly Uber/Lyft pickups inside the congestion zone, 12 weeks (2026-04-06 to 2026-06-28).
We inject a true effect of -2% (a small price test) that takes effect gradually (carryover half-life 1 hour),
then compare designs over 500 random assignments each.

Usage: uv run python scripts/switchback_study.py
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nyc_congestion.switchback import simulate  # noqa: E402
from nyc_congestion.zones import INSIDE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "switchback"
TRUE_EFFECT, HALF_LIFE, SIMS = -0.02, 1.0, 500
START, END = pd.Timestamp("2026-04-06"), pd.Timestamp("2026-06-29")  # Monday to Monday, 12 weeks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    q = f"""select hour, sum(trips) y from '{ROOT}/data/agg/hourly_zone.parquet'
            where service='fhvhv' and direction='pickup' and zone in ({",".join(map(str, sorted(INSIDE)))})
              and hour >= '{START - pd.Timedelta(days=7)}' and hour < '{END}'
            group by hour order by hour"""
    s = duckdb.connect().execute(q).df().set_index("hour")["y"]
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h")).interpolate()  # DST gap hour
    log_y = np.log(s)
    cur = log_y[START:END - pd.Timedelta(hours=1)]
    lag_week = log_y.shift(168)[START:END - pd.Timedelta(hours=1)].to_numpy()  # CUPED covariate
    how = (cur.index.day_of_week * 24 + cur.index.hour).to_numpy()
    base = cur.to_numpy()
    print(f"[data] {len(base)} hours, mean pickups/hour {np.exp(base).mean():,.0f}")

    designs = []
    for L in [1, 2, 4, 8, 24]:
        designs += [
            (L, "difference in means", dict(fixed_effects=False)),
            (L, "+ hour-of-week FE", dict()),
            (L, "+ FE + CUPED", dict(covariate=lag_week)),
        ]
        if L >= 2:
            designs.append((L, "+ FE + CUPED + 1h burn-in", dict(covariate=lag_week, burn_in=1)))
    rows = []
    for L, name, kw in designs:
        r = simulate(base, how, TRUE_EFFECT, L, HALF_LIFE, sims=SIMS, **kw)
        rows.append({"block_hours": L, "estimator": name, **{k: round(v, 4) for k, v in r.items()}})
        print(f"  L={L:2d}h  {name:28s} bias {r['bias']:+.4f}  rmse {r['rmse']:.4f}  "
              f"coverage {r['ci95_coverage']:.2f}  power {r['power']:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "designs.csv", index=False)
    best = df[df["ci95_coverage"] >= 0.9].sort_values("rmse").iloc[0].to_dict()
    (OUT / "results.json").write_text(json.dumps({"true_effect": TRUE_EFFECT, "carryover_half_life_h": HALF_LIFE,
                                                  "sims": SIMS, "hours": len(base), "best_valid_design": best,
                                                  "designs": rows}, indent=2))
    print("best design with >= 90% CI coverage:", best)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, g in df.groupby("estimator", sort=False):
        axes[0].plot(g["block_hours"], g["bias"] * 100, "o-", label=name)
        axes[1].plot(g["block_hours"], g["rmse"] * 100, "o-", label=name)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set(xscale="log", xlabel="block length (hours)", ylabel="bias (percentage points)",
                title=f"Bias (true effect {TRUE_EFFECT:.0%}, carryover half-life {HALF_LIFE:g}h)")
    axes[1].set(xscale="log", xlabel="block length (hours)", ylabel="RMSE (percentage points)",
                title="Error of the estimate")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xticks([1, 2, 4, 8, 24], ["1", "2", "4", "8", "24"])
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "switchback_designs.png", dpi=130)


if __name__ == "__main__":
    main()
