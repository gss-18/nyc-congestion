# Learn nyc-congestion: build it yourself

Causal effect of NYC congestion pricing (tolling started 2025-01-05) on taxi and Uber/Lyft trips,
plus zone-level demand forecasting. One section per step: what, why, exact commands.

---

## Step 1: Data

**Source:** NYC Taxi & Limousine Commission (TLC) trip records,
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

- One parquet file per month per service. We use **yellow taxi** and **fhvhv** (high-volume for-hire = Uber/Lyft),
  Jan 2023 - Sep 2026: 90 files, 24.9 GB, ~1.06 billion trips.
- File pattern: `https://d37ci6vzurychx.cloudfront.net/trip-data/{yellow|fhvhv}_tripdata_YYYY-MM.parquet`
- Trips are located by **taxi zone** (263 zones), not GPS. Names: `taxi_zone_lookup.csv`.

```bash
uv run python scripts/download_data.py      # ~10 min at ~40 MB/s; skips complete files
```

Download design: stream to a `.part` file and rename only when the size matches `content-length`, so an
interrupted download never looks complete. Retries with backoff. Generic user-agent.

## Step 2: Which zones are tolled? Let the data say

The 2025 files have a `cbd_congestion_fee` column. For trips that start and end in the **same** zone,
the share that paid the fee shows whether that zone is inside the Congestion Relief Zone:

- **37 zones** at 95-99% -> `INSIDE` (Manhattan south of 60th St)
- **6 zones** at 14-49% -> `BORDER` (cut by or touching 60th St) -> **excluded** from both groups
- everything else ~0-3% -> `outside`

```bash
uv run python scripts/derive_zones.py
```

Why derive it? A hand-drawn boundary is easy to get wrong at the edges; the fee column is ground truth.
Why exclude border zones? Half-treated units blur the comparison between treated and control.

## Step 3: Aggregate 1 billion rows with DuckDB

DuckDB runs SQL directly on parquet files, reads only the columns it needs, and streams, so it fits in
16 GB RAM (`memory_limit = '6GB'`, 4 threads). ~4 minutes for everything.

- Both services are mapped to one schema (pickup time, zones, miles, fare, fee).
- Cleaning: pickup must fall in the file's own month (raw files contain stray timestamps),
  0 < miles < 100, 0 < duration < 4 h. Keeps ~97-98% of rows in every year.
- Outputs in `data/agg/`:
  - `hourly_zone.parquet`: hour x service x pickup/dropoff x zone -> trips, miles, fare, fee (26.6M rows)
  - `hourly_flow.parquet`: hour x service x flow (inside->inside, inside->outside, outside->inside,
    outside->outside) -> trips (0.3M rows)

```bash
uv run python scripts/aggregate.py
```

## Step 4: First look (EDA)

Change in trips, Jan 6 - Sep 30 of each year:

| Uber/Lyft flow | 2024 vs 2023 (before) | 2025 vs 2024 | 2026 vs 2024 |
|---|---|---|---|
| inside->inside | -3.9% | -7.1% | -8.0% |
| inside->outside | +1.4% | -3.1% | -5.2% |
| outside->inside | +0.1% | -3.6% | -6.2% |
| outside->outside | +5.8% | +3.6% | +13.4% |

**Findings:**

1. **A naive comparison overstates the effect.** In 2025, inside->inside trips fell 7.1% while outside trips grew
   3.6%: a gap of ~11 points. But the gap was **already ~10 points in 2024, before the toll**. The trips inside
   Manhattan were already growing slower. This is a **pre-trend**, and the causal method must account for it.
2. **Yellow taxis grew ~20-40% everywhere in 2025**, including trips that never touch the zone. The vendor mix and
   cleaning filters do not explain it, so it is a market-wide shift, not a toll effect. Outside-zone yellow trips are
   therefore not a clean control; the main analysis uses Uber/Lyft (~85% of trips).
3. Fees recorded in the data: Uber/Lyft $124M and yellow $25M in 2025.

### Check yourself

1. What is a pre-trend and why does it matter? *Treated and control groups already diverging before the policy;
   ignoring it attributes the old trend to the policy.*
2. Why can't yellow outside-zone trips serve as a control here? *They changed a lot for reasons unrelated to the toll.*
3. Why aggregate before analysis? *1 billion rows -> 0.3M; every later step becomes seconds, not minutes.*

---

## Step 5: Causal analysis

**Question:** how much did congestion pricing change Uber/Lyft trips in the zone?
**Data check first:** the Uber/Lyft fee is a flat **$1.50 at all hours** (fee per trip by hour, from the data);
the car toll is what varies by time of day.

### Setup (`panel.py`)

- Unit = taxi zone, period = month. Outcome = log(pickups).
- **Treated:** 37 inside zones. **Control:** 207 outside zones with >= 1,000 Uber/Lyft pickups a month in 2024.
  **Dropped:** airports (driven by flights), unknown zones, border zones (except in the spillover test).
- January 2025 counts as treated (tolling began Jan 5: 27 of 31 days).

### Methods (`causal.py`)

1. **Raw event study (two-way fixed effects).** `log y = zone effect + month effect + b_k x treated x [k months
   from policy]`, with cluster-robust (by zone) errors. It only *shows* the pattern; it does not adjust for trends.
2. **Imputation DiD** (Borusyak, Jaravel & Spiess 2024), the main estimator:
   - fit zone + month effects **plus a treated-group linear trend and treated-group seasonality**, using only
     untreated rows (all controls + treated before 2025);
   - predict the treated zones' counterfactual after 2025; effect = actual vs counterfactual.
   - **95% CI by bootstrapping zones** (resample zones, refit 200 times), which respects correlation over time.
   - Tested on fake data: recovers a known -9.5% despite a pre-trend; ignoring the trend gives < -13%.
3. **Synthetic control** (weekly): non-negative donor weights summing to 1 that match total inside pickups before
   2025. **Placebo in space:** pretend each of 40 big donors was treated; p-value = how often a placebo breaks away
   as much as the real zone (post/pre RMSPE ratio).
4. **Placebo in time:** fake policy in July 2024, using only 2023-2024 data. A valid method should find ~0.

```bash
uv run python scripts/causal_analysis.py   # ~3 min -> runs/causal/results.json, event_study.png, synthetic_control.png
```

### Results (Uber/Lyft pickups inside the zone)

| Method | Effect | 95% CI |
|---|---|---|
| Naive (no trend, no seasonality) | **-15.9%** | - |
| Seasonality only | -16.0% | - |
| **Trend + seasonality (main)** | **-1.4%** | -2.7% to -0.1% |
| Placebo, fake policy Jul 2024 | -0.2% | -1.2% to +1.0% |
| Synthetic control (weekly) | -5.8% | placebo p = 0.39 (not significant) |
| Post window 2025 only | -0.7% | - |

Other outcomes (trend + seasonality): dropoffs -1.6% (CI -3.4 to +0.5); peak hours -1.0% (CI -2.5 to +0.4);
off-peak -2.5% (CI -3.9 to -0.5); border zones (spillover) -1.4% (CI -4.2 to +1.9). Yellow taxi: unstable
(aggregate -6.3%, mean zone +12.7%), not reliable because of the market-wide 2025 taxi surge.

### What it means

1. **The naive answer (-16%) is almost all pre-existing trend.** The raw event study falls ~8% a year relative to
   outside zones from 2023 onward, with **no visible break at January 2025**.
2. **After adjusting, the effect on Uber/Lyft trips is small: about -1.4%,** borderline significant, and not robust
   (synthetic control is not significant; the 2025-only window gives -0.7%).
3. **The method passes its placebo** (-0.2%, CI spans 0), so the adjustment is not just erasing everything.
4. Honest caveat: a linear trend extrapolated 21 months is an assumption. Month-by-month effects swing between
   about -9% and +3%, so treat the point estimate as "small", not as a precise number.
5. Plausible reading: a $1.50 fee on a ~$30 average ride is ~5%; small price changes move rideshare demand little.

### Check yourself

1. Why does the naive estimate say -16%? *It counts a decline that started two years earlier.*
2. What does a placebo test protect against? *A method that "finds" effects where none can exist.*
3. Why bootstrap zones instead of rows? *Rows from one zone are correlated over time; resampling rows overstates
   precision.*
4. Why is synthetic control not significant even though it shows -5.8%? *Many untreated zones break away from
   their synthetic match just as much (p = 0.39).*

---

## Step 6: Day-ahead forecasting (`forecast.py`, `scripts/forecast.py`)

**Task:** every evening, forecast each hour of tomorrow's Uber/Lyft pickups for the 80 busiest zones
(2.57M zone-hours).

**The one rule:** every feature for hour t uses only trips up to **t - 24 h**. Tests check this (`test_forecast.py`).
Break it and the backtest looks great but the model cannot run in real life (leakage).

Features: lags 24/48/168/336 h, same-hour mean of the last 7 days, 24 h mean, a short-vs-long trend ratio,
hour/day/month, NY holidays, inside-zone and post-policy flags, hourly weather (Open-Meteo archive).
Model: LightGBM with a **Poisson** objective (counts), 500 rounds.

**Backtest = rolling origin:** for each month Apr-Sep 2026, train on everything before it, predict that month.
This mimics real use; a random train/test split would let the model peek at neighbouring hours.

**Metric: WAPE** = total absolute error / total trips. Unlike MAPE, it does not blow up on hours with ~0 trips.

| Model | WAPE (Apr-Sep 2026) | Inside zone | Outside |
|---|---|---|---|
| **LightGBM** | **11.7%** | 12.4% | 11.4% |
| Same hour last week | 15.5% | 16.7% | 14.9% |
| 7-day same-hour mean | 22.9% | 24.7% | 21.9% |
| Same hour yesterday | 23.6% | 24.9% | 22.8% |

- **25% less error than the best baseline**, every month (10.5-13.0% vs 13.6-17.5%).
- The 7-day mean is poor because it mixes weekdays and weekends; "same hour last week" respects the weekly cycle.
- **Feature importance:** lag 168 h (same hour last week) 69%, lag 336 h 22%. Weather barely matters day-ahead
  once recent demand is known.
- **Prediction intervals** (p10/p90 quantile models): 78.8% of actuals inside, target 80%: well calibrated.
- **Structural break:** trained only on pre-toll data, forecasting Jan-Mar 2025 still gives 11.7% WAPE (bias +1.3%).
  Day-ahead lags adapt within a week, so a small level shift barely hurts. A month-ahead model would suffer more.

Experiments are logged to **MLflow** (`sqlite:///mlruns.db`): `uv run mlflow ui --backend-store-uri sqlite:///mlruns.db`.

---

## Step 7: Switchback experiment design

**Problem:** to test a price change in a marketplace you cannot randomize riders: treated and control riders
share the same drivers, so one group's treatment changes the other's experience (**interference**).
**Switchback:** randomize **time blocks** instead. The whole area is treated or control for L hours, then a coin
flip decides the next block.

**Study** (`switchback.py`, `scripts/switchback_study.py`): real hourly Uber/Lyft pickups inside the zone,
12 weeks (Apr-Jun 2026, ~6,900 pickups/hour). Inject a known **-2%** effect that kicks in gradually
(**carryover**, half-life 1 hour). 500 random assignments per design. Measure bias, RMSE, 95% CI coverage, power.

Estimators: difference in means; + hour-of-week fixed effects; + CUPED (covariate = same hour last week);
+ 1-hour burn-in (drop the first hour of each block). Standard errors are **clustered by block** (hours in one
block share one coin flip).

| Block | Estimator | Bias (pts) | RMSE (pts) | CI coverage | Power |
|---|---|---|---|---|---|
| 1h | + FE | +1.01 | 1.16 | 0.51 | 0.49 |
| 2h | + FE + CUPED + burn-in | +0.53 | 0.93 | 0.87 | 0.50 |
| **4h** | **+ FE + CUPED + burn-in** | **+0.29** | **1.08** | **0.91** | **0.43** |
| 8h | + FE + CUPED + burn-in | +0.10 | 1.24 | 0.93 | 0.39 |
| 24h | + FE | +0.10 | 1.43 | 0.94 | 0.28 |
| any | difference in means | small | 2.8-6.1 | ~0.95 | ~0.06 |

**Lessons:**

1. **Short blocks are biased toward zero** by carryover (1h blocks recover only half the effect) and their CIs
   are wrong (51% coverage instead of 95%).
2. **Hour-of-week fixed effects are the big win**: RMSE drops 3-5x vs a plain difference in means, because demand
   swings by time of day far more than the effect.
3. **A short burn-in removes most carryover bias** for little cost. Best valid design here: **4-hour blocks +
   FE + burn-in**.
4. **CUPED added nothing here**: hour-of-week FE already capture what "same hour last week" knows. CUPED helps when
   the covariate carries information the design does not already use.
5. **Power is only ~43% for a 2% effect in 12 weeks.** Power grows with the square root of duration, so 80% power
   needs roughly 30 weeks, or a bigger effect. Run this simulation before launching, not after.

---

## Step 8: Drift monitoring (`drift.py`, `scripts/drift_report.py`)

**PSI (Population Stability Index):** bin the reference distribution into deciles, then
`PSI = sum (now% - ref%) * ln(now% / ref%)`. Rule of thumb: < 0.1 stable, 0.1-0.2 watch, > 0.2 alert.

Each month of 2025-2026 is compared with the **same month of 2024**, so normal seasonality is not called drift.

**Findings:**

- **Demand (`y`, `lag_168`) barely drifted:** PSI < 0.05 every month, consistent with the small causal effect.
- **Temperature alerts** come from year-to-year weather. Feb 2026 was far colder than Feb 2024 (PSI 1.53), the same
  month the causal analysis shows a -9% dip: worth noting, not proof.
- **`trend_ratio` alerts often**, but it sits tightly around 1.0, so tiny shifts move many points across decile
  edges. **Lesson: PSI thresholds are not universal**; narrow features need their own thresholds or a different test.

---

## Step 9: Serving (`api.py`, `Dockerfile`)

FastAPI service:

- `GET /forecast?zone=161&date=2026-10-01` -> 24 hourly forecasts with p10 / p50 / p90.
- **Day-ahead only:** refuses dates more than one day past the data, because features need trips >= 24 h old.
- If the day has no recorded weather, it uses that month's average weather per hour and says so in the response
  (`weather_source`). A production version would call a weather forecast API instead.

```bash
uv run uvicorn nyc_congestion.api:app --reload      # http://127.0.0.1:8000/docs
docker build -t nyc-forecast . && docker run -p 8000:8000 -v "$PWD/runs:/app/runs" -v "$PWD/data:/app/data" nyc-forecast
```

CI (`.github/workflows/ci.yml`): every push runs `ruff` and `pytest` (tests use small fake data, no downloads).
