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
