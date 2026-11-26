# Model card: day-ahead zone demand forecast

## What it does

Forecasts Uber/Lyft pickups for each hour of the next day, for the 80 busiest NYC taxi zones, with p10/p50/p90
bands. Served by `nyc_congestion.api` (`GET /forecast?zone=161&date=YYYY-MM-DD`).

## Intended use

- Planning-level demand estimates: staffing, driver positioning, capacity checks, a baseline for experiments.
- **Not for:** individual-trip pricing, zones outside the 80 covered, horizons longer than one day, or any decision
  about individual people. The data has no rider or driver identities.

## Training data

- NYC TLC high-volume for-hire trip records (Uber, Lyft, Via, Juno), Jan 2023 - Sep 2026, aggregated to
  zone x hour pickups (2.57M zone-hours). Public data: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Hourly weather for Midtown Manhattan from the Open-Meteo historical archive.
- Cleaning: pickups outside the file's month, 0 or > 100 miles, or durations <= 0 or > 4 h are dropped (~2-3%).

## Model

LightGBM, Poisson objective (p50) and quantile objective (p10, p90), 500 rounds, 63 leaves.
Features use only data at least 24 hours old (enforced by tests): lags 24/48/168/336 h, rolling means, calendar,
NY holidays, inside-congestion-zone and post-policy flags, weather.

## Performance (rolling-origin backtest, Apr-Sep 2026)

| Metric | Value |
|---|---|
| WAPE, all zones | 11.7% (best baseline: 15.5%, same hour last week) |
| WAPE, congestion-zone / other zones | 12.4% / 11.4% |
| Bias (total forecast vs total actual) | -0.2% |
| p10-p90 interval coverage (Sep 2026) | 78.8% (target 80%) |
| WAPE after the Jan 2025 policy change, model trained only on pre-policy data | 11.7% (bias +1.3%) |

## Limitations and risks

- **Weather on future days:** the API has no weather forecast feed; for a day with no recorded weather it uses that
  month's average weather per hour (flagged in the response as `weather_source`). Weather contributes little to
  day-ahead accuracy, but extreme events (snowstorms) will be under-forecast.
- **Large shocks:** lag features adapt within about a week. Sudden events (storms, transit strikes, big events) are
  not in the features and will be missed on the first day.
- **Coverage:** only the 80 busiest zones; quiet zones have noisy hourly counts and are excluded.
- **Fairness:** zone-level demand reflects where service is used today. Using forecasts to allocate supply can
  reinforce existing gaps between neighbourhoods; check outcomes by borough before such use.
- **Data drift:** monitored monthly with PSI against the same month of 2024 (`scripts/drift_report.py`). Demand
  features have stayed stable (PSI < 0.05); temperature and the narrow `trend_ratio` feature alert often and need
  their own thresholds.

## Maintenance

Retrain monthly when the new TLC file is published (`scripts/download_data.py`, `aggregate.py`, `forecast.py`).
Every run is tracked in MLflow. Retrain immediately if WAPE on the latest month exceeds 15% (the baseline level)
or demand-feature PSI exceeds 0.2.
