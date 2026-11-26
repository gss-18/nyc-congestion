"""Aggregate raw trip records (~1 billion rows) into small hourly tables with DuckDB.

Outputs (data/agg/):
  hourly_zone.parquet   hour, service, direction (pickup/dropoff), zone, trips, miles, fare, cbd_fee
  hourly_flow.parquet   hour, service, flow (inside->inside, inside->outside, outside->inside, outside->outside,
                        other), trips

Each month is filtered to pickups inside that month (raw files contain a few stray timestamps) and to
plausible trips (0 < miles < 100, 0 < duration < 4 h).

Usage: uv run python scripts/aggregate.py
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb

from nyc_congestion.zones import BORDER, INSIDE

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
AGG = ROOT / "data" / "agg"

# Column names differ between the two services; map them to one schema.
SERVICES = {
    "yellow": {"pickup": "tpep_pickup_datetime", "dropoff": "tpep_dropoff_datetime", "miles": "trip_distance",
               "fare": "fare_amount"},
    "fhvhv": {"pickup": "pickup_datetime", "dropoff": "dropoff_datetime", "miles": "trip_miles",
              "fare": "base_passenger_fare"},
}


def trips_sql(service: str) -> str:
    c = SERVICES[service]
    files = f"'{RAW}/{service}_tripdata_*.parquet'"
    # filename=true lets us keep only pickups inside each file's own month.
    return f"""
    select '{service}' as service,
           date_trunc('hour', {c['pickup']}) as hour,
           PULocationID as pu, DOLocationID as dz,
           {c['miles']} as miles, {c['fare']} as fare,
           coalesce(cbd_congestion_fee, 0) as cbd_fee
    from read_parquet({files}, filename=true, union_by_name=true)
    where strftime({c['pickup']}, '%Y-%m') = regexp_extract(filename, '(\\d{{4}}-\\d{{2}})', 1)
      and {c['miles']} > 0 and {c['miles']} < 100
      and {c['dropoff']} > {c['pickup']}
      and {c['dropoff']} < {c['pickup']} + interval 4 hour
    """


def zone_group_sql(col: str) -> str:
    inside = ",".join(map(str, sorted(INSIDE)))
    border = ",".join(map(str, sorted(BORDER)))
    return (f"case when {col} in ({inside}) then 'inside' when {col} in ({border}) then 'border' "
            f"when {col} between 1 and 263 then 'outside' else 'unknown' end")


def main() -> None:
    AGG.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("set threads = 4; set memory_limit = '6GB'; set preserve_insertion_order = false")
    for service in SERVICES:
        t = time.perf_counter()
        con.execute(f"create or replace temp view trips_{service} as {trips_sql(service)}")
        con.execute(f"""
        copy (
          select hour, service, 'pickup' as direction, pu as zone, count(*) trips, sum(miles) miles,
                 sum(fare) fare, sum(cbd_fee) cbd_fee
          from trips_{service} group by all
          union all
          select hour, service, 'dropoff', dz, count(*), sum(miles), sum(fare), sum(cbd_fee)
          from trips_{service} group by all
        ) to '{AGG}/hourly_zone_{service}.parquet' (format parquet)
        """)
        con.execute(f"""
        copy (
          select hour, service,
                 case when g_pu in ('border','unknown') or g_do in ('border','unknown') then 'other'
                      else g_pu || '->' || g_do end as flow,
                 count(*) as trips
          from (select *, {zone_group_sql('pu')} g_pu, {zone_group_sql('dz')} g_do from trips_{service})
          group by all
        ) to '{AGG}/hourly_flow_{service}.parquet' (format parquet)
        """)
        print(f"[aggregate] {service}: {time.perf_counter() - t:.0f}s", flush=True)
    for table in ["hourly_zone", "hourly_flow"]:
        con.execute(f"copy (select * from '{AGG}/{table}_*.parquet') to '{AGG}/{table}.parquet' (format parquet)")
        n = con.execute(f"select count(*), sum(trips) from '{AGG}/{table}.parquet'").fetchall()[0]
        print(f"[aggregate] {table}: {n[0]:,} rows, {int(n[1]):,} trip-ends")


if __name__ == "__main__":
    main()
