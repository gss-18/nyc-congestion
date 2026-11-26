"""Derive which taxi zones are inside the Congestion Relief Zone from the fee actually charged.

Usage: uv run python scripts/derive_zones.py
"""

from __future__ import annotations

from pathlib import Path

import duckdb

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

QUERY = f"""
select z.LocationID, z.Zone, count(*) as intra_zone_trips, avg((cbd_congestion_fee > 0)::int) as fee_share
from '{RAW}/fhvhv_tripdata_2025-03.parquet' t
join '{RAW}/taxi_zone_lookup.csv' z on z.LocationID = t.PULocationID
where t.PULocationID = t.DOLocationID
group by all order by fee_share desc
"""

if __name__ == "__main__":
    rows = duckdb.connect().execute(QUERY).fetchall()
    inside = sorted(r[0] for r in rows if r[3] > 0.9)
    border = sorted(r[0] for r in rows if 0.1 < r[3] <= 0.9)
    print(f"INSIDE ({len(inside)}): {inside}")
    print(f"BORDER ({len(border)}): {border}")
