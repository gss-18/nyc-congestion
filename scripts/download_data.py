"""Download NYC TLC trip records (yellow taxi + high-volume for-hire: Uber/Lyft) and the taxi zone lookup.

Files go to data/raw/. Existing complete files are skipped, so the script can be re-run safely.
Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Usage: uv run python scripts/download_data.py [--start 2023-01] [--end 2026-09]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE = "https://d37ci6vzurychx.cloudfront.net"
HEADERS = {"User-Agent": "nyc-congestion-study/0.1"}
DATASETS = ["yellow", "fhvhv"]


def months(start: str, end: str) -> list[str]:
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def download(client: httpx.Client, url: str, path: Path) -> None:
    """Stream to a temp file, then rename, so an interrupted download never looks complete."""
    for attempt in range(4):
        try:
            with client.stream("GET", url, timeout=300) as r:
                r.raise_for_status()
                expected = int(r.headers.get("content-length", 0))
                if path.exists() and expected and path.stat().st_size == expected:
                    return
                tmp = path.with_suffix(path.suffix + ".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
                if expected and tmp.stat().st_size != expected:
                    raise httpx.HTTPError(f"incomplete download: {tmp.stat().st_size} of {expected} bytes")
                tmp.replace(path)
                return
        except httpx.HTTPError:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2026-09")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
        download(client, f"{BASE}/misc/taxi_zone_lookup.csv", RAW / "taxi_zone_lookup.csv")
        for month in months(args.start, args.end):
            for ds in DATASETS:
                name = f"{ds}_tripdata_{month}.parquet"
                download(client, f"{BASE}/trip-data/{name}", RAW / name)
            print(f"[download] {month} done", flush=True)
    total = sum(p.stat().st_size for p in RAW.glob("*.parquet"))
    print(f"[download] {len(list(RAW.glob('*.parquet')))} files, {total / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
