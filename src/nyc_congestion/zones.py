"""Taxi zones inside the Congestion Relief Zone (CRZ), derived from the data, not drawn by hand.

Method (scripts/derive_zones.py): in March 2025 Uber/Lyft data, take trips that start and end in the same
taxi zone. Inside the CRZ almost all of them pay the CBD congestion fee; outside, almost none do.
  - INSIDE: 37 zones with fee share > 0.9 (all 0.95-0.99; Manhattan south of 60th St)
  - BORDER: 6 zones near 60th St (fee share 0.14-0.49); excluded from both groups in causal analysis
Tolling started 2025-01-05.
"""

from __future__ import annotations

from datetime import date

POLICY_START = date(2025, 1, 5)

INSIDE = frozenset({
    4, 13, 45, 48, 50, 68, 79, 87, 88, 90, 100, 107, 113, 114, 125, 137, 144, 148, 158, 161, 162, 163, 164,
    170, 186, 209, 211, 224, 229, 230, 231, 232, 233, 234, 246, 249, 261,
})
BORDER = frozenset({43, 140, 141, 142, 143, 237})
UNKNOWN = frozenset({264, 265})  # TLC "Unknown" / "Outside of NYC"


def group(zone: int) -> str:
    if zone in INSIDE:
        return "inside"
    if zone in BORDER:
        return "border"
    if zone in UNKNOWN:
        return "unknown"
    return "outside"
