"""Travel and rest disadvantage penalties.

Uses each team's previous game (date + venue) to estimate travel distance, time
-zone change, and days of rest, then applies a small bounded penalty to that
team's offense. Effects are intentionally modest — travel/rest is a real but
second-order factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date as Date

from mlb_engine.data.parks import Park


def _haversine_mi(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@dataclass
class PrevGame:
    game_date: Date
    lat: float
    lon: float


@dataclass
class TravelRestEffect:
    rest_days: int
    travel_mi: float
    tz_shift: float  # positive = traveled east
    offense_mult: float
    note: str = ""

    def multipliers(self) -> dict[str, float]:
        m = self.offense_mult
        return {"1B": m, "2B": m, "3B": m, "HR": m}


def compute(prev: PrevGame | None, today_park: Park, today: Date) -> TravelRestEffect:
    if prev is None:
        return TravelRestEffect(3, 0.0, 0.0, 1.0, note="no prior game found")

    rest_days = (today - prev.game_date).days
    dist = _haversine_mi(prev.lat, prev.lon, today_park.lat, today_park.lon)
    tz_shift = (today_park.lon - prev.lon) / 15.0 * -1  # east travel -> positive

    penalty = 0.0
    # Back-to-back with meaningful travel.
    if rest_days <= 1 and dist > 500:
        penalty += min(0.02, dist / 3000.0 * 0.02)
    # Eastward travel (circadian disadvantage) with short rest.
    if rest_days <= 1 and tz_shift > 0:
        penalty += min(0.015, tz_shift * 0.006)
    # Long rest layoff (rust) — tiny.
    if rest_days >= 4:
        penalty += 0.005

    mult = max(0.96, 1.0 - penalty)
    return TravelRestEffect(
        rest_days=rest_days,
        travel_mi=dist,
        tz_shift=tz_shift,
        offense_mult=mult,
        note=f"rest={rest_days}d travel={dist:.0f}mi tz={tz_shift:+.1f}",
    )
