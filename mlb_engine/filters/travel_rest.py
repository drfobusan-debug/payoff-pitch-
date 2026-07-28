"""Travel, rest and circadian (jet-lag) penalties.

Large multi-decade studies (e.g. Northwestern's ~40k-game analysis) find that
raw travel *mileage* has little effect, but crossing time zones does — and
**eastward** travel (a shortened day) hurts more than westward. The impact is
U-shaped over a road trip: worst on the first game after crossing zones, then it
fades as players adapt (~1 time zone / day). Because we key off each team's
immediately-preceding game, a cross-country flight shows a large ``tz_shift`` on
the first game and ~0 on subsequent same-city games, so the penalty naturally
localizes to the travel game.

Documented manifestations, applied as bounded multipliers:
  * Run creation loss: the jet-lagged offense loses extra-base hits (2B/3B) from
    reduced bat speed and conservative base running; slight overall dip.
  * Run suppression loss: a jet-lagged pitching staff allows more home runs, so
    the opponent's HR rate is boosted (``pitching_multipliers``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date as Date

from mlb_engine.data.parks import Park

EAST_WEIGHT = 1.5  # eastward circadian cost vs. westward
TZ_CAP = 3.0


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
    east: bool
    offense_mult: float  # applied to own singles/HR
    xbh_mult: float  # applied to own 2B/3B (extra XBH suppression)
    hr_allowed_mult: float  # applied to the OPPONENT's HR (run-suppression loss)
    note: str = ""

    def multipliers(self) -> dict[str, float]:
        """Multipliers for this (possibly jet-lagged) team's own offense."""
        return {
            "1B": self.offense_mult,
            "2B": self.xbh_mult,
            "3B": self.xbh_mult,
            "HR": self.offense_mult,
        }

    def pitching_multipliers(self) -> dict[str, float]:
        """HR boost to apply to the OPPONENT's offense (tired pitching staff)."""
        return {"HR": self.hr_allowed_mult}


def compute(prev: PrevGame | None, today_park: Park, today: Date) -> TravelRestEffect:
    if prev is None:
        return TravelRestEffect(3, 0.0, 0.0, False, 1.0, 1.0, 1.0, note="no prior game found")

    rest_days = (today - prev.game_date).days
    dist = _haversine_mi(prev.lat, prev.lon, today_park.lat, today_park.lon)
    # East travel -> longitude increases (toward 0 in the W hemisphere) -> positive.
    tz_shift = (today_park.lon - prev.lon) / 15.0
    east = tz_shift > 0

    # Adaptation: circadian clock realigns ~1 time zone per rest day.
    misalign = max(0.0, min(TZ_CAP, abs(tz_shift)) - max(0, rest_days - 1))
    severity = misalign * (EAST_WEIGHT if east else 1.0)

    # Run-creation loss (own offense).
    off_pen = min(0.04, severity * 0.012)
    if rest_days <= 1 and dist > 1500:  # coast-to-coast next-day burden
        off_pen += 0.01
    offense_mult = max(0.94, 1.0 - off_pen)
    xbh_mult = max(0.90, 1.0 - min(0.06, severity * 0.020))

    # Run-suppression loss: jet-lagged staff -> opponent HR spike.
    hr_allowed_mult = min(1.12, 1.0 + severity * 0.030)

    return TravelRestEffect(
        rest_days=rest_days,
        travel_mi=dist,
        tz_shift=tz_shift,
        east=east,
        offense_mult=offense_mult,
        xbh_mult=xbh_mult,
        hr_allowed_mult=hr_allowed_mult,
        note=(
            f"rest={rest_days}d travel={dist:.0f}mi tz={tz_shift:+.1f} "
            f"{'E' if east else 'W'} misalign={misalign:.1f}"
        ),
    )
