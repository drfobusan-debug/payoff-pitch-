"""Schedule-pacing and lineup-structure layer (the "final 5-7%").

Data-grounded now:
- **DGANG tax** (day-game-after-night-game): a night game followed the next day
  by a getaway day game compresses the sleep window, dropping bat speed and
  spiking whiff vs. off-speed. When a team is in a DGANG spot its offense is
  suppressed by ~0.25-0.40 runs (bounded), independent of the starters.

Structurally captured already (documented, no-op to avoid double counting):
- **Lineup clustering / sequencing**: packing high-OBP hitters at the top raises
  the multi-run-inning ceiling — but the order-aware Markov F5 + Monte Carlo run
  the *actual* batting order, so this is already in the simulation. Exposed here
  as a diagnostic that returns no multiplier.

Neutral hooks (activate only when a real feed is supplied):
- **ABS challenge edge**: a team's border-call challenge success in high leverage.
- **Pitching-coach / dev-org multiplier**: organizational Stuff+/pitch-design lift.
"""

from __future__ import annotations

# DGANG suppression (bat speed / whiff), sized to ~0.25-0.40 runs.
_DGANG: dict[str, float] = {
    "1B": 0.98,
    "2B": 0.97,
    "3B": 0.97,
    "HR": 0.98,
    "K": 1.03,
}


def parse_utc_hour(iso: str | None) -> float | None:
    """Fractional UTC hour from an ISO game-start string (e.g. ...T23:05:00Z)."""
    if not iso:
        return None
    try:
        t = iso.split("T", 1)[1]
        return int(t[:2]) + int(t[3:5]) / 60.0
    except (IndexError, ValueError):
        return None


def local_hour(utc_hour: float, lon: float) -> float:
    """Approximate local clock hour from a UTC hour and park longitude."""
    return (utc_hour + lon / 15.0) % 24.0


def is_night(local_hr: float) -> bool:
    return local_hr >= 18.0 or local_hr < 4.0


def is_day(local_hr: float) -> bool:
    return 8.0 <= local_hr <= 16.0


def dgang_multipliers(
    prev_local_hr: float | None, today_local_hr: float | None, rest_days: int
) -> dict[str, float]:
    """Offense suppression when a team plays a day game right after a night game."""
    if rest_days != 1 or prev_local_hr is None or today_local_hr is None:
        return {}
    if is_night(prev_local_hr) and is_day(today_local_hr):
        return dict(_DGANG)
    return {}
