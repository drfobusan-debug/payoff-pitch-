"""Home-plate umpire strike-zone layer.

Each umpire's zone deviates the ~9.0 run/game league baseline by up to ~1 run:
tight/shrunk zones (hitter-friendly, "Over") push walks and runs up; wide/
expanded zones (pitcher-friendly, "Under") push strikeouts up and slugging down.
This maps a plate umpire to the ``umpire_zone_runs`` input of ``HumanFactors``.

The primary live source is a Rotowire / UmpScorecards umpire leaderboard (runs-
per-game vs. baseline); the curated table below is the fallback for the umpires
with the most documented, consistent tendencies. Returns ``None`` when the plate
umpire is unknown so the layer stays neutral.
"""

from __future__ import annotations

from dataclasses import dataclass

LEAGUE_RUNS_BASELINE = 9.0

# Scales a runs-vs-baseline deviation into the HumanFactors zone-runs coefficient.
# +run_deviation (tight/Over) -> negative zone_runs -> fewer K, more BB.
_DEV_TO_ZONE_RUNS = -6.0


@dataclass(frozen=True)
class UmpireTendency:
    run_deviation: float  # runs vs. 9.0 baseline; + = more runs (tight zone)
    low_consistency: bool = False


CURATED: dict[str, UmpireTendency] = {
    # Tight / shrunk zone -> Over (walks & runs up)
    "edwin moscoso": UmpireTendency(1.0),
    "todd tichenor": UmpireTendency(0.7),
    "jeter downs": UmpireTendency(0.6),
    "andy fletcher": UmpireTendency(0.6),
    # Wide / expanded zone -> Under (strikeouts up, slugging down)
    "lance barrett": UmpireTendency(-0.7),
    "bill miller": UmpireTendency(-0.6),
    "vic carapazza": UmpireTendency(-0.6),
    "ron kulpa": UmpireTendency(-0.6),
    # Low consistency -> variance / chaos
    "cb bucknor": UmpireTendency(0.0, low_consistency=True),
    "laz diaz": UmpireTendency(0.0, low_consistency=True),
}


def _norm(name: str) -> str:
    return " ".join(name.lower().replace(".", "").split())


def lookup(name: str) -> UmpireTendency | None:
    return CURATED.get(_norm(name))


def zone_runs_from_deviation(run_deviation: float) -> float:
    """Convert a runs-vs-baseline deviation into a HumanFactors zone-runs value."""
    return run_deviation * _DEV_TO_ZONE_RUNS


def zone_runs_for_name(name: str) -> float | None:
    """Curated ``umpire_zone_runs`` for a plate umpire, or ``None`` if unknown."""
    t = lookup(name)
    if t is None:
        return None
    return zone_runs_from_deviation(t.run_deviation)
