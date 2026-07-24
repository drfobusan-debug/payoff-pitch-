"""Catcher pitch-framing table (Baseball Savant framing runs).

Elite framers steal called strikes (more K, fewer BB for the pitching side);
poor framers give them back. This maps a starting catcher to the
``catcher_framing_runs`` input of ``HumanFactors`` (see ``filters/human.py``),
which already sizes framing at ~1/5 of the umpire effect.

The curated table below carries the catchers with the most documented, stable
framing values (season framing runs vs. average); it returns ``None`` for
unknown catchers so the layer stays neutral rather than fabricating an edge.
"""

from __future__ import annotations

# Season catcher framing runs vs. average (positive = steals strikes).
CURATED: dict[str, float] = {
    # Elite framers -> more strikes -> Under lean (K up, BB down).
    "patrick bailey": 16.0,
    "austin hedges": 13.0,
    "cal raleigh": 12.0,
    "jose trevino": 11.0,
    "gabriel moreno": 10.0,
    "sean murphy": 9.0,
    "alejandro kirk": 8.0,
    # Poor framers -> gives strikes back -> Over lean (K down, BB up).
    "keibert ruiz": -10.0,
    "salvador perez": -9.0,
    "martin maldonado": -8.0,
    "elias diaz": -7.0,
}


def _norm(name: str) -> str:
    return " ".join(name.lower().replace(".", "").split())


def framing_runs_for_name(name: str) -> float | None:
    """Curated framing runs for a catcher, or ``None`` if unknown."""
    return CURATED.get(_norm(name))
