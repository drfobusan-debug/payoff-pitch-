"""Manager tendency table (MLB Stats API team IDs).

Objective, high-volume managerial vectors translated into simulation weights:
- ``starter_bf_cap``   Third-Time-Through-the-Order hook speed -> the batters-
  faced ceiling before the bullpen takes over (quick hook = low, long leash =
  high).
- ``starter_pitch_cap`` pitch-count hook -> the pitch ceiling before the hook
  (quick hook = low ~88, long leash = high ~105); the sim pulls the starter on
  whichever of the batters-faced or pitch cap trips first.
- ``platoon_aggressive`` late-inning pinch-hit/handedness maximizing -> a small
  bat_vs_pen boost (the better platoon bench bat appears in high-leverage spots).
- ``speed_aggressive`` small-ball / base-stealing engine -> more base
  advancement (2B) and a slightly lower strikeout baseline.

Manager assignments change, so this is a maintained, user-editable map. Teams
without an entry use the neutral default (24 BF, no platoon/speed tilt).
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_BF_CAP = 24
DEFAULT_PITCH_CAP = 95


@dataclass(frozen=True)
class ManagerProfile:
    name: str
    starter_bf_cap: int = DEFAULT_BF_CAP
    starter_pitch_cap: int = DEFAULT_PITCH_CAP
    platoon_aggressive: bool = False
    speed_aggressive: bool = False

    def pen_multipliers(self) -> dict[str, float]:
        """Late-inning (bat_vs_pen) tilt from aggressive platoon matchup use."""
        if not self.platoon_aggressive:
            return {}
        return {"1B": 1.02, "2B": 1.02, "K": 0.98}

    def offense_multipliers(self) -> dict[str, float]:
        """Full-offense tilt from a small-ball / speed engine."""
        if not self.speed_aggressive:
            return {}
        return {"2B": 1.03, "K": 0.98}


MANAGERS: dict[int, ManagerProfile] = {
    # Quick-hook analytical disciples (TTO suppression).
    139: ManagerProfile("Kevin Cash", starter_bf_cap=19, starter_pitch_cap=85),  # TB
    141: ManagerProfile("John Schneider", starter_bf_cap=20, starter_pitch_cap=88),  # TOR
    119: ManagerProfile("Dave Roberts", starter_bf_cap=20, starter_pitch_cap=88),  # LAD
    # Platoon / pinch-hit maximizers.
    142: ManagerProfile("Rocco Baldelli", platoon_aggressive=True),  # MIN
    138: ManagerProfile("Oliver Marmol", platoon_aggressive=True),  # STL
    # Traditional long-leash anchors.
    113: ManagerProfile("Terry Francona", starter_bf_cap=29, starter_pitch_cap=105),  # CIN
    137: ManagerProfile("Bob Melvin", starter_bf_cap=29, starter_pitch_cap=105),  # SF
    # Small-ball / speed chaos.
    114: ManagerProfile("Stephen Vogt", speed_aggressive=True),  # CLE
}

_DEFAULT = ManagerProfile("Unknown")


def get_manager(team_id: int) -> ManagerProfile:
    return MANAGERS.get(team_id, _DEFAULT)
