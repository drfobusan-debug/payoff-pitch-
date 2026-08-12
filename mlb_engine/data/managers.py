"""Manager tendency table (MLB Stats API team IDs).

Managerial vectors translated into simulation weights:
- ``platoon_aggressive`` late-inning pinch-hit/handedness maximizing -> a small
  bat_vs_pen boost (the better platoon bench bat appears in high-leverage spots).
- ``speed_aggressive`` small-ball / base-stealing engine -> more base
  advancement (2B) and a slightly lower strikeout baseline.

The hook used to live here too, as a hand-entered ``starter_bf_cap`` /
``starter_pitch_cap`` per manager. Measured over 3,299 starts in the Statcast
cache, those entries did not describe the hooks they named -- and the reason is
that the real spread between teams is far narrower than a table like this invites
you to write. Every team's 75th-percentile start lands between 23 and 26 batters
faced; the table asserted 19 to 29:

    team  manager           entered BF   measured p75   entered P   measured p75
    LAD   Dave Roberts              20             26          88             99
    TOR   John Schneider            20             24          88             97
    TB    Kevin Cash                19             24          85             89
    CIN   Terry Francona            29             25         105             96
    SF    Bob Melvin                29             25         105             97

Los Angeles has the longest leash in baseball and was entered as the third
quickest; the two "long-leash anchors" are league average. Correlation between
the entered cap and the measured hook was r = +0.22 on batters faced and +0.32 on
pitches. Because the pipeline takes ``min(manager hook, the starter's own recent
depth)``, a cap set six batters too low bound on essentially every start, and the
hook is the ceiling on a starter's simulated K/H/ER line.

So the hook is no longer a manager attribute. The league default (24 BF / 95
pitches, which measures as the p75 of 25 and 96) is the starting point, and
``features.workload.expected_bf_cap`` moves it per start from the pitcher's own
recent outings -- the quantity that actually varies.

Manager assignments change, so this is a maintained, user-editable map. Teams
without an entry use the neutral default (no platoon/speed tilt).
"""

from __future__ import annotations

from dataclasses import dataclass

# League-wide starter hook. Measured p75 over 3,299 starts: 25 BF, 96 pitches.
DEFAULT_BF_CAP = 24
DEFAULT_PITCH_CAP = 95


@dataclass(frozen=True)
class ManagerProfile:
    name: str
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
    # Platoon / pinch-hit maximizers.
    142: ManagerProfile("Rocco Baldelli", platoon_aggressive=True),  # MIN
    138: ManagerProfile("Oliver Marmol", platoon_aggressive=True),  # STL
    # Small-ball / speed chaos.
    114: ManagerProfile("Stephen Vogt", speed_aggressive=True),  # CLE
}

_DEFAULT = ManagerProfile("Unknown")


def get_manager(team_id: int) -> ManagerProfile:
    return MANAGERS.get(team_id, _DEFAULT)
