"""Psychology / strategy / human-element layer.

Bounded, mostly neutral-by-default adjustments for the "hidden" context factors
that Statcast physics misses. Only the division-rivalry signal is derived from
data we already have (team divisions); every other input defaults to a no-op
(multiplier 1.0) and activates only when a real feed supplies its value, so the
model never fabricates coefficients:

- ``catcher_framing_runs``  Savant catcher framing (opp catcher): + steals
  strikes -> more K, fewer BB for this offense.
- ``umpire_zone_runs``      UmpScorecards zone bias: + = wide/pitcher-friendly
  zone -> more K, fewer BB (applied to both offenses at game level).
- ``pitcher_slow_to_plate`` opp starter time-to-plate > 1.35s and
  ``catcher_poor_pop`` opp catcher slow pop time: enable the run game -> a small
  extra-base/advancement (2B) nudge (steals put runners in scoring position).
- ``manager_quick_hook``    opp manager pulls the starter at the TTO line ->
  earlier bullpen exposure, applied as a negative starter batters-faced delta.
- ``divisional``            same-division matchup -> familiarity removes the
  pitcher's novelty edge: fewer K, a touch more contact.
"""

from __future__ import annotations

from dataclasses import dataclass


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class HumanFactors:
    divisional: bool = False
    catcher_framing_runs: float = 0.0
    umpire_zone_runs: float = 0.0
    pitcher_slow_to_plate: bool = False
    catcher_poor_pop: bool = False
    manager_quick_hook: bool = False

    def offense_multipliers(self) -> dict[str, float]:
        """Bounded multipliers on this offense's PA outcomes."""
        k = 1.0
        bb = 1.0
        contact = 1.0  # -> 1B (more balls in play)
        xbh = 1.0  # -> 2B (run-game advancement)

        if self.divisional:
            k *= 0.98
            contact *= 1.01

        if self.catcher_framing_runs:
            f = _clip(self.catcher_framing_runs * 0.004, -0.06, 0.06)
            k *= 1.0 + f
            bb *= 1.0 - f

        if self.umpire_zone_runs:
            u = _clip(self.umpire_zone_runs * 0.01, -0.08, 0.08)
            k *= 1.0 + u
            bb *= 1.0 - u

        if self.pitcher_slow_to_plate:
            xbh *= 1.02
        if self.catcher_poor_pop:
            xbh *= 1.02

        out: dict[str, float] = {}
        if k != 1.0:
            out["K"] = k
        if bb != 1.0:
            out["BB"] = bb
        if contact != 1.0:
            out["1B"] = contact
        if xbh != 1.0:
            out["2B"] = xbh
        return out

    def starter_bf_cap_delta(self) -> int:
        """Batters-faced adjustment for the opposing starter (manager hook)."""
        return -6 if self.manager_quick_hook else 0
