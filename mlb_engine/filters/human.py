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
- ``divisional``            same-division matchup -> familiarity removes the
  pitcher's novelty edge: fewer K, a touch more contact.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- calibration constants (documented run-value / R^2 targets) ------------
# Each factor is sized to its verified contribution so they are NOT weighted
# equally; umpire dominates, framing/manager/battery are minor.
#
# Umpire     ~3-5% R^2, 0.50 run avg / 0.80-1.10 run extreme  -> see umpire.py.
# Catcher framing  ~4-6% R^2, ~0.20 run/9 (elite vs. bottom ~+12-15 FRP).
# Manager    ~1.5-3% R^2, 0.10-0.30 run/game.
# Battery    ~2-4% R^2, ~0.12 run per stolen-base attempt.
UMPIRE_ZONE_PER_RUN = 0.01  # K/BB shift per zone-run unit (bounded below)
UMPIRE_ZONE_MAX = 0.08  # ~1.1-run extreme cap on the K/BB shift
FRAMING_PER_RUN = 0.0013  # ~1/5 of umpire; +15 FRP -> ~0.02 (≈0.20 run)
FRAMING_MAX = 0.02
DIVISION_K = 0.98  # familiarity: fewer Ks
DIVISION_CONTACT = 1.01
BATTERY_ADVANCE = 1.04  # per run-game flag; both flags ~+8% 2B (≈0.25-0.40 run)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class HumanFactors:
    divisional: bool = False
    catcher_framing_runs: float = 0.0
    umpire_zone_runs: float = 0.0
    pitcher_slow_to_plate: bool = False
    catcher_poor_pop: bool = False

    def offense_multipliers(self) -> dict[str, float]:
        """Bounded multipliers on this offense's PA outcomes."""
        k = 1.0
        bb = 1.0
        contact = 1.0  # -> 1B (more balls in play)
        xbh = 1.0  # -> 2B (run-game advancement)

        if self.divisional:
            k *= DIVISION_K
            contact *= DIVISION_CONTACT

        if self.catcher_framing_runs:
            f = _clip(self.catcher_framing_runs * FRAMING_PER_RUN, -FRAMING_MAX, FRAMING_MAX)
            k *= 1.0 + f
            bb *= 1.0 - f

        if self.umpire_zone_runs:
            u = _clip(self.umpire_zone_runs * UMPIRE_ZONE_PER_RUN, -UMPIRE_ZONE_MAX, UMPIRE_ZONE_MAX)
            k *= 1.0 + u
            bb *= 1.0 - u

        if self.pitcher_slow_to_plate:
            xbh *= BATTERY_ADVANCE
        if self.catcher_poor_pop:
            xbh *= BATTERY_ADVANCE

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
