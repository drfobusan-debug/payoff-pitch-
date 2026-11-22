"""Tonight's scoring rates for one game from the §5.1 team rates and §5.2 goalies.

Pure arithmetic on already-shrunk inputs; nothing here re-weights evidence.

Each side gets goal rates *per second* for every manpower block the simulation
visits, already netted against the opponent and the opponent's goalie::

    5v5  lam = L5 x (own xGF/60 / L5) x (opp xGA/60 / L5) x home-ice x finishing x goalie x level
    PP   lam = own PP xGF/60 x opp PK xGA/60 / league PK xGA/60 x finishing x goalie x level
    SH   lam = own SH xGF/60 x opp PP xGA/60 / league PP xGA/60 x goalie x level
    EN   attacking side: league 6v5 goals/60 scaled by the side's 5v5 strength;
         into the empty net: league rate scaled by the shooter's 5v5 strength

``xgf60_5v5`` / ``xga60_5v5`` are MoneyPuck's score- and venue-adjusted rates, so
home ice is applied here (tied-state home xG share from the sim study) rather
than double-counted from the inputs. Finishing is the EB-shrunk 5v5 (G - xG)/60
posterior, additive. The goalie enters once, multiplicatively, as
``1 - GSAx/60 / league xG faced per 60``: a +0.3 GSAx/60 goalie takes ~10% off
every goal rate against him (raw Sv% never appears, §5.2). Penalty rates are
the league effective-minor rate scaled by the taker's ``pen_taken60`` and the
drawer's ``pen_drawn60``. ``level`` (``SimParams.goal_level``) is the measured
gap between a league-average sim game and the league's actual scoring.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from nhl_engine.config import SimParams

PER_60 = 3600.0
# League all-situation xG faced per goalie-60 (sim study, tied-state xG per team-60).
LEAGUE_XG_FACED_60 = 2.82


@dataclass(frozen=True)
class SideRates:
    """Per-second goal and penalty rates for one side against tonight's opponent."""

    g5: float  # 5v5, tied score, mean period
    pp: float  # on the power play
    sh: float  # while shorthanded
    en_for: float  # attacking 6v5 with own net empty
    en_against: float  # opponent's goals into this side's empty net
    minors: float  # minor penalties this side takes, per second
    xg_share: float  # 5v5 xG share (OT strength link)
    goalie_factor: float  # multiplier on goals against this side's goalie (1 = league)

    @property
    def g5_per_60(self) -> float:
        return self.g5 * PER_60


@dataclass(frozen=True)
class GameRates:
    home: SideRates
    away: SideRates

    @property
    def exp_goals_5v5_60(self) -> tuple[float, float]:
        return self.home.g5_per_60, self.away.g5_per_60


def goalie_factor(gsax60: float, league_xg_faced_60: float = LEAGUE_XG_FACED_60) -> float:
    """Multiplier on every goal rate against a goalie: ``1 - GSAx/60 / xG faced/60``."""
    return max(0.5, 1.0 - gsax60 / league_xg_faced_60)


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 and math.isfinite(num) else 1.0


def side_rates(
    own: Mapping[str, float],
    opp: Mapping[str, float],
    league: Mapping[str, float],
    *,
    is_home: bool,
    opp_goalie_gsax60: float,
    own_goalie_gsax60: float,
    params: SimParams,
) -> SideRates:
    l5 = league["xgf60_5v5"]
    hfa = params.home_xg_share / (1.0 - params.home_xg_share)
    venue = math.sqrt(hfa) if is_home else 1.0 / math.sqrt(hfa)
    g_opp = goalie_factor(opp_goalie_gsax60)
    fin = own.get("fin60_5v5", 0.0) - league.get("fin60_5v5", 0.0)

    lvl = params.goal_level
    xg5 = l5 * _ratio(own["xgf60_5v5"], l5) * _ratio(opp["xga60_5v5"], l5) * venue
    g5_60 = max(0.05, xg5 * params.goals_per_xg + fin) * g_opp * lvl

    pp_xg = own["pp_xgf60"] * _ratio(opp["pk_xga60"], league["pk_xga60"])
    pp_60 = max(0.05, pp_xg * params.goals_per_xg + fin) * g_opp * lvl

    sh_xg = own["sh_xgf60"] * _ratio(opp["pp_xga60"], league["pp_xga60"])
    sh_60 = max(0.02, sh_xg * params.goals_per_xg) * g_opp * lvl

    strength = _ratio(g5_60, l5 * params.goals_per_xg * lvl)
    en_for_60 = params.en_goals60_for * strength
    opp_strength = _ratio(opp["xgf60_5v5"], l5)
    en_against_60 = params.en_goals60_against * opp_strength

    minors_60 = (
        params.minors_per_60
        * _ratio(own["pen_taken60"], league["pen_taken60"])
        * _ratio(opp["pen_drawn60"], league["pen_drawn60"])
    )
    xgf, xga = own["xgf60_5v5"], own["xga60_5v5"]
    return SideRates(
        g5=g5_60 / PER_60,
        pp=pp_60 / PER_60,
        sh=sh_60 / PER_60,
        en_for=en_for_60 / PER_60,
        en_against=en_against_60 / PER_60,
        minors=minors_60 / PER_60,
        xg_share=xgf / (xgf + xga) if xgf + xga > 0 else 0.5,
        goalie_factor=goalie_factor(own_goalie_gsax60),
    )


def game_rates(
    home: Mapping[str, float],
    away: Mapping[str, float],
    league: Mapping[str, float],
    *,
    home_goalie_gsax60: float,
    away_goalie_gsax60: float,
    params: SimParams,
) -> GameRates:
    """Both sides' rates; ``home``/``away`` map metric key -> posterior rate."""
    return GameRates(
        home=side_rates(
            home,
            away,
            league,
            is_home=True,
            opp_goalie_gsax60=away_goalie_gsax60,
            own_goalie_gsax60=home_goalie_gsax60,
            params=params,
        ),
        away=side_rates(
            away,
            home,
            league,
            is_home=False,
            opp_goalie_gsax60=home_goalie_gsax60,
            own_goalie_gsax60=away_goalie_gsax60,
            params=params,
        ),
    )


__all__ = ["GameRates", "SideRates", "game_rates", "goalie_factor", "side_rates"]
