"""Finishing as skill + level (``SimParams.finishing_split``)."""

from __future__ import annotations

from dataclasses import replace

from nhl_engine.config import SimParams
from nhl_engine.models import goals

LEAGUE = {
    "xgf60_5v5": 2.6,
    "xga60_5v5": 2.6,
    "pp_xgf60": 6.5,
    "pp_xga60": 0.8,
    "pk_xga60": 6.5,
    "sh_xgf60": 0.8,
    "pen_taken60": 3.4,
    "pen_drawn60": 3.4,
    "fin60_5v5": 0.10,  # league finishing hot: +0.10 g/60 over last season
}
TEAM = {**LEAGUE, "fin60_5v5": 0.0}


def _side(params: SimParams, league: dict[str, float], own: dict[str, float] = TEAM):
    return goals.side_rates(
        own, TEAM, league, is_home=True, opp_goalie_gsax60=0.0, own_goalie_gsax60=0.0, params=params
    )


def test_default_taxes_every_team_when_league_runs_hot() -> None:
    p = SimParams(draws=10)
    assert goals.finishing(TEAM, LEAGUE, p) == -0.10


def test_split_skill_sums_to_zero_and_level_lifts_both_sides() -> None:
    p = replace(SimParams(draws=10), finishing_split=True)
    league = {**LEAGUE, goals.FIN_TEAM_MEAN: 0.0, goals.FIN_LAST: 0.0}
    hot = {**TEAM, "fin60_5v5": 0.05}
    cold = {**TEAM, "fin60_5v5": -0.05}
    assert goals.finishing(hot, league, p) + goals.finishing(cold, league, p) == 2 * 0.10
    assert goals.finishing(TEAM, league, p) == 0.10
    assert _side(p, league).g5 > _side(SimParams(draws=10), LEAGUE).g5


def test_split_needs_slate_keys_else_falls_back() -> None:
    p = replace(SimParams(draws=10), finishing_split=True)
    assert goals.finishing(TEAM, LEAGUE, p) == -0.10
