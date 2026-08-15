"""The likeliest man to homer is an over, not whichever side prices highest."""

from __future__ import annotations

import datetime as dt

from mlb_engine.output.daily_preview import _hr_line, top_hr_prop
from mlb_engine.recommendations import Recommendation


def _hr(name: str, prob: float, side: str) -> Recommendation:
    return Recommendation(
        game_date=dt.date(2026, 8, 14),
        game_pk=1,
        matchup="AAA @ BBB",
        category="Batter Props",
        market="batter_hr",
        selection=f"{name} HR {'o' if side == 'over' else 'u'}0.5",
        model_prob=prob,
        line=0.5,
        stat="HR",
        side=side,
    )


def _both(name: str, p_over: float) -> list[Recommendation]:
    return [_hr(name, p_over, "over"), _hr(name, 1.0 - p_over, "under")]


def test_the_slugger_is_named_and_not_the_man_who_cannot_homer() -> None:
    # A weak bat's under prices at .97, far above any hitter's chance of homering,
    # so ranking on probability alone hands the headline to the wrong player.
    recs = _both("Big Slugger", 0.18) + _both("Slap Hitter", 0.03)
    best = top_hr_prop(recs)
    assert best is not None
    assert best.side == "over"
    assert "Big Slugger" in best.selection


def test_the_quoted_probability_is_the_chance_he_homers() -> None:
    line = _hr_line(_both("Big Slugger", 0.18) + _both("Slap Hitter", 0.03))
    assert "18.0%" in line
    assert "97.0%" not in line
    assert "to go yard" in line


def test_the_side_marker_is_not_left_in_his_name() -> None:
    line = _hr_line(_both("Big Slugger", 0.18))
    assert "Big Slugger —" in line
    assert "o0.5" not in line
    assert "u0.5" not in line


def test_an_unpriced_game_says_so_rather_than_naming_an_under() -> None:
    assert "no home-run market priced" in _hr_line([])
    assert "no home-run market priced" in _hr_line([_hr("Slap Hitter", 0.97, "under")])
