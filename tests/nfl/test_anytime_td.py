"""Anytime-scorer props: priced as research, de-vigged only when both sides exist."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from nfl_engine import props, props_grade
from nfl_engine.data import nflverse
from nfl_engine.features import usage
from nfl_engine.models.player import TOUCHDOWNS, anytime_prob
from tests.nfl.test_props import projection, projections_for, quote

TD = "player_anytime_td"


def td_quote(side: str, book: str, american: float, opposite: float | None):
    return quote(
        player="Kyren Williams",
        market=TD,
        side=side,
        line=None,
        book=book,
        american=american,
        opposite=opposite,
    )


def td_projection(mean: float = 0.5):
    return projection(player="Kyren Williams", stat=TOUCHDOWNS, mean=mean)


def test_anytime_prob_is_poisson_at_least_one() -> None:
    assert anytime_prob(0.5) == pytest.approx(1 - math.exp(-0.5))
    assert anytime_prob(-1.0) == 0.0


def test_a_yes_only_board_is_priced_by_the_model_but_gets_no_fair_price() -> None:
    rows = [td_quote("yes", "draftkings", 110.0, None), td_quote("yes", "betmgm", 105.0, None)]
    (prop,) = props.price_props(rows, projections_for(td_projection()))
    assert prop.model_prob == pytest.approx(anytime_prob(0.5), abs=1e-6)
    assert prop.fair_prob is None
    assert props.UNPAIRED in prop.screens and props.RESEARCH_ONLY in prop.screens


def test_paired_yes_and_no_get_a_fair_price_and_complementary_model() -> None:
    rows = [
        td_quote("yes", "draftkings", 110.0, -140.0),
        td_quote("no", "draftkings", -140.0, 110.0),
        td_quote("yes", "betmgm", 105.0, -135.0),
        td_quote("no", "betmgm", -135.0, 105.0),
    ]
    priced = props.price_props(rows, projections_for(td_projection()), best_price_only=False)
    yes = [p for p in priced if p.side == "yes"]
    no = [p for p in priced if p.side == "no"]
    assert yes and no and all(p.fair_prob is not None for p in priced)
    assert no[0].model_prob == pytest.approx(math.exp(-0.5), abs=1e-6)


def _row(side: str) -> dict[str, str]:
    return {
        "captured_at": "2026-10-29T12:00:00Z",
        "season": "2026",
        "week": "8",
        "matchup": "SEA @ LA",
        "market": TD,
        "player": "Kyren Williams",
        "side": side,
        "line": "",
        "book": "draftkings",
        "american": "110.0",
        "stat": TOUCHDOWNS,
        "projection": "0.5",
        "model_prob": "0.39",
        "fair_prob": "",
        "ev_fair": "",
        "screens": "research_only;unpaired",
        "basis": props.BASIS,
    }


@pytest.mark.parametrize(
    ("side", "tds", "result"),
    [("yes", 1.0, "win"), ("yes", 0.0, "loss"), ("no", 0.0, "win"), ("no", 2.0, "loss")],
)
def test_grading_settles_on_at_least_one_touchdown(side: str, tds: float, result: str) -> None:
    scores = {(usage.normalise("Kyren Williams"), "LA", TOUCHDOWNS): tds}
    graded = props_grade.grade_row(_row(side), scores, "now")
    assert graded.result == result
    assert graded.pnl == pytest.approx(1.1 if result == "win" else -1.0)


def test_a_player_missing_from_the_box_score_is_void() -> None:
    graded = props_grade.grade_row(_row("yes"), {}, "now")
    assert graded.result == props_grade.VOID and graded.reason == props_grade.NO_BOX_SCORE


def test_box_scores_count_rushing_plus_receiving_touchdowns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "player_display_name": ["Kyren Williams"],
            "team": ["LA"],
            "week": [8],
            "season_type": ["REG"],
            "rushing_tds": [1.0],
            "receiving_tds": [float("nan")],
            "passing_tds": [3.0],
        }
    )
    monkeypatch.setattr(nflverse, "player_week", lambda season: frame)
    scores = props_grade.box_scores(2026, 8)
    assert scores[(usage.normalise("Kyren Williams"), "LA", TOUCHDOWNS)] == 1.0
