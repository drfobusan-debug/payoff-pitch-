"""Opta's read travels onto the card beside our own.

The outside benchmark (#101) was only ever scored months later in a study. The
card is where it is worth something: a second model's probability on the same
selection is the one check the price cannot give us, since the market only
quotes what it chooses to and CLV grades the number rather than the forecast.
"""

from __future__ import annotations

from datetime import date as Date

from mlb_engine.data.opta import OptaRow, annotate
from mlb_engine.recommendations import Recommendation


def _row(**kw) -> OptaRow:
    base = dict(
        date="2026-08-12",
        matchup="BAL @ MIN",
        market="batter_hr",
        selection="Byron Buxton HR o0.5",
        player="Byron Buxton",
        player_id=1,
        stat="HR",
        line=0.5,
        projection=0.21,
        over_odds=320.0,
        under_odds=-420.0,
        over_prob=0.30,
        edge=0.04,
        bet="over",
        confidence=3,
        result=None,
        actual=None,
    )
    base.update(kw)
    return OptaRow(**base)


def _rec(**kw) -> Recommendation:
    base = dict(
        game_date=Date(2026, 8, 12),
        game_pk=1,
        matchup="BAL @ MIN",
        category="batter",
        market="batter_hr",
        selection="Byron Buxton HR o0.5",
        model_prob=0.28,
        side="over",
    )
    base.update(kw)
    return Recommendation(**base)


def test_agreement_shows_optas_own_stars() -> None:
    rec = _rec()
    assert annotate([rec], [_row()]) == 1
    assert rec.opta_prob == 0.30
    assert rec.opta_stars == 3
    assert rec.opta_agrees is True
    assert rec.opta_mark == "\u2605\u2605\u2605"


def test_stars_on_the_other_side_are_not_ours() -> None:
    """Three stars on the under is evidence against the over, not for it."""
    rec = _rec()
    annotate([rec], [_row(bet="under")])
    assert rec.opta_agrees is False
    assert rec.opta_mark == "fade \u2605\u2605\u2605"


def test_no_bet_earns_no_mark() -> None:
    rec = _rec()
    annotate([rec], [_row(bet=None)])
    assert rec.opta_prob == 0.30  # the probability is still worth showing
    assert rec.opta_mark == ""


def test_an_under_gets_the_complement_of_optas_over() -> None:
    rec = _rec(side="under", selection="Byron Buxton HR o0.5")
    annotate([rec], [_row(bet="under")])
    assert rec.opta_prob == 0.70
    assert rec.opta_agrees is True


def test_names_join_across_spelling_differences() -> None:
    """The two feeds punctuate and accent names differently."""
    rec = _rec(selection="Andrés Giménez HR o0.5")
    assert annotate([rec], [_row(selection="Andres Gimenez HR o0.5")]) == 1
    assert rec.opta_stars == 3


def test_a_market_opta_does_not_cover_is_left_blank() -> None:
    rec = _rec(market="game_ml", selection="MIN ML", side="win", category="game")
    assert annotate([rec], [_row()]) == 0
    assert rec.opta_prob is None
    assert rec.opta_mark == ""


def test_the_mark_reaches_the_workbook_row() -> None:
    from mlb_engine.output.excel import COLUMNS

    rec = _rec()
    annotate([rec], [_row()])
    row = rec.as_row()
    assert row["AI"] == "\u2605\u2605\u2605"
    assert row["Opta %"] == 30.0
    assert set(COLUMNS) <= set(row)
