"""The priced board bolted to the power screen.

The join is the whole feature, so the tests pin what it must never do: invent a
price, cross-price two hitters on a shared surname, silently drop a survivor the
engine never priced, or let a number leak into a matchup rating.
"""

from __future__ import annotations

from datetime import date as Date

from mlb_engine.market.tiers import Tier
from mlb_engine.output import power_board, power_report
from mlb_engine.recommendations import Recommendation, save_json
from tests.test_power_screen import _result


def _rec(
    name: str,
    stat: str,
    line: float,
    side: str = "over",
    *,
    player_id: int | None = None,
    american: float | None = -115.0,
    opposite: float | None = -105.0,
    model: float = 0.55,
    fair: float | None = 0.50,
    ev: float | None = 0.04,
    edge: float | None = 0.05,
    tier: Tier = Tier.MODERATE,
) -> Recommendation:
    return Recommendation(
        game_date=Date(2026, 8, 17),
        game_pk=1,
        matchup="ATL @ MIN",
        category="batter",
        market=f"batter_{stat.lower()}",
        selection=f"{name} {stat} {'o' if side == 'over' else 'u'}{line}",
        model_prob=model,
        line=line,
        book="DraftKings",
        market_american=american,
        opposite_american=opposite,
        ev=ev,
        edge=edge,
        fair_prob=fair,
        tier=tier,
        player_id=player_id,
        stat=stat,
        side=side,
    )


def _pid(result) -> int:
    return result.sections[0].hitters[0].line.mlbam_id


# --- the join -------------------------------------------------------------


def test_a_survivor_keeps_his_own_rows_best_expected_value_first() -> None:
    result = _result()
    pid = _pid(result)
    board = power_board.build(
        result,
        [
            _rec("Matt Olson", "HR", 0.5, player_id=pid, ev=0.02),
            _rec("Matt Olson", "TB", 1.5, player_id=pid, ev=0.09),
        ],
    )
    assert [r.label for r in board.rows] == ["TB o1.5", "HR o0.5"]
    assert board.unpriced == []
    assert board.best_for_batter("Matt Olson").label == "TB o1.5"


def test_another_players_row_is_not_priced_onto_this_one() -> None:
    """An id mismatch is a different hitter, whatever the book spells."""
    result = _result()
    board = power_board.build(
        result, [_rec("Matt Olson", "HR", 0.5, player_id=_pid(result) + 1)]
    )
    assert board.rows == []
    assert board.unpriced == ["Matt Olson"]


def test_an_id_less_row_falls_back_to_the_name() -> None:
    result = _result()
    board = power_board.build(result, [_rec("Matt Olson", "H", 1.5, player_id=None)])
    assert len(board.rows) == 1


def test_a_market_with_no_quote_is_not_a_bet() -> None:
    result = _result()
    board = power_board.build(
        result,
        [_rec("Matt Olson", "HR", 0.5, player_id=_pid(result), american=None, ev=None)],
    )
    assert board.rows == []
    assert board.unpriced == ["Matt Olson"]


def test_a_pitcher_row_never_reaches_the_batter_board() -> None:
    result = _result()
    rec = _rec("Matt Olson", "K", 5.5, player_id=_pid(result))
    rec.category = "pitcher"
    assert power_board.build(result, [rec]).rows == []


def test_only_the_best_rows_per_hitter_are_printed_and_the_rest_are_counted() -> None:
    result = _result()
    pid = _pid(result)
    recs = [
        _rec("Matt Olson", "TB", 1.5, player_id=pid, ev=0.01 * i) for i in range(7)
    ]
    board = power_board.build(result, recs, rows_per_batter=2)
    assert len(board.rows) == 2
    assert board.dropped == 5


def test_a_one_sided_quote_is_flagged_as_undevigged() -> None:
    result = _result()
    board = power_board.build(
        result,
        [_rec("Matt Olson", "HR", 0.5, player_id=_pid(result), opposite=None, fair=None)],
    )
    assert board.rows[0].devigged is False
    html = power_report.render_html(result, board=board)
    assert "one-way" in html
    assert "could not be stripped" in html


def test_an_empty_ledger_leaves_every_survivor_unpriced() -> None:
    result = _result()
    board = power_board.build(result, [])
    assert board.rows == []
    assert board.unpriced == ["Matt Olson"]
    assert board.buys == []


# --- the note -------------------------------------------------------------


def test_the_note_prints_the_board_and_still_refuses_to_price_the_rating() -> None:
    result = _result()
    board = power_board.build(
        result,
        [_rec("Matt Olson", "TB", 1.5, player_id=_pid(result))],
        source="predictions_2026-08-17.json",
    )
    html = power_report.render_html(result, board=board)
    assert "The board" in html
    assert "DraftKings" in html
    assert "-115" in html
    assert "Moderate buy" in html
    assert "predictions_2026-08-17.json" in html  # provenance
    assert "TB o1.5" in html
    # the rating is still scored on the matchup alone
    assert "contains no price" in html
    assert html.index("The board") < html.index("Recommendations")


def test_a_note_with_no_board_is_the_note_it_was_before() -> None:
    result = _result()
    plain = power_report.render_html(result)
    assert "The board" not in plain
    assert "This note reads no market" in plain


def test_an_unpriced_survivor_is_named_rather_than_dropped() -> None:
    result = _result()
    board = power_board.build(result, [])
    html = power_report.render_html(result, board=board)
    assert "Priced by nobody: Matt Olson" in html
    assert "not priced" in html  # in the recommendation table's price column


def test_the_market_disagreeing_with_the_screen_is_called_out() -> None:
    result = _result()
    board = power_board.build(
        result,
        [
            _rec(
                "Matt Olson", "H", 1.5, player_id=_pid(result),
                model=0.38, fair=0.50, edge=-0.12, ev=-0.10, tier=Tier.PASS,
            )
        ],
    )
    html = power_report.render_html(result, board=board)
    assert "The market disagrees hardest on Matt Olson" in html


# --- the file on disk -----------------------------------------------------


def test_the_board_round_trips_through_the_predictions_file(tmp_path) -> None:
    result = _result()
    path = power_board.default_predictions_path(tmp_path, result.as_of)
    assert path.name == "predictions_2026-08-17.json"
    save_json([_rec("Matt Olson", "TB", 1.5, player_id=_pid(result))], path)
    from mlb_engine.recommendations import load_json

    board = power_board.build(result, load_json(path), source=path.name)
    assert [r.label for r in board.rows] == ["TB o1.5"]
    assert board.rows[0].is_buy
