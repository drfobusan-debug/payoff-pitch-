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


def test_the_homer_never_wins_the_price_quoted_beside_a_rating() -> None:
    """EV on a one-way longshot is measured against a price nobody devigged.

    Left alone it wins this column on most of the board -- +480 against a modelled
    21% prints an enormous expected value -- and the note would then quote the
    screen's worst market as its recommendation.
    """
    result = _result()
    pid = _pid(result)
    board = power_board.build(
        result,
        [
            _rec("Matt Olson", "HR", 0.5, player_id=pid, american=480.0, opposite=None, ev=0.40),
            _rec("Matt Olson", "TB", 1.5, player_id=pid, ev=0.04),
        ],
    )
    assert [r.label for r in board.rows] == ["HR o0.5", "TB o1.5"]
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
        _rec("Matt Olson", "TB", 0.5 + i, player_id=pid, ev=0.01 * i) for i in range(7)
    ]
    board = power_board.build(result, recs, rows_per_batter=2)
    assert len(board.rows) == 2
    assert board.dropped == 5


def test_the_same_bet_at_two_books_is_one_row_at_the_better_price() -> None:
    """Two prices on one bet is a line-shopping question, not a board row."""
    result = _result()
    pid = _pid(result)
    board = power_board.build(
        result,
        [
            _rec("Matt Olson", "HR", 0.5, player_id=pid, american=280.0, ev=0.30),
            _rec("Matt Olson", "HR", 0.5, player_id=pid, american=255.0, ev=0.22),
        ],
    )
    assert [r.american for r in board.rows] == [280.0]
    assert board.dropped == 0  # a second quote on one bet is not a row the note withheld


def test_the_homer_and_the_hits_runs_rbis_are_both_printed_for_every_hitter() -> None:
    """The two markets the note is read for cannot be sorted off the page.

    A long HR price inflates EV by construction, so ranking on EV alone drops
    H+R+RBI on nearly every hitter -- which is the comparison the reader wants.
    """
    result = _result()
    pid = _pid(result)
    board = power_board.build(
        result,
        [
            _rec("Matt Olson", "HR", 0.5, player_id=pid, american=280.0, ev=0.96),
            _rec("Matt Olson", "RBI", 0.5, player_id=pid, ev=0.48),
            _rec("Matt Olson", "TB", 1.5, player_id=pid, ev=0.35),
            _rec("Matt Olson", "R", 0.5, player_id=pid, ev=0.26),
            _rec("Matt Olson", "HRR", 1.5, player_id=pid, ev=0.13),
        ],
        rows_per_batter=3,
    )
    labels = [r.label for r in board.rows]
    assert labels == ["HR o0.5", "RBI o0.5", "H+R+RBI o1.5"]  # HRR kept over the better TB and R
    assert board.dropped == 2


def test_the_better_of_two_hits_runs_rbis_lines_is_the_one_anchored() -> None:
    result = _result()
    pid = _pid(result)
    board = power_board.build(
        result,
        [
            _rec("Matt Olson", "HRR", 1.5, player_id=pid, ev=0.02),
            _rec("Matt Olson", "HRR", 2.5, player_id=pid, ev=0.28),
        ],
        rows_per_batter=1,
    )
    assert [r.label for r in board.rows] == ["H+R+RBI o2.5"]


def test_a_hitter_the_book_never_hung_a_homer_on_still_shows_his_other_markets() -> None:
    result = _result()
    pid = _pid(result)
    board = power_board.build(
        result, [_rec("Matt Olson", "HRR", 2.5, player_id=pid, ev=0.28)], rows_per_batter=2
    )
    assert [r.label for r in board.rows] == ["H+R+RBI o2.5"]
    assert board.dropped == 0


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
