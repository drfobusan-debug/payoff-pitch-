"""The power screen's own ledger, and the scorecard it grades out of it.

The point of the record is that it cannot flatter the screen, so the tests pin
the ways it could: a re-run must overwrite the day rather than count it twice, a
scratched hitter must void rather than lose, units must be paid at the price the
note showed, and the model-versus-market comparison must be scored only where the
vig could actually be stripped.
"""

from __future__ import annotations

from datetime import date as Date

from mlb_engine.audit import power_ledger
from mlb_engine.data.results import GameResult, PlayerLine
from mlb_engine.market.tiers import Tier
from mlb_engine.output import power_board, power_report
from mlb_engine.recommendations import Recommendation
from tests.test_power_board import _pid, _rec
from tests.test_power_screen import _result

DAY = Date(2026, 8, 17)


def _position(
    stat: str = "HRR",
    line: float = 1.5,
    *,
    batter: str = "Matt Olson",
    player_id: int | None = 7,
    game_pk: int | None = 1,
    side: str = "over",
    odds: float | None = 100.0,
    model: float = 0.60,
    bet: float | None = None,
    fair: float | None = 0.50,
    tier: str = "Moderate buy",
    rating: str = "BUY",
    devigged: bool = True,
) -> power_ledger.Position:
    return power_ledger.Position(
        date=DAY.isoformat(),
        batter=batter,
        player_id=player_id,
        game_pk=game_pk,
        stat=stat,
        line=line,
        side=side,
        book="DraftKings",
        odds=odds,
        model_prob=model,
        bet_prob=bet,
        fair_prob=fair,
        edge=None if fair is None else round(model - fair, 4),
        ev=0.05,
        tier=tier,
        rating=rating,
        devigged=devigged,
    )


def _line(**batting: int) -> PlayerLine:
    base = {"PA": 4, "H": 0, "1B": 0, "2B": 0, "3B": 0, "HR": 0, "R": 0, "RBI": 0}
    base.update(batting)
    return PlayerLine(batting=base)


def _game(players: dict[int, PlayerLine], *, final: bool = True) -> GameResult:
    return GameResult(
        game_pk=1,
        final=final,
        home_runs=4,
        away_runs=2,
        f5_home=2,
        f5_away=1,
        players=players,
    )


# --- the record -----------------------------------------------------------


def test_the_board_is_recorded_with_the_price_it_was_shown_at(tmp_path) -> None:
    result = _result()
    pid = _pid(result)
    board = power_board.build(
        result, [_rec("Matt Olson", "TB", 1.5, player_id=pid, american=255.0)]
    )
    positions = power_ledger.positions_from_board(board, DAY, power_report.ratings(result))

    assert [p.batter for p in positions] == ["Matt Olson"]
    p = positions[0]
    assert (p.odds, p.stat, p.line, p.player_id, p.game_pk) == (255.0, "TB", 1.5, pid, 1)
    assert p.rating in ("BUY", "HOLD", "AVOID")

    path = tmp_path / power_ledger.LEDGER_NAME
    power_ledger.record(path, positions, DAY)
    assert power_ledger.load(path) == positions


def test_rerunning_the_screen_replaces_the_day_instead_of_doubling_it(tmp_path) -> None:
    path = tmp_path / power_ledger.LEDGER_NAME
    power_ledger.record(path, [_position(), _position("TB", 1.5)], DAY)
    power_ledger.record(path, [_position(odds=-110.0)], DAY)

    kept = power_ledger.positions_for(path, DAY)
    assert len(kept) == 1
    assert kept[0].odds == -110.0


def test_an_earlier_day_survives_a_later_recording(tmp_path) -> None:
    path = tmp_path / power_ledger.LEDGER_NAME
    power_ledger.record(path, [_position()], DAY)
    later = _position(batter="Drake Baldwin")
    power_ledger.record(
        path, [power_ledger.Position(**{**later.__dict__, "date": "2026-08-18"})], Date(2026, 8, 18)
    )

    assert len(power_ledger.positions_for(path, DAY)) == 1
    assert len(power_ledger.positions_for(path, Date(2026, 8, 18))) == 1
    assert len(power_ledger.load(path)) == 2


def test_the_homer_is_shown_on_the_board_and_held_by_nobody(tmp_path) -> None:
    """The screen's worst market is also its most eye-catching one.

    Graded, its HR rows went 2-13 for -7.3 units while every other market
    together lost 3.1, and no price band rescues them: the book's home-run overs
    lose 34.5% above +300 and more the longer the price, because the quote is
    one-way and the edge measured against it is mostly the hold. The arsenal work
    is still the reason to watch the hitter, so the row stays on the board and out
    of the record.
    """
    result = _result()
    pid = _pid(result)
    board = power_board.build(
        result,
        [
            _rec("Matt Olson", "HR", 0.5, player_id=pid, american=480.0, ev=0.40),
            _rec("Matt Olson", "TB", 1.5, player_id=pid, ev=0.04),
        ],
    )
    positions = power_ledger.positions_from_board(board, DAY)

    assert [r.stat for r in board.rows] == ["HR", "TB"]
    assert [p.stat for p in positions] == ["TB"]


def test_a_homer_already_written_down_stops_scoring_the_screen(tmp_path) -> None:
    """Filtered on the way out too, so an old board grades like a new one."""
    path = tmp_path / power_ledger.LEDGER_NAME
    power_ledger.record(path, [_position("HR", 0.5, odds=480.0), _position("TB", 1.5)], DAY)

    assert [p.stat for p in power_ledger.load(path)] == ["HR", "TB"]
    assert [p.stat for p in power_ledger.positions_for(path, DAY)] == ["TB"]


def test_an_absent_ledger_reads_as_empty(tmp_path) -> None:
    assert power_ledger.load(tmp_path / "nothing.csv") == []
    assert power_ledger.positions_for(tmp_path / "nothing.csv", DAY) == []


# --- grading --------------------------------------------------------------


def test_the_summed_markets_are_derived_rather_than_looked_up() -> None:
    res = _game({7: _line(PA=5, H=2, **{"1B": 1, "2B": 1}, R=1, RBI=0)})
    graded, voided = power_ledger.grade_positions([_position("HRR", 2.5), _position("TB", 2.5)], {1: res})

    assert voided == 0
    assert [(g.position.stat, g.actual, g.result) for g in graded] == [
        ("HRR", 3, "win"),
        ("TB", 3, "win"),
    ]


def test_units_are_paid_at_the_recorded_price() -> None:
    res = _game({7: _line(H=1, R=1)})
    graded, _ = power_ledger.grade_positions(
        [_position("HRR", 1.5, odds=255.0), _position("HR", 0.5, odds=255.0)], {1: res}
    )

    by_stat = {g.position.stat: g for g in graded}
    assert by_stat["HRR"].result == "win"
    assert by_stat["HRR"].units == 2.55
    assert by_stat["HR"].result == "loss"
    assert by_stat["HR"].units == -1.0


def test_a_line_landed_exactly_pushes_and_pays_nothing() -> None:
    res = _game({7: _line(H=1, R=1)})
    graded, _ = power_ledger.grade_positions([_position("HRR", 2.0)], {1: res})

    assert (graded[0].result, graded[0].units) == ("push", 0.0)


def test_a_hitter_who_never_batted_is_voided_not_lost() -> None:
    res = _game({7: PlayerLine()})
    graded, voided = power_ledger.grade_positions([_position()], {1: res})

    assert (graded, voided) == ([], 1)


def test_an_unfinished_game_is_voided() -> None:
    res = _game({7: _line(H=3, R=2, RBI=2)}, final=False)
    graded, voided = power_ledger.grade_positions([_position()], {1: res})

    assert (graded, voided) == ([], 1)


def test_a_position_with_no_box_score_is_voided() -> None:
    graded, voided = power_ledger.grade_positions([_position()], {})

    assert (graded, voided) == ([], 1)


# --- the scorecard --------------------------------------------------------


def _graded(*positions: power_ledger.Position, players: dict[int, PlayerLine]) -> list:
    graded, _ = power_ledger.grade_positions(list(positions), {1: _game(players)})
    return graded


def test_the_scorecard_counts_the_record_and_the_units() -> None:
    players = {7: _line(H=2, R=1, RBI=1, **{"1B": 2})}
    graded = _graded(
        _position("HRR", 1.5, odds=100.0),
        _position("HR", 0.5, odds=255.0),
        _position("TB", 1.5, odds=-110.0),
        players=players,
    )
    card = power_ledger.scorecard(DAY, graded, voided=2)

    assert (card.overall.wins, card.overall.losses) == (2, 1)
    assert card.overall.units == 0.9091  # +1.00 even money, -1.00 on the homer, +0.909 at -110
    assert card.voided == 2
    assert card.overall.win_pct == 2 / 3
    assert card.day == DAY.isoformat()


def test_the_model_is_scored_against_the_no_vig_line_only_where_it_was_devigged() -> None:
    players = {7: _line(H=1, R=1)}
    graded = _graded(
        # Won, and the model was more confident than the price: model wins here.
        _position("HRR", 1.5, model=0.70, fair=0.50),
        # One-sided, so it carries no honest market probability and is excluded.
        _position("HR", 0.5, model=0.90, fair=None, devigged=False),
        players=players,
    )
    card = power_ledger.scorecard(DAY, graded)

    assert card.scored_probs == 1
    assert card.model_brier == round((0.70 - 1) ** 2, 4)
    assert card.market_brier == round((0.50 - 1) ** 2, 4)
    assert card.model_beat_market is True
    assert card.mean_model_prob == 0.70


def test_an_overconfident_loser_hands_the_comparison_to_the_market() -> None:
    graded = _graded(_position("HR", 0.5, model=0.60, fair=0.30), players={7: _line(H=1)})
    card = power_ledger.scorecard(DAY, graded)

    assert card.model_beat_market is False


def test_no_devigged_row_leaves_the_comparison_unanswered_rather_than_tied() -> None:
    graded = _graded(
        _position("HR", 0.5, fair=None, devigged=False), players={7: _line(H=1, R=1)}
    )
    card = power_ledger.scorecard(DAY, graded)

    assert (card.model_brier, card.market_brier, card.model_beat_market) == (None, None, None)


def test_the_tier_and_rating_cuts_are_reported_separately() -> None:
    players = {7: _line(H=1, R=1)}
    graded = _graded(
        _position("HRR", 1.5, tier="Moderate buy", rating="BUY"),
        _position("HR", 0.5, tier="Pass", rating="AVOID"),
        players=players,
    )
    card = power_ledger.scorecard(DAY, graded)

    tiers = {r.label: (r.wins, r.losses) for r in card.by_tier}
    ratings = {r.label: (r.wins, r.losses) for r in card.by_rating}
    markets = {r.label: (r.wins, r.losses) for r in card.by_market}
    assert tiers == {"Moderate buy": (1, 0), "Pass": (0, 1)}
    assert ratings == {"BUY": (1, 0), "AVOID": (0, 1)}
    assert markets == {"H+R+RBI": (1, 0), "HR": (0, 1)}


def test_an_empty_day_scores_to_nothing_rather_than_to_zero_wins() -> None:
    card = power_ledger.scorecard(DAY, [], voided=3)

    assert card.graded == 0
    assert card.overall.win_pct is None
    assert card.overall.roi is None
    assert card.voided == 3


# --- the note -------------------------------------------------------------


def test_the_note_prints_yesterdays_record_above_todays_board() -> None:
    result = _result()
    graded = _graded(_position("HRR", 1.5, model=0.70), players={7: _line(H=1, R=1)})
    card = power_ledger.scorecard(DAY, graded)

    doc = power_report.render_html(result, review=(card, graded))

    assert "Yesterday&#x27;s board, graded" in doc or "Yesterday's board, graded" in doc
    assert DAY.isoformat() in doc
    assert "Brier" in doc
    assert "H+R+RBI o1.5" in doc


def test_a_day_with_nothing_gradeable_says_so_rather_than_showing_a_blank_table() -> None:
    result = _result()
    card = power_ledger.scorecard(DAY, [], voided=4)

    doc = power_report.render_html(result, review=(card, []))

    assert "could be graded" in doc
    assert "4 rows voided" in doc


def test_a_note_with_no_review_is_unchanged() -> None:
    result = _result()

    assert "graded" not in power_report.render_html(result)


def test_a_row_recorded_before_the_anchor_shows_the_model_it_was_shown_at() -> None:
    assert _position(model=0.60, bet=None).shown_prob == 0.60
    assert _position(model=0.60, bet=0.54).shown_prob == 0.54


def test_the_anchored_probability_survives_the_round_trip(tmp_path) -> None:
    path = tmp_path / "power.csv"
    power_ledger.record(path, [_position(model=0.60, bet=0.54)], DAY)

    (back,) = power_ledger.load(path)

    assert back.model_prob == 0.60
    assert back.bet_prob == 0.54


def test_the_printed_number_is_scored_beside_the_model_not_instead_of_it() -> None:
    # Lost, and the anchored number was the more modest of the two, so the
    # anchored score must be the better one and both must be reported.
    graded = _graded(_position("HR", 0.5, model=0.60, bet=0.45, fair=0.30), players={7: _line(H=1)})
    card = power_ledger.scorecard(DAY, graded)

    assert card.model_brier == round(0.60**2, 4)
    assert card.shown_brier == round(0.45**2, 4)
    assert card.mean_shown_prob == 0.45
    assert card.shown_beat_market is False


def test_the_note_prints_the_number_the_card_bet_not_the_raw_model() -> None:
    result = _result()
    graded = _graded(
        _position("HRR", 1.5, model=0.70, bet=0.61), players={7: _line(H=1, R=1)}
    )
    card = power_ledger.scorecard(DAY, graded)

    doc = power_report.render_html(result, review=(card, graded))

    assert "61.0%" in doc
    assert "70.0%" not in doc


def test_the_grade_is_lettered_and_never_called_a_buy() -> None:
    doc = power_report.render_html(_result())

    assert "MATCHUP " in doc
    assert "rate a buy on the matchup" not in doc


def test_the_ratings_helper_names_every_survivor() -> None:
    result = _result()
    rated = power_report.ratings(result)

    assert set(rated) == {v.line.name for s in result.sections for v in s.hitters}
    assert set(rated.values()) <= {"BUY", "HOLD", "AVOID"}


def test_a_recorded_position_knows_whether_the_card_bet_it() -> None:
    assert _position(tier="Strong buy").is_buy is True
    assert _position(tier="Pass").is_buy is False


def test_only_priced_rows_reach_the_ledger() -> None:
    result = _result()
    pid = _pid(result)
    unpriced: Recommendation = _rec(
        "Matt Olson", "HR", 0.5, player_id=pid, american=None, tier=Tier.PASS
    )
    board = power_board.build(result, [unpriced])

    assert power_ledger.positions_from_board(board, DAY) == []
    assert board.unpriced == ["Matt Olson"]
