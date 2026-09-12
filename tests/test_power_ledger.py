"""The power screen's own ledger, and the scorecard it grades out of it.

The point of the record is that it cannot flatter the screen, so the tests pin
the ways it could: a re-run must overwrite the day rather than count it twice, a
scratched hitter must void rather than lose, units must be paid at the price the
note showed, and the model-versus-market comparison must be scored only where the
vig could actually be stripped.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, replace
from datetime import date as Date

from mlb_engine.audit import power_ledger
from mlb_engine.data.results import GameResult, PlayerLine
from mlb_engine.features import arm
from mlb_engine.features.arm import ArmProfile
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
    delivery: str = "",
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
        delivery=delivery,
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


def test_the_labels_follow_the_money() -> None:
    """The bucket with the best ROI on enough rows is the Strong Buy; the rest are Buys."""
    rec = power_ledger.Record
    records = {
        "STRONG BUY": rec("STRONG BUY", wins=2, losses=10, units=-8.83),
        "BUY": rec("BUY", wins=50, losses=79, units=-17.64),
        "HOLD": rec("HOLD", wins=127, losses=155, units=-15.80),
        "AVOID": rec("AVOID", wins=39, losses=36, units=7.26),
    }
    assert power_report.strong_bucket(records) == "AVOID"
    assert power_report.labels(records) == {
        "STRONG BUY": "BUY", "BUY": "BUY", "HOLD": "BUY", "AVOID": "STRONG BUY",
    }

    # Once HOLD's record is the best one, the word moves with it.
    records["HOLD"] = rec("HOLD", wins=160, losses=122, units=30.0)
    assert power_report.strong_bucket(records) == "HOLD"

    # A bucket under LABEL_MIN_ROWS cannot take the label however good its ROI.
    records["STRONG BUY"] = rec("STRONG BUY", wins=11, losses=1, units=9.5)
    assert power_report.strong_bucket(records) == "HOLD"

    # No record: the default bucket, and the note prints buckets beside the words.
    assert power_report.strong_bucket(None) == power_report.DEFAULT_STRONG_BUCKET
    doc = power_report.render_html(_result())
    assert "MATCHUP " not in doc
    assert "(contact " in doc or "(rank 1-2)" in doc


def test_each_grade_gets_its_whole_ledger_record() -> None:
    """One record per grade string, hitters only, wins, losses and units."""
    players = {7: _line(H=1, R=1, **{"1B": 1})}
    graded = _graded(
        _position("HRR", 1.5, rating="STRONG BUY", odds=100.0),
        _position("HR", 0.5, rating="STRONG BUY", odds=300.0),
        _position("TB", 0.5, rating="AVOID", odds=-110.0),
        _position("H", 0.5, rating="", odds=-120.0),
        players=players,
    )
    records = power_ledger.records_by_rating(graded)

    assert set(records) == {"STRONG BUY", "AVOID"}
    assert (records["STRONG BUY"].wins, records["STRONG BUY"].losses) == (1, 1)
    assert records["STRONG BUY"].units == 0.0
    assert (records["AVOID"].wins, records["AVOID"].losses) == (1, 0)
    assert abs(records["AVOID"].units - 100 / 110) < 1e-3


def test_the_note_prints_each_grades_record_beside_the_label() -> None:
    """The labels stay; what each has been worth on the ledger sits next to them."""
    result = _result()
    grade = next(iter(power_report.ratings(result).values()))
    records = {grade: power_ledger.Record(grade, wins=5, losses=16, units=-11.89)}

    doc = power_report.render_html(result, grade_records=records)

    assert "grade's ledger record" in doc
    assert "5-16" in doc
    assert "-11.89u" in doc
    assert "-57% ROI" in doc
    assert "What each bucket has been worth" in doc
    assert "no record yet" in doc or len(set(power_report.ratings(result).values())) == 1
    assert "grade's ledger record" not in power_report.render_html(result)


def test_the_ratings_helper_names_every_survivor() -> None:
    result = _result()
    rated = power_report.ratings(result)

    assert set(rated) == {v.line.name for s in result.sections for v in s.hitters}
    assert set(rated.values()) <= {"BUY", "HOLD", "AVOID"}


def _armed(pvelo: float) -> ArmProfile:
    """A measured delivery, thrown at the perceived velocity asked for."""
    return ArmProfile(pitches=500, pvelo=pvelo)


def test_the_delivery_verdict_follows_the_arm_down_to_the_hitters_row() -> None:
    """The flag is the starter's and the ledger's rows are hitters, so it rides
    down to the bat that faces him -- otherwise it can never be graded."""
    result = _result()
    starter = result.sections[0].starter
    starter.index = 1.0
    starter.arm = _armed(99.0)  # above league under a soft read: the delivery argues
    assert power_report.deliveries(result)[result.sections[0].hitters[0].line.name] == (
        arm.CONTRADICTED
    )

    starter.arm = _armed(88.0)
    board = power_board.build(result, [_rec("Matt Olson", "TB", 1.5, player_id=_pid(result))])
    (p,) = power_ledger.positions_from_board(
        board, DAY, power_report.ratings(result), power_report.deliveries(result)
    )
    assert p.delivery == arm.CONFIRMED


def test_an_arm_nobody_measured_is_recorded_unmeasured_and_not_as_agreement() -> None:
    result = _result()
    result.sections[0].starter.arm = ArmProfile(pitches=0)
    assert set(power_report.deliveries(result).values()) == {arm.UNMEASURED}


def test_an_arm_with_no_profile_at_all_leaves_the_field_blank() -> None:
    """No profile is no reading; the row says nothing rather than the common case."""
    result = _result()
    for section in result.sections:
        section.starter.arm = None
    assert power_report.deliveries(result) == {}

    board = power_board.build(result, [_rec("Matt Olson", "TB", 1.5, player_id=_pid(result))])
    (p,) = power_ledger.positions_from_board(board, DAY, deliveries={})
    assert p.delivery == ""


def test_the_delivery_verdict_survives_the_round_trip(tmp_path) -> None:
    path = tmp_path / "power.csv"
    power_ledger.record(path, [_position(delivery=arm.CONTRADICTED)], DAY)

    (back,) = power_ledger.load(path)

    assert back.delivery == arm.CONTRADICTED


def test_a_row_recorded_before_the_flag_existed_still_loads(tmp_path) -> None:
    path = tmp_path / "power.csv"
    power_ledger.record(path, [_position()], DAY)
    old = path.read_text().splitlines()
    header = old[0].split(",")
    drop = header.index("delivery")
    path.write_text(
        "\n".join(
            ",".join(v for i, v in enumerate(row.split(",")) if i != drop) for row in old
        )
        + "\n"
    )

    (back,) = power_ledger.load(path)

    assert back.delivery == ""
    assert back == _position()


def test_a_recorded_position_knows_whether_the_card_bet_it() -> None:
    assert _position(tier="Strong buy").is_buy is True
    assert _position(tier="Pass").is_buy is False


def test_a_display_only_row_recorded_before_the_hold_is_not_a_ticket() -> None:
    """Rows in a display-only market stopped being written, but the ones already
    in the ledger carry a buy tier and must not roll up as bets the card took."""
    assert _position("HR", 0.5, tier="Strong buy").is_buy is False
    assert _position("HR", 0.5, tier="Moderate buy").is_buy is False


def test_only_priced_rows_reach_the_ledger() -> None:
    result = _result()
    pid = _pid(result)
    unpriced: Recommendation = _rec(
        "Matt Olson", "HR", 0.5, player_id=pid, american=None, tier=Tier.PASS
    )
    board = power_board.build(result, [unpriced])

    assert power_ledger.positions_from_board(board, DAY) == []
    assert board.unpriced == ["Matt Olson"]


# --- the run, and the ordering it captured --------------------------------


def test_a_second_run_of_a_day_is_kept_beside_the_first(tmp_path) -> None:
    """Two captures of a day are two boards, and the file loses neither.

    The morning run and the re-run once lineups post showed different rows at
    different prices, and replacing the day meant the record depended on which
    ran last -- on this machine and, through the state branch, across machines.
    """
    path = tmp_path / power_ledger.LEDGER_NAME
    power_ledger.record(path, [_position()], DAY, run_id="20260817T1500Z")
    power_ledger.record(path, [_position(odds=-110.0)], DAY, run_id="20260817T2200Z")

    assert len(power_ledger.load(path)) == 2
    assert power_ledger.runs_for(path, DAY) == ["20260817T1500Z", "20260817T2200Z"]


def test_the_day_grades_its_last_board_and_can_be_pinned_to_an_earlier_one(tmp_path) -> None:
    path = tmp_path / power_ledger.LEDGER_NAME
    power_ledger.record(path, [_position()], DAY, run_id="20260817T1500Z")
    power_ledger.record(path, [_position(odds=-110.0)], DAY, run_id="20260817T2200Z")

    (last,) = power_ledger.positions_for(path, DAY)
    assert (last.run_id, last.odds) == ("20260817T2200Z", -110.0)

    (pinned,) = power_ledger.positions_for(path, DAY, "20260817T1500Z")
    assert (pinned.run_id, pinned.odds) == ("20260817T1500Z", 100.0)


def test_rerunning_one_run_overwrites_itself_rather_than_doubling_it(tmp_path) -> None:
    """Append-only across runs, idempotent within one."""
    path = tmp_path / power_ledger.LEDGER_NAME
    power_ledger.record(path, [_position(), _position("TB", 1.5)], DAY, run_id="r1")
    power_ledger.record(path, [_position(odds=-110.0)], DAY, run_id="r1")

    kept = power_ledger.positions_for(path, DAY)
    assert [p.odds for p in kept] == [-110.0]


def test_the_scorecard_names_the_run_it_graded(tmp_path) -> None:
    graded, voided = power_ledger.grade_positions(
        [replace(_position(), run_id="20260817T2200Z")],
        {1: _game({7: _line(H=2, **{"1B": 1, "2B": 1})})},
    )
    card = power_ledger.scorecard(DAY, graded, voided)
    assert card.run_id == "20260817T2200Z"


def _composite(rank: int = 1) -> power_ledger.Composite:
    return power_ledger.Composite(rank=rank, points=18, fit_pts=11, fit_rv=3.6)


def test_the_ordering_is_recorded_with_the_row_so_it_can_be_graded() -> None:
    """The screen's claim is the order it put the bats in, and until now the only
    thing surviving to the CSV was the tier, so the claim could not be scored."""
    result = _result()
    board = power_board.build(result, [_rec("Matt Olson", "TB", 1.5, player_id=_pid(result))])

    (p,) = power_ledger.positions_from_board(
        board, DAY, composites={"Matt Olson": _composite()}, run_id="r1"
    )

    assert (p.rank, p.points, p.fit_pts, p.fit_rv) == (1, 18, 11, 3.6)
    assert p.run_id == "r1"


def test_the_ordering_survives_the_round_trip(tmp_path) -> None:
    path = tmp_path / power_ledger.LEDGER_NAME
    written = replace(
        _position(), run_id="r1", rank=3, points=12, fit_pts=5, fit_rv=-1.25
    )
    power_ledger.record(path, [written], DAY, run_id="r1")

    (back,) = power_ledger.load(path)

    assert back == written


def test_a_row_recorded_before_the_run_and_the_ordering_existed_still_loads(tmp_path) -> None:
    path = tmp_path / power_ledger.LEDGER_NAME
    power_ledger.record(path, [_position()], DAY)
    old = path.read_text().splitlines()
    header = old[0].split(",")
    drop = {header.index(c) for c in ("run_id", "rank", "points", "fit_pts", "fit_rv")}
    path.write_text(
        "\n".join(
            ",".join(v for i, v in enumerate(row.split(",")) if i not in drop) for row in old
        )
        + "\n"
    )

    (back,) = power_ledger.load(path)

    assert (back.run_id, back.rank, back.points, back.fit_pts) == ("", None, None, None)
    assert back.fit_rv is None
    assert back == _position()


# --- one hitter, one key --------------------------------------------------


def test_an_accent_is_not_a_second_hitter() -> None:
    assert power_ledger.name_key("Eugenio Suárez") == power_ledger.name_key("Eugenio Suarez")
    assert power_ledger.name_key("Ronald Acuña Jr.") != power_ledger.name_key("Ronald Acuna")


def test_a_rating_reaches_the_row_whichever_source_spelled_the_name() -> None:
    """The lineup feed accents him and the box score does not, and the row must
    still carry the note's rating rather than falling back to blank."""
    result = _result()
    board = power_board.build(result, [_rec("Matt Olson", "TB", 1.5, player_id=_pid(result))])

    (p,) = power_ledger.positions_from_board(
        board,
        DAY,
        ratings={"matt olson": "AVOID"},
        composites={"MATT OLSON": _composite(rank=4)},
    )

    assert (p.rating, p.rank) == ("AVOID", 4)


def test_a_row_is_keyed_by_the_hitter_and_not_by_the_spelling() -> None:
    accented = replace(_position(), batter="Eugenio Suárez", player_id=None)
    plain = replace(_position(), batter="Eugenio Suarez", player_id=None)
    assert accented.key == plain.key


# --- the arms' positions ---------------------------------------------------


def _arm_position(stat: str = "BB", line: float = 1.5, *, side: str = "over", **kw):
    return replace(
        _position(stat, line, batter="Bailey Ober", player_id=641927, side=side, rating="", **kw),
        category="pitcher",
        gate="prob_floor",
    )


def _arm_line(**pitching: int) -> PlayerLine:
    base = {"BF": 22, "outs": 15, "K": 4, "BB": 1, "H": 5, "ER": 2}
    base.update(pitching)
    return PlayerLine(pitching=base)


def test_an_arms_row_is_graded_off_his_pitching_line() -> None:
    res = _game({641927: _arm_line(BB=2)})
    (g,), voided = power_ledger.grade_positions([_arm_position()], {1: res})
    assert voided == 0
    assert g.result == power_ledger.WIN
    assert g.actual == 2
    assert g.units == 1.0


def test_an_arm_who_never_pitched_is_voided_not_lost() -> None:
    res = _game({641927: PlayerLine()})
    graded, voided = power_ledger.grade_positions([_arm_position()], {1: res})
    assert graded == []
    assert voided == 1


def test_the_arm_and_the_gate_survive_the_round_trip(tmp_path) -> None:
    path = tmp_path / "power.csv"
    power_ledger.record(path, [_position(), _arm_position()], DAY)

    hitter, arm = power_ledger.load(path)

    assert hitter.category == "batter" and hitter.gate == ""
    assert arm.category == "pitcher" and arm.gate == "prob_floor"
    assert arm.market == "SP BB"
    assert arm.label == "SP BB o1.5"


def test_a_ledger_written_before_the_arms_reads_every_row_as_a_hitter(tmp_path) -> None:
    path = tmp_path / "power.csv"
    old = [f for f in power_ledger.FIELDS if f not in ("category", "gate")]
    row = {k: v for k, v in asdict(_position()).items() if k in old}
    row["devigged"] = "1"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=old)
        w.writeheader()
        w.writerow({k: "" if v is None else v for k, v in row.items()})

    (back,) = power_ledger.load(path)
    assert back.category == "batter"
    assert back.gate == ""


def test_the_scorecard_splits_the_two_halves_of_the_screen() -> None:
    players = {7: _line(H=2, R=1), 641927: _arm_line(BB=0)}
    graded, _ = power_ledger.grade_positions(
        [_position(), _arm_position()], {1: _game(players)}
    )
    card = power_ledger.scorecard(DAY, graded)
    assert [(r.label, r.wins, r.losses) for r in card.by_category] == [
        ("hitters", 1, 0),
        ("arms", 0, 1),
    ]
    assert [r.label for r in card.by_market] == ["H+R+RBI", "SP BB"]


def test_the_arms_rows_are_recorded_beside_the_hitters() -> None:
    result = _result()
    hitter = _rec("Matt Olson", "HRR", 1.5, player_id=_pid(result))
    arm = _rec("Bailey Ober", "K", 5.5, player_id=641927, tier=Tier.PASS, ev=0.02)
    arm.category = "pitcher"
    arm.pass_gate = "edge_ceiling"
    board = power_board.build(result, [hitter, arm])
    positions = power_ledger.positions_from_board(
        board,
        DAY,
        power_report.ratings(result),
        {**power_report.deliveries(result), **power_report.arm_deliveries(result)},
        power_report.composites(result),
        "run",
    )
    assert [(p.batter, p.category, p.gate) for p in positions] == [
        ("Matt Olson", "batter", ""),
        ("Bailey Ober", "pitcher", "edge_ceiling"),
    ]
    assert positions[1].rating == ""


def test_the_arm_tier_survives_the_round_trip_and_old_rows_are_soft(tmp_path) -> None:
    path = tmp_path / "power.csv"
    elite = replace(_position(batter="Juan Soto"), arm_tier="elite")
    power_ledger.record(path, [_position(), elite], DAY)
    soft, back = power_ledger.load(path)
    assert soft.arm_tier == "" and back.arm_tier == "elite"

    old = [f for f in power_ledger.FIELDS if f != "arm_tier"]
    row = {k: v for k, v in asdict(_position()).items() if k in old}
    row["devigged"] = "1"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=old)
        w.writeheader()
        w.writerow({k: "" if v is None else v for k, v in row.items()})
    (legacy,) = power_ledger.load(path)
    assert legacy.arm_tier == ""


def test_the_scorecard_grades_soft_and_elite_arms_apart() -> None:
    soft = _position()
    elite = replace(_position(batter="Juan Soto", player_id=8), arm_tier="elite")
    players = {7: _line(H=2, R=1), 8: _line(H=0, R=0)}
    graded, _ = power_ledger.grade_positions([soft, elite], {1: _game(players)})
    card = power_ledger.scorecard(DAY, graded)
    assert [(r.label, r.wins, r.losses) for r in card.by_arm_tier] == [
        ("elite arms", 0, 1),
        ("soft arms", 1, 0),
    ]
    only_soft, _ = power_ledger.grade_positions([soft], {1: _game(players)})
    assert power_ledger.scorecard(DAY, only_soft).by_arm_tier == []


def test_positions_from_board_tag_the_tier_of_arm() -> None:
    result = _result()
    board = power_board.build(result, [_rec("Matt Olson", "HRR", 1.5, player_id=_pid(result))])
    (pos,) = power_ledger.positions_from_board(
        board, DAY, power_report.ratings(result), {}, None, "run", arm_tier="elite"
    )
    assert pos.arm_tier == "elite"
