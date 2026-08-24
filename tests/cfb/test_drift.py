"""Line movement: the sign conventions, the CLV bug it fixes, and the gate."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from cfb_engine.audit import snapshot
from cfb_engine.audit.clv import ClosingQuote, compute_clv
from cfb_engine.audit.snapshot import SideQuote
from cfb_engine.config import Config
from cfb_engine.data.cfbd import CFBDClient
from cfb_engine.market import keys
from cfb_engine.market.board import GameOdds
from cfb_engine.market.drift import DriftGate
from cfb_engine.market.ev import MarketQuote
from cfb_engine.market.linevalue import drift_probability, line_drift_points, value_points
from cfb_engine.pipeline import Pipeline
from cfb_engine.schemas import Game, Slate, TeamGameInfo

MATCHUP = "UGA @ ALA"

# -- keys ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ("ALA ML", "ALA"),
        ("ALA -7.0", "ALA"),
        ("ALA +7.0", "ALA"),
        ("ALA +0.0", "ALA"),
        ("Over 54.5", "Over"),
        ("Under 54.5", "Under"),
        ("Miami OH -3.5", "Miami OH"),  # abbrevs contain spaces
    ],
)
def test_side_of_strips_the_handicap_and_keeps_the_side(selection: str, expected: str) -> None:
    assert keys.side_of(selection) == expected


def test_side_of_leaves_a_selection_with_no_handicap_alone() -> None:
    assert keys.side_of("Georgia") == "Georgia"


def test_both_ends_of_a_moved_spread_share_one_key() -> None:
    assert snapshot.key(MATCHUP, "game_ats", keys.game_ats("ALA", -7.0)) == snapshot.key(
        MATCHUP, "game_ats", keys.game_ats("ALA", -7.5)
    )


def test_two_games_totals_do_not_share_one_key() -> None:
    """Every game's over is called "Over", so the key has to name the game: without
    the matchup one game's total became the baseline for the whole board."""
    over = keys.game_total(True, 55.0)
    assert snapshot.key(MATCHUP, "game_total", over) != snapshot.key(
        "OSU @ MICH", "game_total", over
    )


# -- sign conventions ----------------------------------------------------


def test_laying_fewer_points_is_the_better_number() -> None:
    assert value_points("game_ats", "cover", -7.0, -7.5) == 0.5
    assert value_points("game_ats", "cover", -7.5, -7.0) == -0.5


def test_a_dog_taking_more_points_is_the_better_number() -> None:
    assert value_points("game_ats", "cover", 7.5, 7.0) == 0.5
    assert value_points("game_ats", "cover", 7.0, 7.5) == -0.5


def test_over_wants_the_lower_total_and_under_the_higher() -> None:
    assert value_points("game_total", "over", 54.5, 55.0) == 0.5
    assert value_points("game_total", "over", 55.0, 54.5) == -0.5
    assert value_points("game_total", "under", 55.0, 54.5) == 0.5
    assert value_points("game_total", "under", 54.5, 55.0) == -0.5


def test_a_moneyline_has_no_handicap_to_value() -> None:
    assert value_points("game_ml", "win", None, None) is None
    assert value_points("game_ml", "win", -7.0, -7.5) is None


def test_drift_is_positive_when_the_market_comes_to_our_side() -> None:
    # -7 -> -7.5: the market thinks more of Alabama than it did.
    assert line_drift_points("game_ats", "cover", -7.0, -7.5) == 0.5
    # The other end of that same move is the away side losing opinion.
    assert line_drift_points("game_ats", "cover", 7.0, 7.5) == -0.5
    # 54.5 -> 55.0: the market came to the over.
    assert line_drift_points("game_total", "over", 54.5, 55.0) == 0.5
    assert line_drift_points("game_total", "under", 54.5, 55.0) == -0.5


def test_a_point_of_spread_is_worth_more_probability_than_the_price_usually_moves() -> None:
    # The failure this whole module exists for: -110 to -110 across a half-point
    # move is real movement, and measuring only price calls it zero.
    drift = drift_probability(
        "game_ats",
        "cover",
        from_prob=0.5,
        to_prob=0.5,
        from_line=-7.0,
        to_line=-7.5,
        margin_sd=16.0,
        total_sd=13.0,
    )
    assert drift is not None and drift == pytest.approx(0.0125, abs=1e-4)


def test_moneyline_drift_is_pure_price() -> None:
    drift = drift_probability(
        "game_ml", "win", from_prob=0.50, to_prob=0.57, margin_sd=16.0, total_sd=13.0
    )
    assert drift == pytest.approx(0.07, abs=1e-6)


def test_drift_is_none_without_a_baseline() -> None:
    assert (
        drift_probability(
            "game_ml", "win", from_prob=None, to_prob=0.57, margin_sd=16.0, total_sd=13.0
        )
        is None
    )


# -- the CLV bug ---------------------------------------------------------


def _closing_ats(line: float) -> dict[str, ClosingQuote]:
    sel = keys.game_ats("ALA", line)
    return {snapshot.key(MATCHUP, "game_ats", sel): SideQuote(-110.0, 0.50, line)}


def test_clv_is_found_when_the_spread_moved() -> None:
    """The regression: an entry at -7 against a close of -7.5 used to return nothing."""
    res = compute_clv(
        MATCHUP,
        "game_ats",
        keys.game_ats("ALA", -7.0),
        -110.0,
        0.48,
        _closing_ats(-7.5),
        bet_line=-7.0,
        side="cover",
    )
    assert res.close_odds == -110.0
    assert res.clv_pts == 0.5
    # We hold the better number, so our side's closing probability is above the
    # 0.500 the -7.5 close was priced at, and the CLV is positive.
    assert res.close_prob is not None and res.close_prob > 0.50
    assert res.clv is not None and res.clv > 0.0


def test_clv_is_negative_when_the_spread_moved_against_us() -> None:
    res = compute_clv(
        MATCHUP,
        "game_ats",
        keys.game_ats("ALA", -7.5),
        -110.0,
        0.50,
        _closing_ats(-7.0),
        bet_line=-7.5,
        side="cover",
    )
    assert res.clv_pts == -0.5
    assert res.close_prob is not None and res.close_prob < 0.50
    assert res.clv is not None and res.clv < 0.0


def test_clv_on_an_unmoved_line_is_pure_price() -> None:
    res = compute_clv(
        MATCHUP,
        "game_ats",
        keys.game_ats("ALA", -7.0),
        -110.0,
        0.48,
        _closing_ats(-7.0),
        bet_line=-7.0,
        side="cover",
    )
    assert res.clv_pts == 0.0
    assert res.close_prob == 0.50
    assert res.clv == pytest.approx(0.02)


def test_clv_totals_respect_the_over_under_direction() -> None:
    close = {
        snapshot.key(MATCHUP, "game_total", keys.game_total(True, 55.0)): SideQuote(
            -110.0, 0.50, 55.0
        )
    }
    over = compute_clv(
        MATCHUP, "game_total", keys.game_total(True, 54.5), -110.0, 0.50, close,
        bet_line=54.5, side="over",
    )
    assert over.clv_pts == 0.5 and over.clv is not None and over.clv > 0

    under_close = {
        snapshot.key(MATCHUP, "game_total", keys.game_total(False, 55.0)): SideQuote(
            -110.0, 0.50, 55.0
        )
    }
    under = compute_clv(
        MATCHUP, "game_total", keys.game_total(False, 54.5), -110.0, 0.50, under_close,
        bet_line=54.5, side="under",
    )
    assert under.clv_pts == -0.5 and under.clv is not None and under.clv < 0


def test_clv_still_reads_a_legacy_selection_keyed_snapshot() -> None:
    legacy = {"game_ml|Georgia ML": SideQuote(-150.0, 0.60)}
    res = compute_clv(MATCHUP, "game_ml", "Georgia ML", -120.0, 0.55, legacy)
    assert res.close_odds == -150.0
    assert res.clv == pytest.approx(0.05)


def test_clv_is_empty_for_a_side_the_close_never_priced() -> None:
    assert compute_clv(MATCHUP, "game_ml", "Nobody ML", -120.0, 0.5, {}).as_tuple() == (
        None,
        None,
        None,
        None,
    )


# -- snapshot persistence ------------------------------------------------


def test_snapshot_round_trips_the_line(tmp_path: Path) -> None:
    path = tmp_path / "board.json"
    quotes = {snapshot.key(MATCHUP, "game_ats", "ALA -7.0"): SideQuote(-110.0, 0.5, -7.0)}
    snapshot.save(quotes, path)
    assert snapshot.load(path) == quotes


def test_loading_a_legacy_snapshot_recovers_the_line_from_the_key(tmp_path: Path) -> None:
    path = tmp_path / "closing.json"
    path.write_text('{"game_ats|ALA -7.5": {"american": -110.0, "no_vig_prob": 0.5}}')
    # A legacy key has no matchup to normalise onto, so it keeps its own shape and
    # compute_clv falls back to it; the line is recovered from the selection.
    assert snapshot.load(path) == {"game_ats|ALA -7.5": SideQuote(-110.0, 0.5, -7.5)}


def test_the_baseline_keeps_the_first_quote_and_the_close_keeps_the_last() -> None:
    home = snapshot.key(MATCHUP, "game_ml", "ALA ML")
    away = snapshot.key(MATCHUP, "game_ml", "UGA ML")
    first = {home: SideQuote(-110.0, 0.50, None)}
    later = {home: SideQuote(-140.0, 0.57, None), away: SideQuote(120.0, 0.43, None)}

    baseline = snapshot.merge_first_wins(first, later)
    assert baseline[home].american == -110.0  # earliest wins
    assert away in baseline  # a late arrival is still recorded

    close = snapshot.merge_last_wins(first, later)
    assert close[home].american == -140.0  # latest wins
    assert away in close


def test_snapshot_load_of_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert snapshot.load(tmp_path / "nope.json") == {}


# -- the gate ------------------------------------------------------------


def test_the_gate_measures_but_refuses_nothing_by_default() -> None:
    gate = DriftGate()
    keep, reason, which = gate.verdict(-0.05)
    assert keep is True
    assert which is None
    assert "moved" in reason


def test_no_baseline_is_neutral_and_silent() -> None:
    assert DriftGate(enabled=True).verdict(None) == (True, "", None)


def test_an_enabled_gate_refuses_a_large_adverse_move() -> None:
    keep, reason, which = DriftGate(enabled=True).verdict(-0.05)
    assert keep is False
    assert which == "clv_drift"
    assert "PASS" in reason


def test_an_enabled_gate_keeps_a_small_move() -> None:
    keep, _, which = DriftGate(enabled=True).verdict(-0.005)
    assert keep is True and which is None


def test_the_momentum_tail_is_separately_switchable() -> None:
    assert DriftGate(enabled=True).verdict(0.05)[0] is True  # off by default
    keep, _, which = DriftGate(enabled=True, momentum=True).verdict(0.05)
    assert keep is False and which == "momentum_run_up"


def test_from_env_reads_the_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CFBE_DRIFT_GATE", "1")
    monkeypatch.setenv("CFBE_DRIFT_MOMENTUM", "1")
    monkeypatch.setenv("CFBE_DRIFT_MAX_ADVERSE", "0.04")
    gate = DriftGate.from_env()
    assert gate.enabled and gate.momentum and gate.max_adverse == 0.04
    assert gate.verdict(-0.03)[0] is True  # inside the widened band


# -- the pipeline's baseline --------------------------------------------


def _slate_and_board() -> tuple[Slate, dict[str, GameOdds]]:
    game = Game(
        game_id="1",
        game_date=date(2026, 9, 5),
        home=TeamGameInfo(name="Alabama", abbrev="ALA", is_home=True),
        away=TeamGameInfo(name="Georgia", abbrev="UGA", is_home=False),
    )
    return Slate(slate_date=date(2026, 9, 5), games=[game]), _board(game.matchup(), -7.0)


def _board(matchup: str, home_point: float) -> dict[str, GameOdds]:
    """A board with both ends of one home spread (``add_spread`` keys on the home
    number, so the away side is filed under it too)."""
    odds = GameOdds(matchup=matchup)
    quote = MarketQuote(book="b", american=-110, opposite_american=-110)
    odds.add_spread(home_point, "ALA", quote)
    odds.add_spread(home_point, "UGA", quote)
    return {matchup: odds}


def _moved_board(slate: Slate, point: float) -> dict[str, GameOdds]:
    return _board(slate.games[0].matchup(), point)


def _pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Pipeline, Config]:
    monkeypatch.setenv("CFBE_DATA_DIR", str(tmp_path))
    cfg = Config()
    return Pipeline(cfg, cfbd=CFBDClient(None)), cfg


def test_the_first_board_is_written_once_and_not_redefined_by_a_later_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe, cfg = _pipeline(tmp_path, monkeypatch)
    day = date(2026, 9, 5)
    slate, board = _slate_and_board()

    pipe._baseline_board(day, slate, board)
    first = cfg.board_file(day).read_text()

    # The market has since moved a point; the baseline must still be -7.
    pipe._baseline_board(day, slate, _moved_board(slate, -8.0))
    assert cfg.board_file(day).read_text() == first
    assert pipe._first_board[snapshot.key(MATCHUP, "game_ats", "ALA -7.0")].line == -7.0


def test_a_side_the_board_posts_later_still_gets_a_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe, cfg = _pipeline(tmp_path, monkeypatch)
    day = date(2026, 9, 5)
    slate, board = _slate_and_board()
    pipe._baseline_board(day, slate, board)

    later = _moved_board(slate, -7.0)
    later[slate.games[0].matchup()].add_ml(
        "ALA", MarketQuote(book="b", american=-250, opposite_american=200)
    )
    baseline = pipe._baseline_board(day, slate, later)
    assert snapshot.key(MATCHUP, "game_ml", "ALA ML") in baseline


def test_drift_reads_the_market_coming_to_the_favourite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe, _ = _pipeline(tmp_path, monkeypatch)
    slate, board = _slate_and_board()
    pipe._baseline_board(date(2026, 9, 5), slate, board)

    # Now the board is ALA -8: a point of movement toward Alabama, and away
    # from Georgia, at an unchanged -110.
    toward = pipe._drift(MATCHUP, "game_ats", keys.game_ats("ALA", -8.0), "cover", -8.0, 0.5)
    away = pipe._drift(MATCHUP, "game_ats", keys.game_ats("UGA", 8.0), "cover", 8.0, 0.5)
    assert toward is not None and toward > 0
    assert away is not None and away < 0
    assert toward == pytest.approx(-away, abs=1e-9)


def test_drift_is_none_for_a_side_with_no_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe, _ = _pipeline(tmp_path, monkeypatch)
    assert pipe._drift(MATCHUP, "game_ats", "ALA -7.0", "cover", -7.0, 0.5) is None


def test_a_baseline_that_cannot_be_written_does_not_cost_the_slate_its_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe, cfg = _pipeline(tmp_path, monkeypatch)
    day = date(2026, 9, 5)
    slate, board = _slate_and_board()

    def boom(*_: object, **__: object) -> None:
        raise OSError("read-only")

    monkeypatch.setattr(snapshot, "save", boom)
    baseline = pipe._baseline_board(day, slate, board)
    assert baseline  # still usable in memory for this run
    assert not cfg.board_file(day).exists()
