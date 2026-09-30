"""Phase 0: archive rows, snapshots, Odds API parsing, NHL API parsing, book rules."""

from __future__ import annotations

import json
from datetime import date as Date
from pathlib import Path

import pytest

from nhl_engine.data import capture, teamnames
from nhl_engine.data.book_rules import UNVERIFIED, BookRule, BookRules
from nhl_engine.data.capture import MARKET_MAP, QuoteRow
from nhl_engine.data.nhlapi import parse_result, parse_schedule
from nhl_engine.data.oddsapi import OddsAPIClient, event_rows
from nhl_engine.schemas import Game

FIXTURES = Path(__file__).parent / "fixtures"
DAY = Date(2026, 9, 29)


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _game() -> Game:
    return Game(game_date=DAY, home="TOR", away="MTL", event_id="485b295347cb22f002e014cb87813ed7")


def _row(**over: object) -> QuoteRow:
    base: dict[str, object] = dict(
        captured_at="2026-09-29T15:00:00Z",
        game_date=DAY.isoformat(),
        matchup="MTL @ TOR",
        event_id="e1",
        market="game_ml",
        side="TOR",
        entity="",
        line=None,
        book="fanduel",
        american=-130.0,
        opposite_american=110.0,
    )
    base.update(over)
    return QuoteRow(**base)  # type: ignore[arg-type]


# -- team names -------------------------------------------------------------
def test_team_names_cover_all_32_and_accents():
    assert len(teamnames.CODES) == 32
    assert teamnames.code_for("Montréal Canadiens") == "MTL"
    assert teamnames.code_for("St Louis Blues") == "STL"
    assert teamnames.code_for("Utah Mammoth") == "UTA"
    assert teamnames.code_for("Quebec Nordiques") is None
    assert teamnames.canonical("TB") == "TBL"


# -- Odds API parsing -------------------------------------------------------
def test_event_rows_cover_period_and_prop_markets():
    rows = event_rows(_fixture("event_odds.json"), _game(), "2026-09-28T20:00:00Z")
    markets = {r.market for r in rows}
    for expected in (
        "game_ml3",
        "p1_ml",
        "p1_total",
        "p2_pl",
        "p3_total",
        "team_total",
        "sk_sog",
        "ags",
    ):
        assert expected in markets, expected
    # Every row is stamped and keyed to the game.
    assert all(r.matchup == "MTL @ TOR" and r.event_id == _game().event_id for r in rows)


def test_prop_pairs_by_player_and_line_only():
    rows = event_rows(_fixture("event_odds.json"), _game(), "t")
    sog = [r for r in rows if r.market == "sk_sog" and r.book == "williamhill_us"]
    assert sog, "fixture carries William Hill SOG props"
    for r in sog:
        partner = [p for p in sog if p.entity == r.entity and p.line == r.line and p.side != r.side]
        if partner:
            assert r.opposite_american == partner[0].american
        else:
            assert r.opposite_american is None
    # An anytime scorer is a one-way "yes" with no invented "no".
    ags = [r for r in rows if r.market == "ags"]
    assert ags and all(r.side == "yes" and r.opposite_american is None and r.entity for r in ags)


def test_three_way_has_draw_and_no_partner():
    rows = [
        r for r in event_rows(_fixture("event_odds.json"), _game(), "t") if r.market == "game_ml3"
    ]
    sides = {r.side for r in rows}
    assert sides == {"MTL", "TOR", "draw"}
    assert all(r.opposite_american is None for r in rows)


def test_spread_ladder_pairs_mirror_lines():
    rows = [
        r
        for r in event_rows(_fixture("event_odds.json"), _game(), "t")
        if r.market == "game_pl_alt" and r.book == "draftkings"
    ]
    by_key = {(r.side, r.line): r for r in rows}
    mtl = by_key.get(("MTL", -1.5))
    tor = by_key.get(("TOR", 1.5))
    assert mtl is not None and tor is not None
    assert mtl.opposite_american == tor.american
    assert tor.opposite_american == mtl.american
    # Rungs pair only with their mirror line.
    rung = by_key.get(("MTL", -2.5))
    assert rung is not None
    mirror = by_key.get(("TOR", 2.5))
    assert rung.opposite_american == (mirror.american if mirror else None)


def test_team_totals_name_the_team():
    rows = [
        r for r in event_rows(_fixture("event_odds.json"), _game(), "t") if r.market == "team_total"
    ]
    assert {r.entity for r in rows} == {"MTL", "TOR"}
    paired = [r for r in rows if r.opposite_american is not None]
    assert paired
    for r in paired:
        partner = [
            p
            for p in rows
            if p.book == r.book and p.entity == r.entity and p.line == r.line and p.side != r.side
        ]
        assert len(partner) == 1 and partner[0].american == r.opposite_american


def test_all_provider_keys_have_ot_rule():
    assert all(
        rule in ("incl_ot_so", "reg_only", "incl_ot", "book_rule")
        for _, rule in MARKET_MAP.values()
    )
    assert MARKET_MAP["team_totals"][1] == "book_rule"
    assert MARKET_MAP["h2h_3_way"][1] == "reg_only"


def test_client_without_key_is_inert(tmp_path: Path):
    client = OddsAPIClient(None, cache_dir=tmp_path)
    slate, rows = client.fetch_board(slate_date=DAY)
    assert slate.games == [] and rows == []
    assert client.fetch_events(slate_date=DAY).games == []
    assert client.fetch_event_markets([_game()]) == []


# -- snapshots --------------------------------------------------------------
def test_snapshot_is_idempotent_and_keeps_changes(tmp_path: Path):
    rows = [_row(), _row(side="MTL", american=110.0, opposite_american=-130.0)]
    first = capture.write_snapshot(rows, tmp_path, DAY, captured_at="2026-09-29T15:00:00Z")
    assert first is not None
    again = [
        _row(captured_at="2026-09-29T15:30:00Z"),
        _row(
            side="MTL", american=110.0, opposite_american=-130.0, captured_at="2026-09-29T15:30:00Z"
        ),
    ]
    assert capture.write_snapshot(again, tmp_path, DAY, captured_at="2026-09-29T15:30:00Z") is None
    moved = [_row(american=-135.0, captured_at="2026-09-29T16:00:00Z")]
    second = capture.write_snapshot(moved, tmp_path, DAY, captured_at="2026-09-29T16:00:00Z")
    assert second is not None and second != first
    assert len(capture.snapshot_paths(tmp_path, DAY)) == 2
    assert capture.write_snapshot([], tmp_path, DAY) is None


def test_snapshot_round_trip_and_summary(tmp_path: Path):
    rows = [
        _row(),
        _row(
            market="sk_sog",
            side="over",
            entity="Auston Matthews",
            line=3.5,
            american=-115.0,
            opposite_american=None,
        ),
    ]
    path = capture.write_snapshot(rows, tmp_path, DAY)
    assert path is not None
    back = capture.read_snapshot(path)
    assert back == rows
    summary = capture.archive_summary(capture.read_day(tmp_path, DAY))
    assert summary["games"] == 1 and summary["markets"] == {"game_ml": 1, "sk_sog": 1}


def test_read_snapshot_skips_bad_rows(tmp_path: Path):
    path = tmp_path / "board_x.csv"
    path.write_text(
        ",".join(capture.FIELDS) + "\nbad\n2026,2026-09-29,MTL @ TOR,e,game_ml,TOR,,,fd,nan?,\n"
    )
    assert capture.read_snapshot(path) == []


def test_last_quotes_keeps_latest_per_key():
    rows = [_row(captured_at="a", american=-130.0), _row(captured_at="b", american=-140.0)]
    latest = capture.last_quotes(rows)
    assert len(latest) == 1 and next(iter(latest.values())).american == -140.0


# -- NHL API ---------------------------------------------------------------
def test_parse_schedule_filters_to_day():
    games = parse_schedule(_fixture("schedule.json"), DAY)
    assert games and all(g.game_date == DAY for g in games)
    assert games[0].away == "FLA" and games[0].home == "CAR" and games[0].nhl_game_id == 2026020001
    assert parse_schedule(_fixture("schedule.json"), Date(2020, 1, 1)) == []


def test_parse_result_shootout_settlement():
    game = Game(game_date=Date(2026, 4, 15), home="BUF", away="DAL", nhl_game_id=2025021301)
    res = parse_result(_fixture("boxscore.json"), _fixture("right_rail.json"), game)
    assert res.is_final and res.decided == "SO"
    assert (res.reg_away, res.reg_home) == (3, 3)
    assert (res.final_away, res.final_home) == (4, 3)
    p1 = res.period(1)
    assert p1 is not None and (p1.away, p1.home) == (1, 1)
    assert res.home_starter == "C. Ellis"


def test_parse_result_tolerates_missing_rail():
    game = Game(game_date=Date(2026, 4, 15), home="BUF", away="DAL", nhl_game_id=2025021301)
    res = parse_result(_fixture("boxscore.json"), {}, game)
    assert res.periods == () and res.final_home == 0


# -- book rules ------------------------------------------------------------
def test_book_rules_gate_and_expiry(tmp_path: Path):
    rules = BookRules()
    today = Date(2026, 9, 29)
    assert rules.gate("fanduel", "team_total", today) == UNVERIFIED
    assert rules.gate("fanduel", "game_ml", today) is None
    rules.add(BookRule("fanduel", "team_total", "incl_ot_so", "2026-09-01"))
    assert rules.gate("fanduel", "team_total", today) is None
    assert rules.resolve("fanduel", "team_total", Date(2026, 12, 1)) == UNVERIFIED
    path = tmp_path / "rules.json"
    rules.save(path)
    loaded = BookRules.load(path)
    assert loaded.resolve("fanduel", "team_total", today) == "incl_ot_so"
    with pytest.raises(ValueError):
        rules.add(BookRule("dk", "team_total", "whatever", "2026-09-01"))
    assert len(BookRules.load(tmp_path / "missing.json")) == 0


# -- Mac shortcuts ---------------------------------------------------------
def test_mac_shortcuts_present_and_executable():
    root = Path(__file__).resolve().parents[2] / "scripts" / "nhl" / "macos"
    for name in (
        "nhl_capture.command",
        "run_predictions.command",
        "run_audit.command",
        "run_results.command",
        "open_ledger.command",
        "install_shortcuts.command",
        "install_schedule.command",
    ):
        path = root / name
        assert path.exists(), name
        assert path.stat().st_mode & 0o111, f"{name} not executable"
