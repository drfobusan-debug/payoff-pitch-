"""Per-book totals capture and the best-number / Kalshi grading of totals rows."""

from __future__ import annotations

import gzip
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from cfb_engine.audit import shop
from cfb_engine.audit.ledger import LedgerEntry, load_ledger, update_ledger
from cfb_engine.data.kalshi import Ladder, Strike, match, parse_markets
from cfb_engine.market.board import GameOdds
from cfb_engine.market.ev import MarketQuote
from cfb_engine.schemas import Game, Slate, TeamGameInfo
from cfb_engine.state import merge_books

DAY = date(2026, 10, 17)
PULLED = datetime(2026, 10, 17, 11, 0, tzinfo=timezone.utc)


def _game(gid: str, home: str, away: str, kick: str) -> Game:
    return Game(
        game_id=gid,
        game_date=DAY,
        commence_time_utc=kick,
        home=TeamGameInfo(name=home, abbrev=home.split()[0].upper()[:4], is_home=True),
        away=TeamGameInfo(name=away, abbrev=away.split()[0].upper()[:4], is_home=False),
    )


def _board(game: Game, lines: dict[str, tuple[float, float, float]]) -> GameOdds:
    """lines: book -> (total, over price, under price)."""
    odds = GameOdds(matchup=game.matchup())
    for book, (line, o, u) in lines.items():
        odds.add_total(line, True, MarketQuote(book=book, american=o, opposite_american=u))
        odds.add_total(line, False, MarketQuote(book=book, american=u, opposite_american=o))
    return odds


def _ladder() -> list[list[float]]:
    return [[54.5, 0.54, 0.56, 10.0], [55.5, 0.50, 0.52, 795.0], [56.5, 0.47, 0.49, 360.0]]


def test_capture_keeps_every_book_and_number_and_skips_started_games() -> None:
    later = _game("1", "Tulane Green Wave", "Memphis Tigers", "2026-10-17T23:30:00Z")
    started = _game("2", "Houston Cougars", "Oklahoma State Cowboys", "2026-10-17T10:00:00Z")
    board = {
        later.matchup(): _board(
            later, {"fanduel": (55.5, -110, -110), "betrivers": (55.0, -110, -110)}
        ),
        started.matchup(): _board(started, {"fanduel": (62.5, -110, -110)}),
    }
    ladders = [Ladder(("Memphis", "Tulane"), None, tuple(Strike(*r) for r in _ladder()))]
    rows = shop.capture_rows(
        Slate(slate_date=DAY, games=[later, started]),
        board,
        shop.CARD,
        pulled=PULLED,
        ladders=ladders,
    )
    assert [r["matchup"] for r in rows] == [later.matchup()]
    assert sorted(map(tuple, rows[0]["quotes"])) == sorted(
        [
            ("betrivers", 55.0, "over", -110.0),
            ("betrivers", 55.0, "under", -110.0),
            ("fanduel", 55.5, "over", -110.0),
            ("fanduel", 55.5, "under", -110.0),
        ]
    )
    assert rows[0]["kalshi"][1] == [55.5, 0.50, 0.52, 795.0]
    assert rows[0]["source"] == "card"


def test_kalshi_ladders_parse_and_match_by_school_and_date() -> None:
    def mk(strike: float, teams: str, when: str) -> dict[str, object]:
        return {
            "event_ticker": f"E-{teams}-{when[:10]}",
            "rules_primary": f"If the teams collectively score more than {strike} points in the "
            f"{teams} college football game originally scheduled for Oct 17, 2026, then ...",
            "floor_strike": strike,
            "yes_bid_dollars": "0.5000",
            "yes_ask_dollars": "0.5200",
            "open_interest_fp": "12.00",
            "occurrence_datetime": when,
        }

    ladders = parse_markets(
        [
            mk(56.5, "Kennesaw St. vs Liberty", "2026-10-18T02:00:00Z"),
            mk(55.5, "Kennesaw St. vs Liberty", "2026-10-18T02:00:00Z"),
            mk(48.5, "Kennesaw St. vs Liberty", "2026-11-28T02:00:00Z"),
        ]
    )
    assert len(ladders) == 2  # the November rematch is its own event
    game = _game("1", "Liberty Flames", "Kennesaw State Owls", "2026-10-17T23:00:00Z")
    found = match(game, ladders)
    assert found is not None
    assert [s.strike for s in found.strikes] == [55.5, 56.5]
    assert (
        match(_game("2", "Tulane Green Wave", "Memphis Tigers", "2026-10-17T23:00:00Z"), ladders)
        is None
    )


def test_best_offer_takes_the_better_number_and_respects_the_allow_list() -> None:
    quotes = [
        ["fanduel", 55.5, "under", -110.0],
        ["betrivers", 56.5, "under", -110.0],
        ["bovada", 57.5, "under", -110.0],
    ]
    best = shop.best_offer(quotes, "under", 55.5, 0.5, 15.5, None)
    assert best is not None and (best.book, best.line) == ("bovada", 57.5)
    allowed = shop.best_offer(quotes, "under", 55.5, 0.5, 15.5, ("fanduel", "betrivers"))
    assert allowed is not None and (allowed.book, allowed.line) == ("betrivers", 56.5)
    # A half point is worth about 1.3 points of probability at SD 15.5, not a
    # 25-cent price cut: -110 at the card's number beats -135 half a point better.
    priced = shop.best_offer(
        [["a", 55.5, "under", -110.0], ["b", 56.0, "under", -135.0]], "under", 55.5, 0.5, 15.5, None
    )
    assert priced is not None and priced.book == "a"
    assert shop.side_prob("under", 56.5, 55.5, 0.5, 15.5) == pytest.approx(0.5257, abs=1e-3)


def _entry(**kw: object) -> LedgerEntry:
    base: dict[str, object] = dict(
        date=DAY.isoformat(),
        matchup="MEMP @ TULA",
        category="Totals",
        market="game_total",
        selection="Under 55.5",
        line=55.5,
        book="fanduel",
        odds=-110.0,
        tier="Pass",
        model_prob=0.55,
        ev=0.03,
        result="win",
        pnl=0.9091,
        fair_prob=0.5,
        pass_gate="graded_only_game_total",
    )
    base.update(kw)
    return LedgerEntry(**base)  # type: ignore[arg-type]


def test_stamp_grades_the_best_number_and_kalshi_against_the_final() -> None:
    rows = [
        {
            "pulled": "2026-10-17T11:00:00+00:00",
            "source": "card",
            "matchup": "MEMP @ TULA",
            "quotes": [["fanduel", 55.5, "under", -110.0], ["betrivers", 56.0, "under", -110.0]],
            "kalshi": _ladder(),
        },
        # A later close pull is not what the card could have bet.
        {
            "pulled": "2026-10-17T23:00:00+00:00",
            "source": "close",
            "matchup": "MEMP @ TULA",
            "quotes": [["fanduel", 60.5, "under", -110.0]],
        },
    ]
    e = _entry()
    assert shop.stamp(e, "under", 56.0, rows, None, 15.5)
    # The card's 55.5 lost 56-point final; the 56 at BetRivers pushes.
    assert (e.shop_book, e.shop_line, e.shop_result, e.shop_pnl) == ("betrivers", 56.0, "push", 0.0)
    # Kalshi Under 55.5 = buy No at 1 - 0.50 bid; the 56 final goes over 55.5.
    assert (e.kalshi_strike, e.kalshi_ask, e.kalshi_bid, e.kalshi_oi) == (55.5, 0.5, 0.48, 795.0)
    assert e.kalshi_result == "loss" and e.kalshi_pnl == -1.0
    win = _entry(selection="Over 55.5")
    assert shop.stamp(win, "over", 56.0, rows, None, 15.5)
    # Yes at 0.52 plus 0.07 x 0.52 x 0.48 fee.
    assert win.kalshi_pnl == pytest.approx(1 / (0.52 + 0.07 * 0.52 * 0.48) - 1, abs=1e-4)
    assert not shop.stamp(_entry(market="game_ats"), "cover", 56.0, rows, None, 15.5)
    assert not shop.stamp(_entry(matchup="X @ Y"), "under", 56.0, rows, None, 15.5)


def test_new_columns_survive_the_ledger_round_trip(tmp_path: Path) -> None:
    e = _entry(
        shop_book="betrivers",
        shop_line=56.0,
        shop_odds=-110.0,
        shop_result="push",
        shop_pnl=0.0,
        kalshi_strike=55.5,
        kalshi_bid=0.48,
        kalshi_ask=0.5,
        kalshi_oi=795.0,
        kalshi_result="loss",
        kalshi_pnl=-1.0,
    )
    plain = _entry(matchup="A @ B")
    path = tmp_path / "ledger.csv"
    update_ledger(path, [e, plain], DAY)
    back = {x.matchup: x for x in load_ledger(path)}
    got = back["MEMP @ TULA"]

    def cols(x: LedgerEntry) -> tuple[object, ...]:
        return (
            x.shop_book,
            x.shop_line,
            x.shop_odds,
            x.shop_result,
            x.shop_pnl,
            x.kalshi_strike,
            x.kalshi_bid,
            x.kalshi_ask,
            x.kalshi_oi,
            x.kalshi_result,
            x.kalshi_pnl,
        )

    assert cols(got) == cols(e)
    assert back["A @ B"].shop_result is None and back["A @ B"].kalshi_pnl is None


def test_records_compare_the_same_rows_and_default_to_graded_only() -> None:
    graded = _entry(
        result="loss",
        pnl=-1.0,
        shop_result="push",
        shop_pnl=0.0,
        kalshi_result="win",
        kalshi_pnl=0.9,
    )
    live = _entry(matchup="A @ B", pass_gate=None, tier="Moderate", shop_result="win", shop_pnl=0.9)
    card, best, kal_card, kal = shop.records([graded, live])
    assert (card.n, card.losses, best.pushes, kal.wins, kal_card.n) == (1, 1, 1, 1, 1)
    assert shop.records([graded, live], graded_only=False)[0].n == 2


def test_books_logs_merge_into_one_gzip_on_the_branch(tmp_path: Path) -> None:
    local = tmp_path / "totals_books_2026-10-17.jsonl"
    remote = tmp_path / "branch" / "totals_books_2026-10-17.jsonl.gz"
    local.write_text('{"a": 1}\n{"b": 2}\n')
    remote.parent.mkdir()
    remote.write_bytes(gzip.compress(b'{"a": 1}\n{"c": 3}\n'))
    merge_books(remote, local, remote)
    assert gzip.decompress(remote.read_bytes()).decode().splitlines() == [
        '{"a": 1}',
        '{"c": 3}',
        '{"b": 2}',
    ]
    merge_books(remote, local, local)
    assert local.read_text().splitlines() == ['{"a": 1}', '{"c": 3}', '{"b": 2}']
    shop.append([{"x": 1}], local)
    assert shop.load(local)[-1] == {"x": 1}
