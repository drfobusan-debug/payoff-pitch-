"""Alternate spreads/totals: captured as tracked-only rows, read bought vs sold."""

from __future__ import annotations

from datetime import date

from nba_engine.audit import altlines
from nba_engine.audit.ledger import LedgerRow
from nba_engine.audit.settle import settle
from nba_engine.data.oddsapi import event_rows, games_from
from nba_engine.schemas import GameResult


def _event() -> dict:
    return {
        "id": "abc123",
        "commence_time": "2025-12-11T03:30:00Z",
        "home_team": "Los Angeles Lakers",
        "away_team": "San Antonio Spurs",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "alternate_spreads",
                        "outcomes": [
                            {"name": "Los Angeles Lakers", "point": -1.5, "price": -150},
                            {"name": "San Antonio Spurs", "point": 1.5, "price": 125},
                        ],
                    },
                    {
                        "key": "alternate_totals",
                        "outcomes": [
                            {"name": "Over", "point": 228.5, "price": 140},
                            {"name": "Under", "point": 228.5, "price": -170},
                        ],
                    },
                ],
            }
        ],
    }


def test_alternate_markets_parse_to_tracked_rows_paired_per_number() -> None:
    event = _event()
    game = games_from([event], date(2025, 12, 10))[0]
    rows = event_rows(event, game, "2025-12-11T03:20:00Z")
    assert {r.market for r in rows} == {"alt_ats", "alt_total"}
    assert all(r.opposite_american is not None for r in rows)


def _final(home: int, away: int) -> GameResult:
    return GameResult(
        espn_id="1",
        game_date=date(2025, 12, 10),
        away="SAS",
        home="LAL",
        state="STATUS_FINAL",
        away_q=(away, 0, 0, 0),
        home_q=(home, 0, 0, 0),
    )


def test_alternates_settle_like_the_main_market() -> None:
    game = _final(110, 108)
    assert settle("alt_ats", "LAL", "", -1.5, game) == "win"
    assert settle("alt_ats", "SAS", "", 1.5, game) == "loss"
    assert settle("alt_total", "under", "", 218.0, game) == "push"


def _row(
    market: str, side: str, line: float, outcome: str, pnl: float, books: int = 6
) -> LedgerRow:
    return LedgerRow(
        slate_date="2025-12-10",
        tip_utc="2025-12-11T03:30:00Z",
        event_id="abc123",
        espn_id="1",
        matchup="SAS @ LAL",
        market=market,
        ot_rule="incl_ot",
        side=side,
        entity="",
        line=line,
        book="draftkings",
        american=-110.0,
        opposite_american=-110.0,
        exec_books=1,
        exec_prices="draftkings:-110",
        books=books,
        paired_books=books,
        fair=0.5,
        model_prob=None,
        edge=None,
        ev=0.0,
        tier="Pass",
        gates="no_model",
        priced_at="2025-12-11T01:00:00Z",
        pass_tag="card",
        versions="",
        outcome=outcome,
        pnl=pnl,
    )


def test_the_report_splits_points_bought_from_points_sold() -> None:
    rows = [
        _row("game_ats", "LAL", -3.5, "loss", -1.0),
        _row("game_ats", "SAS", 3.5, "win", 0.91),
        _row("game_ats", "LAL", -4.5, "loss", -1.0, books=2),  # an off-market book
        _row("game_total", "over", 226.5, "win", 0.91),
        _row("alt_ats", "SAS", 5.5, "win", 0.5),  # +2 over the main: bought
        _row("alt_ats", "LAL", -6.5, "loss", -1.0),  # 3 fewer: sold
        _row("alt_ats", "SAS", 3.5, "win", 0.91),  # the main number: skipped
        _row("alt_total", "over", 224.5, "win", 0.4),  # lower Over: bought
        _row("alt_total", "under", 224.5, "loss", -1.0),  # lower Under: sold
    ]
    main = altlines.main_lines(rows)
    assert main[("abc123", "game_ats")] == -3.5
    table = {r[0]: r for r in altlines.report(rows)}
    assert table["alt_ats bought (all)"][1:5] == (1, 1, 0, 0)
    assert table["alt_ats bought 2 pts"][5] == 0.5
    assert table["alt_ats sold 3 pts"][1:5] == (1, 0, 1, 0)
    assert table["alt_total bought 2 pts"][1] == 1
    assert table["alt_total sold (all)"][1] == 1
    assert not any("game_" in k for k in table)
