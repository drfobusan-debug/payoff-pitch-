"""EV Analytics' saved prop board: parsing, the join, and its display-only scope."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date

import pytest

from mlb_engine.data.evanalytics import EVProp, annotate, load_board, parse_board
from mlb_engine.market.tiers import Tier
from mlb_engine.recommendations import Recommendation

HEADERS = [
    "STATUS", "SITE", "DATE", "TIME", "MARKET", "PLAYER", "TM", "GAME",
    "OFFICAL LINEUP", "BATTING ORDER", "PITCH COUNT CHECKED",
    "THE BAT X PITCH COUNT", "LINE MOVE", "OVER", "UNDER", "BEST LINE",
    "IMPLIED PROJECTION", "THE BAT X PROJECTION", "IMPLIED VS BATX % DIFFERENCE",
    "SUGGESTED BET", "EXPECTED VALUE",
]


def _row(market: str, player: str, over: str, implied: str, batx: str, bet: str) -> str:
    cells = [
        "", "DraftKings", "Aug 15", "1:10 PM", market, player, "CWS", "CWS@DET",
        "", "9", "", "", "", over, "", "", implied, batx, "", bet, "",
    ]
    return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def _page(*rows: str) -> str:
    head = "<tr>" + "".join(f"<th>{h}</th>" for h in HEADERS) + "</tr>"
    return f"<table id='dataTable'>{head}{''.join(rows)}</table>"


def _rec(market: str, selection: str, line: float, side: str) -> Recommendation:
    return Recommendation(
        game_pk=1,
        game_date=Date(2026, 8, 15),
        matchup="CWS @ DET",
        category="batter" if market.startswith("batter_") else "game",
        market=market,
        selection=selection,
        line=line,
        side=side,
        model_prob=0.5,
        tier=Tier.PASS,
    )


def test_parses_the_columns_that_matter() -> None:
    props = parse_board(
        _page(_row("Hits", "Jake Rogers", "0.5 (+172)", "0.57", "0.81", "OVER")), year=2026
    )
    assert len(props) == 1
    prop = props[0]
    assert prop.player == "Jake Rogers"
    assert prop.market == "batter_h"
    assert prop.line == 0.5
    assert prop.projection == 0.81
    assert prop.implied == 0.57
    assert prop.date == "2026-08-15"
    assert prop.matchup == "CWS @ DET"


def test_markets_we_do_not_price_are_dropped_not_guessed() -> None:
    """Their board carries stolen bases and pitcher win; the engine prices neither."""
    props = parse_board(
        _page(
            _row("Stolen Bases", "Luis Robert", "0.5 (+250)", "0.28", "0.35", "OVER"),
            _row("Total Bases", "Luis Robert", "1.5 (+120)", "1.10", "1.42", "OVER"),
        ),
        year=2026,
    )
    assert [p.market for p in props] == ["batter_tb"]


def test_side_prefers_their_call_then_the_numbers() -> None:
    stated = parse_board(_page(_row("Hits", "A B", "0.5 (-110)", "0.86", "0.86", "UNDER")))[0]
    assert stated.side == "under"
    derived = parse_board(_page(_row("Hits", "A B", "0.5 (-110)", "0.70", "0.81", "")))[0]
    assert derived.side == "over"
    silent = parse_board(_page(_row("Hits", "A B", "0.5 (-110)", "0.81", "0.81", "")))[0]
    assert silent.side == ""


def test_projection_is_read_against_the_implied_number_not_the_line() -> None:
    """0.86 hits clears a 0.5 line and is still under a market implying 0.90.

    Comparing their mean to the line rather than to the market's own mean would
    call that a star -- an agreement they never expressed.
    """
    prop = parse_board(_page(_row("Hits", "A B", "0.5 (-110)", "0.90", "0.86", "")))[0]
    assert prop.side == "under"
    assert prop.reads("over", 0.5) is False


def test_their_call_carries_only_the_way_it_points() -> None:
    theirs = EVProp(
        date="2026-08-15", player="A B", team="CWS", matchup="CWS @ DET",
        market="batter_h", line=0.5, projection=0.4, implied=0.7, suggestion="under",
    )
    assert theirs.reads("under", 0.5) is True  # their own number
    assert theirs.reads("under", 1.5) is True  # harder under is implied by the easier
    assert theirs.reads("over", 1.5) is False
    over = EVProp(
        date="2026-08-15", player="A B", team="CWS", matchup="CWS @ DET",
        market="batter_h", line=0.5, projection=0.9, implied=0.7, suggestion="over",
    )
    assert over.reads("over", 0.5) is True
    assert over.reads("over", 2.5) is None  # liking over 0.5 says nothing about over 2.5


def test_annotate_marks_both_sides_of_the_same_forecast() -> None:
    props = parse_board(_page(_row("Hits", "Jake Rogers", "0.5 (+172)", "0.57", "0.81", "OVER")))
    recs = [
        _rec("batter_h", "Jake Rogers H o0.5", 0.5, "over"),
        _rec("batter_h", "Jake Rogers H u0.5", 0.5, "under"),
    ]
    assert annotate(recs, props) == 2
    assert recs[0].ev_agrees is True
    assert recs[0].ev_mark == "\u2605"
    assert recs[1].ev_agrees is False
    assert recs[1].ev_mark == "\u2717"
    assert recs[0].ev_proj == 0.81
    assert "BATX 0.81 vs 0.57 implied" in (recs[0].ev_pick or "")


def test_game_markets_are_never_marked() -> None:
    props = parse_board(_page(_row("Hits", "Jake Rogers", "0.5 (+172)", "0.57", "0.81", "OVER")))
    total = _rec("game_total", "Over 7.5", 7.5, "over")
    total.market = "game_total"
    ml = _rec("game_ml", "DET ML", 0.0, "win")
    annotate([total, ml], props)
    assert ml.ev_pick is None
    # A game total has a line and a side but no player, so nothing joins to it.
    assert total.ev_pick is None


def test_annotation_touches_nothing_that_prices_a_bet() -> None:
    props = parse_board(_page(_row("Hits", "Jake Rogers", "0.5 (+172)", "0.57", "0.81", "OVER")))
    rec = _rec("batter_h", "Jake Rogers H o0.5", 0.5, "over")
    rec.fair_prob, rec.edge, rec.ev = 0.44, 0.06, 0.12
    before = (rec.model_prob, rec.fair_prob, rec.edge, rec.ev, rec.tier)
    annotate([rec], props)
    assert (rec.model_prob, rec.fair_prob, rec.edge, rec.ev, rec.tier) == before


def test_board_is_dated_by_its_rows_not_by_the_file(tmp_path) -> None:
    """A stale download in the folder must not be joined to tonight's slate."""
    (tmp_path / "yesterday.html").write_text(
        _page(_row("Hits", "Old Guy", "0.5 (-110)", "0.60", "0.80", "OVER")).replace(
            "Aug 15", "Aug 14"
        )
    )
    (tmp_path / "today.html").write_text(
        _page(_row("Hits", "New Guy", "0.5 (-110)", "0.60", "0.80", "OVER"))
    )
    props = load_board(tmp_path, date=f"{Date.today().year}-08-15")
    assert [p.player for p in props] == ["New Guy"]


def test_pages_of_the_same_board_are_merged(tmp_path) -> None:
    """Their table pages at 250 rows, so a full slate arrives as several files."""
    year = Date.today().year
    (tmp_path / "p1.html").write_text(
        _page(_row("Hits", "Player One", "0.5 (-110)", "0.60", "0.80", "OVER"))
    )
    (tmp_path / "p2.html").write_text(
        _page(
            _row("Hits", "Player Two", "0.5 (-110)", "0.60", "0.80", "OVER"),
            _row("Hits", "Player One", "0.5 (-110)", "0.60", "0.80", "OVER"),
        )
    )
    props = load_board(tmp_path, date=f"{year}-08-15")
    assert sorted(p.player for p in props) == ["Player One", "Player Two"]


def test_empty_folder_and_unreadable_page_are_survivable(tmp_path) -> None:
    assert load_board(tmp_path / "nope") == []
    (tmp_path / "notatable.html").write_text("<html><body>subscribe</body></html>")
    assert load_board(tmp_path) == []


def test_a_batters_walks_and_strikeouts_are_their_own_markets() -> None:
    """Their two largest sections after runs+RBIs, and both are the batter's own
    line -- ``Walks`` must never collide with the pitcher's ``Walks Allowed``."""
    props = parse_board(
        _page(
            _row("Walks", "Salvador Perez", "0.5 (+291)", "0.21", "0.21", "UNDER"),
            _row("Hitter Strikeouts", "Ronny Simon", "0.5 (-154)", "0.84", "0.69", ""),
            _row("Walks Allowed", "Troy Melton", "1.5 (-110)", "1.60", "1.40", "UNDER"),
        ),
        year=2026,
    )
    assert [p.market for p in props] == ["batter_bb", "batter_k", "pitcher_bb"]
    # Equal projection and implied: their own board suggests a side, so take it.
    assert props[0].side == "under"
    # No suggestion: 0.69 projected against 0.84 implied is their under.
    assert props[1].side == "under"


def test_their_walks_join_our_batter_walk_rows() -> None:
    props = parse_board(
        _page(_row("Hitter Strikeouts", "Ronny Simon", "0.5 (-154)", "0.84", "0.69", "")),
        year=2026,
    )
    over = _rec("batter_k", "Ronny Simon K o0.5", 0.5, "over")
    under = _rec("batter_k", "Ronny Simon K u0.5", 0.5, "under")
    assert annotate([over, under], props) == 2
    assert over.ev_agrees is False
    assert under.ev_agrees is True


def test_a_batters_walks_and_strikeouts_are_priced_but_never_bought() -> None:
    from mlb_engine.data.oddsapi import (
        DEFAULT_PROP_MARKETS,
        PRICE_ONLY_MARKETS,
    )
    from mlb_engine.models.props import BATTER_PROP_LINES

    assert "batter_walks" in DEFAULT_PROP_MARKETS
    assert "batter_strikeouts" in DEFAULT_PROP_MARKETS
    assert BATTER_PROP_LINES["BB"] and BATTER_PROP_LINES["K"]
    assert {"batter_bb", "batter_k"} <= PRICE_ONLY_MARKETS


def test_a_batters_walks_and_strikeouts_are_gradeable() -> None:
    """A market priced off a box score that does not carry the stat grades every
    over as a loss, so the batting line has to read both back."""
    from mlb_engine.audit.grade import grade
    from mlb_engine.data.results import GameResult, PlayerLine

    res = GameResult(
        game_pk=1, final=True, home_runs=3, away_runs=1, f5_home=2, f5_away=1,
        players={7: PlayerLine(batting={"PA": 4, "H": 1, "BB": 2, "K": 1})},
    )
    walks = _rec("batter_bb", "Someone BB o1.5", 1.5, "over")
    walks.player_id, walks.stat = 7, "BB"
    ks = _rec("batter_k", "Someone K o1.5", 1.5, "over")
    ks.player_id, ks.stat = 7, "K"
    assert grade(walks, res) == "win"
    assert grade(ks, res) == "loss"


@dataclass
class _Sheet:
    columns: list[str]


def test_workbook_carries_the_columns() -> None:
    from mlb_engine.output.excel import COLUMNS, GRID_COLUMNS, GRID_WIDTHS

    for name in ("EVA", "EVA Proj", "EVA Pick"):
        assert name in COLUMNS
    assert "EVA" in GRID_COLUMNS
    assert set(GRID_COLUMNS) <= set(GRID_WIDTHS)


def test_card_play_shows_their_number() -> None:
    from mlb_engine.output.card import Play

    play = Play(
        selection="Jake Rogers H o0.5",
        market="batter_h",
        odds=-110,
        model_prob=0.6,
        implied_prob=0.55,
        edge=0.05,
        ev=0.1,
        tier=Tier.STRONG,
        ev_pick="OVER: BATX 0.81 vs 0.57 implied",
        ev_agrees=True,
    )
    assert play.ev_bit.startswith("\u2605")
    assert "BATX 0.81" in play.ev_bit
    play.ev_agrees = None
    assert play.ev_bit == ""


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
