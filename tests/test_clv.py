"""Closing line value: capture, scoring, and the anchor that uses the same price."""

from __future__ import annotations

from pathlib import Path

from mlb_engine.audit.clv import (
    ClosingQuote,
    attach_clv,
    closing_quotes,
    clv_ev,
    clv_points,
    clv_rows,
    load_closing,
    merge_closing,
    save_closing,
    summarize,
)
from mlb_engine.audit.ledger import LedgerEntry
from mlb_engine.market.ev import MarketQuote, anchor_to_market


def _entry(**kw: object) -> LedgerEntry:
    base: dict[str, object] = {
        "date": "2026-07-27",
        "matchup": "KC@DET",
        "category": "Batter Props",
        "market": "batter_h",
        "selection": "Riley Greene 0.5+ H",
        "line": 0.5,
        "book": "draftkings",
        "odds": -150.0,
        "tier": "Strong buy",
        "model_prob": 0.64,
        "ev": 0.06,
        "result": "win",
        "pnl": 0.6667,
    }
    base.update(kw)
    return LedgerEntry(**base)  # type: ignore[arg-type]


def test_closing_quotes_devigs_and_line_shops() -> None:
    board = {
        ("KC@DET", "game_ml", "DET"): [
            MarketQuote(book="draftkings", american=-120, opposite_american=100),
            MarketQuote(book="circa", american=-110, opposite_american=-110),
        ]
    }
    (q,) = closing_quotes(board)
    assert q.market == "game_ml"
    # Best price is the shortest juice we could actually have taken.
    assert q.american == -110
    # Circa carries double weight: .5 on Circa, .5455 on DK -> nearer .5.
    assert 0.50 < q.no_vig_prob < 0.52


def test_closing_quotes_skips_unpriced_selections() -> None:
    assert closing_quotes({("KC@DET", "game_ml", "DET"): []}) == []


def test_save_load_roundtrip(tmp_path: Path) -> None:
    quotes = [
        ClosingQuote("KC@DET", "game_ml", "DET", -135.0, 0.5712),
        ClosingQuote("KC@DET", "game_total", "Over 9.0", 105.0, 0.4801),
    ]
    path = tmp_path / "closing.json"
    save_closing(path, quotes)
    loaded = load_closing(path)
    assert set(loaded) == {"KC@DET|game_ml|DET", "KC@DET|game_total|Over 9.0"}
    assert loaded["KC@DET|game_total|Over 9.0"].no_vig_prob == 0.4801


def test_load_closing_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_closing(tmp_path / "nope.json") == {}


def test_second_capture_keeps_the_day_games_it_can_no_longer_see(tmp_path: Path) -> None:
    """The afternoon close must survive the evening capture.

    A game in progress has left the pre-match board, so the evening snapshot
    returns nothing for it. Overwriting would trade the day slate's CLV for the
    night slate's; merging keeps both, latest price winning per selection.
    """
    path = tmp_path / "closing.json"
    save_closing(path, [ClosingQuote("KC@DET", "game_ml", "DET", -135.0, 0.5712)])

    evening = [
        ClosingQuote("KC@DET", "game_ml", "DET", -150.0, 0.5901),  # a later look
        ClosingQuote("LAD@SF", "game_ml", "LAD", -180.0, 0.6350),  # a night game
    ]
    merged = merge_closing(load_closing(path), evening)
    save_closing(path, merged)

    final = load_closing(path)
    assert set(final) == {"KC@DET|game_ml|DET", "LAD@SF|game_ml|LAD"}
    assert final["KC@DET|game_ml|DET"].no_vig_prob == 0.5901

    # A third capture that no longer sees either game changes nothing.
    assert merge_closing(final, []) == sorted(final.values(), key=lambda q: q.key)


def test_clv_points_and_ev() -> None:
    # Bet a side at a devigged .58; it closes at .62 -> the market came to us.
    assert clv_points(0.58, 0.62) == 0.04
    # -150 pays 0.6667; at a closing probability of .62 that is positive EV.
    assert clv_ev(-150, 0.62) > 0
    # The same price is negative when the close disagrees.
    assert clv_ev(-150, 0.55) < 0
    # Break-even at a -150 price is exactly 0.6.
    assert abs(clv_ev(-150, 0.6)) < 1e-9


def test_attach_clv_uses_bet_time_fair_prob() -> None:
    e = _entry(fair_prob=0.58, bet_prob=0.64)
    closing = {"KC@DET|batter_h|Riley Greene 0.5+ H": ClosingQuote(
        "KC@DET", "batter_h", "Riley Greene 0.5+ H", -170.0, 0.62
    )}
    assert attach_clv([e], closing) == 1
    assert e.close_odds == -170.0
    assert e.close_prob == 0.62
    # Against the price we bet (.58 devigged), not the .64 we predicted.
    assert e.clv == 0.04
    assert e.clv_ev is not None and e.clv_ev > 0


def test_attach_clv_leaves_unmatched_and_unpriced_rows_alone() -> None:
    unpriced = _entry(odds=None, fair_prob=None)
    unmatched = _entry(selection="Bobby Witt 0.5+ H", fair_prob=0.58)
    assert attach_clv([unpriced, unmatched], {}) == 0
    for e in (unpriced, unmatched):
        assert e.clv is None and e.clv_ev is None and e.close_prob is None


def test_attach_clv_matches_a_close_that_spells_the_name_differently() -> None:
    """The close carries the book's spelling and the ledger the lineup feed's."""
    e = _entry(selection="Ronald Acuna H o0.5", fair_prob=0.58)
    close = ClosingQuote("KC@DET", "batter_h", "Ronald Acu\u00f1a Jr. H o0.5", -170.0, 0.62)
    assert attach_clv([e], {close.key: close}) == 1
    assert e.close_odds == -170.0
    # The ledger keeps the spelling it was written with.
    assert e.selection == "Ronald Acuna H o0.5"


def test_attach_clv_will_not_guess_between_two_players_of_the_same_name() -> None:
    a = ClosingQuote("WSH@HOU", "batter_h", "Luis Garcia Jr. H o0.5", -170.0, 0.62)
    b = ClosingQuote("WSH@HOU", "batter_h", "Luis Garc\u00eda H o0.5", 120.0, 0.44)
    e = _entry(matchup="WSH@HOU", selection="Luis Garcia H o0.5", fair_prob=0.58)
    assert attach_clv([e], {a.key: a, b.key: b}) == 0
    assert e.clv is None


def test_attach_clv_falls_back_to_implied_for_legacy_rows() -> None:
    """Rows written before the devig fix have no fair_prob; CLV still scores."""
    e = _entry(fair_prob=None, bet_prob=None)
    closing = {"KC@DET|batter_h|Riley Greene 0.5+ H": ClosingQuote(
        "KC@DET", "batter_h", "Riley Greene 0.5+ H", -150.0, 0.60
    )}
    assert attach_clv([e], closing) == 1
    # -150 implies .60 with the vig in, so a .60 close reads as zero CLV.
    assert e.clv == 0.0


def test_summarize_rolls_up_per_market_plus_all() -> None:
    rows = [
        ("batter_h", 0.02, 0.04),
        ("batter_h", -0.01, -0.02),
        ("game_ml", 0.03, 0.05),
    ]
    out = summarize(rows)
    assert [s.label for s in out] == ["batter_h", "game_ml", "ALL"]
    hits = out[0]
    assert hits.n == 2
    assert hits.mean_clv == 0.005
    assert hits.beat_close_pct == 0.5
    assert out[1].positive is True
    assert out[2].n == 3


def test_summarize_empty() -> None:
    assert summarize([]) == []


def test_clv_rows_only_scored_entries() -> None:
    scored = _entry(clv=0.01, clv_ev=0.02)
    unscored = _entry()
    assert clv_rows([scored, unscored]) == [("batter_h", 0.01, 0.02)]


def test_ledger_roundtrips_clv_columns(tmp_path: Path) -> None:
    from datetime import date

    from mlb_engine.audit.ledger import load_ledger, update_ledger

    path = tmp_path / "ledger.csv"
    scored = _entry(fair_prob=0.58, bet_prob=0.64, close_odds=-170.0, close_prob=0.62,
                    clv=0.04, clv_ev=0.0333)
    unscored = _entry(selection="Bobby Witt 0.5+ H")
    update_ledger(path, [scored, unscored], date(2026, 7, 27))
    back = {e.selection: e for e in load_ledger(path)}
    assert back["Riley Greene 0.5+ H"].clv == 0.04
    assert back["Riley Greene 0.5+ H"].close_prob == 0.62
    assert back["Riley Greene 0.5+ H"].bet_prob == 0.64
    # A row with no close stays empty rather than reading back as zero CLV.
    assert back["Bobby Witt 0.5+ H"].clv is None
    assert back["Bobby Witt 0.5+ H"].close_odds is None


def test_anchor_to_market_blends_and_clamps() -> None:
    # Off by default: the model is untouched.
    assert anchor_to_market(0.64, 0.55, 0.0) == 0.64
    # Fully anchored bets the market itself.
    assert anchor_to_market(0.64, 0.55, 1.0) == 0.55
    # Halfway lands halfway, and out-of-range weights are clamped, not scaled.
    assert abs(anchor_to_market(0.64, 0.55, 0.5) - 0.595) < 1e-12
    assert anchor_to_market(0.64, 0.55, 2.5) == 0.55
    assert anchor_to_market(0.64, 0.55, -1.0) == 0.64


def test_anchor_scales_the_edge_requirement() -> None:
    """A weight is a stricter edge threshold, not a deferral to the market.

    Anchoring keeps the model's biggest disagreements and drops the small ones,
    which is the opposite of how "market as prior" reads. Pinned here so the
    mechanism cannot be quietly misdescribed.
    """
    fair = 0.50
    for model, w in ((0.56, 0.6), (0.62, 0.4), (0.51, 0.5)):
        edge = anchor_to_market(model, fair, w) - fair
        assert abs(edge - (1 - w) * (model - fair)) < 1e-12
    # At w=.6 a .02 edge screen passes only disagreements of .05 or more.
    assert anchor_to_market(0.549, fair, 0.6) - fair < 0.02
    assert anchor_to_market(0.551, fair, 0.6) - fair > 0.02
