"""EV, screens, tiers and the ledger, on a hand-built board."""

from __future__ import annotations

import numpy as np
import pytest

from nfl_engine.audit.ledger import (
    LedgerEntry,
    apply_close,
    entry_from_bet,
    grade,
    load_ledger,
    metrics,
    save_ledger,
    screen_metrics,
    update_ledger,
)
from nfl_engine.market.board import GameOdds, MarketQuote
from nfl_engine.market.ev import best_by_line, ev, price_game
from nfl_engine.market.screens import Thresholds, apply_screens, buys, screen, tier_of
from nfl_engine.models.distribution import ScoreDistribution

HOME, AWAY = "KC", "BUF"
MATCHUP = f"{AWAY} @ {HOME}"


WIN_MARGINS = (3, 7, 10, 14)
LOSS_MARGINS = (-3, -7, -10)


def distribution(home_win_share: float = 0.58, n: int = 200) -> ScoreDistribution:
    """Deterministic scores with a chosen home win share and no ties.

    Margins are spread over the key numbers so the spread and total ladders have
    something to price; ties are excluded so a push does not silently become the
    thing under test.
    """
    wins = int(round(home_win_share * n))
    margins = [WIN_MARGINS[i % len(WIN_MARGINS)] for i in range(wins)]
    margins += [LOSS_MARGINS[i % len(LOSS_MARGINS)] for i in range(n - wins)]
    home = np.full(n, 24)
    away = home - np.array(margins)
    return ScoreDistribution(home=home, away=away)


def board(
    *,
    home_ml: float = -150,
    away_ml: float = 130,
    books: int = 3,
    paired: bool = True,
) -> GameOdds:
    odds = GameOdds(matchup=MATCHUP)
    for index in range(books):
        name = f"book{index}"
        odds.add_ml(
            HOME,
            MarketQuote(
                book=name,
                american=home_ml,
                opposite_american=away_ml if paired else None,
            ),
        )
        odds.add_ml(
            AWAY,
            MarketQuote(
                book=name,
                american=away_ml,
                opposite_american=home_ml if paired else None,
            ),
        )
    return odds


def test_ev_is_zero_at_the_break_even_price() -> None:
    assert ev(0.5, 2.0) == pytest.approx(0.0)
    assert ev(0.6, 2.0) == pytest.approx(0.2)


def test_every_quote_on_the_board_is_priced() -> None:
    odds = board()
    odds.add_spread(-3.0, HOME, MarketQuote(book="dk", american=-110, opposite_american=-110))
    odds.add_spread(-3.0, AWAY, MarketQuote(book="dk", american=-110, opposite_american=-110))
    odds.add_total(47.5, True, MarketQuote(book="dk", american=-105, opposite_american=-115))
    odds.add_total(47.5, False, MarketQuote(book="dk", american=-115, opposite_american=-105))
    bets = price_game(odds, distribution(), home=HOME, away=AWAY)
    assert {bet.market for bet in bets} == {"moneyline", "spread", "total"}
    assert len(bets) == 6 + 2 + 2


def test_the_same_line_at_two_books_collapses_to_the_better_price() -> None:
    odds = GameOdds(matchup=MATCHUP)
    odds.add_ml(HOME, MarketQuote(book="a", american=-150, opposite_american=130))
    odds.add_ml(HOME, MarketQuote(book="b", american=-135, opposite_american=115))
    bets = best_by_line(price_game(odds, distribution(), home=HOME, away=AWAY))
    assert len(bets) == 1
    assert bets[0].book == "b"


def test_a_price_off_the_market_is_a_buy() -> None:
    odds = board(home_ml=-150, away_ml=130)
    # A fourth book hangs the same side much cheaper: that is the execution edge.
    odds.add_ml(HOME, MarketQuote(book="outlier", american=110, opposite_american=-130))
    bets = buys(best_by_line(price_game(odds, distribution(), home=HOME, away=AWAY)))
    assert [bet.book for bet in bets] == ["outlier"]
    assert bets[0].ev_fair is not None and bets[0].ev_fair > 0.015
    assert tier_of(bets[0]).value == "Strong buy"


def test_a_price_at_the_consensus_is_not_a_bet() -> None:
    bets = apply_screens(best_by_line(price_game(board(), distribution(), home=HOME, away=AWAY)))
    assert all("no_execution_edge" in bet.screens for bet in bets)
    assert all(tier_of(bet).value == "Pass" for bet in bets)


def test_an_unpaired_board_prices_nothing() -> None:
    odds = board(paired=False)
    bets = apply_screens(best_by_line(price_game(odds, distribution(), home=HOME, away=AWAY)))
    for bet in bets:
        assert "unpaired" in bet.screens
        assert "model_only" in bet.screens


def test_one_paired_book_is_screened_as_thin() -> None:
    odds = board(books=1)
    bets = apply_screens(best_by_line(price_game(odds, distribution(), home=HOME, away=AWAY)))
    assert all("thin_market" in bet.screens for bet in bets)


def test_a_longshot_moneyline_is_capped() -> None:
    odds = GameOdds(matchup=MATCHUP)
    for index in range(3):
        odds.add_ml(
            AWAY, MarketQuote(book=f"b{index}", american=600, opposite_american=-900)
        )
    odds.add_ml(AWAY, MarketQuote(book="outlier", american=900, opposite_american=-1200))
    bets = apply_screens(best_by_line(price_game(odds, distribution(), home=HOME, away=AWAY)))
    assert len(bets) == 1
    assert "longshot" in bets[0].screens


def test_the_model_disagreeing_with_a_liquid_market_is_a_veto() -> None:
    """Phase 3: our disagreement explains none of the line's residual."""
    odds = board(home_ml=-110, away_ml=-110)
    odds.add_ml(HOME, MarketQuote(book="outlier", american=150, opposite_american=-180))
    # Home wins 75% of trials here, against a consensus near 50%.
    dist = distribution(home_win_share=0.75)
    bets = apply_screens(best_by_line(price_game(odds, dist, home=HOME, away=AWAY)))
    outlier = next(bet for bet in bets if bet.book == "outlier")
    assert "model_disagrees" in outlier.screens


def test_a_negative_model_price_vetoes_an_execution_edge() -> None:
    odds = board(home_ml=-150, away_ml=130)
    odds.add_ml(AWAY, MarketQuote(book="outlier", american=175, opposite_american=-210))
    # Away is 36% here against a 41.7% consensus: inside the disagreement cap, but
    # short of the 36.4% the +175 needs to pay.
    dist = distribution(home_win_share=0.64, n=100)
    bets = apply_screens(best_by_line(price_game(odds, dist, home=HOME, away=AWAY)))
    outlier = next(bet for bet in bets if bet.book == "outlier")
    assert outlier.ev_fair is not None and outlier.ev_fair > 0
    assert "model_negative" in outlier.screens


def test_screens_are_reported_together_not_short_circuited() -> None:
    odds = GameOdds(matchup=MATCHUP)
    odds.add_ml(AWAY, MarketQuote(book="only", american=900))
    bet = price_game(odds, distribution(), home=HOME, away=AWAY)[0]
    reasons = screen(bet)
    assert "unpaired" in reasons
    assert "model_only" in reasons
    assert "longshot" in reasons


def test_thresholds_are_configurable() -> None:
    odds = board(home_ml=-150, away_ml=130)
    odds.add_ml(HOME, MarketQuote(book="outlier", american=-138, opposite_american=118))
    strict = Thresholds(min_execution_ev=0.10)
    bets = apply_screens(
        best_by_line(price_game(odds, distribution(), home=HOME, away=AWAY)), strict
    )
    assert all("no_execution_edge" in bet.screens for bet in bets)


# -- ledger ---------------------------------------------------------------
def priced_entry(**overrides: object) -> LedgerEntry:
    odds = board(home_ml=-150, away_ml=130)
    odds.add_ml(HOME, MarketQuote(book="outlier", american=110, opposite_american=-130))
    bet = buys(best_by_line(price_game(odds, distribution(), home=HOME, away=AWAY)))[0]
    entry = entry_from_bet(bet, season=2025, week=3, date="2025-09-21")
    for key, value in overrides.items():
        setattr(entry, key, value)
    return entry


def test_a_bought_row_carries_both_edges_and_the_fade_price() -> None:
    entry = priced_entry()
    assert entry.tier == "Strong buy"
    assert entry.screens == ""
    assert entry.opposite_odds == -130
    assert entry.ev_fair is not None and entry.ev_model is not None


def test_moneyline_grades_on_the_side_bet() -> None:
    entry = grade(priced_entry(), 27, 20, home=HOME)
    assert entry.result == "win"
    assert entry.pnl == pytest.approx(1.1)
    assert grade(priced_entry(), 20, 27, home=HOME).result == "loss"
    assert grade(priced_entry(), 20, 20, home=HOME).result == "push"


def test_spread_and_total_pushes_are_refunded_not_lost() -> None:
    spread = LedgerEntry(
        season=2025,
        week=1,
        date="2025-09-07",
        matchup=MATCHUP,
        market="spread",
        side=HOME,
        line=-3.0,
        book="dk",
        odds=-110,
        opposite_odds=-110,
        tier="Moderate buy",
        model_prob=0.5,
        fair_prob=0.5,
        ev_model=0.0,
        ev_fair=0.0,
        paired_books=3,
    )
    assert grade(spread, 24, 21, home=HOME).result == "push"
    assert spread.pnl == 0.0
    total = LedgerEntry(**{**spread.__dict__, "market": "total", "side": "over", "line": 45.0})
    assert grade(total, 24, 21, home=HOME).result == "push"
    assert grade(
        LedgerEntry(**{**spread.__dict__, "market": "total", "side": "under", "line": 45.0}),
        24,
        20,
        home=HOME,
    ).result == "win"


def test_clv_is_scored_against_the_devigged_close_on_the_side_taken() -> None:
    entry = apply_close(priced_entry(), -120, 100)
    assert entry.close_prob is not None and entry.clv is not None
    # The close is much shorter than the +110 taken, so the market came our way.
    assert entry.clv > 0
    assert entry.clv_ev is not None and entry.clv_ev > 0


def test_an_unpaired_close_is_recorded_not_invented() -> None:
    """With no other side at the close, the raw implied price is used -- and it
    overstates the closing probability by about half the hold, so CLV reads high.
    Recording the number that exists beats fabricating one; the paired close is
    lower, which is the bias being made visible rather than hidden."""
    unpaired = apply_close(priced_entry(), -120, None)
    paired = apply_close(priced_entry(), -120, 100)
    assert unpaired.close_prob == pytest.approx(0.545455, abs=1e-5)
    assert paired.close_prob is not None and unpaired.close_prob is not None
    assert unpaired.close_prob > paired.close_prob
    assert unpaired.clv is not None and paired.clv is not None
    assert unpaired.clv > paired.clv


def test_ledger_round_trips_through_csv(tmp_path) -> None:
    entry = grade(priced_entry(), 27, 20, home=HOME)
    path = tmp_path / "ledger.csv"
    save_ledger(path, [entry])
    loaded = load_ledger(path)
    assert len(loaded) == 1
    assert loaded[0].matchup == MATCHUP
    assert loaded[0].result == "win"
    assert loaded[0].fair_prob == pytest.approx(entry.fair_prob)


def test_regrading_a_week_replaces_it_instead_of_doubling_it(tmp_path) -> None:
    path = tmp_path / "ledger.csv"
    update_ledger(path, [priced_entry()])
    update_ledger(path, [priced_entry(), priced_entry(book="second")])
    rows = load_ledger(path)
    assert len(rows) == 2
    update_ledger(path, [priced_entry(season=2024, week=1)])
    assert len(load_ledger(path)) == 3


def test_ppv_is_measured_as_lift_over_the_base_rate() -> None:
    entries = [
        grade(priced_entry(tier="Strong buy"), 27, 20, home=HOME),
        grade(priced_entry(tier="Strong buy"), 27, 20, home=HOME),
        grade(priced_entry(tier="Pass"), 20, 27, home=HOME),
        grade(priced_entry(tier="Pass"), 20, 27, home=HOME),
    ]
    row = metrics(entries, lambda e: e.tier == "Strong buy", "Strong")
    assert row.base_rate == pytest.approx(0.5)
    assert row.ppv == pytest.approx(1.0)
    assert row.ppv_lift == pytest.approx(0.5)
    assert row.npv_lift == pytest.approx(0.5)
    assert row.required_win_pct == pytest.approx(1 / 2.1, abs=1e-3)


def test_a_screen_is_graded_by_what_it_rejected() -> None:
    rejected = grade(priced_entry(tier="Pass", screens="longshot"), 20, 27, home=HOME)
    kept = grade(priced_entry(), 27, 20, home=HOME)
    rows = {row.label: row for row in screen_metrics([rejected, kept])}
    assert "screen:longshot" in rows
    # The rejection lost, which is a screen doing its job.
    assert rows["screen:longshot"].wins == 0
    assert rows["screen:longshot"].losses == 1
