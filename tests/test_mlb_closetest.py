"""The MLB close test and graduation bar: display-only rules for when a market may be bought."""

from __future__ import annotations

import random

from mlb_engine.audit import closetest
from mlb_engine.audit.ledger import LedgerEntry


def _e(i: int, model: float, close: float, won: bool, *, market: str = "game_ml",
       tier: str = "Pass", gate: str = "", pnl: float | None = None,
       clv_ev: float | None = None, game: int | None = None) -> LedgerEntry:
    return LedgerEntry(
        date=f"2026-{7 + i // 900:02d}-{1 + (i // 30) % 28:02d}",
        matchup=f"G{game if game is not None else i}", category="game", market=market,
        selection=f"S{i}", line=None, book="draftkings", odds=-110.0, tier=tier,
        model_prob=model, ev=0.0, result="win" if won else "loss",
        pnl=(pnl if pnl is not None else (0.909 if won else -1.0)), pass_gate=gate,
        close_prob=close, clv_ev=clv_ev, clv_quarantine="clean" if clv_ev is not None else "",
    )


def _rows(weight: float, n: int = 4000, seed: int = 7) -> list[LedgerEntry]:
    """The truth is the close plus ``weight`` of the model's disagreement with it."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        close = rng.uniform(0.35, 0.65)
        model = min(0.95, max(0.05, close + rng.gauss(0.0, 0.08)))
        truth = close + weight * (model - close)
        out.append(_e(i, model, close, rng.random() < truth))
    return out


def test_a_model_with_no_information_stays_graded_only() -> None:
    fit = closetest.close_fit(_rows(0.0), "game")
    assert abs(fit.slope) < 3 * fit.se


def test_a_model_with_information_earns_an_anchor() -> None:
    fit = closetest.close_fit(_rows(0.8), "game")
    assert fit.slope > fit.se
    assert fit.verdict.startswith("anchor")


def test_stacked_tickets_on_one_game_widen_the_se() -> None:
    flat = _rows(0.0, n=2000)
    stacked = [_e(i, e.model_prob, e.close_prob or 0.5, e.result == "win", game=i // 20)
               for i, e in enumerate(flat)]
    assert closetest.close_fit(stacked, "x").games < closetest.close_fit(flat, "x").games
    assert closetest.close_fit(stacked, "x").se != closetest.close_fit(flat, "x").se


def test_too_few_games_is_underpowered() -> None:
    assert closetest.close_fit([_e(0, 0.6, 0.5, True)], "x").verdict == "underpowered"


def test_graded_only_would_be_buys_count_toward_graduation() -> None:
    rows = [_e(i, 0.6, 0.5, i % 2 == 0, gate="graded_only_game_ml", clv_ev=0.01) for i in range(120)]
    rows.append(_e(999, 0.6, 0.5, True))  # an ordinary pass: not a would-be buy
    (g,) = closetest.graduation(rows)
    assert g.market == "game_ml" and g.n == 120 and g.status == closetest.NOT_YET
    assert "margin" in g.failed  # 50% at -110 loses


def test_a_winning_market_with_clv_meets_the_bar() -> None:
    rows = [_e(i, 0.6, 0.5, i % 3 != 0, tier="Strong buy", clv_ev=0.03 + 0.001 * (i % 5))
            for i in range(150)]
    (g,) = closetest.graduation(rows)
    assert g.status == closetest.MEETS_BAR and not g.failed


def test_under_the_volume_bar_is_watching() -> None:
    (g,) = closetest.graduation([_e(i, 0.6, 0.5, True, tier="Moderate buy") for i in range(20)])
    assert g.status == closetest.WATCHING and g.failed[0] == "volume (20 of 100)"
    assert closetest.graduation_lines([])[0].startswith("  no would-be buys")
