"""A losing dog and a winning favorite are both judged against their price."""

from __future__ import annotations

from mlb_engine.audit.analysis import (
    dog_vs_favorite,
    price_bucket_findings,
    price_buckets,
)
from mlb_engine.audit.ledger import LedgerEntry
from mlb_engine.market.odds import american_to_decimal
from mlb_engine.market.tiers import Tier


def _bet(
    odds: float | None,
    result: str,
    *,
    tier: str = Tier.STRONG.value,
    date: str = "2026-08-07",
) -> LedgerEntry:
    dec = american_to_decimal(odds) if odds is not None else 1.91
    return LedgerEntry(
        date=date,
        matchup="KC @ DET",
        category="Moneyline",
        market="game_ml",
        selection="KC ML",
        line=None,
        book="dk",
        odds=odds,
        tier=tier,
        model_prob=0.55,
        ev=0.05,
        result=result,
        pnl=(dec - 1.0) if result == "win" else (-1.0 if result == "loss" else 0.0),
    )


def test_a_dog_winning_under_half_can_still_be_profitable() -> None:
    """+200 dogs at 40% beat the 33.3% the price demands."""
    bets = [_bet(200.0, "win")] * 4 + [_bet(200.0, "loss")] * 6
    dogs = next(b for b in dog_vs_favorite(bets) if "Underdog" in b.label)
    assert dogs.n == 10
    assert dogs.win_rate == 0.4
    assert round(dogs.breakeven, 4) == 0.3333
    assert dogs.shortfall > 0
    assert round(dogs.roi, 3) == 0.2  # 4 x +2u - 6 x 1u = +2u over 10 bets


def test_a_favorite_winning_over_half_can_still_be_losing() -> None:
    """-200 favorites at 60% miss the 66.7% the price demands."""
    bets = [_bet(-200.0, "win")] * 6 + [_bet(-200.0, "loss")] * 4
    favs = next(b for b in dog_vs_favorite(bets) if "Favorite" in b.label)
    assert favs.win_rate == 0.6
    assert favs.shortfall < 0
    assert favs.roi < 0


def test_buckets_split_by_price_length_and_skip_empty_bands() -> None:
    bets = [
        _bet(-250.0, "win"),
        _bet(-150.0, "win"),
        _bet(-105.0, "loss"),
        _bet(150.0, "win"),
        _bet(600.0, "loss"),
    ]
    labels = [b.label for b in price_buckets(bets)]
    assert labels == [
        "Heavy favorite (-200 and shorter)",
        "Favorite (-199 to -110)",
        "Pick'em (-109 to +109)",
        "Short dog (+110 to +199)",
        "Longshot (+400 and up)",
    ]
    assert all(b.n == 1 for b in price_buckets(bets))


def test_assumed_prices_and_non_buys_are_excluded() -> None:
    """Only rows with a real book price and a buy tier can be measured."""
    bets = [
        _bet(None, "win"),  # graded at an assumed -110
        _bet(-120.0, "win", tier=Tier.PASS.value),
        _bet(-120.0, "loss", tier=Tier.PASS.value),
    ]
    assert price_buckets(bets) == []
    assert dog_vs_favorite(bets) == []


def test_pushes_do_not_count_as_losses() -> None:
    bets = [_bet(110.0, "win")] * 5 + [_bet(110.0, "push")] * 5
    dogs = next(b for b in dog_vs_favorite(bets) if "Underdog" in b.label)
    assert dogs.n == 5
    assert dogs.win_rate == 1.0


def test_findings_name_a_leaking_band_and_stay_quiet_on_thin_ones() -> None:
    thin = [_bet(150.0, "loss")] * 5
    assert price_bucket_findings(thin) == []

    # 20 short dogs at 20%, needing 40%: a real leak, worth naming.
    leaking = [_bet(150.0, "win")] * 4 + [_bet(150.0, "loss")] * 16
    found = " ".join(price_bucket_findings(leaking))
    assert "Underdog buys win 20.0%" in found
    assert "need 40.0%" in found
    assert "Short dog (+110 to +199)" in found


def test_a_band_that_clears_its_price_is_not_flagged() -> None:
    winning = [_bet(150.0, "win")] * 10 + [_bet(150.0, "loss")] * 10
    found = " ".join(price_bucket_findings(winning))
    assert "clearing the bar the price sets" in found
    assert "cap buys in this band" not in found


def test_the_report_renders_the_price_section_off_the_whole_ledger() -> None:
    """One slate is too thin to bucket, so the section reads the history."""
    from mlb_engine.output.report import (
        build_report_data,
        render_html_report,
        render_markdown_report,
    )

    today = [_bet(150.0, "loss", date="2026-08-08")]
    history = today + [_bet(150.0, "win")] * 4 + [_bet(150.0, "loss")] * 16
    data = build_report_data(
        today, period_label="Daily", subtitle="x", history=history
    )
    assert data.price_n_dates == 2
    dogs = next(b for b in data.price_sides if "Underdog" in b.label)
    assert dogs.n == 21  # the whole ledger, not tonight's single bet

    md = render_markdown_report(data)
    assert "## Price buckets" in md
    assert "| Underdogs (plus money) | 21 |" in md
    assert "Short dog (+110 to +199)" in md
    assert "<h2>Price buckets" in render_html_report(data)


def test_the_price_section_falls_back_to_the_period_when_no_history_is_given() -> None:
    from mlb_engine.output.report import build_report_data

    bets = [_bet(150.0, "win")] * 3
    data = build_report_data(bets, period_label="Daily", subtitle="x")
    assert [b.n for b in data.price_sides] == [3]
