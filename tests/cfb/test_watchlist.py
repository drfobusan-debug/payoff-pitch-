"""The slate's "What we're watching" snapshot of every graded-only rule."""

from __future__ import annotations

from datetime import date, timedelta

from cfb_engine.audit import watchlist
from cfb_engine.audit.ledger import LedgerEntry
from cfb_engine.market import lineagree, lowtotal
from cfb_engine.market.tiers import Tier
from cfb_engine.output.card import build_article
from cfb_engine.recommendations import Recommendation

DAY = date(2026, 10, 17)


def _entry(i: int, **kw: object) -> LedgerEntry:
    base: dict[str, object] = dict(
        date=(DAY - timedelta(days=30 - i)).isoformat(),
        matchup=f"A{i} @ H{i}",
        category="Totals",
        market="game_total",
        selection="Under 46.5",
        line=46.5,
        book="b",
        odds=-110.0,
        tier=Tier.MODERATE.value,
        model_prob=0.55,
        ev=0.03,
        result="loss",
        pnl=-1.0,
        fair_prob=0.5,
    )
    base.update(kw)
    return LedgerEntry(**base)  # type: ignore[arg-type]


def _ledger() -> list[LedgerEntry]:
    low = [_entry(i) for i in range(6)]  # Under buys below 50: the skip screen's rows
    graded = [
        _entry(
            10 + i,
            selection="Over 55.5",
            line=55.5,
            tier=Tier.PASS.value,
            pass_gate="graded_only_game_total",
            result="win",
            pnl=0.9091,
        )
        for i in range(4)
    ]
    live = [_entry(20, market="game_ats", category="Spread (ATS)", selection="H20 -3.0", line=-3.0)]
    return low + graded + live


def test_rows_cover_graded_only_rules_and_markets_but_not_live_markets() -> None:
    rows = {w.name: w for w in watchlist.watch_rows(_ledger())}
    assert "graded_only_game_total" in rows and "game_ats" not in rows and "game_total" not in rows
    assert {lowtotal.RULE_NAME, lineagree.RULE_NAME} <= set(rows)
    skip = rows[lowtotal.RULE_NAME]
    assert (skip.kind, skip.wants, skip.n, skip.status) == (
        "Skip screen",
        "rows lose",
        6,
        "WATCHING",
    )
    assert skip.needs == "94 more graded" and skip.leaning  # its rows lost: the skip looks right
    totals = rows["graded_only_game_total"]
    assert totals.n == 4 and totals.wants == "rows win" and totals.leaning
    assert "would-be total buys" in totals.what
    assert next(iter(rows.values())).name == "graded_only_game_total"  # markets first


def test_the_slate_pdf_leads_with_the_snapshot() -> None:
    rec = Recommendation(
        game_date=DAY,
        game_id="g1",
        matchup="Away @ Home",
        market="game_total",
        selection="Under 46.5",
        model_prob=0.55,
        line=46.5,
        market_american=-110,
        fair_prob=0.5,
        tier=Tier.STRONG,
        home_abbrev="Home",
        away_abbrev="Away",
    )
    html, _ = build_article(DAY, [rec], watching=watchlist.build(_ledger()))
    assert "What we're watching" in html and lowtotal.RULE_NAME in html
    assert html.index("What we're watching") < html.index("class='game'")
    assert "no graded totals with a per-book capture yet" in html
    bare, _ = build_article(DAY, [rec])
    assert "What we're watching" not in bare
