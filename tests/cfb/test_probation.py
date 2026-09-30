"""Probation: the three tests, and the findings they are and are not allowed to make."""

from __future__ import annotations

from cfb_engine.audit.ledger import LedgerEntry
from cfb_engine.audit.probation import (
    CANDIDATE_SCREENS,
    CLEAR,
    LIFT,
    SHIP,
    SHUT,
    WATCHING,
    CandidateScreen,
    candidate_probation,
    market_probation,
    probation_findings,
    screen_probation,
)
from cfb_engine.market.tiers import Tier


def _entry(
    *,
    result: str,
    date: str,
    market: str = "game_ml",
    odds: float = 100.0,
    tier: str = Tier.MODERATE.value,
    drift: float | None = None,
    pass_gate: str | None = None,
    sharp_div: float | None = None,
) -> LedgerEntry:
    pnl = {"win": 1.0, "loss": -1.0}.get(result, 0.0)
    return LedgerEntry(
        date=date,
        matchup="Alabama vs Georgia",
        category="Moneyline",
        market=market,
        selection="Georgia ML",
        line=None,
        book="pinnacle",
        odds=odds,
        under_odds=-110,
        tier=tier,
        model_prob=0.55,
        ev=0.05,
        result=result,
        pnl=pnl,
        drift=drift,
        pass_gate=pass_gate,
        sharp_div=sharp_div,
    )


def _run(pattern: str, *, start_day: int = 1, **kw: object) -> list[LedgerEntry]:
    """One row per character of ``pattern`` (``w``/``l``), oldest first."""
    out: list[LedgerEntry] = []
    for i, ch in enumerate(pattern):
        day = start_day + i
        out.append(
            _entry(
                result="win" if ch == "w" else "loss",
                date=f"2025-09-{day % 28 + 1:02d}",
                **kw,  # type: ignore[arg-type]
            )
        )
    # Dates repeat across a long pattern, so the order the caller wrote is what
    # the halves must respect; _halves sorts stably on date.
    for i, e in enumerate(out):
        e.date = f"2025-{9 + i // 28:02d}-{i % 28 + 1:02d}"
    return out


def test_a_bad_number_on_a_small_sample_does_nothing() -> None:
    """The condition that matters most, because it is the one people talk round.

    Twenty straight losers is as bad as a market can look, and it still buys no
    action: below the volume bar the verdict is WATCHING, full stop.
    """
    rows = _run("l" * 20)
    verdicts = market_probation(rows, min_n=100)
    assert [v.status for v in verdicts] == [WATCHING]
    assert verdicts[0].roi == -1.0
    assert probation_findings(rows) == []


def test_a_market_losing_consistently_is_shut() -> None:
    rows = _run("l" * 30 + "w" * 5 + "l" * 30 + "w" * 5)
    (verdict,) = market_probation(rows, min_n=50)
    assert verdict.status == SHUT
    assert verdict.first_half < 0 and verdict.second_half < 0
    assert "shut game_ml until the refit" in verdict.finding


def test_a_market_whose_halves_disagree_is_left_alone() -> None:
    """The test that caught every false finding the MLB engine produced.

    Pooled, this market is a loser. Split, it won in the first half and lost in
    the second -- which is what a window cut in the wrong place looks like, so
    nothing happens.
    """
    rows = _run("w" * 24 + "l" * 6 + "l" * 26 + "w" * 4)
    (verdict,) = market_probation(rows, min_n=50)
    assert verdict.status == CLEAR
    assert verdict.roi < 0
    assert verdict.first_half > 0 > verdict.second_half
    assert "the halves disagree" in verdict.finding


def test_a_market_inside_one_standard_error_is_left_alone() -> None:
    """Barely negative in both halves is not evidence of anything."""
    rows = _run(("wl" * 24 + "ll") * 2)
    (verdict,) = market_probation(rows, min_n=50)
    assert verdict.status == CLEAR
    assert "inside one standard error of zero" in verdict.finding


def test_a_screen_refusing_winners_is_lifted() -> None:
    """A screen deleting money is as expensive as a market losing it, and it
    shows up nowhere in a scorecard that only counts the bets we made."""
    rows = _run("w" * 30 + "l" * 5 + "w" * 30 + "l" * 5, pass_gate="drift_gate")
    (verdict,) = screen_probation(rows, min_n=50)
    assert verdict.status == LIFT
    assert "deleting money" in verdict.finding


def test_the_absence_of_a_market_is_not_a_screen() -> None:
    """``unpriced`` and ``tier_downgrade`` are not decisions anyone can lift."""
    rows = _run("w" * 60, pass_gate="unpriced") + _run("w" * 60, pass_gate="tier_downgrade")
    assert screen_probation(rows, min_n=10) == []


def test_a_candidate_screen_is_graded_before_it_refuses_anything() -> None:
    """The drift gate's veto half, judged on the buys it would have refused.

    The rows it targets lose in both halves and by more than a standard error,
    so it ships; the rows it does not target are winners and must not count
    toward the verdict.
    """
    adverse = _run("l" * 25 + "w" * 3 + "l" * 25 + "w" * 3, drift=-0.05)
    benign = _run("w" * 40, drift=0.0)
    candidate = CandidateScreen(
        "drift_refuse_adverse_2pct",
        lambda e: e.drift is not None and e.drift <= -0.02,
        "the number moved against us before we bet it",
    )
    (verdict,) = candidate_probation(adverse + benign, (candidate,), min_n=50)
    assert verdict.status == SHIP
    assert verdict.n == 56
    assert "ship drift_refuse_adverse_2pct" in verdict.finding


def test_a_candidate_that_would_refuse_winners_is_not_shipped() -> None:
    candidate = CandidateScreen(
        "ml_refuse_dogs_longer_than_+200",
        lambda e: e.market == "game_ml" and e.odds is not None and e.odds >= 200,
        "a long dog's EV is dominated by the tail we model worst",
    )
    rows = _run("w" * 40 + "l" * 20, odds=250.0)
    (verdict,) = candidate_probation(rows, (candidate,), min_n=50)
    assert verdict.status != SHIP


def test_the_stricter_sharp_money_bar_accrues_as_a_candidate() -> None:
    """The live gate refuses at 0; whether it belongs at +5 is the ledger's call,
    and only moneyline rows carrying a split can answer it."""
    (candidate,) = [c for c in CANDIDATE_SCREENS if c.name == "ml_refuse_divergence_under_+5"]
    thin = _run("l" * 25 + "w" * 3 + "l" * 25 + "w" * 3, sharp_div=2.0)
    backed = _run("w" * 40, sharp_div=19.0)
    ats = _run("l" * 40, market="game_ats", sharp_div=2.0)
    (verdict,) = candidate_probation(thin + backed + ats, (candidate,), min_n=50)
    assert verdict.status == SHIP
    assert verdict.n == 56


def test_a_moneyline_row_with_no_split_is_not_evidence_about_the_bar() -> None:
    (candidate,) = [c for c in CANDIDATE_SCREENS if c.name == "ml_refuse_divergence_under_+5"]
    (verdict,) = candidate_probation(_run("l" * 60), (candidate,), min_n=10)
    assert (verdict.status, verdict.n) == (WATCHING, 0)


def test_the_window_can_start_at_the_day_a_screen_was_changed() -> None:
    """A screen's record before it was changed is a different engine's record."""
    rows = _run("l" * 60)
    for e in rows[:30]:
        e.date = "2025-08-01"
    for e in rows[30:]:
        e.date = "2025-10-01"
    assert market_probation(rows, min_n=25)[0].n == 60
    assert market_probation(rows, "2025-09-01", min_n=25)[0].n == 30


def test_pushes_and_unpriced_rows_are_excluded() -> None:
    """A push returned nothing and an unpriced row was graded at a price nobody
    offered; both would dilute every ROI here toward zero."""
    rows = _run("l" * 20)
    rows.append(_entry(result="push", date="2025-09-01"))
    rows.append(_entry(result="loss", date="2025-09-02", odds=None))  # type: ignore[arg-type]
    assert market_probation(rows, min_n=10)[0].n == 20


def test_a_shipped_candidate_carries_the_reason_it_was_proposed() -> None:
    candidate = CandidateScreen(
        "drift_refuse_adverse_2pct",
        lambda e: e.drift is not None and e.drift <= -0.02,
        "the number moved against us before we bet it",
    )
    rows = _run("l" * 25 + "w" * 3 + "l" * 25 + "w" * 3, drift=-0.05)
    (verdict,) = candidate_probation(rows, (candidate,), min_n=50)
    assert verdict.finding.endswith("(the number moved against us before we bet it)")
