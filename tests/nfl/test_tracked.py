"""Graded-only NFL rules: recorded on the model leans, judged, never bet."""

from __future__ import annotations

from nfl_engine.audit.graduation import MEETS_BAR, NOT_YET, WATCHING
from nfl_engine.audit.ledger import LedgerEntry
from nfl_engine.audit.tracked import (
    ML_LONGSHOT_RULE,
    SPREAD_DOG_GAP_RULE,
    SPREAD_DOG_REFUSED_RULE,
    SPREAD_LEAN_RULE,
    ml_longshot,
    spread_dog_gap,
    spread_dog_refused,
    spread_lean,
    tracked,
    tracked_lines,
)


def lean(i: int = 0, *, result: str = "win", **overrides: object) -> LedgerEntry:
    day = 1 + i // 24
    hour = i % 24
    entry = LedgerEntry(
        season=2026,
        week=1 + i // 16,
        date=f"2026-10-{day:02d}",
        matchup=f"G{i} @ H{i}",
        market="spread",
        side=f"G{i}",
        line=3.5,
        book="dk",
        odds=100.0,
        opposite_odds=-120.0,
        tier="Pass",
        model_prob=0.53,
        fair_prob=0.50,
        ev_model=0.06,
        ev_fair=0.0,
        paired_books=3,
        screens="no_execution_edge",
        result=result,
        pnl=1.0 if result == "win" else -1.0,
        captured_at=f"2026-10-{day:02d}T{hour:02d}:00:00Z",
        kickoff_utc=f"2026-10-{day:02d}T{hour:02d}:50:00Z",
        close_captured_at=f"2026-10-{day:02d}T{hour:02d}:40:00Z",
        clv=0.01,
    )
    for key, value in overrides.items():
        setattr(entry, key, value)
    return entry


def test_spread_rule_takes_dogs_with_a_two_to_four_point_gap_only() -> None:
    assert spread_dog_gap(lean())
    assert spread_dog_gap(lean(model_prob=0.52))
    assert not spread_dog_gap(lean(model_prob=0.54))  # the band is half-open, as in the audit
    assert not spread_dog_gap(lean(model_prob=0.515))
    assert not spread_dog_gap(lean(line=-3.5))  # laying points
    assert not spread_dog_gap(lean(fair_prob=None))
    assert not spread_dog_gap(lean(market="total", side="over", line=44.5))


def test_longshot_rule_takes_leans_the_longshot_screen_refuses() -> None:
    ml = {"market": "moneyline", "line": None, "model_prob": 0.24, "fair_prob": 0.21}
    assert ml_longshot(lean(odds=395.0, **ml))
    assert not ml_longshot(lean(odds=300.0, **ml))  # the cap itself is allowed
    assert not ml_longshot(lean(odds=395.0, **{**ml, "model_prob": 0.20}))
    assert not ml_longshot(lean(odds=395.0))  # a spread at a long price is not a moneyline


def test_rules_watch_until_volume_then_apply_the_graduation_bar() -> None:
    few = tracked([lean(i) for i in range(10)])
    by = {t.name: t for t in few}
    assert by[SPREAD_DOG_GAP_RULE].verdict.status == WATCHING
    assert by[SPREAD_DOG_GAP_RULE].wins == 10
    assert by[ML_LONGSHOT_RULE].verdict.n == 0

    winning = [lean(i, result="win" if i % 3 else "loss") for i in range(120)]
    assert tracked(winning)[0].verdict.status == MEETS_BAR
    late_collapse = [lean(i, result="win" if i < 60 else "loss") for i in range(120)]
    verdict = tracked(late_collapse)[0].verdict
    assert verdict.status == NOT_YET
    assert "consistency" in verdict.failed


def test_pushes_and_post_kickoff_prices_are_not_evidence() -> None:
    rows = [
        lean(0),
        lean(1, result="push", pnl=0.0),
        lean(2, captured_at="2026-10-01T03:00:00Z", kickoff_utc="2026-10-01T02:50:00Z"),
    ]
    assert tracked(rows)[0].verdict.n == 1


def test_lines_say_the_rules_change_no_selection() -> None:
    text = "\n".join(tracked_lines([lean(0)]))
    assert SPREAD_DOG_GAP_RULE in text and ML_LONGSHOT_RULE in text
    assert "no selection changes" in text


def test_refused_twin_drops_the_zone_leans_the_engine_bought() -> None:
    assert spread_dog_refused(lean())
    assert not spread_dog_refused(lean(tier="Moderate buy"))
    assert not spread_dog_refused(lean(model_prob=0.54))
    rows = {t.name: t for t in tracked([lean(0), lean(1, tier="Moderate buy")], min_n=2)}
    assert rows[SPREAD_DOG_GAP_RULE].wins == 2
    assert rows[SPREAD_DOG_REFUSED_RULE].wins == 1


def test_spread_lean_takes_any_positive_gap_and_waits_for_the_band_bar() -> None:
    assert spread_lean(lean(model_prob=0.505, fair_prob=0.50))
    assert spread_lean(lean(line=-6.5, model_prob=0.58, fair_prob=0.50))  # favourites too
    assert not spread_lean(lean(model_prob=0.49, fair_prob=0.50))
    assert not spread_lean(lean(market="total"))
    rows = [lean(i) for i in range(120)]
    by_name = {t.name: t for t in tracked(rows)}
    assert by_name[SPREAD_LEAN_RULE].min_n == 150
    assert by_name[SPREAD_LEAN_RULE].verdict.status == WATCHING
    assert by_name[SPREAD_DOG_GAP_RULE].min_n == 100
    line = next(x for x in tracked_lines(rows) if SPREAD_LEAN_RULE in x)
    assert "need 150" in line and "120-0" in line
