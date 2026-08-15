"""A market or screen is condemned only on volume, size and consistency.

The tests that matter here are the ones that refuse to act. Every bad screen
this engine has shipped looked good on a pooled sample of 30-60 bets whose
halves disagreed, so the cases below pin the monitor's silence as tightly as its
verdicts: a losing market with 40 bets, or 200 bets whose halves point opposite
ways, must produce no finding at all.
"""

from __future__ import annotations

from datetime import date as Date
from datetime import timedelta

from mlb_engine.audit.ledger import LedgerEntry
from mlb_engine.audit.probation import (
    ALL_HISTORY,
    CANDIDATE_SCREENS,
    CLEAR,
    DEFAULT_SINCE,
    LIFT,
    SHIP,
    SHUT,
    WATCHING,
    candidate_probation,
    market_probation,
    probation_findings,
    screen_probation,
)
from mlb_engine.calibration import FEATURE_BASIS
from mlb_engine.market.tiers import Tier

# The window probation grades by default: bets before the current feature basis
# belong to a different engine.
START = Date.fromisoformat(DEFAULT_SINCE)


def _e(
    day: int,
    pnl: float,
    *,
    market: str = "batter_2b",
    tier: str = Tier.STRONG.value,
    gate: str = "",
    result: str | None = None,
) -> LedgerEntry:
    return LedgerEntry(
        date=(START + timedelta(days=day)).isoformat(),
        matchup="AWAY @ HOME",
        category="prop",
        market=market,
        selection="Someone TB o1.5",
        line=1.5,
        book="dk",
        odds=-110.0,
        tier=tier,
        model_prob=0.6,
        ev=0.05,
        result=result or ("win" if pnl > 0 else "loss"),
        pnl=pnl,
        pass_gate=gate,
    )


def _run(n: int, pnl: float, *, start_day: int = 0, **kw) -> list[LedgerEntry]:
    return [_e(start_day + i, pnl, **kw) for i in range(n)]


def _ml(
    n: int,
    pnl: float,
    *,
    odds: float,
    start_day: int = 0,
    home: bool = True,
    model_prob: float = 0.6,
    fair_prob: float | None = None,
) -> list[LedgerEntry]:
    """Graded moneyline buys on one side of the matchup, for the candidates."""
    rows = []
    for i in range(n):
        e = _e(start_day + i, pnl, market="game_ml")
        e.selection = f"{'HOME' if home else 'AWAY'} ML"
        e.odds = odds
        e.model_prob = model_prob
        e.fair_prob = fair_prob
        rows.append(e)
    return rows


(_HOME_FLOOR,) = (c for c in CANDIDATE_SCREENS if c.name.startswith("home_ml"))


# --- condition 1: volume ----------------------------------------------------
def test_a_market_losing_badly_on_a_small_sample_is_only_watched() -> None:
    rows = _run(40, -0.5)
    (p,) = market_probation(rows, min_n=100)
    assert p.status == WATCHING
    assert p.n == 40
    assert not p.actionable
    assert "under the 100 needed to judge" in p.finding


def test_watching_verdicts_never_reach_the_findings_list() -> None:
    assert probation_findings(_run(40, -0.5)) == []


# --- condition 2: size ------------------------------------------------------
def test_a_market_within_one_standard_error_of_zero_is_cleared() -> None:
    # Alternating +0.91/-1.00 is a near-coin-flip: the mean is a fraction of the
    # spread, so the SE swamps it.
    rows = [_e(i, 0.91 if i % 2 else -1.0) for i in range(120)]
    (p,) = market_probation(rows, min_n=100)
    assert p.status == CLEAR
    assert "inside one standard error of zero" in p.finding


# --- condition 3: the halves must agree -------------------------------------
def test_a_market_whose_halves_disagree_is_not_shut() -> None:
    # The +1.5 run line's shape: strongly positive, then strongly negative.
    rows = [*_run(60, 0.91, start_day=0), *_run(60, -1.0, start_day=60)]
    (p,) = market_probation(rows, min_n=100)
    assert p.status == CLEAR
    assert "the halves disagree" in p.finding
    assert p.first_half > 0 > p.second_half


def test_a_market_losing_in_both_halves_with_volume_is_shut() -> None:
    rows = [*_run(60, -1.0), *_run(60, -0.2, start_day=60)]
    (p,) = market_probation(rows, min_n=100)
    assert p.status == SHUT
    assert p.actionable
    assert "shut batter_2b until the refit" in p.finding
    assert probation_findings(rows) == [p.finding]


# --- screens are the same test with the sign flipped ------------------------
def test_a_screen_whose_refusals_keep_winning_is_lifted() -> None:
    rows = [
        *_run(60, 0.91, tier=Tier.PASS.value, gate="singles_price_floor"),
        *_run(60, 0.4, start_day=60, tier=Tier.PASS.value, gate="singles_price_floor"),
    ]
    (p,) = screen_probation(rows, min_n=100)
    assert p.status == LIFT
    assert "refusing winners" in p.finding
    assert "lift singles_price_floor" in p.finding


def test_a_screen_deleting_losers_is_cleared_not_lifted() -> None:
    rows = _run(120, -1.0, tier=Tier.PASS.value, gate="hr_price_band")
    (p,) = screen_probation(rows, min_n=100)
    assert p.status == CLEAR
    assert not p.actionable


def test_tier_downgrades_and_unpriced_rows_are_not_screens() -> None:
    rows = [
        *_run(120, 0.91, tier=Tier.PASS.value, gate="tier_downgrade"),
        *_run(120, 0.91, start_day=120, tier=Tier.PASS.value, gate="unpriced"),
    ]
    assert screen_probation(rows, min_n=100) == []


# --- scoping ----------------------------------------------------------------
def test_only_buys_count_toward_a_market_verdict() -> None:
    rows = [
        *_run(120, -1.0, tier=Tier.PASS.value, gate="ev_floor"),
        *_run(30, -1.0),
    ]
    (p,) = market_probation(rows, min_n=100)
    assert p.n == 30  # the 120 passes are the screen's rows, not the market's
    assert p.status == WATCHING


def test_since_excludes_the_record_the_market_had_before_it_changed() -> None:
    rows = [*_run(120, -1.0), *_run(20, 0.91, start_day=200)]
    since = (START + timedelta(days=200)).isoformat()
    (p,) = market_probation(rows, since=since, min_n=100)
    assert p.n == 20
    assert p.status == WATCHING


def test_the_default_window_starts_at_the_feature_basis() -> None:
    # 300 buys from the retired basis plus 20 on the current one: the old record
    # must not be able to condemn (or vouch for) the market as it now stands.
    old = [_e(-i - 1, -1.0) for i in range(300)]
    (p,) = market_probation([*old, *_run(20, -1.0)], min_n=100)
    assert p.n == 20
    assert p.status == WATCHING
    assert FEATURE_BASIS in p.finding


def test_all_history_is_an_explicit_opt_in() -> None:
    old = [_e(-i - 1, -1.0) for i in range(300)]
    (p,) = market_probation([*old, *_run(20, -1.0)], since=ALL_HISTORY, min_n=100)
    assert p.n == 320
    assert p.status == SHUT


def test_pushes_and_unpriced_rows_are_excluded_from_the_sample() -> None:
    rows = [*_run(110, -1.0), *_run(10, 0.0, start_day=110, result="push")]
    (p,) = market_probation(rows, min_n=100)
    assert p.n == 110


def test_each_market_is_judged_on_its_own_rows() -> None:
    rows = [
        *_run(120, -1.0, market="batter_rbi"),
        *_run(120, 0.91, market="batter_h"),
    ]
    verdicts = {p.name: p.status for p in market_probation(rows, min_n=100)}
    assert verdicts == {"batter_rbi": SHUT, "batter_h": CLEAR}


def test_a_proposed_screen_is_graded_on_the_buys_it_would_refuse() -> None:
    """The candidate ships only if the rows it deletes lose in both halves."""
    losing = [
        *_ml(60, -1.0, odds=-110.0),
        *_ml(60, -0.4, start_day=60, odds=-110.0),
    ]
    (p,) = candidate_probation(losing, candidates=(_HOME_FLOOR,), min_n=100)
    assert p.status == SHIP
    assert p.kind == "candidate"
    assert "ship home_ml_refuse_longer_than_-120" in p.finding


def test_a_proposed_screen_whose_halves_disagree_is_not_shipped() -> None:
    """The shape of every floor this engine has had to unship."""
    mixed = [
        *_ml(60, 0.91, odds=-110.0),
        *_ml(60, -1.0, start_day=60, odds=-110.0),
    ]
    (p,) = candidate_probation(mixed, candidates=(_HOME_FLOOR,), min_n=100)
    assert p.status == CLEAR
    assert not p.actionable
    assert "the halves disagree" in p.finding
    assert probation_findings(mixed) == []


def test_a_proposed_screen_only_sees_the_rows_it_would_refuse() -> None:
    rows = [
        *_ml(120, -1.0, odds=-110.0),  # inside the band the floor refuses
        *_ml(120, -1.0, start_day=120, odds=-160.0),  # shorter than the floor
        *_ml(120, -1.0, start_day=240, odds=-110.0, home=False),  # road side
    ]
    (p,) = candidate_probation(rows, candidates=(_HOME_FLOOR,), min_n=100)
    assert p.n == 120


def test_the_registered_candidates_cover_the_two_that_were_asked_for() -> None:
    assert [c.name for c in CANDIDATE_SCREENS] == [
        "home_ml_refuse_longer_than_-120",
        "game_ml_market_anchor_0.5",
    ]


def test_the_market_anchor_candidate_only_refuses_what_the_toll_reprices() -> None:
    """Anchoring costs a buy only when the toll drags it under the vig."""
    keep = _ml(1, -1.0, odds=-110.0, model_prob=0.70, fair_prob=0.60)[0]
    # -110 breaks even at 52.4%, so half the way back to a 50% price is under it.
    refuse = _ml(1, -1.0, odds=-110.0, model_prob=0.545, fair_prob=0.50)[0]
    (anchor,) = (c for c in CANDIDATE_SCREENS if c.name == "game_ml_market_anchor_0.5")
    assert not anchor.refuses(keep)
    assert anchor.refuses(refuse)
    # No devigged price recorded means the question cannot be asked of the row.
    assert not anchor.refuses(_ml(1, -1.0, odds=-110.0, model_prob=0.56)[0])


def test_the_standard_error_shrinks_as_the_sample_grows() -> None:
    small = market_probation([_e(i, 0.91 if i % 2 else -1.0) for i in range(100)], min_n=100)
    big = market_probation([_e(i, 0.91 if i % 2 else -1.0) for i in range(900)], min_n=100)
    assert big[0].se < small[0].se
