"""Hierarchical batter splits: a split regresses toward the hitter, not the league."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from mlb_engine.features.rolling import (
    LEAGUE_RATES,
    build_batter_profile,
    rates_from_events,
)

AS_OF = date(2026, 8, 2)


def _pa_frame(batter_id: int, hit_rate: float, n: int = 200) -> pd.DataFrame:
    """A batter's plate appearances at a fixed single rate, over the last 20 days.

    Alternates home/away and RHP/LHP so every split gets a share of the sample.
    """
    n_hits = int(round(n * hit_rate))
    events = ["single"] * n_hits + ["field_out"] * (n - n_hits)
    return pd.DataFrame(
        {
            "batter": batter_id,
            "events": events,
            "game_date": [AS_OF - timedelta(days=1 + i % 20) for i in range(n)],
            "inning_topbot": ["Bot" if i % 2 else "Top" for i in range(n)],
            "p_throws": ["R" if i % 3 else "L" for i in range(n)],
        }
    )


def test_split_regresses_toward_the_hitter_not_the_league() -> None:
    # A genuinely poor contact hitter: .120 singles/PA against a .140 league.
    weak = _pa_frame(1, hit_rate=0.05)

    flat = build_batter_profile(weak, 1, AS_OF, 21, 21, 42, split_prior=False)
    hier = build_batter_profile(weak, 1, AS_OF, 21, 21, 42, split_prior=True)

    # Under the flat league prior the thin home split is dragged up toward the
    # league single rate; the hierarchical prior leaves him near his own level.
    assert flat.home.p_1b > hier.home.p_1b
    assert hier.home.p_1b < LEAGUE_RATES["1B"]
    assert abs(hier.home.p_1b - hier.overall.p_1b) < abs(
        flat.home.p_1b - flat.overall.p_1b
    )


def test_hierarchy_widens_the_gap_between_a_good_and_a_bad_bat() -> None:
    # The measured failure: the model compressed a 9.3-point realised gap
    # between the worst and best bats into 3.7. The hierarchy widens it.
    weak = build_batter_profile(
        _pa_frame(1, hit_rate=0.05), 1, AS_OF, 21, 21, 42
    )
    strong = build_batter_profile(
        _pa_frame(2, hit_rate=0.22), 2, AS_OF, 21, 21, 42
    )
    weak_flat = build_batter_profile(
        _pa_frame(1, hit_rate=0.05), 1, AS_OF, 21, 21, 42, split_prior=False
    )
    strong_flat = build_batter_profile(
        _pa_frame(2, hit_rate=0.22), 2, AS_OF, 21, 21, 42, split_prior=False
    )

    hier_gap = strong.for_context(True, "R").p_1b - weak.for_context(True, "R").p_1b
    flat_gap = (
        strong_flat.for_context(True, "R").p_1b
        - weak_flat.for_context(True, "R").p_1b
    )
    assert hier_gap > flat_gap


def test_overall_still_regresses_toward_the_league() -> None:
    # Only the splits change target; the top of the hierarchy is unchanged.
    weak = _pa_frame(1, hit_rate=0.05, n=40)
    prof = build_batter_profile(weak, 1, AS_OF, 21, 21, 42)
    assert prof.overall.p_1b > 0.05
    assert prof.overall.p_1b < LEAGUE_RATES["1B"]


def test_rates_from_events_still_sums_to_one_with_a_custom_prior() -> None:
    ev = pd.Series(["single", "home_run", "strikeout", "walk", "field_out"] * 20)
    prior = rates_from_events(pd.Series(["single"] * 10 + ["field_out"] * 90))
    r = rates_from_events(ev, prior.as_dict(), 60.0)
    assert abs(sum(r.as_dict().values()) - 1.0) < 1e-9
