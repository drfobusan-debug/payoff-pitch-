"""The hitter's baseline reads its own window, not the longest split's."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from mlb_engine.config import RollingWindows
from mlb_engine.features.rolling import build_batter_profile

AS_OF = date(2026, 8, 2)


def _two_era_frame(batter_id: int = 1) -> pd.DataFrame:
    """A hitter who hits .400 for three weeks after hitting .100 all season.

    300 PA between 90 and 22 days out at a .100 single rate, 100 PA inside the
    last three weeks at .400 -- so a 21-day read and a 90-day read disagree by
    a distance no shrinkage can hide.
    """
    rows = []
    for i in range(300):
        rows.append(
            {
                "events": "single" if i % 10 == 0 else "field_out",
                "game_date": AS_OF - timedelta(days=22 + i % 68),
            }
        )
    for i in range(100):
        rows.append(
            {
                "events": "single" if i % 10 < 4 else "field_out",
                "game_date": AS_OF - timedelta(days=1 + i % 20),
            }
        )
    return pd.DataFrame(
        [
            {
                "batter": batter_id,
                "inning_topbot": "Bot" if i % 2 else "Top",
                "p_throws": "R" if i % 3 else "L",
                **r,
            }
            for i, r in enumerate(rows)
        ]
    )


def test_overall_window_is_read_over_its_own_days() -> None:
    df = _two_era_frame()
    hot = build_batter_profile(df, 1, AS_OF, 21, 21, 21, overall_days=21)
    season = build_batter_profile(df, 1, AS_OF, 21, 21, 21, overall_days=90)

    assert hot.overall.pa == 100
    assert season.overall.pa == 400
    # The three-week read sees the hot streak; the season read barely moves off
    # the hitter's real level, which is the point of the longer baseline.
    assert hot.overall.p_1b > season.overall.p_1b + 0.10


def test_overall_window_defaults_to_the_longest_split() -> None:
    df = _two_era_frame()
    implicit = build_batter_profile(df, 1, AS_OF, 21, 42, 42)
    explicit = build_batter_profile(df, 1, AS_OF, 21, 42, 42, overall_days=42)

    assert implicit.overall.pa == explicit.overall.pa
    assert implicit.overall.p_1b == explicit.overall.p_1b


def test_configured_batter_windows() -> None:
    w = RollingWindows()

    assert w.batter_overall_days == 90
    assert w.batter_vs_rhp_days == 90
    assert w.batter_vs_lhp_days == 90
    # The baseline must not be shorter than the splits that regress toward it.
    assert w.batter_overall_days >= max(
        w.batter_home_away_days, w.batter_vs_rhp_days, w.batter_vs_lhp_days
    )
    # And the slate must already be fetching a frame that long.
    assert w.batter_overall_days <= w.team_split_days
