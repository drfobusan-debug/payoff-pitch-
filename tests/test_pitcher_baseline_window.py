"""The starter's rate profile and the bullpen read have their own windows.

Graded walk-forward on four cutoffs against the next three weeks, the six-week
starter read carries nothing the 90-day read does not (jointly K +0.15 vs +0.49,
OUT +0.04 vs +0.38), and a three-week relief read carries negative weight next
to a 60-day one (OUT -0.19). Both defaults would regress silently.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from mlb_engine.config import Config, RollingWindows
from mlb_engine.features.rolling import build_pitcher_profile

AS_OF = date(2026, 8, 2)


def _two_era_frame(pitcher_id: int = 7) -> pd.DataFrame:
    """A starter who stopped missing bats: 300 batters faced at a .300 strikeout
    rate between 90 and 22 days out, 100 inside the last three weeks at .050."""
    rows = []
    for i in range(300):
        rows.append(
            {
                "events": "strikeout" if i % 10 < 3 else "field_out",
                "game_date": AS_OF - timedelta(days=22 + i % 68),
            }
        )
    for i in range(100):
        rows.append(
            {
                "events": "strikeout" if i % 20 == 0 else "field_out",
                "game_date": AS_OF - timedelta(days=1 + i % 20),
            }
        )
    return pd.DataFrame([{"pitcher": pitcher_id, **r} for r in rows])


def test_baseline_window_is_read_over_its_own_days() -> None:
    df = _two_era_frame()
    form = build_pitcher_profile(df, 7, AS_OF, 42, 21)
    season = build_pitcher_profile(df, 7, AS_OF, 42, 90)

    assert form.allowed.pa == 100
    assert season.allowed.pa == 400
    # The three-week read prices tonight off the cold stretch alone; the season
    # read stays near the arm's real level, which is what the grading favoured.
    assert season.allowed.p_k > form.allowed.p_k + 0.10


def test_baseline_window_defaults_to_the_form_window() -> None:
    df = _two_era_frame()
    implicit = build_pitcher_profile(df, 7, AS_OF, 42)
    explicit = build_pitcher_profile(df, 7, AS_OF, 42, 42)

    assert implicit.allowed.pa == explicit.allowed.pa
    assert implicit.allowed.p_k == explicit.allowed.p_k


def test_configured_pitcher_and_bullpen_windows() -> None:
    w = RollingWindows()

    assert w.pitcher_baseline_days == 90
    assert w.bullpen_days == 60
    # The form window stays six weeks: it now only drives the trend read, the
    # batters-faced cap and pitch efficiency.
    assert w.pitcher_form_days == 42
    assert Config().windows.pitcher_baseline_days == 90
    # Both reads must fit inside the frame the slate already fetches.
    assert w.pitcher_baseline_days <= w.team_split_days
    assert w.bullpen_days <= w.team_split_days


def test_windows_are_env_overridable(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_PITCHER_BASELINE_DAYS", "42")
    monkeypatch.setenv("MLBE_BULLPEN_DAYS", "21")

    assert RollingWindows().pitcher_baseline_days == 42
    assert RollingWindows().bullpen_days == 21
