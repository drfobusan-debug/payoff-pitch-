"""The starter form window is six weeks, and it is what the pipeline reads.

Measured on 2,894 starts with every profile rebuilt from the days before the
start, the 42-day read beats the 28-day one on every held-out target, and a
54-slate replay moves favoured PPV .5831 -> .5867 (95% CI [+0.04, +0.64] pp).
A regression back to 28 would be silent, so it is pinned here.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from mlb_engine.config import Config, RollingWindows
from mlb_engine.features.rolling import build_pitcher_profile

AS_OF = date(2026, 7, 28)


def test_starter_window_defaults_to_six_weeks() -> None:
    assert RollingWindows().pitcher_form_days == 42
    assert Config().windows.pitcher_form_days == 42


def test_starter_window_is_env_overridable(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_PITCHER_FORM_DAYS", "28")
    assert RollingWindows().pitcher_form_days == 28


def _starter_frame() -> pd.DataFrame:
    """One starter: strikeouts 30-40 days back, contact inside the last 3 weeks."""
    rows: list[dict[str, object]] = []
    for back, event in [(d, "strikeout") for d in range(30, 41)] + [
        (d, "single") for d in range(2, 20)
    ]:
        rows.append(
            {
                "game_date": AS_OF - pd.Timedelta(days=back),
                "pitcher": 7,
                "batter": 900 + back,
                "events": event,
                "description": "swinging_strike" if event == "strikeout" else "hit_into_play",
                "inning": 1,
                "inning_topbot": "Top",
                "release_speed": 94.0,
                "estimated_woba_using_speedangle": 0.200 if event == "strikeout" else 0.500,
                "woba_value": 0.0 if event == "strikeout" else 0.9,
                "woba_denom": 1.0,
                "launch_speed": None if event == "strikeout" else 100.0,
                "launch_angle": None if event == "strikeout" else 12.0,
            }
        )
    return pd.DataFrame(rows)


def test_six_week_window_sees_starts_a_four_week_window_misses() -> None:
    frame = _starter_frame()
    four = build_pitcher_profile(frame, 7, AS_OF, 28)
    six = build_pitcher_profile(frame, 7, AS_OF, 42)
    assert six.allowed.pa > four.allowed.pa
    # The strikeouts live 30-40 days back, so only the six-week read carries them.
    assert six.allowed.p_k > four.allowed.p_k
