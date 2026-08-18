"""Per-metric bullpen windows and xwOBA shrinkage.

A three-week relief read is ~270 batters faced across a dozen arms. Measured on
30 pens over 2026-06-16..07-27, that read repeats at r=0.66 for K% and r=0.37 for
xwOBA allowed, so the two belong on different windows and only one of them should
be trusted at face value.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from mlb_engine.config import Config
from mlb_engine.features.rolling import (
    LEAGUE_PEN_XWOBA,
    build_bullpen_profile,
    shrink_pen_xwoba,
)

AS_OF = date(2026, 7, 28)


def _relief_frame() -> pd.DataFrame:
    """NYY relief rows: strikeouts inside 21 days, walks 22-40 days back."""
    rows: list[tuple[date, int, int, str, str, float]] = []
    for back in range(2, 20):  # recent window: whiffs and weak contact
        gd = AS_OF - pd.Timedelta(days=back).to_pytimedelta()
        rows.append((gd, 100, 1, "Top", "single", 0.300))  # starter, excluded
        rows.append((gd, 200, 7, "Top", "strikeout", 0.100))
        rows.append((gd, 201, 8, "Top", "field_out", 0.150))
    for back in range(22, 40):  # skill window only: walks
        gd = AS_OF - pd.Timedelta(days=back).to_pytimedelta()
        rows.append((gd, 200, 7, "Top", "walk", 0.700))
        rows.append((gd, 201, 8, "Top", "walk", 0.700))
    df = pd.DataFrame(
        rows,
        columns=[
            "game_date",
            "pitcher",
            "inning",
            "inning_topbot",
            "events",
            "estimated_woba_using_speedangle",
        ],
    )
    df["batter"] = 1
    df["home_team"] = "NYY"
    df["away_team"] = "BOS"
    return df


def test_shrink_interpolates_between_league_and_observed() -> None:
    raw = 0.360
    assert shrink_pen_xwoba(raw, 1.0) == raw
    assert shrink_pen_xwoba(raw, 0.0) == LEAGUE_PEN_XWOBA
    half = shrink_pen_xwoba(raw, 0.5)
    assert LEAGUE_PEN_XWOBA < half < raw
    # The measured reliability keeps 37% of the distance from league average.
    assert shrink_pen_xwoba(raw, 0.37) == LEAGUE_PEN_XWOBA + 0.37 * (raw - LEAGUE_PEN_XWOBA)


def test_shrink_clamps_the_weight() -> None:
    raw = 0.250
    assert shrink_pen_xwoba(raw, 5.0) == raw
    assert shrink_pen_xwoba(raw, -2.0) == LEAGUE_PEN_XWOBA


def test_shrink_pulls_both_directions() -> None:
    assert shrink_pen_xwoba(0.400, 0.37) < 0.400  # bad pen looks less bad
    assert shrink_pen_xwoba(0.250, 0.37) > 0.250  # good pen looks less good


def test_skill_window_widens_only_the_skill_frame() -> None:
    df = _relief_frame()
    short = build_bullpen_profile(df, "NYY", AS_OF, 21, min_inning=6)
    split = build_bullpen_profile(df, "NYY", AS_OF, 21, min_inning=6, skill_days=42)

    # The results-based rates are identical: both read the same 21 days.
    assert split.allowed.as_dict() == short.allowed.as_dict()
    # The skill frame reaches further back and so picks up the walk-heavy rows.
    assert len(split.skill_frame) > len(short.skill_frame)
    assert (split.skill_frame["events"] == "walk").any()
    assert not (short.skill_frame["events"] == "walk").any()


def test_skill_frame_falls_back_to_the_short_window() -> None:
    df = _relief_frame()
    pen = build_bullpen_profile(df, "NYY", AS_OF, 21, min_inning=6)
    assert pen.skill is None
    assert pen.skill_frame is pen.relief
    # A skill window no longer than the rate window is not worth a second slice.
    same = build_bullpen_profile(df, "NYY", AS_OF, 21, min_inning=6, skill_days=21)
    assert same.skill is None


def test_profile_keeps_the_raw_xwoba_alongside_the_shrunk_one() -> None:
    df = _relief_frame()
    raw_pen = build_bullpen_profile(df, "NYY", AS_OF, 21, min_inning=6)
    assert raw_pen.xwoba_raw is not None
    # Default weight of 1.0 leaves the observed mean untouched.
    assert raw_pen.xwoba_allowed == raw_pen.xwoba_raw

    shrunk = build_bullpen_profile(df, "NYY", AS_OF, 21, min_inning=6, xwoba_shrink=0.37)
    assert shrunk.xwoba_raw == raw_pen.xwoba_raw
    assert shrunk.xwoba_allowed is not None
    assert shrunk.xwoba_raw is not None
    # This pen allows well under league average, so shrinkage moves it up.
    assert shrunk.xwoba_allowed > shrunk.xwoba_raw
    assert shrunk.xwoba_allowed < LEAGUE_PEN_XWOBA


def test_defaults_read_the_pen_at_its_measured_reliability() -> None:
    w = Config().windows
    assert w.bullpen_days == 21
    assert w.bullpen_skill_days == 0
    # A three-week xwOBA repeats at 0.37, and is now used at 0.37.
    assert w.bullpen_xwoba_shrink == 0.37


def test_pen_rates_ship_shrunk_to_the_fitted_priors() -> None:
    """The pen vector the simulator receives is the empirical-Bayes one.

    Out of time (330 team-windows, a 21-day read against the next 21 days), the
    flat-60 read forecast the next three weeks worse than assuming every pen is
    league average; the fitted priors do not.
    """
    cfg = Config()
    assert cfg.pen_shrink is True


def test_the_run_line_gate_still_has_a_threshold_it_can_reach() -> None:
    """Why the underdog-pen gate keeps reading ``xwoba_raw``.

    Its .330 cut was calibrated against unshrunk three-week means. At the
    measured 0.37 weight even the worst pen in baseball lands short of it, so
    reading the shrunk value would retire the gate silently rather than on
    evidence.
    """
    worst = 0.353  # the league's worst pen on 2026-08-16, raw (the 30 span .243-.353)
    cut = Config().runline_gates.dog_pen_xwoba_max
    assert worst > cut
    assert shrink_pen_xwoba(worst, 0.37) < cut
