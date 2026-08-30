"""The article scripts read their day and their frame off state, not off a constant."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

import pytest

from scripts.slate_inputs import (
    predictions_path,
    resolve_day,
    slate_days,
    statcast_frame,
)


def _slate(audit: Path, day: str, *, pregame: bool = False) -> None:
    (audit / f"previews_{day}.json").write_text("[]")
    suffix = ".pregame.json" if pregame else ".json"
    (audit / f"predictions_{day}{suffix}").write_text("[]")


def test_the_day_defaults_to_the_latest_slate_state_holds(tmp_path):
    _slate(tmp_path, "2026-08-26")
    _slate(tmp_path, "2026-08-29")
    assert slate_days(tmp_path) == [Date(2026, 8, 26), Date(2026, 8, 29)]
    assert resolve_day(tmp_path) == Date(2026, 8, 29)
    assert resolve_day(tmp_path, "2026-08-26") == Date(2026, 8, 26)


def test_a_previews_file_with_no_predictions_is_not_a_slate(tmp_path):
    (tmp_path / "previews_2026-08-29.json").write_text("[]")
    _slate(tmp_path, "2026-08-26")
    assert resolve_day(tmp_path) == Date(2026, 8, 26)


def test_an_ungraded_slate_is_read_from_its_pregame_capture(tmp_path):
    _slate(tmp_path, "2026-08-28", pregame=True)
    assert resolve_day(tmp_path) == Date(2026, 8, 28)
    assert predictions_path(tmp_path, Date(2026, 8, 28)).name == (
        "predictions_2026-08-28.pregame.json"
    )


def test_the_graded_file_wins_over_the_pregame_capture(tmp_path):
    _slate(tmp_path, "2026-08-29", pregame=True)
    (tmp_path / "predictions_2026-08-29.json").write_text("[]")
    assert predictions_path(tmp_path, Date(2026, 8, 29)).name == "predictions_2026-08-29.json"


def test_an_incomplete_day_is_refused_by_name(tmp_path):
    _slate(tmp_path, "2026-08-26")
    (tmp_path / "predictions_2026-08-28.pregame.json").write_text("[]")  # no previews
    with pytest.raises(FileNotFoundError, match="2026-08-28"):
        resolve_day(tmp_path, "2026-08-28")


def test_no_slate_at_all_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_day(tmp_path)


def _frames(cache: Path, names: list[str]) -> None:
    for n in names:
        (cache / n).write_text("")


def test_the_frame_is_the_widest_window_that_ends_before_the_slate(tmp_path):
    # A window running past the slate would describe a hitter with games he had
    # not played when the bet was priced.
    _frames(
        tmp_path,
        [
            "statcast_2026-08-17_2026-08-25.pkl",
            "statcast_2026-07-13_2026-08-26.pkl",
            "statcast_2026-07-13_2026-09-10.pkl",
        ],
    )
    assert statcast_frame(tmp_path, Date(2026, 8, 27)).name == "statcast_2026-07-13_2026-08-26.pkl"
    assert statcast_frame(tmp_path, Date(2026, 8, 25)).name == "statcast_2026-08-17_2026-08-25.pkl"


def test_a_cache_that_is_all_newer_than_the_slate_still_yields_a_frame(tmp_path):
    _frames(tmp_path, ["statcast_2026-07-13_2026-08-26.pkl", "statcast_2026-08-17_2026-08-25.pkl"])
    assert statcast_frame(tmp_path, Date(2026, 6, 1)).name == "statcast_2026-07-13_2026-08-26.pkl"


def test_an_explicit_frame_is_taken_as_given(tmp_path):
    _frames(tmp_path, ["statcast_2026-07-13_2026-08-26.pkl"])
    named = statcast_frame(tmp_path, Date(2026, 8, 29), "statcast_2026-07-13_2026-08-26.pkl")
    assert named == tmp_path / "statcast_2026-07-13_2026-08-26.pkl"
    absolute = statcast_frame(tmp_path, Date(2026, 8, 29), str(tmp_path / "elsewhere.pkl"))
    assert absolute == tmp_path / "elsewhere.pkl"


def test_an_empty_cache_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        statcast_frame(tmp_path, Date(2026, 8, 29))
