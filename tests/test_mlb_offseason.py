"""The Mac runner skips MLB steps in the off-season window (scripts/macos/run_engine.sh)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "macos" / "run_engine.sh"


def _helper() -> str:
    text = RUNNER.read_text()
    m = re.search(r"^MLB_OFFSEASON_FROM=.*?^mlb_off_note\(\).*?$", text, re.S | re.M)
    assert m is not None
    return m.group(0)


def _in_season(today: str, **env: str) -> bool:
    script = f"{_helper()}\nmlb_in_season"
    run_env = {"MLB_TODAY_MMDD": today, "PATH": "/usr/bin:/bin", **env}
    return subprocess.run(["bash", "-c", script], env=run_env).returncode == 0


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        ("11-04", True),
        ("11-05", False),
        ("12-31", False),
        ("01-15", False),
        ("03-14", False),
        ("03-15", True),
        ("07-04", True),
    ],
)
def test_the_default_window_wraps_the_new_year(today: str, expected: bool) -> None:
    assert _in_season(today) is expected


def test_force_and_a_moved_window_reopen_mlb() -> None:
    assert _in_season("12-01", MLB_SEASON_FORCE="1")
    assert _in_season("03-01", MLB_OFFSEASON_TO="02-20")
    assert not _in_season("06-01", MLB_OFFSEASON_FROM="05-01", MLB_OFFSEASON_TO="07-01")


def test_every_mlb_mode_is_guarded() -> None:
    text = RUNNER.read_text()
    assert text.count("mlb_in_season") >= 5  # close, react, slate-*, morning (+ the definition)
