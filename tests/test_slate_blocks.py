"""The slate passes tile the day: every first pitch is priced by exactly one."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mlb_engine import slate_blocks
from mlb_engine.features.lineup_lock import LineupLockGate
from mlb_engine.pipeline import Pipeline
from mlb_engine.slate_blocks import BLOCK_NAMES, BLOCKS, WINDOW_HOURS

ROOT = Path(__file__).resolve().parents[1]


def _run_times() -> list[datetime]:
    day = datetime(2026, 8, 24).astimezone()
    out = []
    for b in BLOCKS.values():
        hh, mm = map(int, b.run_at.split(":"))
        out.append(day.replace(hour=hh, minute=mm, second=0, microsecond=0))
    return out


def _owners(first_pitch: datetime) -> list[str]:
    """Which passes would price a game at ``first_pitch``: the ones it starts inside
    WINDOW_HOURS of, not yet underway -- the pipeline's ``--within-hours`` rule."""
    owners = []
    for name, run in zip(BLOCK_NAMES, _run_times(), strict=True):
        hours = (first_pitch - run).total_seconds() / 3600
        if 0.0 <= hours < WINDOW_HOURS:
            owners.append(name)
    return owners


def test_the_pipeline_prices_inside_the_window_and_not_at_its_edge() -> None:
    now = datetime.now().astimezone()
    inside = (now + timedelta(hours=WINDOW_HOURS - 0.05)).isoformat()
    edge = (now + timedelta(hours=WINDOW_HOURS + 0.001)).isoformat()
    started = (now - timedelta(minutes=5)).isoformat()
    assert Pipeline._starts_within(inside, WINDOW_HOURS) is True
    assert Pipeline._starts_within(edge, WINDOW_HOURS) is False
    assert Pipeline._starts_within(started, WINDOW_HOURS) is False


@pytest.mark.parametrize(
    "hhmm, owner",
    [
        ("12:05", "matinee"),
        ("13:10", "matinee"),
        ("14:54", "matinee"),
        ("14:55", "afternoon"),  # exactly on the boundary: the later pass owns it
        ("16:05", "afternoon"),
        ("17:54", "afternoon"),
        ("17:55", "evening"),
        ("19:05", "evening"),
        ("20:54", "evening"),
        ("20:55", "late"),
        ("21:40", "late"),
        ("22:10", "late"),
        ("23:50", "late"),
    ],
)
def test_every_first_pitch_belongs_to_exactly_one_pass(hhmm: str, owner: str) -> None:
    hh, mm = map(int, hhmm.split(":"))
    pitch = datetime(2026, 8, 24, hh, mm).astimezone()
    assert _owners(pitch) == [owner]


def test_the_passes_are_one_window_apart_and_start_before_the_first_pitch() -> None:
    runs = _run_times()
    assert runs[0].hour == 11 and runs[0].minute < 60
    for earlier, later in zip(runs, runs[1:], strict=False):
        assert later - earlier == timedelta(hours=WINDOW_HOURS)


def test_the_window_is_the_clock_gate() -> None:
    assert LineupLockGate.from_env().stale_hours == WINDOW_HOURS


def test_an_unknown_block_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="matinee, afternoon, evening, late"):
        slate_blocks.block("night")


def test_the_installer_schedules_the_same_passes() -> None:
    text = (ROOT / "setup_engine_autorun.sh").read_text()
    m = re.search(r'^SLATE_RUNS="([^"]+)"', text, re.M)
    assert m is not None
    runs = dict(kv.split("=") for kv in m.group(1).split())
    assert runs == {b.name: b.run_at for b in BLOCKS.values()}
    m = re.search(r"^SLATE_WINDOW_HOURS=(\d+)", text, re.M)
    assert m is not None and float(m.group(1)) == WINDOW_HOURS
