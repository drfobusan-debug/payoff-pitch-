"""A card shortcut's podcast read is capped, so a stalled read never holds the card back."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HELPER = REPO / "scripts" / "macos" / "time_limit.sh"


def _run_for(cmd: str) -> tuple[subprocess.CompletedProcess[str], float]:
    t0 = time.monotonic()
    proc = subprocess.run(
        ["bash", "-c", f'set -uo pipefail; . "{HELPER}"; {cmd}; echo "rc=$?"; echo next step'],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc, time.monotonic() - t0


def test_a_stalled_step_is_stopped_and_the_next_one_runs() -> None:
    proc, took = _run_for("run_for 1 sleep 30")
    assert "rc=124" in proc.stdout
    assert "next step" in proc.stdout
    assert "still running after 1s" in proc.stderr
    assert took < 10


@pytest.mark.parametrize(("cmd", "rc"), [("true", 0), ("false", 1)])
def test_a_step_that_finishes_keeps_its_status_and_is_not_waited_on(cmd: str, rc: int) -> None:
    proc, took = _run_for(f"run_for 30 {cmd}")
    assert f"rc={rc}" in proc.stdout
    assert proc.stderr == ""
    assert took < 10


@pytest.mark.parametrize(
    ("script", "engine"),
    [
        ("scripts/cbb/macos/cbb_card.command", "cbb-engine"),
        ("scripts/nba/macos/nba_card.command", "nba-engine"),
    ],
)
def test_every_card_shortcut_caps_its_podcast_read(script: str, engine: str) -> None:
    lines = (REPO / script).read_text().splitlines()
    calls = [ln for ln in lines if f"{engine} podcast" in ln and not ln.lstrip().startswith("#")]
    assert calls and all(
        ln.lstrip().startswith('run_for "${PODCAST_MAX_SECONDS:-3600}"') for ln in calls
    )
