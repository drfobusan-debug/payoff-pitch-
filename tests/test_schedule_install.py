"""A schedule that cannot run must fail while somebody is watching.

Franz's CFB schedule was installed from ``~/Documents/GitHub/payoff-pitch-``.
All three agents loaded, and every one of them died at exec:

    /bin/bash: .../autorun.command: Operation not permitted

macOS protects ``~/Documents`` against processes launchd starts without a
foreground session. The installer itself is fine -- it runs from Terminal, which
has been granted access -- so the only evidence was a line in an error log,
found because the 09:00 card never arrived.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INSTALLERS = (
    ("nfl", "com.payoffpitch.nfl.week"),
    ("cfb", "com.payoffpitch.cfb.predictions"),
)


def _checkout(at: Path) -> Path:
    """A copy of the scripts tree, deep enough for an installer to run."""
    repo = at / "payoff-pitch-"
    (repo / "scripts").mkdir(parents=True)
    for sport in ("nfl", "cfb", "macos"):
        shutil.copytree(ROOT / "scripts" / sport, repo / "scripts" / sport)
    return repo


def _run(repo: Path, sport: str, home: Path) -> subprocess.CompletedProcess[str]:
    """Install, with a stub ``launchctl`` so nothing is loaded on the test box."""
    bin_dir = home / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "launchctl").write_text("#!/bin/bash\nexit 0\n")
    (bin_dir / "launchctl").chmod(0o755)
    env = dict(os.environ, HOME=str(home), PATH=f"{bin_dir}:/usr/bin:/bin")
    return subprocess.run(
        ["bash", str(repo / "scripts" / sport / "macos" / "install_schedule.command")],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


@pytest.mark.parametrize(("sport", "label"), INSTALLERS)
def test_a_checkout_macos_shields_is_refused_before_a_plist_is_written(
    tmp_path: Path, sport: str, label: str
) -> None:
    repo = _checkout(tmp_path / "Documents" / "GitHub")

    proc = _run(repo, sport, tmp_path)

    assert proc.returncode == 1, proc.stdout
    assert "Operation not permitted" in proc.stderr
    assert str(tmp_path / "Documents") in proc.stderr
    # Refused *before* anything was installed: a loaded agent that never runs is
    # worse than none, because the schedule looks present.
    assert not (tmp_path / "Library" / "LaunchAgents" / f"{label}.plist").exists()


@pytest.mark.parametrize(("sport", "label"), INSTALLERS)
def test_an_unprotected_checkout_installs_and_points_at_itself(
    tmp_path: Path, sport: str, label: str
) -> None:
    repo = _checkout(tmp_path)

    proc = _run(repo, sport, tmp_path)

    assert proc.returncode == 0, proc.stderr
    plist = (tmp_path / "Library" / "LaunchAgents" / f"{label}.plist").read_text()
    assert f"{repo}/scripts/{sport}/macos/autorun.command" in plist
    assert f"<string>{repo}</string>" in plist


@pytest.mark.parametrize(("sport", "_label"), INSTALLERS)
def test_the_credentials_file_is_seeded_but_never_overwritten(
    tmp_path: Path, sport: str, _label: str
) -> None:
    repo = _checkout(tmp_path)
    engine_env = tmp_path / f".{sport}_engine" / "engine.env"

    _run(repo, sport, tmp_path)
    assert "GMAIL_APP_PASSWORD=" in engine_env.read_text()

    engine_env.write_text("GMAIL_APP_PASSWORD=already-mine\n")
    _run(repo, sport, tmp_path)
    assert engine_env.read_text() == "GMAIL_APP_PASSWORD=already-mine\n"
