"""The one-click icons have to run the checkout, or refuse.

Franz's Desktop icon was a *copy* of ``run_predictions.command``, so its
``dirname $0/../..`` resolved to ``/Users``: activating the venv failed, nothing
was set -e, and a whole slate got priced by whatever ``mlb-engine`` was on PATH
-- an install from PR #99. The workbook came out with 21 columns instead of 34
and looked like a missing feature rather than a stale binary.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts" / "macos"
GUARDED = ("run_predictions.command", "run_audit.command")


def _fake_checkout(root: Path) -> Path:
    """A tree with the two things the guard looks for, and no engine to run."""
    repo = root / "payoff-pitch-"
    (repo / "mlb_engine").mkdir(parents=True)
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "activate").write_text("export PP_ACTIVATED=1\n")
    (repo / "scripts" / "macos").mkdir(parents=True)
    shutil.copy(SCRIPTS / "_repo.sh", repo / "scripts" / "macos" / "_repo.sh")
    return repo


def _run(script: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, HOME=str(cwd), PATH="/usr/bin:/bin")
    return subprocess.run(
        ["bash", str(script)], cwd=cwd, capture_output=True, text=True, env=env, timeout=60
    )


@pytest.mark.parametrize("name", GUARDED)
def test_a_copy_of_the_icon_refuses_rather_than_pricing_with_anything(
    tmp_path: Path, name: str
) -> None:
    """The failure that matters is the silent one, so this one must not be."""
    _fake_checkout(tmp_path)
    copy = tmp_path / "Desktop" / "PAYOFF PITCH.command"
    copy.parent.mkdir()
    shutil.copy(SCRIPTS / name, copy)

    proc = _run(copy, tmp_path)
    assert proc.returncode == 1, proc.stdout
    assert "cannot find the engine" in proc.stderr
    assert "ln -s" in proc.stderr
    # And it stopped before doing any of the work.
    assert "propsheet" not in proc.stdout and "audit" not in proc.stdout


@pytest.mark.parametrize("name", GUARDED)
def test_a_symlinked_icon_runs_the_checkout_it_points_at(tmp_path: Path, name: str) -> None:
    """Which is why a link is the right way to put one on the Desktop."""
    repo = _fake_checkout(tmp_path)
    shutil.copy(SCRIPTS / name, repo / "scripts" / "macos" / name)
    link = tmp_path / "Desktop" / "PAYOFF PITCH.command"
    link.parent.mkdir()
    link.symlink_to(repo / "scripts" / "macos" / name)

    proc = _run(link, tmp_path)
    assert "cannot find the engine" not in proc.stderr
    # Past the guard, into the checkout, with the venv sourced -- and then failing
    # on the engine itself, which this tree does not have.
    assert "command not found" in proc.stderr or "No module named" in proc.stderr


def test_the_guard_names_where_it_looked(tmp_path: Path) -> None:
    """A checkout without a venv is setup not yet run, not a copied icon."""
    repo = _fake_checkout(tmp_path)
    (repo / ".venv" / "bin" / "activate").unlink()
    shutil.copy(SCRIPTS / "run_predictions.command", repo / "scripts" / "macos" / "x.command")

    proc = _run(repo / "scripts" / "macos" / "x.command", tmp_path)
    assert proc.returncode == 1
    assert str(repo) in proc.stderr
    assert "setup.command" in proc.stderr


ROOT = Path(__file__).resolve().parent.parent
CALLERS = (ROOT / "setup_engine_autorun.sh", ROOT / "scripts" / "macos" / "run_predictions.command")


def _invocations(module: str) -> list[list[str]]:
    """Every ``python -m <module> ...`` line in the shell callers, as argv.

    The autorun script writes its job through a heredoc, so ``\\$day`` is the
    escaped form there; both are read as one placeholder, then a plausible value.
    """
    found = []
    for path in CALLERS:
        for line in path.read_text().splitlines():
            if f"python -m {module}" not in line:
                continue
            tail = line.split(f"python -m {module}", 1)[1].replace("\\", "").rstrip(" \\")
            tail = tail.replace('"$(basename "$pkl")"', "statcast.pkl").replace('"$day"', "2026-09-09")
            found.append(tail.split())
    assert found, f"no call to {module} in {[p.name for p in CALLERS]}"
    return found


def test_the_shell_callers_pass_regen_regression_what_it_parses() -> None:
    """The morning job calls the script the way the script reads its arguments.

    #300 moved ``regen_regression`` to ``--date``/``--statcast`` and left both
    callers on positionals, so the step exited with a usage error every morning
    and no regression PDF went out. The parser is the contract; check it here.
    """
    import scripts.comprehensive_report as cr

    for argv in _invocations("scripts.regen_regression"):
        args = cr.parse_args(argv)
        assert args.date == "2026-09-09"
        assert args.statcast == "statcast.pkl"
