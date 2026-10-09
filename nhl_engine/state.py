"""Carry the NHL price archive and audit ledger between machines.

Same orphan ``engine-state`` branch as the other engines, under ``nhl/prices/``
and ``nhl/ledger/``. Snapshot files are immutable and content-addressed
(``<label>_<stamp>_<fp>.csv``), so merging is a set union of file names: pull
copies what the branch has and this machine lacks, push the reverse. Nothing is
rewritten; two machines capturing the same slate produce two interleaved,
equally true histories.

The ledger (``predictions_<date>.json`` write-once, ``graded_<date>.json``
rewritten on every audit) is pushed when missing *or changed* -- the machine
that grades is the authority -- and pulled only where missing locally.

Only data goes on the branch, never code. The git plumbing is the MLB module's.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from mlb_engine.state import (
    STATE_BRANCH,
    SyncReport,
    _commit,
    _git,
    _git_ok,
    _worktree,
    repo_root,
)
from nhl_engine.data.capture import prices_dir

PREFIX = "nhl"
PRICES_DIR = "prices"
LEDGER_DIR = "ledger"
LEDGER_GLOBS = ("predictions_*.json", "graded_*.json")
_PUSH_ATTEMPTS = 3
log = logging.getLogger(__name__)


def _copy_missing(src_root: Path, dest_root: Path) -> list[str]:
    copied: list[str] = []
    if not src_root.is_dir():
        return copied
    for src in sorted(src_root.glob("*/*.csv")):
        rel = src.relative_to(src_root)
        dest = dest_root / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        copied.append(str(rel))
    return copied


def _copy_ledger(src_root: Path, dest_root: Path, *, overwrite: bool) -> list[str]:
    copied: list[str] = []
    if not src_root.is_dir():
        return copied
    for pattern in LEDGER_GLOBS:
        for src in sorted(src_root.glob(pattern)):
            dest = dest_root / src.name
            if dest.exists() and (not overwrite or dest.read_bytes() == src.read_bytes()):
                continue
            dest_root.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            copied.append(f"{LEDGER_DIR}/{src.name}")
    return copied


def pull_state(data_dir: Path, repo: Path | None = None, branch: str = STATE_BRANCH) -> SyncReport:
    repo = repo or repo_root()
    state = _worktree(repo, branch)
    pulled = _copy_missing(state / PREFIX / PRICES_DIR, prices_dir(data_dir))
    pulled += _copy_ledger(state / PREFIX / LEDGER_DIR, data_dir / LEDGER_DIR, overwrite=False)
    return SyncReport(pulled=tuple(pulled))


def push_state(
    data_dir: Path, message: str, repo: Path | None = None, branch: str = STATE_BRANCH
) -> SyncReport:
    repo = repo or repo_root()
    for attempt in range(_PUSH_ATTEMPTS):
        state = _worktree(repo, branch)
        if attempt:
            pull_state(data_dir, repo=repo, branch=branch)
        pushed = _copy_missing(prices_dir(data_dir), state / PREFIX / PRICES_DIR)
        pushed += _copy_ledger(data_dir / LEDGER_DIR, state / PREFIX / LEDGER_DIR, overwrite=True)
        if not pushed:
            return SyncReport()
        _git(["add", "-A", PREFIX], state)
        if not _git(["status", "--porcelain"], state):
            return SyncReport(pushed=tuple(pushed))
        _commit(state, message)
        if _git_ok(["push", "origin", f"HEAD:{branch}"], state):
            return SyncReport(pushed=tuple(pushed))
    raise RuntimeError(f"push to {branch} rejected after {_PUSH_ATTEMPTS} attempts")


def auto_pull(data_dir: Path, branch: str = STATE_BRANCH) -> SyncReport | None:
    """Best effort: no remote, branch or credentials means capture locally, not fail."""
    try:
        return pull_state(data_dir, branch=branch)
    except Exception as exc:  # noqa: BLE001 - state sync is never the point of the run
        log.warning("state pull skipped: %s", exc)
        return None


def auto_push(data_dir: Path, message: str, branch: str = STATE_BRANCH) -> SyncReport | None:
    try:
        return push_state(data_dir, message, branch=branch)
    except Exception as exc:  # noqa: BLE001
        log.warning("state push skipped: %s", exc)
        return None


__all__ = [
    "LEDGER_DIR",
    "PREFIX",
    "STATE_BRANCH",
    "SyncReport",
    "auto_pull",
    "auto_push",
    "pull_state",
    "push_state",
]
