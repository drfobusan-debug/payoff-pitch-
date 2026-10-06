"""Carry the podcast store between machines on the ``engine-state`` branch.

Every league's podcast picks travel together under ``podcasts/``: each episode's
checked picks (one file per episode) and each league's graded ledger. The
transcripts stay on the machine that made them; the checked picks are what a
second machine needs to print and grade a slate.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from engine_common.podcasts.extract import PICKS_DIR
from mlb_engine.state import (
    STATE_BRANCH,
    SyncReport,
    _commit,
    _git,
    _git_ok,
    _worktree,
    merge_dated_csv,
    repo_root,
)

PREFIX = "podcasts"
LEDGERS = "ledgers"
LEDGER_KEY = ("date", "pick_id")
_PUSH_ATTEMPTS = 3
log = logging.getLogger(__name__)


def fill_in_file(remote: Path, local: Path) -> bool:
    """Copy ``remote`` only where this machine has no copy of its own."""
    if not remote.exists() or local.exists():
        return False
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(remote, local)
    return True


def pull_state(root: Path, repo: Path | None = None, branch: str = STATE_BRANCH) -> SyncReport:
    """Bring the branch's podcast picks and ledgers into ``root``, never overwriting a read."""
    state = _worktree(repo or repo_root(), branch) / PREFIX
    pulled: list[str] = []
    for src in sorted((state / PICKS_DIR).glob("*.json")):
        if fill_in_file(src, root / PICKS_DIR / src.name):
            pulled.append(src.name)
    for src in sorted((state / LEDGERS).glob("*.csv")):
        if merge_dated_csv(src, root / LEDGERS / src.name, LEDGER_KEY):
            pulled.append(f"{LEDGERS}/{src.name}")
    return SyncReport(pulled=tuple(pulled))


def push_state(
    root: Path, message: str, repo: Path | None = None, branch: str = STATE_BRANCH
) -> SyncReport:
    """Publish ``root``'s podcast picks and ledgers, re-merging if the branch moved."""
    repo = repo or repo_root()
    for attempt in range(_PUSH_ATTEMPTS):
        state = _worktree(repo, branch)
        if attempt:
            pull_state(root, repo=repo, branch=branch)
        dest_root = state / PREFIX
        pushed: list[str] = []
        for src in sorted((root / PICKS_DIR).glob("*.json")):
            dest = dest_root / PICKS_DIR / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            pushed.append(src.name)
        for src in sorted((root / LEDGERS).glob("*.csv")):
            dest = dest_root / LEDGERS / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            merge_dated_csv(dest, src, LEDGER_KEY)
            shutil.copyfile(src, dest)
            pushed.append(f"{LEDGERS}/{src.name}")
        if not pushed:
            return SyncReport()
        _git(["add", "-A", PREFIX], state)
        if not _git(["status", "--porcelain"], state):
            return SyncReport(pushed=tuple(pushed))
        _commit(state, message)
        if _git_ok(["push", "origin", f"HEAD:{branch}"], state):
            return SyncReport(pushed=tuple(pushed))
    raise RuntimeError(f"push to {branch} rejected after {_PUSH_ATTEMPTS} attempts")


def auto_pull(root: Path, branch: str = STATE_BRANCH) -> SyncReport | None:
    """Sync down, best effort: a podcast store that cannot sync still prints locally."""
    try:
        return pull_state(root, branch=branch)
    except Exception as exc:  # noqa: BLE001 - state sync is never the point of the run
        log.warning("podcast state pull skipped: %s", exc)
        return None


def auto_push(root: Path, message: str, branch: str = STATE_BRANCH) -> SyncReport | None:
    try:
        return push_state(root, message, branch=branch)
    except Exception as exc:  # noqa: BLE001
        log.warning("podcast state push skipped: %s", exc)
        return None


__all__ = ["PREFIX", "auto_pull", "auto_push", "fill_in_file", "pull_state", "push_state"]
