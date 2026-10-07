"""Carry the NBA archives between machines on the shared ``engine-state`` branch.

Under ``nba/``: ``prices/`` (live snapshots), ``injuries/`` (official reports
and ESPN feed states), ``alerts/`` (the news alarm), ``history/`` (the
historical pull), ``params/`` (versioned fitted values) and ``ledger/`` (priced passes
and graded days). Every file is immutable and
content-addressed or keyed by (event, anchor), so a merge is a set union of
paths: pull copies what the branch has and this machine lacks, push the
reverse. The git plumbing is the MLB module's.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from mlb_engine.state import STATE_BRANCH, SyncReport, _commit, _git, _git_ok, _worktree, repo_root
from nba_engine.config import preseason

PREFIX = "nba"
PRESEASON_PREFIX = "nba/preseason"
# Subtree -> glob of the immutable files it holds.
TREES: dict[str, tuple[str, ...]] = {
    "prices": ("*/*.csv",),
    "injuries": ("*/*.pdf", "*/*.csv"),
    "history": ("events/*.json.gz", "raw/*/*.json.gz"),
    "alerts": ("*/*.csv",),
    "params": ("*/*.json",),
    "ledger": ("*/*.csv.gz",),
}
LIVE: tuple[str, ...] = ("prices", "injuries", "alerts", "ledger")
_PUSH_ATTEMPTS = 3
log = logging.getLogger(__name__)


def _copy_missing(src_root: Path, dest_root: Path, trees: tuple[str, ...]) -> list[str]:
    copied: list[str] = []
    for tree in trees:
        src_tree = src_root / tree
        if not src_tree.is_dir():
            continue
        for pattern in TREES[tree]:
            for src in sorted(src_tree.glob(pattern)):
                rel = src.relative_to(src_root)
                dest = dest_root / rel
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
                copied.append(str(rel))
    return copied


def prefix() -> str:
    """Preseason archives sync under their own subtree, never beside the season's."""
    return PRESEASON_PREFIX if preseason() else PREFIX


def pull_state(
    data_dir: Path,
    repo: Path | None = None,
    branch: str = STATE_BRANCH,
    trees: tuple[str, ...] = LIVE,
) -> SyncReport:
    state = _worktree(repo or repo_root(), branch)
    return SyncReport(pulled=tuple(_copy_missing(state / prefix(), data_dir, trees)))


def push_state(
    data_dir: Path,
    message: str,
    repo: Path | None = None,
    branch: str = STATE_BRANCH,
    trees: tuple[str, ...] = LIVE,
) -> SyncReport:
    repo = repo or repo_root()
    for attempt in range(_PUSH_ATTEMPTS):
        state = _worktree(repo, branch)
        if attempt:
            pull_state(data_dir, repo=repo, branch=branch, trees=trees)
        pushed = _copy_missing(data_dir, state / prefix(), trees)
        if not pushed:
            return SyncReport()
        _git(["add", "-A", prefix()], state)
        if not _git(["status", "--porcelain"], state):
            return SyncReport(pushed=tuple(pushed))
        _commit(state, message)
        if _git_ok(["push", "origin", f"HEAD:{branch}"], state):
            return SyncReport(pushed=tuple(pushed))
    raise RuntimeError(f"push to {branch} rejected after {_PUSH_ATTEMPTS} attempts")


def auto_pull(
    data_dir: Path, branch: str = STATE_BRANCH, trees: tuple[str, ...] = LIVE
) -> SyncReport | None:
    """Best effort: no remote, branch or credentials means work locally, not fail."""
    try:
        return pull_state(data_dir, branch=branch, trees=trees)
    except Exception as exc:  # noqa: BLE001 - state sync is never the point of the run
        log.warning("state pull skipped: %s", exc)
        return None


def auto_push(
    data_dir: Path,
    message: str,
    branch: str = STATE_BRANCH,
    trees: tuple[str, ...] = LIVE,
) -> SyncReport | None:
    try:
        return push_state(data_dir, message, branch=branch, trees=trees)
    except Exception as exc:  # noqa: BLE001
        log.warning("state push skipped: %s", exc)
        return None


__all__ = [
    "LIVE",
    "PREFIX",
    "PRESEASON_PREFIX",
    "STATE_BRANCH",
    "TREES",
    "SyncReport",
    "auto_pull",
    "auto_push",
    "prefix",
    "pull_state",
    "push_state",
]
