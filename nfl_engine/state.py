"""Carry the NFL prop audit between machines.

The prop board is priced and graded where the Thursday/Sunday job runs, and
``~/.nfl_engine/props`` does not travel: the research rows and their grades are
the only record of whether a book's prop number can be beaten, and until they
sit somewhere a second machine can read, that record is one laptop's log. So the
same orphan ``engine-state`` branch the MLB and CFB engines use carries the prop
files too, under ``nfl/props/``. Pull before the prop leg, push after.

Only data goes on the branch, never code. The git plumbing is the MLB module's;
what differs is the file map and how each kind of file merges. Both prop files
are keyed by week and union row by row: a research file is appended on every
pricing pass, so two machines' copies are two captures and both are true; a
graded file is rewritten whole from the research it was cut from, so a row that
one machine graded and another did not is kept, and the same row graded twice
is the same row.
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
    merge_dated_csv,
    repo_root,
)
from nfl_engine.props import FIELDS as RESEARCH_FIELDS

PREFIX = "nfl"
PROPS_DIR = "props"
_PUSH_ATTEMPTS = 3
# What identifies one priced quote across the two files. The price and the time
# it was captured are part of it: the same rung re-priced an hour later at a moved
# number is a second research row, and its grade is a second grade.
PROP_KEY: tuple[str, ...] = (
    "captured_at",
    "season",
    "week",
    "matchup",
    "market",
    "player",
    "side",
    "line",
    "book",
    "american",
    "basis",
)
_PROP_FILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Research rows repeat every column of a priced quote, so the whole row is
    # its own key and a byte-identical append on two machines collapses to one.
    ("research_*.csv", tuple(RESEARCH_FIELDS)),
    ("graded_*.csv", PROP_KEY),
)
log = logging.getLogger(__name__)


def merge_prop_files(remote: Path, local: Path, key: tuple[str, ...]) -> bool:
    """Union two copies of one week's prop file, row by row, this machine's winning."""
    return merge_dated_csv(remote, local, key, by_date=False)


def _props_dir(data_dir: Path) -> Path:
    return data_dir / PROPS_DIR


def pull_state(
    data_dir: Path,
    repo: Path | None = None,
    branch: str = STATE_BRANCH,
) -> SyncReport:
    """Bring the branch's prop files onto this machine, merging rather than replacing."""
    repo = repo or repo_root()
    state = _worktree(repo, branch)
    root = state / PREFIX / PROPS_DIR
    local = _props_dir(data_dir)
    pulled: list[str] = []
    for pattern, key in _PROP_FILES:
        for src in sorted(root.glob(pattern)):
            if merge_prop_files(src, local / src.name, key):
                pulled.append(src.name)
    return SyncReport(pulled=tuple(pulled))


def push_state(
    data_dir: Path,
    message: str,
    repo: Path | None = None,
    branch: str = STATE_BRANCH,
) -> SyncReport:
    """Publish this machine's prop files, re-merging if the branch moved underneath."""
    repo = repo or repo_root()
    local = _props_dir(data_dir)
    for attempt in range(_PUSH_ATTEMPTS):
        state = _worktree(repo, branch)
        if attempt:
            pull_state(data_dir, repo=repo, branch=branch)
        root = state / PREFIX / PROPS_DIR
        pushed: list[str] = []
        for pattern, key in _PROP_FILES:
            for src in sorted(local.glob(pattern)) if local.exists() else []:
                dest = root / src.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Fold the branch's rows into ours before overwriting it.
                merge_prop_files(dest, src, key)
                shutil.copyfile(src, dest)
                pushed.append(src.name)
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
    """Sync down, best effort: no remote, branch or credentials is a reason to
    grade with local files, never to fail the job."""
    try:
        return pull_state(data_dir, branch=branch)
    except Exception as exc:  # noqa: BLE001 - state sync is never the point of the run
        log.warning("state pull skipped: %s", exc)
        return None


def auto_push(data_dir: Path, message: str, branch: str = STATE_BRANCH) -> SyncReport | None:
    """Sync up, best effort. The local files are written either way."""
    try:
        return push_state(data_dir, message, branch=branch)
    except Exception as exc:  # noqa: BLE001
        log.warning("state push skipped: %s", exc)
        return None


__all__ = [
    "PREFIX",
    "PROP_KEY",
    "STATE_BRANCH",
    "SyncReport",
    "auto_pull",
    "auto_push",
    "merge_prop_files",
    "pull_state",
    "push_state",
]
