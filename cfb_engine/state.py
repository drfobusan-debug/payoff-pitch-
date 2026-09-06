"""Carry the CFB audit's memory between machines.

The card is priced on one box (the Mac's 09:00 launchd job) and read, audited or
re-audited on another, and ``~/.cfb_engine`` does not travel: the machine that
did not price a slate cannot grade it, and its ledger is a single Saturday. So
the same orphan ``engine-state`` branch the MLB engine uses carries the CFB
state too, under its own ``cfb/`` prefix: the pregame predictions, the first-seen
board, the closing snapshots, the availability log, the ledger and the scorecard.
Pull before a command, push after.

Only data goes on the branch, never code. The git plumbing is the MLB module's;
what differs is the file map and how each kind of file merges.
"""

from __future__ import annotations

import gzip
import logging
import shutil
from pathlib import Path

from cfb_engine.audit import snapshot
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

PREFIX = "cfb"
# A slate is ~100 KB; a season fits, and the audit only ever grades yesterday.
PREDICTION_KEEP_DAYS = 120
_PUSH_ATTEMPTS = 3
# The accumulating records and the columns identifying one row of each.
_MERGED_CSVS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ledger.csv", ("date", "matchup", "category", "market", "selection", "line")),
    ("scorecard.csv", ("date", "market", "tier")),
)
AVAILABILITY_NAME = "availability.jsonl"
log = logging.getLogger(__name__)


# --- file-level merges -------------------------------------------------------


def merge_board_files(remote: Path, local: Path) -> bool:
    """Union two first-seen boards, the earlier-published quote winning."""
    if not remote.exists():
        return False
    snapshot.save(snapshot.merge_first_wins(snapshot.load(remote), snapshot.load(local)), local)
    return True


def merge_closing_files(remote: Path, local: Path) -> bool:
    """Union two closing snapshots, the latest capture per side winning.

    Both sides matter: a Saturday's closes are captured four times, and the
    machine pulling may hold a capture the branch has not seen.
    """
    if not remote.exists():
        return False
    snapshot.save(snapshot.merge_last_wins(snapshot.load(remote), snapshot.load(local)), local)
    return True


def merge_jsonl_files(remote: Path, local: Path) -> bool:
    """Union two append-only logs, keeping first appearance order."""
    if not remote.exists():
        return False
    seen: dict[str, None] = {}
    for path in (local, remote):
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    seen.setdefault(line, None)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text("".join(f"{line}\n" for line in seen))
    return True


# --- the state map -----------------------------------------------------------


def _audit_dir(data_dir: Path) -> Path:
    return data_dir / "audit"


def _pull_predictions(state: Path, audit: Path, dates: tuple[str, ...] | None) -> list[str]:
    """Restore the pregame cards this machine never priced.

    Fill-in only: a card already here was priced here, at the prices the email
    went out with, and the branch must not replace it with another machine's.
    """
    src_dir = state / PREFIX / "predictions"
    wanted = (
        [f"predictions_{d}.json.gz" for d in dates]
        if dates is not None
        else sorted(p.name for p in src_dir.glob("predictions_*.json.gz"))
    )
    moved: list[str] = []
    for name in wanted:
        src = src_dir / name
        dest = audit / name[: -len(".gz")]
        if not src.exists() or dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(src, "rb") as fin, dest.open("wb") as fout:
            shutil.copyfileobj(fin, fout)
        moved.append(dest.name)
    return moved


def pull_state(
    data_dir: Path,
    repo: Path | None = None,
    branch: str = STATE_BRANCH,
    dates: tuple[str, ...] | None = None,
) -> SyncReport:
    """Bring the branch's memory onto this machine, merging rather than replacing."""
    repo = repo or repo_root()
    state = _worktree(repo, branch)
    root = state / PREFIX
    audit = _audit_dir(data_dir)
    audit.mkdir(parents=True, exist_ok=True)
    pulled: list[str] = []
    for src in sorted((root / "board").glob("board_*.json")):
        if merge_board_files(src, audit / src.name):
            pulled.append(src.name)
    for src in sorted((root / "closing").glob("closing_*.json")):
        if merge_closing_files(src, audit / src.name):
            pulled.append(src.name)
    for name, key in _MERGED_CSVS:
        if merge_dated_csv(root / name, audit / name, key):
            pulled.append(name)
    if merge_jsonl_files(root / AVAILABILITY_NAME, audit / AVAILABILITY_NAME):
        pulled.append(AVAILABILITY_NAME)
    pulled.extend(_pull_predictions(state, audit, dates))
    return SyncReport(pulled=tuple(pulled))


def _stage_predictions(state: Path, audit: Path) -> tuple[list[str], int]:
    """Publish this machine's cards; the latest pricing of a slate is its record."""
    out = state / PREFIX / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    for src in sorted(audit.glob("predictions_*.json")):
        with src.open("rb") as fin, gzip.open(out / f"{src.name}.gz", "wb", 6) as fout:
            shutil.copyfileobj(fin, fout)
        staged.append(src.name)
    keep = sorted(p.name for p in out.glob("predictions_*.json.gz"))[-PREDICTION_KEEP_DAYS:]
    pruned = 0
    for path in out.glob("predictions_*.json.gz"):
        if path.name not in keep:
            path.unlink()
            pruned += 1
    return staged, pruned


def push_state(
    data_dir: Path,
    message: str,
    repo: Path | None = None,
    branch: str = STATE_BRANCH,
) -> SyncReport:
    """Publish this machine's state, re-merging if the branch moved underneath."""
    repo = repo or repo_root()
    audit = _audit_dir(data_dir)
    for attempt in range(_PUSH_ATTEMPTS):
        state = _worktree(repo, branch)
        if attempt:
            pull_state(data_dir, repo=repo, branch=branch)
        root = state / PREFIX
        pushed: list[str] = []
        for pattern, sub, merge in (
            ("board_*.json", "board", merge_board_files),
            ("closing_*.json", "closing", merge_closing_files),
        ):
            for src in sorted(audit.glob(pattern)):
                dest = root / sub / src.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Fold the branch's quotes into ours before overwriting it.
                merge(dest, src)
                shutil.copyfile(src, dest)
                pushed.append(src.name)
        for name, key in _MERGED_CSVS:
            src = audit / name
            if src.exists():
                dest = root / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                merge_dated_csv(dest, src, key)
                shutil.copyfile(src, dest)
                pushed.append(name)
        src = audit / AVAILABILITY_NAME
        if src.exists():
            dest = root / AVAILABILITY_NAME
            dest.parent.mkdir(parents=True, exist_ok=True)
            merge_jsonl_files(dest, src)
            shutil.copyfile(src, dest)
            pushed.append(AVAILABILITY_NAME)
        staged, pruned = _stage_predictions(state, audit)
        pushed.extend(staged)
        if not pushed:
            return SyncReport()
        _git(["add", "-A", PREFIX], state)
        if not _git(["status", "--porcelain"], state):
            return SyncReport(pushed=tuple(pushed), pruned=pruned)
        _commit(state, message)
        if _git_ok(["push", "origin", f"HEAD:{branch}"], state):
            return SyncReport(pushed=tuple(pushed), pruned=pruned)
    raise RuntimeError(f"push to {branch} rejected after {_PUSH_ATTEMPTS} attempts")


def auto_pull(
    data_dir: Path, branch: str = STATE_BRANCH, dates: tuple[str, ...] | None = None
) -> SyncReport | None:
    """Sync down, best effort: no remote, branch or credentials is a reason to
    run with local state, never to fail a priced slate."""
    try:
        return pull_state(data_dir, branch=branch, dates=dates)
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
    "STATE_BRANCH",
    "SyncReport",
    "auto_pull",
    "auto_push",
    "merge_board_files",
    "merge_closing_files",
    "merge_jsonl_files",
    "pull_state",
    "push_state",
]
