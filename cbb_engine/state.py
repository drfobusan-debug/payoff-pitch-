"""Carry the CBB record between machines on the shared ``engine-state`` branch.

Under ``cbb/``: ``quotes/<day>/`` (Parquet shards and the day's manifest),
``ledger/<day>/`` (write-once passes) and ``cards/``. Shards, passes and cards
are immutable, so they merge as a set union of paths; a manifest is
append-only, so its lines are unioned. The git plumbing is the MLB module's.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

from cbb_engine.config import SPORT, data_dir, runtime_root
from engine_runtime.ledger import GENESIS
from mlb_engine.state import (
    STATE_BRANCH,
    SyncReport,
    _commit,
    _git,
    _git_ok,
    _worktree,
    push_refusal,
    repo_root,
    synced,
)

PREFIX = "cbb"
TREES: dict[str, tuple[str, ...]] = {
    "quotes": ("*/*.parquet",),
    "ledger": ("*/*.jsonl",),
    "cards": ("*.txt", "*.pdf"),
}
LIVE: tuple[str, ...] = ("quotes", "ledger", "cards")
_PUSH_ATTEMPTS = 3
log = logging.getLogger(__name__)


def local_trees(runtime: Path | None = None, data: Path | None = None) -> dict[str, Path]:
    runtime, data = runtime or runtime_root(), data or data_dir()
    return {"quotes": runtime / "quotes" / SPORT, "ledger": runtime / "ledger" / SPORT, "cards": data / "cards"}


def _union_manifests(src: Path, dest: Path) -> list[str]:
    changed: list[str] = []
    for man in sorted(src.glob("*/manifest.jsonl")):
        rel = man.relative_to(src)
        target = dest / rel
        have = target.read_text().splitlines() if target.exists() else []
        seen = {json.dumps(json.loads(x), sort_keys=True) for x in have if x.strip()}
        new = [x for x in man.read_text().splitlines() if x.strip() and json.dumps(json.loads(x), sort_keys=True) not in seen]
        if new:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a") as fh:
                fh.write("".join(x + "\n" for x in new))
            changed.append(str(rel))
    return changed


def _copy(src: dict[str, Path], dest: dict[str, Path], trees: tuple[str, ...]) -> list[str]:
    """Copy what ``dest`` lacks, in name order (pass ids sort by time, and the ledger chain reads mtime)."""
    copied: list[str] = []
    for tree in trees:
        s, d = src[tree], dest[tree]
        if not s.is_dir():
            continue
        for pattern in TREES[tree]:
            for f in sorted(s.glob(pattern)):
                target = d / f.relative_to(s)
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, target)
                copied.append(f"{tree}/{f.relative_to(s)}")
        if tree == "quotes":
            copied += [f"quotes/{x}" for x in _union_manifests(s, d)]
        if tree == "ledger":
            for day in {(d / c.split("/", 1)[1]).parent for c in copied if c.startswith("ledger/")}:
                _restamp_chain(day)
    return copied


def _restamp_chain(day: Path) -> None:
    """Give a day's ledger passes mtimes in hash-chain order: ``ledger.read_day`` and
    ``ledger.last_hash`` order passes by mtime, and a copy stamps them in copy order."""
    heads: dict[str, Path] = {}
    tails: dict[Path, str] = {}
    for f in day.glob("*.jsonl"):
        lines = [x for x in f.read_text().splitlines() if x.strip()]
        if lines:
            heads[json.loads(lines[0])["prev_hash"]] = f
            tails[f] = json.loads(lines[-1])["row_hash"]
    order: list[Path] = []
    prev = GENESIS
    while prev in heads and heads[prev] not in order:
        order.append(heads[prev])
        prev = tails[heads[prev]]
    if len(order) != len(tails):
        log.warning("ledger chain for %s is broken; mtimes left as copied", day)
        return
    base = time.time() - len(order)
    for i, f in enumerate(order):
        os.utime(f, (base + i, base + i))


def _branch_trees(state: Path) -> dict[str, Path]:
    return {t: state / PREFIX / t for t in TREES}


@synced
def pull_state(
    runtime: Path, data: Path, repo: Path | None = None, branch: str = STATE_BRANCH, trees: tuple[str, ...] = LIVE
) -> SyncReport:
    state = _worktree(repo or repo_root(), branch)
    return SyncReport(pulled=tuple(_copy(_branch_trees(state), local_trees(runtime, data), trees)))


@synced
def push_state(
    runtime: Path,
    data: Path,
    message: str,
    repo: Path | None = None,
    branch: str = STATE_BRANCH,
    trees: tuple[str, ...] = LIVE,
) -> SyncReport:
    repo = repo or repo_root()
    refusal = push_refusal(runtime, Path.home() / ".engine_runtime", "CBBE")
    if refusal:
        raise RuntimeError(f"state push refused: {refusal}")
    for attempt in range(_PUSH_ATTEMPTS):
        state = _worktree(repo, branch)
        if attempt:
            pull_state(runtime, data, repo=repo, branch=branch, trees=trees)
        pushed = _copy(local_trees(runtime, data), _branch_trees(state), trees)
        if not pushed:
            return SyncReport()
        _git(["add", "-A", PREFIX], state)
        if not _git(["status", "--porcelain"], state):
            return SyncReport(pushed=tuple(pushed))
        _commit(state, message)
        if _git_ok(["push", "origin", f"HEAD:{branch}"], state):
            return SyncReport(pushed=tuple(pushed))
    raise RuntimeError(f"push to {branch} rejected after {_PUSH_ATTEMPTS} attempts")


def _off() -> bool:
    return os.getenv("CBBE_STATE_SYNC", "1").lower() in ("0", "false", "no")


def auto_pull(trees: tuple[str, ...] = LIVE) -> SyncReport | None:
    """Best effort: no remote, branch or credentials means work locally, not fail."""
    if _off():
        return None
    try:
        return pull_state(runtime_root(), data_dir(), trees=trees)
    except Exception as exc:  # noqa: BLE001 - state sync is never the point of the run
        log.warning("state pull skipped: %s", exc)
        return None


def auto_push(message: str, trees: tuple[str, ...] = LIVE) -> SyncReport | None:
    if _off():
        return None
    try:
        return push_state(runtime_root(), data_dir(), message, trees=trees)
    except Exception as exc:  # noqa: BLE001
        log.warning("state push skipped: %s", exc)
        return None


__all__ = ["LIVE", "PREFIX", "TREES", "auto_pull", "auto_push", "local_trees", "pull_state", "push_state"]
