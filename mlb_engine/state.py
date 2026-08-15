"""Carry the audit's memory between machines.

Every scheduled run is a fresh box, so ``~/.mlb_engine`` starts empty: the
evening close capture, the morning card and the 2:30am audit never see each
other's files. Two things break as a result. Closing line value can never be
scored -- the snapshot the capture wrote is on a machine the audit does not
have -- and the ledger never accumulates, so "N graded bets across all dates"
is always a single slate.

So the state that has to outlive a machine lives on an orphan branch of the
repo instead: the pregame predictions (the picks actually sent, at the prices
they were sent at), the closing snapshots, the ledger and the scorecard. Pull
before a run, push after.

Only data goes on that branch, never code, and it is deliberately shallow:
predictions are the bulk and are pruned to the most recent few weeks.
"""

from __future__ import annotations

import csv
import gzip
import logging
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from mlb_engine.audit.clv import load_closing, merge_closing, save_closing
from mlb_engine.data.opta import load_rows, merge_rows, save_rows
from mlb_engine.data.propicks import load_picks, merge_picks, save_picks

STATE_BRANCH = "engine-state"
# Predictions dominate the branch's size (~5 MB a slate before gzip). A month
# is far more history than the audit reads and keeps the branch clonable.
PREDICTION_KEEP_DAYS = 35
# Pulled predictions keep their own name. The nightly audit regenerates the
# slate before grading it, which would otherwise overwrite the pregame picks
# with an after-the-fact re-price, so what the card actually sent is kept
# under a name nothing else writes.
PREGAME_SUFFIX = ".pregame.json"
log = logging.getLogger(__name__)
# A nightly run grades yesterday, so restoring more slates than that only
# spends time expanding megabytes nothing will read.
_PULL_PREDICTION_DAYS = 2
_PUSH_ATTEMPTS = 3


@dataclass(frozen=True)
class SyncReport:
    """What a sync moved, for the automation's log line."""

    pulled: tuple[str, ...] = ()
    pushed: tuple[str, ...] = ()
    pruned: int = 0

    def describe(self) -> str:
        if self.pulled:
            return f"pulled {len(self.pulled)} state file(s): {', '.join(self.pulled)}"
        if self.pushed:
            extra = f", pruned {self.pruned} stale prediction file(s)" if self.pruned else ""
            return f"pushed {len(self.pushed)} state file(s): {', '.join(self.pushed)}{extra}"
        return "nothing to sync"


def _git(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=300
    )
    return proc.stdout.strip()


def _git_ok(args: list[str], cwd: Path) -> bool:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=300)
    return proc.returncode == 0


def _commit(state: Path, message: str) -> None:
    """Commit as the engine when the box has no git identity of its own."""
    ident: list[str] = []
    if not _git(["config", "--get", "--default", "", "user.email"], state):
        ident = [
            "-c",
            "user.name=payoff-pitch engine",
            "-c",
            "user.email=engine@payoffpitch.local",
        ]
    _git([*ident, "commit", "-m", message], state)


def repo_root(start: Path | None = None) -> Path:
    """The checkout the state branch is fetched from and pushed to."""
    return Path(_git(["rev-parse", "--show-toplevel"], start or Path.cwd()))


def _remote_has_branch(repo: Path, branch: str) -> bool:
    proc = subprocess.run(
        ["git", "ls-remote", "--heads", "origin", branch],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return bool(proc.stdout.strip())


def _fetch(repo: Path, branch: str) -> None:
    """Fetch the state branch into a tracking ref.

    Spelled out rather than ``fetch origin <branch>`` because a scheduled run
    clones one branch: the configured refspec then covers only that branch, so
    a bare fetch updates FETCH_HEAD and leaves ``origin/<branch>`` undefined.
    """
    _git(["fetch", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}"], repo)


def _worktree(repo: Path, branch: str) -> Path:
    """A checkout of the state branch, sharing the repo's remote and credentials.

    A worktree rather than a second clone so the authenticated remote, and
    nothing else, is inherited. Created empty when the branch does not exist
    yet, which is the first run of a fresh installation.

    Discarded and recreated when one is already there, rather than reset in
    place: it holds nothing but a copy of the branch, and every git command
    that could rescue a half-finished sync would also be one that can destroy
    work if this path ever pointed somewhere it should not.
    """
    path = repo.parent / f".{repo.name}-{branch}"
    if path.exists():
        if not _git_ok(["worktree", "remove", "--force", str(path)], repo):
            shutil.rmtree(path, ignore_errors=True)
    _git_ok(["worktree", "prune"], repo)
    if _remote_has_branch(repo, branch):
        _fetch(repo, branch)
        _git(
            [
                "worktree",
                "add",
                "--force",
                "-B",
                branch,
                str(path),
                f"refs/remotes/origin/{branch}",
            ],
            repo,
        )
    else:
        _git(["worktree", "add", "--detach", str(path), "HEAD"], repo)
        _git(["checkout", "--orphan", branch], path)
        _git_ok(["rm", "-rf", "--quiet", "."], path)
    return path


# --- file-level merges -------------------------------------------------------
# Pure so they can be tested without a repo: the git plumbing above is thin on
# purpose and these decide what actually survives a sync.


def merge_closing_files(remote: Path, local: Path) -> bool:
    """Union two closing snapshots, latest price per selection winning.

    Both sides matter: the machine pulling may hold an afternoon capture the
    branch has not seen, and the branch may hold one this machine never made.
    """
    if not remote.exists():
        return False
    merged = merge_closing(load_closing(local), list(load_closing(remote).values()))
    save_closing(local, merged)
    return True


def merge_opta_files(remote: Path, local: Path) -> bool:
    """Union two Opta captures, the later view of a projection winning.

    The morning capture holds the projections and the evening one holds the
    graded outcomes, and they are usually made on different machines.
    """
    if not remote.exists():
        return False
    save_rows(local, merge_rows(load_rows(local), load_rows(remote)))
    return True


def merge_propick_files(remote: Path, local: Path) -> bool:
    """Union two captures of a day's VSiN model picks.

    The pages are same-day and get restated as the board moves, so whichever
    machine ran the card first holds picks the other never saw.
    """
    if not remote.exists():
        return False
    save_picks(local, merge_picks(load_picks(local), load_picks(remote)))
    return True


def _rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), [dict(r) for r in reader]


def _write_rows(path: Path, fields: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def merge_dated_csv(remote: Path, local: Path, key: tuple[str, ...]) -> bool:
    """Union the ledger (or scorecard) by date, this machine's rows winning.

    A re-audit of a date is authoritative for that date -- it has the results
    and the closes -- so local rows replace the branch's for any date both
    hold, and the branch supplies every date this machine never graded.
    """
    if not remote.exists():
        return False
    fields, remote_rows = _rows(remote)
    if not local.exists():
        _write_rows(local, fields, remote_rows)
        return True
    local_fields, local_rows = _rows(local)
    fields = local_fields if len(local_fields) >= len(fields) else fields
    local_dates = {r.get("date", "") for r in local_rows}
    kept = [r for r in remote_rows if r.get("date", "") not in local_dates]
    merged = {tuple(r.get(k, "") for k in key): r for r in [*kept, *local_rows]}
    _write_rows(local, fields, [merged[k] for k in sorted(merged)])
    return True


# --- the state map -----------------------------------------------------------


# The accumulating records, and the columns identifying one row of each. Both
# directions merge on these: a machine only ever contributes the dates it
# graded, and never speaks for the ones it did not.
_MERGED_CSVS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ledger.csv", ("date", "matchup", "category", "market", "selection", "line")),
    ("scorecard.csv", ("date", "tier")),
)


def _audit_dir(data_dir: Path) -> Path:
    return data_dir / "audit"


def _pull_predictions(state: Path, data_dir: Path, dates: tuple[str, ...] | None) -> list[str]:
    """Restore the pregame picks, which only the machine that ran them has.

    The audit must grade what the card actually recommended, at the prices it
    was recommended at; re-running the slate hours later grades a different
    set of picks against prices that no longer exist.

    A slate is ~5 MB expanded, so only the dates asked for are restored --
    by default the couple a nightly run could plausibly grade.
    """
    wanted = (
        [f"predictions_{d}.json.gz" for d in dates]
        if dates is not None
        else sorted(p.name for p in (state / "mlb" / "predictions").glob("predictions_*.json.gz"))[
            -_PULL_PREDICTION_DAYS:
        ]
    )
    moved = []
    for name in wanted:
        src = state / "mlb" / "predictions" / name
        dest = _audit_dir(data_dir) / (name[: -len(".json.gz")] + PREGAME_SUFFIX)
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
    audit = _audit_dir(data_dir)
    audit.mkdir(parents=True, exist_ok=True)
    pulled: list[str] = []

    for src in sorted((state / "mlb" / "closing").glob("closing_*.json")):
        if merge_closing_files(src, audit / src.name):
            pulled.append(src.name)
    for src in sorted((state / "mlb" / "opta").glob("opta_*.json")):
        if merge_opta_files(src, audit / src.name):
            pulled.append(src.name)
    for src in sorted((state / "mlb" / "propicks").glob("propicks_*.json")):
        if merge_propick_files(src, audit / src.name):
            pulled.append(src.name)
    for name, key in _MERGED_CSVS:
        if merge_dated_csv(state / "mlb" / name, audit / name, key):
            pulled.append(name)
    pulled.extend(_pull_predictions(state, data_dir, dates))
    return SyncReport(pulled=tuple(pulled))


def _stage_predictions(state: Path, data_dir: Path) -> tuple[list[str], int]:
    """Publish pregame picks, write-once per slate.

    A date already on the branch is never republished: the machine holding a
    second copy is usually the audit, whose local file is a re-price made after
    the games finished. Only the run that actually produced the card publishes.
    """
    out = state / "mlb" / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    staged = []
    for src in sorted(_audit_dir(data_dir).glob("predictions_*.json")):
        if src.name.endswith(PREGAME_SUFFIX):
            continue
        dest = out / f"{src.name}.gz"
        if dest.exists():
            continue
        with src.open("rb") as fin, gzip.open(dest, "wb", compresslevel=6) as fout:
            shutil.copyfileobj(fin, fout)
        staged.append(dest.name)
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
    last_error = ""
    for attempt in range(_PUSH_ATTEMPTS):
        state = _worktree(repo, branch)
        if attempt:
            # Someone else pushed between our read and our write: fold their
            # rows into ours and try again rather than overwrite them.
            pull_state(data_dir, repo=repo, branch=branch)
        pushed: list[str] = []
        for src in sorted(audit.glob("closing_*.json")):
            dest = state / "mlb" / "closing" / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            merge_closing_files(dest, src)
            shutil.copyfile(src, dest)
            pushed.append(src.name)
        for src in sorted(audit.glob("opta_*.json")):
            dest = state / "mlb" / "opta" / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            merge_opta_files(dest, src)
            shutil.copyfile(src, dest)
            pushed.append(src.name)
        for src in sorted(audit.glob("propicks_*.json")):
            dest = state / "mlb" / "propicks" / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            merge_propick_files(dest, src)
            shutil.copyfile(src, dest)
            pushed.append(src.name)
        for name, key in _MERGED_CSVS:
            src = audit / name
            if src.exists():
                dest = state / "mlb" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Fold the branch's dates into ours before overwriting it, the
                # same way the closing snapshots above are merged. A push that
                # git accepts as a fast-forward is not evidence that this
                # machine's ledger is a superset of the branch's: a run that
                # pulled before last night's audit landed will happily publish
                # a ledger missing that slate, and the ledger is a growing
                # record, not this machine's opinion.
                merge_dated_csv(dest, src, key)
                shutil.copyfile(src, dest)
                pushed.append(name)
        staged, pruned = _stage_predictions(state, data_dir)
        pushed.extend(staged)
        if not pushed:
            return SyncReport()

        _git(["add", "-A", "mlb"], state)
        if not _git(["status", "--porcelain"], state):
            return SyncReport(pushed=tuple(pushed), pruned=pruned)
        _commit(state, message)
        if _git_ok(["push", "origin", f"HEAD:{branch}"], state):
            return SyncReport(pushed=tuple(pushed), pruned=pruned)
        last_error = f"push to {branch} rejected"
    raise RuntimeError(f"{last_error} after {_PUSH_ATTEMPTS} attempts")


def auto_pull(
    data_dir: Path,
    branch: str = STATE_BRANCH,
    dates: tuple[str, ...] | None = None,
) -> SyncReport | None:
    """Sync down, best effort: no remote, no branch and no credentials are all
    reasons to run without shared state rather than to fail a priced slate."""
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
    "PREDICTION_KEEP_DAYS",
    "PREGAME_SUFFIX",
    "STATE_BRANCH",
    "auto_pull",
    "auto_push",
    "SyncReport",
    "merge_closing_files",
    "merge_opta_files",
    "merge_propick_files",
    "merge_dated_csv",
    "pull_state",
    "push_state",
    "repo_root",
]
