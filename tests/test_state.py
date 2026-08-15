"""The audit's memory has to survive the machine it was written on."""

from __future__ import annotations

import csv
import gzip
import json
import subprocess
from pathlib import Path

import pytest

from mlb_engine.audit.clv import ClosingQuote, load_closing, save_closing
from mlb_engine.config import load_config
from mlb_engine.state import (
    PREDICTION_KEEP_DAYS,
    PREGAME_SUFFIX,
    STATE_BRANCH,
    auto_pull,
    auto_push,
    merge_board_files,
    merge_closing_files,
    merge_dated_csv,
    pull_state,
    push_state,
)

LEDGER_FIELDS = ("date", "matchup", "category", "market", "selection", "line", "pnl")
LEDGER_KEY = ("date", "matchup", "category", "market", "selection", "line")


def _ledger(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _row(date: str, selection: str, pnl: str = "0.91") -> dict[str, str]:
    return {
        "date": date,
        "matchup": "KC@DET",
        "category": "game",
        "market": "game_ml",
        "selection": selection,
        "line": "",
        "pnl": pnl,
    }


def test_pulling_a_ledger_keeps_history_and_lets_a_reaudit_win(tmp_path: Path) -> None:
    """Yesterday's dates come from the branch; today's rows are ours.

    The machine grading a slate has the results and the captured close for it,
    so it is authoritative for that date -- but every earlier date exists only
    on the branch, and dropping them is what reset the ledger to one slate.
    """
    remote = tmp_path / "remote.csv"
    local = tmp_path / "local.csv"
    _ledger(remote, [_row("2026-08-01", "DET"), _row("2026-08-04", "DET", pnl="-1.0")])
    _ledger(local, [_row("2026-08-04", "DET", pnl="0.72")])

    assert merge_dated_csv(remote, local, LEDGER_KEY)

    with local.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["date"] for r in rows] == ["2026-08-01", "2026-08-04"]
    assert [r["pnl"] for r in rows] == ["0.91", "0.72"]


def test_pulling_a_ledger_onto_an_empty_machine_takes_the_branch(tmp_path: Path) -> None:
    remote = tmp_path / "remote.csv"
    _ledger(remote, [_row("2026-08-01", "DET")])
    assert merge_dated_csv(remote, tmp_path / "local.csv", LEDGER_KEY)
    assert not merge_dated_csv(tmp_path / "absent.csv", tmp_path / "local.csv", LEDGER_KEY)


def test_closing_snapshots_union_across_machines(tmp_path: Path) -> None:
    """Neither side of a close sync may lose a price the other captured."""
    remote = tmp_path / "remote.json"
    local = tmp_path / "local.json"
    save_closing(remote, [ClosingQuote("LAD@SF", "game_ml", "LAD", -180.0, 0.6350)])
    save_closing(local, [ClosingQuote("KC@DET", "game_ml", "DET", -150.0, 0.5901)])

    assert merge_closing_files(remote, local)
    assert set(load_closing(local)) == {"KC@DET|game_ml|DET", "LAD@SF|game_ml|LAD"}


def test_opening_boards_keep_the_earliest_price_across_machines(tmp_path: Path) -> None:
    """The morning run's price is the open, whichever machine captured it."""
    remote = tmp_path / "remote.json"
    local = tmp_path / "local.json"
    save_closing(remote, [ClosingQuote("KC@DET", "game_ml", "DET", -130.0, 0.5600)])
    save_closing(local, [ClosingQuote("KC@DET", "game_ml", "DET", -170.0, 0.6200)])

    assert merge_board_files(remote, local)
    assert load_closing(local)["KC@DET|game_ml|DET"].no_vig_prob == 0.56


# --- git round trip ----------------------------------------------------------


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def machines(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Two checkouts of one origin, standing in for two scheduled runs."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True)
    checkouts = []
    for name in ("box_a", "box_b"):
        repo = tmp_path / name / "repo"
        repo.parent.mkdir()
        _git(["clone", "-q", str(origin), str(repo)], tmp_path)
        (repo / "README.md").write_text(name)
        _git(["add", "README.md"], repo)
        _git(["commit", "-qm", "seed"], repo)
        checkouts.append(repo)
    data_a = tmp_path / "box_a" / "data"
    data_b = tmp_path / "box_b" / "data"
    return checkouts[0], data_a, checkouts[1], data_b


def test_the_evening_capture_reaches_the_overnight_audit(
    machines: tuple[Path, Path, Path, Path],
) -> None:
    """The whole point: two machines, one closing snapshot.

    The capture runs at 6:50pm on one box and the audit at 2:30am on another,
    so without the state branch the audit finds no close and scores no CLV --
    which is exactly what happened every night until now.
    """
    repo_a, data_a, repo_b, data_b = machines
    audit_a = data_a / "audit"
    save_closing(
        audit_a / "closing_2026-08-04.json",
        [ClosingQuote("KC@DET", "game_ml", "DET", -150.0, 0.5901)],
    )
    (audit_a / "predictions_2026-08-04.json").write_text(json.dumps([{"selection": "DET"}]))

    pushed = push_state(data_a, "evening capture", repo=repo_a, branch="engine-state")
    assert "closing_2026-08-04.json" in pushed.pushed
    assert "predictions_2026-08-04.json.gz" in pushed.pushed

    report = pull_state(data_b, repo=repo_b, branch="engine-state")
    assert "closing_2026-08-04.json" in report.pulled
    assert load_closing(data_b / "audit" / "closing_2026-08-04.json")["KC@DET|game_ml|DET"]
    # The pregame picks come back too, under a name the nightly re-price cannot
    # clobber, so the audit grades what was actually recommended.
    pregame = data_b / "audit" / f"predictions_2026-08-04{PREGAME_SUFFIX}"
    assert pregame.name in report.pulled
    assert json.loads(pregame.read_text())


def test_a_second_machine_cannot_erase_the_first(
    machines: tuple[Path, Path, Path, Path],
) -> None:
    """Concurrent pushes merge: the later run re-pulls and re-applies."""
    repo_a, data_a, repo_b, data_b = machines
    _ledger((data_a / "audit" / "ledger.csv"), [_row("2026-08-03", "DET")])
    push_state(data_a, "audit 08-03", repo=repo_a, branch="engine-state")

    _ledger((data_b / "audit" / "ledger.csv"), [_row("2026-08-04", "KC")])
    pull_state(data_b, repo=repo_b, branch="engine-state")
    push_state(data_b, "audit 08-04", repo=repo_b, branch="engine-state")

    pull_state(data_a, repo=repo_a, branch="engine-state")
    with (data_a / "audit" / "ledger.csv").open(newline="") as f:
        dates = [r["date"] for r in csv.DictReader(f)]
    assert dates == ["2026-08-03", "2026-08-04"]


def test_a_single_branch_clone_can_still_read_the_state(
    machines: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    """A scheduled run clones one branch, which hides every other remote ref.

    ``git fetch origin engine-state`` then updates FETCH_HEAD only and leaves
    ``origin/engine-state`` undefined, so the sync has to name the refspec.
    """
    repo_a, data_a, _repo_b, _data_b = machines
    _ledger(data_a / "audit" / "ledger.csv", [_row("2026-08-03", "DET")])
    push_state(data_a, "audit 08-03", repo=repo_a, branch="engine-state")

    shallow = tmp_path / "box_c" / "repo"
    shallow.parent.mkdir()
    origin = tmp_path / "origin.git"
    _git(["clone", "-q", "--single-branch", "--depth", "1", str(origin), str(shallow)], tmp_path)

    report = pull_state(tmp_path / "box_c" / "data", repo=shallow, branch="engine-state")
    assert "ledger.csv" in report.pulled


def test_a_pregame_slate_is_never_republished_by_the_audit(
    machines: tuple[Path, Path, Path, Path],
) -> None:
    """The card's picks are write-once; a later re-price must not replace them."""
    repo_a, data_a, repo_b, data_b = machines
    (data_a / "audit").mkdir(parents=True)
    (data_a / "audit" / "predictions_2026-08-04.json").write_text('[{"selection": "DET"}]')
    push_state(data_a, "card", repo=repo_a, branch="engine-state")

    # box_b is the audit: it pulls the pregame copy and regenerates its own.
    pull_state(data_b, repo=repo_b, branch="engine-state")
    (data_b / "audit" / "predictions_2026-08-04.json").write_text('[{"selection": "KC"}]')
    push_state(data_b, "audit", repo=repo_b, branch="engine-state")

    state = repo_a.parent / f".{repo_a.name}-engine-state"
    with gzip.open(state / "mlb" / "predictions" / "predictions_2026-08-04.json.gz", "rt") as f:
        assert json.load(f) == [{"selection": "DET"}]


def test_a_run_outside_a_checkout_is_not_a_failed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sync is best effort: an engine installed outside a checkout still runs."""
    monkeypatch.chdir(tmp_path)
    assert auto_pull(tmp_path / "data") is None
    assert auto_push(tmp_path / "data", "nothing") is None


def test_state_sync_is_on_by_default_and_agrees_with_the_module() -> None:
    cfg = load_config()
    assert cfg.state_sync
    assert cfg.state_branch == STATE_BRANCH


def test_predictions_are_compressed_and_pruned(
    machines: tuple[Path, Path, Path, Path],
) -> None:
    """Predictions are the branch's whole weight, so they expire."""
    repo_a, data_a, _repo_b, _data_b = machines
    audit = data_a / "audit"
    audit.mkdir(parents=True)
    payload = json.dumps([{"selection": "DET", "prob": 0.57}] * 200)
    for day in range(1, PREDICTION_KEEP_DAYS + 4):
        (audit / f"predictions_2026-06-{day:02d}.json").write_text(payload)

    push_state(data_a, "many slates", repo=repo_a, branch="engine-state")
    state = repo_a.parent / f".{repo_a.name}-engine-state"
    kept = sorted(p.name for p in (state / "mlb" / "predictions").glob("*.json.gz"))
    assert len(kept) == PREDICTION_KEEP_DAYS
    assert kept[-1] == f"predictions_2026-06-{PREDICTION_KEEP_DAYS + 3:02d}.json.gz"
    with gzip.open(state / "mlb" / "predictions" / kept[0], "rt") as f:
        assert json.load(f)
