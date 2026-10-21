"""The NFL prop research and its grades travel on the shared state branch."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import pytest

from nfl_engine.props import FIELDS as RESEARCH_FIELDS
from nfl_engine.props_grade import GRADED_FIELDS
from nfl_engine.state import PREFIX, PROP_KEY, merge_prop_files, pull_state, push_state


@pytest.fixture(autouse=True)
def _scratch_dirs_may_push(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NFLE_STATE_PUSH", "1")


def _research_row(player: str, captured_at: str = "2026-09-10T18:00:00Z") -> dict[str, str]:
    row = dict.fromkeys(RESEARCH_FIELDS, "")
    row.update(
        captured_at=captured_at,
        season="2026",
        week="2",
        matchup="BUF @ KC",
        market="player_pass_yds",
        player=player,
        side="over",
        line="250.5",
        book="dk",
        american="-110",
        model_prob="0.55",
        screens="research_only",
        basis="usage-shrunk-2016-2021",
    )
    return row


def _graded_row(player: str, result: str, captured_at: str = "2026-09-10T18:00:00Z"):
    row = dict.fromkeys(GRADED_FIELDS, "")
    row.update(_research_row(player, captured_at))
    row.update(result=result, pnl="0.91" if result == "win" else "-1", graded_at="x")
    return {k: row[k] for k in GRADED_FIELDS}


def _write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_research_merge_is_a_union_that_collapses_the_same_append(tmp_path: Path):
    remote, local = tmp_path / "r.csv", tmp_path / "l.csv"
    _write(remote, RESEARCH_FIELDS, [_research_row("Allen"), _research_row("Mahomes")])
    _write(local, RESEARCH_FIELDS, [_research_row("Mahomes"), _research_row("Mahomes", "T2")])

    assert merge_prop_files(remote, local, tuple(RESEARCH_FIELDS))
    kept = {(r["player"], r["captured_at"]) for r in _read(local)}
    # Allen came from the branch; Mahomes at the first stamp is one row, not two;
    # the later re-pricing of Mahomes is its own row.
    assert kept == {
        ("Allen", "2026-09-10T18:00:00Z"),
        ("Mahomes", "2026-09-10T18:00:00Z"),
        ("Mahomes", "T2"),
    }


def test_graded_merge_keeps_this_machines_grade_for_the_same_quote(tmp_path: Path):
    remote, local = tmp_path / "r.csv", tmp_path / "l.csv"
    _write(remote, GRADED_FIELDS, [_graded_row("Allen", "loss"), _graded_row("Mahomes", "win")])
    _write(local, GRADED_FIELDS, [_graded_row("Allen", "win")])

    assert merge_prop_files(remote, local, PROP_KEY)
    by_player = {r["player"]: r["result"] for r in _read(local)}
    assert by_player == {"Allen": "win", "Mahomes": "win"}


def test_a_settled_grade_beats_this_machines_void_for_the_same_quote(tmp_path: Path):
    # A machine that graded before the box scores posted pushes its voids: the
    # branch's settled rows must survive, and the reverse must hold on a pull.
    remote, local = tmp_path / "r.csv", tmp_path / "l.csv"
    _write(remote, GRADED_FIELDS, [_graded_row("Allen", "win"), _graded_row("Mahomes", "void")])
    _write(local, GRADED_FIELDS, [_graded_row("Allen", "void"), _graded_row("Mahomes", "loss")])

    assert merge_prop_files(remote, local, PROP_KEY)
    by_player = {r["player"]: r["result"] for r in _read(local)}
    assert by_player == {"Allen": "win", "Mahomes": "loss"}


def test_a_stale_machines_push_does_not_void_the_branch(tmp_path: Path):
    repo = _repo_pair(tmp_path)
    graded = "graded_2026_wk02.csv"
    mac, stale = tmp_path / "mac", tmp_path / "stale"
    _write(mac / "props" / graded, GRADED_FIELDS, [_graded_row("Allen", "win")])
    push_state(mac, "mac grades", repo=repo, branch="engine-state")
    _write(stale / "props" / graded, GRADED_FIELDS, [_graded_row("Allen", "void")])
    push_state(stale, "stale push", repo=repo, branch="engine-state")

    origin = str(tmp_path / "origin.git")
    branch = _git(["--git-dir", origin, "show", f"engine-state:{PREFIX}/props/{graded}"], tmp_path)
    assert ",win," in branch and ",void," not in branch


def test_merge_with_nothing_on_the_branch_changes_nothing(tmp_path: Path):
    local = tmp_path / "l.csv"
    _write(local, GRADED_FIELDS, [_graded_row("Allen", "win")])
    assert not merge_prop_files(tmp_path / "missing.csv", local, PROP_KEY)
    assert len(_read(local)) == 1


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo_pair(tmp_path: Path) -> Path:
    origin = tmp_path / "origin.git"
    _git(["init", "--bare", "-q", str(origin)], tmp_path)
    repo = tmp_path / "repo"
    _git(["init", "-q", str(repo)], tmp_path)
    (repo / "README").write_text("x")
    _git(["add", "README"], repo)
    _git(["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"], repo)
    _git(["remote", "add", "origin", str(origin)], repo)
    _git(["push", "-q", "origin", "HEAD:main"], repo)
    return repo


def test_round_trip_moves_the_prop_audit_between_two_machines(tmp_path: Path):
    """The Mac prices and grades the week; another box pulls and holds the same files."""
    repo = _repo_pair(tmp_path)
    mac, box = tmp_path / "mac", tmp_path / "box"
    _write(mac / "props" / "research_2026_wk02.csv", RESEARCH_FIELDS, [_research_row("Allen")])
    _write(mac / "props" / "graded_2026_wk02.csv", GRADED_FIELDS, [_graded_row("Allen", "win")])

    pushed = push_state(mac, "test push", repo=repo, branch="engine-state")
    assert set(pushed.pushed) == {"research_2026_wk02.csv", "graded_2026_wk02.csv"}
    assert _git(["ls-remote", "--heads", "origin", "engine-state"], repo)
    state = repo.parent / f".{repo.name}-engine-state"
    assert (state / PREFIX / "props" / "graded_2026_wk02.csv").exists()

    pulled = pull_state(box, repo=repo, branch="engine-state")
    assert set(pulled.pulled) == {"research_2026_wk02.csv", "graded_2026_wk02.csv"}
    for name in ("research_2026_wk02.csv", "graded_2026_wk02.csv"):
        assert (box / "props" / name).read_text() == (mac / "props" / name).read_text()


def test_a_second_machines_research_is_folded_in_rather_than_overwritten(tmp_path: Path):
    repo = _repo_pair(tmp_path)
    mac, box = tmp_path / "mac", tmp_path / "box"
    _write(mac / "props" / "research_2026_wk02.csv", RESEARCH_FIELDS, [_research_row("Allen")])
    push_state(mac, "mac", repo=repo, branch="engine-state")

    _write(box / "props" / "research_2026_wk02.csv", RESEARCH_FIELDS, [_research_row("Mahomes")])
    push_state(box, "box", repo=repo, branch="engine-state")

    pull_state(mac, repo=repo, branch="engine-state")
    players = sorted(r["player"] for r in _read(mac / "props" / "research_2026_wk02.csv"))
    assert players == ["Allen", "Mahomes"]


def test_push_with_no_prop_files_is_a_no_op(tmp_path: Path):
    repo = _repo_pair(tmp_path)
    report = push_state(tmp_path / "empty", "nothing", repo=repo, branch="engine-state")
    assert report.pushed == ()


def test_only_the_production_data_dir_may_push(tmp_path: Path, monkeypatch) -> None:
    from nfl_engine import state

    monkeypatch.delenv("NFLE_STATE_SYNC", raising=False)
    monkeypatch.delenv("NFLE_STATE_PUSH", raising=False)
    with pytest.raises(RuntimeError, match="refused"):
        state.push_state(tmp_path, "nfl audit test")
    assert state.auto_push(tmp_path, "nfl audit test") is None
    monkeypatch.setattr(state, "production_data_dir", lambda: tmp_path)
    monkeypatch.setenv("NFLE_STATE_SYNC", "0")
    with pytest.raises(RuntimeError, match="NFLE_STATE_SYNC=0"):
        state.push_state(tmp_path, "nfl audit test")
