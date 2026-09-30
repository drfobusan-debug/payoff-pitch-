"""The CFB audit directory travels on the shared state branch."""

from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path

from cfb_engine.audit import snapshot
from cfb_engine.audit.snapshot import SideQuote
from cfb_engine.state import (
    PREFIX,
    merge_board_files,
    merge_closing_files,
    merge_jsonl_files,
    pull_state,
    push_state,
)


def _snap(path: Path, quotes: dict[str, tuple[float, float]]) -> None:
    snapshot.save({k: SideQuote(a, p, None) for k, (a, p) in quotes.items()}, path)


def test_board_merge_keeps_the_earlier_quote(tmp_path: Path):
    remote, local = tmp_path / "r.json", tmp_path / "l.json"
    _snap(remote, {"A @ B|game_ats|B": (-110, 0.52), "A @ B|game_ats|A": (-110, 0.48)})
    _snap(local, {"A @ B|game_ats|B": (-120, 0.55), "C @ D|game_ml|D": (-150, 0.6)})

    assert merge_board_files(remote, local)
    merged = snapshot.load(local)
    assert merged["A @ B|game_ats|B"].american == -110  # the branch published first
    assert set(merged) == {"A @ B|game_ats|B", "A @ B|game_ats|A", "C @ D|game_ml|D"}


def test_closing_merge_keeps_the_later_quote(tmp_path: Path):
    remote, local = tmp_path / "r.json", tmp_path / "l.json"
    _snap(remote, {"A @ B|game_ats|B": (-110, 0.52)})
    _snap(local, {"A @ B|game_ats|B": (-120, 0.55)})

    assert merge_closing_files(remote, local)
    assert snapshot.load(local)["A @ B|game_ats|B"].american == -120


def test_jsonl_merge_is_a_union_in_first_seen_order(tmp_path: Path):
    remote, local = tmp_path / "r.jsonl", tmp_path / "l.jsonl"
    local.write_text('{"a": 1}\n{"b": 2}\n')
    remote.write_text('{"b": 2}\n{"c": 3}\n')

    assert merge_jsonl_files(remote, local)
    assert local.read_text().splitlines() == ['{"a": 1}', '{"b": 2}', '{"c": 3}']


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo_pair(tmp_path: Path) -> Path:
    """A clone with an authenticated-looking origin, both on disk."""
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


def test_round_trip_moves_the_audit_between_two_machines(tmp_path: Path):
    """The Mac prices and audits; another box pulls and holds the same record."""
    repo = _repo_pair(tmp_path)
    mac, box = tmp_path / "mac", tmp_path / "box"
    audit = mac / "audit"
    audit.mkdir(parents=True)
    (audit / "predictions_2026-09-05.json").write_text(json.dumps([{"matchup": "A @ B"}]))
    (audit / "ledger.csv").write_text(
        "date,matchup,category,market,selection,line,result\n"
        "2026-09-05,A @ B,game,game_ats,B -3.0,-3.0,win\n"
    )
    (audit / "availability.jsonl").write_text('{"team": "A"}\n')
    _snap(audit / "closing_2026-09-05.json", {"A @ B|game_ats|B": (-110, 0.52)})

    pushed = push_state(mac, "test push", repo=repo, branch="engine-state")
    assert "ledger.csv" in pushed.pushed
    assert _git(["ls-remote", "--heads", "origin", "engine-state"], repo)

    pulled = pull_state(box, repo=repo, branch="engine-state")
    assert set(pulled.pulled) >= {
        "ledger.csv",
        "availability.jsonl",
        "closing_2026-09-05.json",
        "predictions_2026-09-05.json",
    }
    assert (box / "audit" / "ledger.csv").read_text() == (audit / "ledger.csv").read_text()
    assert json.loads((box / "audit" / "predictions_2026-09-05.json").read_text()) == [
        {"matchup": "A @ B"}
    ]
    # Published gzipped, under the CFB prefix, apart from the MLB state.
    state = repo.parent / f".{repo.name}-engine-state"
    with gzip.open(state / PREFIX / "predictions" / "predictions_2026-09-05.json.gz") as fz:
        assert json.load(fz) == [{"matchup": "A @ B"}]


def test_pull_does_not_overwrite_a_card_priced_here(tmp_path: Path):
    repo = _repo_pair(tmp_path)
    mac, box = tmp_path / "mac", tmp_path / "box"
    (mac / "audit").mkdir(parents=True)
    (mac / "audit" / "predictions_2026-09-05.json").write_text('[{"who": "mac"}]')
    push_state(mac, "mac", repo=repo, branch="engine-state")

    (box / "audit").mkdir(parents=True)
    (box / "audit" / "predictions_2026-09-05.json").write_text('[{"who": "box"}]')
    pull_state(box, repo=repo, branch="engine-state")

    assert (box / "audit" / "predictions_2026-09-05.json").read_text() == '[{"who": "box"}]'
