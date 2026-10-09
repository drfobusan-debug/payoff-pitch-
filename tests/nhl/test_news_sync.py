from __future__ import annotations

from pathlib import Path

from nhl_engine import state


def test_news_logs_push_when_changed_and_pull_only_missing(tmp_path: Path):
    local, branch, other = tmp_path / "local", tmp_path / "branch", tmp_path / "other"
    (local / "availability").mkdir(parents=True)
    (local / "starters").mkdir()
    (local / "cards").mkdir()
    (local / "availability" / "2026-10-10.jsonl").write_text("a\n")
    (local / "starters" / "2026-10-10.json").write_text("{}")
    (local / "starters" / "history_2026-10-10.jsonl").write_text("r1\n")
    (local / "cards" / "card_2026-10-10_goalie.json").write_text("[]")
    (local / "cards" / "notes.txt").write_text("x")
    pushed = state._copy_news(local, branch, overwrite=True)
    assert sorted(pushed) == [
        "availability/2026-10-10.jsonl",
        "cards/card_2026-10-10_goalie.json",
        "starters/2026-10-10.json",
        "starters/history_2026-10-10.jsonl",
    ]
    assert state._copy_news(local, branch, overwrite=True) == []
    (local / "starters" / "history_2026-10-10.jsonl").write_text("r1\nr2\n")
    assert state._copy_news(local, branch, overwrite=True) == ["starters/history_2026-10-10.jsonl"]

    (other / "availability").mkdir(parents=True)
    (other / "availability" / "2026-10-10.jsonl").write_text("mine\n")
    pulled = state._copy_news(branch, other, overwrite=False)
    assert "availability/2026-10-10.jsonl" not in pulled and len(pulled) == 3
    assert (other / "availability" / "2026-10-10.jsonl").read_text() == "mine\n"
