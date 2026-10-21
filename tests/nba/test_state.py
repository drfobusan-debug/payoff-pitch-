from pathlib import Path

import pytest

from nba_engine import state


@pytest.fixture(autouse=True)
def _scratch_dirs_may_push(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NBAE_STATE_PUSH", "1")


def test_copy_missing_is_a_set_union(tmp_path):
    src, dest = tmp_path / "a", tmp_path / "b"
    for rel in (
        "prices/2025-12-10/board_x.csv",
        "alerts/2025-12-10/t_fp.csv",
        "injuries/2025-12-10/1700_fp.pdf",
    ):
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(rel)
    (dest / "prices/2025-12-10").mkdir(parents=True)
    (dest / "prices/2025-12-10/board_x.csv").write_text("kept")
    copied = state._copy_missing(src, dest, state.LIVE)
    assert sorted(copied) == ["alerts/2025-12-10/t_fp.csv", "injuries/2025-12-10/1700_fp.pdf"]
    assert (dest / "prices/2025-12-10/board_x.csv").read_text() == "kept"
    assert state._copy_missing(src, dest, state.LIVE) == []


def test_history_is_not_a_live_tree():
    assert "history" not in state.LIVE
    assert isinstance(state.TREES["history"], tuple)
    assert Path("x")  # keeps pathlib import used
