"""State sync: only the production data dir may push the ledger to engine-state."""

from pathlib import Path

import pytest

from nhl_engine import state


def test_scratch_data_dir_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("NHLE_STATE_SYNC", raising=False)
    monkeypatch.delenv("NHLE_STATE_PUSH", raising=False)
    assert state.push_refusal(tmp_path) is not None
    with pytest.raises(RuntimeError, match="refused"):
        state.push_state(tmp_path, "nhl audit test")
    assert state.auto_push(tmp_path, "nhl audit test") is None


def test_production_dir_allowed_and_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv("NHLE_STATE_SYNC", raising=False)
    monkeypatch.delenv("NHLE_STATE_PUSH", raising=False)
    monkeypatch.setattr(state, "production_data_dir", lambda: Path(tmp_path))
    assert state.push_refusal(tmp_path) is None
    monkeypatch.setenv("NHLE_STATE_SYNC", "0")
    assert state.push_refusal(tmp_path) == "NHLE_STATE_SYNC=0"
    monkeypatch.delenv("NHLE_STATE_SYNC")
    monkeypatch.setenv("NHLE_STATE_PUSH", "1")
    assert state.push_refusal(tmp_path / "other") is None
