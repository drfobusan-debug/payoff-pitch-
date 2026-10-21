"""Shared push guard: scratch data dirs never publish to engine-state."""

from pathlib import Path

import pytest

from cfb_engine import state as cfb
from mlb_engine.state import push_refusal
from nba_engine import state as nba


def test_refusal_rules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XE_STATE_SYNC", raising=False)
    monkeypatch.delenv("XE_STATE_PUSH", raising=False)
    assert push_refusal(tmp_path, tmp_path, "XE") is None
    assert push_refusal(tmp_path / "scratch", tmp_path, "XE") is not None
    monkeypatch.setenv("XE_STATE_PUSH", "1")
    assert push_refusal(tmp_path / "scratch", tmp_path, "XE") is None
    monkeypatch.setenv("XE_STATE_SYNC", "0")
    assert push_refusal(tmp_path, tmp_path, "XE") == "XE_STATE_SYNC=0"


@pytest.mark.parametrize("mod,prefix", [(cfb, "CFBE"), (nba, "NBAE")])
def test_cfb_and_nba_refuse_a_scratch_dir(tmp_path: Path, monkeypatch, mod, prefix) -> None:
    monkeypatch.delenv(f"{prefix}_STATE_SYNC", raising=False)
    monkeypatch.delenv(f"{prefix}_STATE_PUSH", raising=False)
    with pytest.raises(RuntimeError, match="refused"):
        mod.push_state(tmp_path, "audit test")
    assert mod.auto_push(tmp_path, "audit test") is None
