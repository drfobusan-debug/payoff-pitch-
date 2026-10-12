from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine_runtime import runner


def test_runner_stamps_steps_and_records_failure(tmp_path: Path) -> None:
    p = runner.Pass("cbb", "2026-11-10", "featured", tmp_path)
    assert p.pass_id.startswith("featured_")
    assert p.run("capture", lambda ps: ps.spend(12) or "ok") == "ok"
    assert p.run("optional", lambda ps: 1 / 0, required=False) is None
    with pytest.raises(ZeroDivisionError):
        p.run("price", lambda ps: 1 / 0)
    log = json.loads(p.log_path.read_text())
    assert log["credits"] == 12
    assert [s["status"] for s in log["steps"]] == ["ok", "failed", "failed"]
    assert "ZeroDivisionError" in log["steps"][-1]["error"]


def test_bad_pass_kind_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        runner.Pass("cbb", "d", "watch", tmp_path)
