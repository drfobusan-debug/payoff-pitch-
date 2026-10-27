from __future__ import annotations

from pathlib import Path

import pytest

from cbb_engine import checklist
from cbb_engine.config import Config, Delivery
from cbb_engine.models import distribution as dist
from engine_runtime import ledger


def test_checklist_fails_closed_without_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real params (width.json, a filled yaml) live in the package; the test stands in a repo without them."""
    monkeypatch.setattr(dist, "load_width", lambda: None)
    (tmp_path / "p.yaml").write_text("# TODO-PILOT slots are filled by the pilot\nmde: TODO-PILOT\n")
    monkeypatch.setattr(checklist, "PREREG", tmp_path / "p.yaml")
    items = checklist.run(Config(delivery=Delivery(gmail_user=None, gmail_app_password=None, email_to=None)), tmp_path, "2026-11-10")
    by = {i.name: i for i in items}
    assert not by["width_fitted"].ok and not by["preregistered_mdes"].ok and not by["emails_configured"].ok
    assert "RESEARCH MODE" in checklist.render(items)
    assert not by["credit_budget"].ok and "no daily limit set" in by["credit_budget"].detail
    assert by["write_once_green"].ok


def test_checklist_ignores_todo_pilot_in_comments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "p.yaml").write_text("# TODO-PILOT markers are filled by the pilot\nmde: 1.75\n")
    monkeypatch.setattr(checklist, "PREREG", tmp_path / "p.yaml")
    items = checklist.run(Config(delivery=Delivery(gmail_user=None, gmail_app_password=None, email_to=None)), tmp_path, "2026-11-10")
    assert {i.name: i for i in items}["preregistered_mdes"].ok


def _cfg() -> Config:
    return Config(delivery=Delivery(gmail_user=None, gmail_app_password=None, email_to=None))


def test_credit_budget_reads_the_cbb_cap_from_the_runtime_budget(tmp_path: Path) -> None:
    (tmp_path / "credit_budget.yaml").write_text("enforce: false\nsports:\n  mlb: 3000\n  cbb: 6000\n")
    item = {i.name: i for i in checklist.run(_cfg(), tmp_path, "2026-11-10")}["credit_budget"]
    assert item.ok and "of 6000 (credit_budget.yaml sports.cbb)" in item.detail
    (tmp_path / "credit_budget.yaml").write_text("sports:\n  cbb: 400\n")
    assert not {i.name: i for i in checklist.run(_cfg(), tmp_path, "2026-11-10")}["credit_budget"].ok


def test_credit_budget_flag_overrides_the_yaml(tmp_path: Path) -> None:
    (tmp_path / "credit_budget.yaml").write_text("sports:\n  cbb: 6000\n")
    item = {i.name: i for i in checklist.run(_cfg(), tmp_path, "2026-11-10", daily_credits=400)}["credit_budget"]
    assert not item.ok and "of 400 (--daily-credits)" in item.detail


def test_write_once_green_fails_when_the_ledger_stops_refusing(monkeypatch: pytest.MonkeyPatch) -> None:
    assert checklist.write_once_check()[0]
    monkeypatch.setattr(ledger, "write_once", lambda *a, **k: Path("x"))
    ok, detail = checklist.write_once_check()
    assert not ok and "duplicate pass accepted" in detail and "close pass accepted" in detail
