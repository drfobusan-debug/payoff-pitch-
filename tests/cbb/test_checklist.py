from __future__ import annotations

from pathlib import Path

import pytest

from cbb_engine import checklist
from cbb_engine.config import Config, Delivery
from cbb_engine.models import distribution as dist


def test_checklist_fails_closed_without_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real params (width.json, a filled yaml) live in the package; the test stands in a repo without them."""
    monkeypatch.setattr(dist, "load_width", lambda: None)
    (tmp_path / "p.yaml").write_text("# TODO-PILOT slots are filled by the pilot\nmde: TODO-PILOT\n")
    monkeypatch.setattr(checklist, "PREREG", tmp_path / "p.yaml")
    items = checklist.run(Config(delivery=Delivery(gmail_user=None, gmail_app_password=None, email_to=None)), tmp_path, "2026-11-10")
    by = {i.name: i for i in items}
    assert not by["width_fitted"].ok and not by["preregistered_mdes"].ok and not by["emails_configured"].ok
    assert "RESEARCH MODE" in checklist.render(items)
    assert by["credit_budget"].ok


def test_checklist_ignores_todo_pilot_in_comments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "p.yaml").write_text("# TODO-PILOT markers are filled by the pilot\nmde: 1.75\n")
    monkeypatch.setattr(checklist, "PREREG", tmp_path / "p.yaml")
    items = checklist.run(Config(delivery=Delivery(gmail_user=None, gmail_app_password=None, email_to=None)), tmp_path, "2026-11-10")
    assert {i.name: i for i in items}["preregistered_mdes"].ok
