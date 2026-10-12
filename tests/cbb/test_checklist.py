from __future__ import annotations

from pathlib import Path

from cbb_engine import checklist
from cbb_engine.config import Config, Delivery


def test_checklist_fails_closed_without_data(tmp_path: Path) -> None:
    items = checklist.run(Config(delivery=Delivery(gmail_user=None, gmail_app_password=None, email_to=None)), tmp_path, "2026-11-10")
    by = {i.name: i for i in items}
    assert not by["width_fitted"].ok and not by["preregistered_mdes"].ok and not by["emails_configured"].ok
    assert "RESEARCH MODE" in checklist.render(items)
    assert by["credit_budget"].ok
