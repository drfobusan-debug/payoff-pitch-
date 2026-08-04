"""VSiN per-team home-field-advantage table."""

from __future__ import annotations

from cfb_engine.data.vsin import hfa_for


def test_listed_teams_use_guide_values():
    # 3.5-pt tier
    assert hfa_for("Ohio State", 2.4) == 3.5
    assert hfa_for("Notre Dame Fighting Irish", 2.4) == 3.5
    # 3.0-pt tier, incl. distinct Miami programs
    assert hfa_for("Miami (FL)", 2.4) == 3.0
    assert hfa_for("LSU", 2.4) == 3.0
    # 1.5-pt tier
    assert hfa_for("UCLA", 2.4) == 1.5
    # 1.0-pt tier
    assert hfa_for("Purdue", 2.4) == 1.0


def test_ole_miss_alias_resolves():
    # Guide lists "Mississippi"; CFBD/odds use "Ole Miss" -> same key.
    assert hfa_for("Ole Miss", 2.4) == 3.5
    assert hfa_for("Mississippi", 2.4) == 3.5


def test_miami_oh_distinct_from_miami_fl():
    assert hfa_for("Miami (OH)", 2.4) == 3.5


def test_unlisted_team_keeps_default():
    assert hfa_for("Vanderbilt", 2.4) == 2.4
    assert hfa_for("Some Nonexistent School", 3.1) == 3.1


def test_disabled_returns_default():
    assert hfa_for("Ohio State", 2.4, enabled=False) == 2.4
