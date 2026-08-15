"""VSiN per-team home-field-advantage table."""

from __future__ import annotations

from cfb_engine.config import Config
from cfb_engine.data.vsin import hfa_for, hfa_note


def test_override_is_off_by_default():
    """The guide's tiers are fitted on their own three-year home ATS record."""
    assert Config().vsin_hfa is False


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


def test_note_names_the_price_it_did_not_charge():
    note = hfa_note("Ohio State", 2.4, enabled=False)
    assert note is not None
    assert "3.5" in note and "2.4" in note and "not scored" in note


def test_note_is_silent_while_the_override_is_live():
    assert hfa_note("Ohio State", 2.4, enabled=True) is None


def test_note_is_silent_for_unlisted_teams_and_matching_values():
    assert hfa_note("Vanderbilt", 2.4, enabled=False) is None
    assert hfa_note("Ohio State", 3.5, enabled=False) is None
