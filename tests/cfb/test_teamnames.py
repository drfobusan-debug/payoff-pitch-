"""Cross-source school-name normalization."""

from __future__ import annotations

from cfb_engine.data.teamnames import school_key, short_code


def test_mascot_and_bare_school_collapse():
    assert school_key("Alabama Crimson Tide") == school_key("Alabama") == "alabama"
    assert school_key("Ohio State Buckeyes") == school_key("Ohio State") == "ohio state"


def test_state_word_is_not_truncated():
    # "State" is a real school-name token, not a mascot, and must survive.
    assert school_key("Ohio State") == "ohio state"
    assert school_key("Michigan State Spartans") == "michigan state"


def test_st_abbreviation_expands():
    assert school_key("San Jose St") == "san jose state"


def test_aliases_resolve():
    assert school_key("Ole Miss") == "mississippi"
    assert school_key("Pitt") == "pittsburgh"
    assert school_key("NC State Wolfpack") == "north carolina state"
    assert school_key("UCF") == "central florida"


def test_notre_dame_multiword_mascot():
    assert school_key("Notre Dame Fighting Irish") == "notre dame"


def test_short_code_drops_mascot():
    assert short_code("Alabama Crimson Tide") == "Alabama"
