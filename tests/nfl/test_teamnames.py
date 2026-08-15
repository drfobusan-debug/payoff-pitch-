"""Board names to nflverse codes, including the franchises that moved."""

from __future__ import annotations

from nfl_engine.data import teamnames


def test_every_current_team_maps_from_its_board_name():
    codes = {code for code in teamnames.BY_NAME.values() if teamnames.is_team(code)}
    assert len(codes & teamnames.TEAMS) == 32


def test_odds_board_names_resolve():
    assert teamnames.code_for("Kansas City Chiefs") == "KC"
    assert teamnames.code_for("Los Angeles Rams") == "LA"
    assert teamnames.code_for("Los Angeles Chargers") == "LAC"
    assert teamnames.code_for("  washington commanders  ") == "WAS"


def test_an_unknown_name_returns_none_rather_than_a_guess():
    """Pricing the wrong team is worse than skipping a game."""
    assert teamnames.code_for("Toronto Argonauts") is None
    assert teamnames.code_for("") is None


def test_historical_names_keep_their_own_code():
    """A 2015 Rams game was played by STL, and the history has to agree."""
    assert teamnames.code_for("St. Louis Rams") == "STL"
    assert teamnames.code_for("Oakland Raiders") == "OAK"
    assert teamnames.canonical("STL") == "STL"


def test_relocations_resolve_to_the_current_franchise():
    assert teamnames.franchise("STL") == "LA"
    assert teamnames.franchise("OAK") == "LV"
    assert teamnames.franchise("SD") == "LAC"
    assert teamnames.franchise("KC") == "KC"


def test_alias_spellings_are_normalized():
    for alias, code in (("LAR", "LA"), ("WSH", "WAS"), ("JAC", "JAX")):
        assert teamnames.canonical(alias) == code
