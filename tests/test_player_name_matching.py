"""A price is dropped when the book and the lineup feed spell a name differently.

The quote key is `(matchup, market, selection)` matched by string equality, and
the selection carries the player's name. The book writes "Ronald Acuna Jr." and
"Jose Ramirez"; the lineup feed writes "Ronald Acuna" and "Jose Ramirez" with
the accents. Over 2026-08-04..08 that left 52 hitters -- Acuna, Witt, Tatis,
Guerrero, Robert, Pena, Suarez, O'Neill, Ha-Seong Kim -- unpriced every single
time they appeared, so every prop on them was passed for want of a quote.
"""

from __future__ import annotations

from mlb_engine.market import keys


def test_the_names_that_were_never_priced_now_match() -> None:
    """Every failure mode seen on the real board: suffix, accent, apostrophe,
    hyphen, and the period on "Jr."."""
    pairs = [
        ("Ronald Acuña Jr. H o0.5", "Ronald Acuna H o0.5"),
        ("Bobby Witt Jr. H o1.5", "Bobby Witt H o1.5"),
        ("José Ramírez 1B o0.5", "Jose Ramirez 1B o0.5"),
        ("Jeremy Peña H o0.5", "Jeremy Pena H o0.5"),
        ("Michael Harris II H o0.5", "Michael Harris H o0.5"),
        ("Tyler O'Neill H o0.5", "Tyler ONeill H o0.5"),
        ("Ha-Seong Kim H o0.5", "Ha Seong Kim H o0.5"),
        ("Pete Crow-Armstrong H o0.5", "Pete Crow Armstrong H o0.5"),
        ("Ke'Bryan Hayes 1B o0.5", "KeBryan Hayes 1B o0.5"),
    ]
    for book, engine in pairs:
        assert keys.canonical(book) == keys.canonical(engine), book


def test_a_run_line_never_matches_its_mirror() -> None:
    """The sign and the point are the bet. Canonicalizing them away would price
    a +1.5 at the -1.5 quote, which is the opposite side.
    """
    assert keys.canonical("BOS +1.5") != keys.canonical("BOS -1.5")
    assert keys.canonical("Over 8.5") != keys.canonical("Over 9.5")
    assert keys.canonical("Jacob Wilson H o0.5") != keys.canonical("Jacob Wilson H o1.5")


def test_two_different_players_are_not_conflated() -> None:
    """The Nationals' Luis Garcia Jr. and the Astros' Luis Garcia collapse to
    the same canonical form. Rather than guess, the ambiguous form is dropped
    and both stay unpriced -- exactly what happens today.
    """
    quotes = {
        ("WSH @ HOU", "batter_h", "Luis Garcia Jr. H o0.5"): "a",
        ("WSH @ HOU", "batter_h", "Luis García H o0.5"): "b",
        ("WSH @ HOU", "batter_h", "CJ Abrams H o0.5"): "c",
    }
    idx = keys.canonical_index(quotes)
    assert ("WSH @ HOU", "batter_h", "luis garcia h o0.5") not in idx
    assert idx[("WSH @ HOU", "batter_h", "cj abrams h o0.5")] == "c"


def test_the_same_name_in_another_game_is_a_different_key() -> None:
    """Canonicalizing the selection must not merge across matchups or markets."""
    quotes = {
        ("WSH @ HOU", "batter_h", "José Altuve H o0.5"): "a",
        ("NYY @ BOS", "batter_h", "Jose Altuve H o0.5"): "b",
        ("WSH @ HOU", "batter_1b", "José Altuve 1B o0.5"): "c",
    }
    idx = keys.canonical_index(quotes)
    assert len(idx) == 3
    assert idx[("NYY @ BOS", "batter_h", "jose altuve h o0.5")] == "b"


def test_a_lone_v_is_left_alone() -> None:
    """"V" is a generational suffix and also a surname; dropping it would erase
    a real player, so only jr/sr/ii/iii/iv are removed.
    """
    assert keys.canonical("Robert V H o0.5") == "robert v h o0.5"
    assert keys.canonical("Robert IV H o0.5") == "robert h o0.5"
