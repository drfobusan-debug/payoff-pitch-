"""Transfer-portal churn: aggregation, the pre-season cutoff, and the card note."""

from __future__ import annotations

from cfb_engine.data.portal import build_portal_book, portal_for, portal_note


def _move(origin, destination, stars, position, date="2025-01-10"):
    return {
        "origin": origin,
        "destination": destination,
        "stars": stars,
        "position": position,
        "transferDate": f"{date}T00:00:00.000Z",
    }


def test_net_talent_counts_both_ends_of_a_move():
    book = build_portal_book([_move("Rice", "Texas", 4, "QB")], 2025)
    texas, rice = portal_for(book, "Texas"), portal_for(book, "Rice")
    assert texas is not None and rice is not None
    # A 4-star QB is 3.0 star points at the 2.0 quarterback weight.
    assert texas.net == 6.0
    assert rice.net == -6.0
    assert texas.players_in == 1 and rice.players_out == 1
    assert texas.qb_net == 3.0 and rice.qb_net == -3.0


def test_position_weighting_ranks_a_quarterback_above_a_safety():
    qb = build_portal_book([_move(None, "Texas", 4, "QB")], 2025)
    safety = build_portal_book([_move(None, "Texas", 4, "S")], 2025)
    assert portal_for(qb, "Texas").net > portal_for(safety, "Texas").net


def test_in_season_transfers_are_dropped():
    """August onwards is not information the pre-season roster had."""
    rows = [
        _move(None, "Texas", 5, "QB", date="2025-01-05"),
        _move(None, "Texas", 5, "QB", date="2025-09-20"),
    ]
    book = build_portal_book(rows, 2025)
    assert portal_for(book, "Texas").players_in == 1


def test_unrated_and_unknown_positions_do_not_crash_or_score():
    book = build_portal_book(
        [{"origin": "Rice", "destination": "Texas", "stars": None, "position": None}], 2025
    )
    assert portal_for(book, "Texas").net == 0.0
    assert portal_for(book, "Texas").players_in == 1


def test_churn_sums_both_directions():
    rows = [_move(None, "Texas", 4, "WR"), _move("Texas", None, 4, "WR")]
    book = build_portal_book(rows, 2025)
    team = portal_for(book, "Texas")
    assert team.net == 0.0
    assert team.churn == 6.0


def test_note_names_the_gap_and_marks_it_unscored():
    rows = [_move("Rice", "Texas", 5, "QB"), _move("Rice", "Texas", 4, "WR")]
    book = build_portal_book(rows, 2025)
    note = portal_note(book, "Texas", "Rice")
    assert note is not None
    assert "Texas" in note and "not scored" in note


def test_note_is_silent_when_the_rosters_churned_evenly():
    rows = [_move("Rice", "Texas", 3, "WR"), _move("Texas", "Rice", 3, "WR")]
    book = build_portal_book(rows, 2025)
    assert portal_note(book, "Texas", "Rice") is None


def test_note_is_silent_for_teams_with_no_portal_row():
    book = build_portal_book([_move("Rice", "Texas", 4, "QB")], 2025)
    assert portal_note(book, "Texas", "Navy") is None
