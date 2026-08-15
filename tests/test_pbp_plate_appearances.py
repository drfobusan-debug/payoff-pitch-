"""A runner thrown out is not a plate appearance, whatever the feed calls it.

Batting order is recovered by counting plate appearances, so anything miscounted
here shifts every slot after it in that game onto the wrong man. The feed types a
caught stealing as ``atBat``, which is the trap: these tests pin the distinction
so a hazard or a substitute-rate fit cannot be trained on phantom substitutions
again.
"""

from __future__ import annotations

from typing import Any

from mlb_engine.data.pbp import RUNNER_EVENT_TYPES, is_plate_appearance, plate_appearances


def _play(event_type: str, kind: str = "atBat") -> dict[str, Any]:
    return {"result": {"type": kind, "eventType": event_type, "event": event_type}}


def test_a_runner_thrown_out_does_not_end_the_appearance() -> None:
    """The batter is still standing there; his turn finishes on the next play."""
    for event in ("caught_stealing_2b", "pickoff_1b", "pickoff_caught_stealing_2b",
                  "other_out", "wild_pitch", "stolen_base_2b"):
        assert not is_plate_appearance(_play(event)), event


def test_the_appearances_that_do_count_are_kept() -> None:
    """Including the ones that are not at-bats: a walk and a sacrifice still count."""
    for event in ("single", "double", "home_run", "strikeout", "walk", "hit_by_pitch",
                  "sac_fly", "sac_bunt", "field_error", "fielders_choice_out",
                  "grounded_into_double_play", "catcher_interf"):
        assert is_plate_appearance(_play(event)), event


def test_a_pitching_substitution_is_not_a_plate_appearance() -> None:
    """Only ``atBat`` plays are candidates at all."""
    assert not is_plate_appearance(_play("pitching_substitution", kind="pitchingSubstitution"))
    assert not is_plate_appearance({})
    assert not is_plate_appearance({"result": None})


def test_the_order_is_preserved_and_only_runner_events_are_dropped() -> None:
    """The pointer advances on the survivors, so feed order has to hold."""
    plays = [
        _play("single"),
        _play("caught_stealing_2b"),
        _play("strikeout"),
        _play("pickoff_1b"),
        _play("walk"),
    ]
    kept = plate_appearances(plays)
    assert [p["result"]["eventType"] for p in kept] == ["single", "strikeout", "walk"]


def test_the_dropped_events_are_named_by_event_type_not_display_text() -> None:
    """``event`` is prose and changes; ``eventType`` is the stable key."""
    assert "caught_stealing_2b" in RUNNER_EVENT_TYPES
    assert "Caught Stealing 2B" not in RUNNER_EVENT_TYPES
    # A real plate appearance is never in the set, or every game loses its order.
    assert not RUNNER_EVENT_TYPES & {"single", "walk", "strikeout", "home_run"}
