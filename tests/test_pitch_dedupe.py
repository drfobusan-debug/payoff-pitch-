"""Overlapping Statcast caches must not count the same plate appearance twice."""

from __future__ import annotations

import pandas as pd

from mlb_engine.data.statcast import dedupe_pitches
from mlb_engine.features.siera import pitcher_siera


def _pitch(inning: int, balls: int, strikes: int, event: str | None) -> dict[str, object]:
    return {
        "game_date": "2026-07-04",
        "home_team": "SEA",
        "away_team": "TEX",
        "batter": 100 + inning,
        "pitcher": 7,
        "inning": inning,
        "inning_topbot": "Top",
        "balls": balls,
        "strikes": strikes,
        "pitch_type": "FF",
        "release_speed": 94.1,
        "description": "called_strike" if event is None else "hit_into_play",
        "events": event,
        "bb_type": "ground_ball" if event == "field_out" else None,
    }


def _start() -> pd.DataFrame:
    """One inning per plate appearance: nine outs, one walk."""
    rows = [_pitch(i, 0, 0, None) for i in range(1, 10)]
    rows += [_pitch(i, 3, 0, "field_out") for i in range(1, 10)]
    rows.append(_pitch(10, 3, 0, "walk"))
    return pd.DataFrame(rows)


def test_a_pitch_scraped_twice_is_counted_once() -> None:
    one = _start()
    both = pd.concat([one, one], ignore_index=True)
    assert len(both) == 2 * len(one)
    assert len(dedupe_pitches(both)) == len(one)


def test_deduping_leaves_a_single_scrape_untouched() -> None:
    one = _start()
    assert len(dedupe_pitches(one)) == len(one)
    assert dedupe_pitches(one).index.is_unique


def test_distinct_pitches_in_the_same_at_bat_survive() -> None:
    """Same batter and inning, different count: two pitches, not one."""
    df = pd.DataFrame([_pitch(1, 0, 0, None), _pitch(1, 0, 1, None)])
    assert len(dedupe_pitches(df)) == 2


def test_duplication_would_have_moved_the_rates_it_protects() -> None:
    """The bug this guards: overlapping caches trebled a starter's walk rate."""
    one = _start()
    dupes = pd.concat([one, one.drop(columns=["bb_type"])], ignore_index=True)
    clean = pitcher_siera(dedupe_pitches(dupes))
    dirty = pitcher_siera(dupes)
    assert clean.pa == pitcher_siera(one).pa
    assert dirty.pa > clean.pa
    # Ground balls vanish from half the rows, so the arm reads worse than it is.
    assert dirty.siera > clean.siera


def test_a_reduced_frame_without_the_key_columns_is_returned_as_is() -> None:
    df = pd.DataFrame({"launch_speed": [95.0, 95.0], "launch_angle": [12.0, 12.0]})
    assert len(dedupe_pitches(df)) == 2
