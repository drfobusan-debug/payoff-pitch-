"""Tests for the observe-only per-pitcher inning run-attribution capture."""

from __future__ import annotations

import csv
from pathlib import Path

from mlb_engine.data.collapse import game_inning_lines, write_collapse_ledger


def _score_runner(pid: int, earned: bool = True) -> dict:
    return {
        "movement": {"end": "score"},
        "details": {"responsiblePitcher": {"id": pid}, "earned": earned},
    }


def _synthetic_pbp() -> dict:
    """Away starter (10) throws the 1st; in the 2nd he leaves a runner that scores
    off reliever 11 -- the run is charged to the responsible pitcher 10, not 11.
    Home starter (20) gives up a 3-run inning (a 'collapse') in the bottom 1st."""
    return {
        "allPlays": [
            # top 1: away batting, home pitcher 20 -- 3 runs charged to 20
            {"about": {"inning": 1, "halfInning": "top"}, "matchup": {"pitcher": {"id": 20, "fullName": "Home SP"}},
             "runners": [_score_runner(20), _score_runner(20), _score_runner(20)]},
            {"about": {"inning": 1, "halfInning": "top"}, "matchup": {"pitcher": {"id": 20, "fullName": "Home SP"}},
             "runners": []},
            # bottom 1: home batting, away pitcher 10 -- no runs
            {"about": {"inning": 1, "halfInning": "bottom"}, "matchup": {"pitcher": {"id": 10, "fullName": "Away SP"}},
             "runners": []},
            # bottom 2: away starter 10 faces one batter (leaves a runner), then reliever 11
            {"about": {"inning": 2, "halfInning": "bottom"}, "matchup": {"pitcher": {"id": 10, "fullName": "Away SP"}},
             "runners": []},
            # reliever 11 on the mound, but the run scores off starter 10 (inherited)
            {"about": {"inning": 2, "halfInning": "bottom"}, "matchup": {"pitcher": {"id": 11, "fullName": "Away RP"}},
             "runners": [_score_runner(10, earned=True)]},
        ]
    }


def test_inherited_runner_credited_to_responsible_pitcher() -> None:
    lines = game_inning_lines(_synthetic_pbp(), game_pk=1, date="2026-08-01", home_abbr="HHH", away_abbr="AAA")
    by = {(x.inning, x.half, x.pitcher_id): x for x in lines}

    # The run scored while reliever 11 pitched is charged to starter 10, not 11.
    assert by[(2, "bottom", 10)].runs_charged == 1
    assert by[(2, "bottom", 11)].runs_charged == 0
    assert by[(2, "bottom", 11)].batters_faced == 1

    # Home starter's collapse inning is captured as a >=2-run inning.
    assert by[(1, "top", 20)].runs_charged == 3
    assert by[(1, "top", 20)].is_start is True
    assert by[(1, "top", 20)].pitch_team == "HHH"
    assert by[(1, "top", 20)].bat_team == "AAA"

    # Starter/reliever roles resolve from who threw the first inning.
    assert by[(1, "bottom", 10)].is_start is True
    assert by[(2, "bottom", 11)].is_start is False


def test_write_collapse_ledger_dedupes(tmp_path: Path) -> None:
    lines = game_inning_lines(_synthetic_pbp(), game_pk=1, date="2026-08-01", home_abbr="HHH", away_abbr="AAA")
    write_collapse_ledger(lines, "2026-08-01", tmp_path)
    write_collapse_ledger(lines, "2026-08-01", tmp_path)  # re-run must not double-count
    rows = list(csv.DictReader((tmp_path / "collapse_ledger.csv").open()))
    assert len(rows) == len(lines)
