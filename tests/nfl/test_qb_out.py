"""A starter the official report has ruled out is priced as a fill-in and shown out."""

from __future__ import annotations

from datetime import date as Date

import pandas as pd
import pytest

from nfl_engine.data import injuries
from nfl_engine.features import books as books_mod
from nfl_engine.features import quarterback as qb_mod
from nfl_engine.output.brief import GameBrief, TeamBrief, _apply_report
from nfl_engine.output.card import _out_line
from nfl_engine.schemas import Game, TeamGameInfo


def _report() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026] * 4,
            "week": [6, 6, 6, 5],
            "team": ["BAL", "BAL", "BAL", "CLE"],
            "gsis_id": ["00-0034796", "00-JURGENS", "00-ANDREWS", "00-WATSON"],
            "full_name": ["Lamar Jackson", "Cam Jurgens", "Mark Andrews", "Deshaun Watson"],
            "position": ["QB", "C", "TE", "QB"],
            "report_status": ["Out", "Out", "Questionable", "Out"],
            "report_primary_injury": ["Ankle", "Concussion", None, "Knee"],
        }
    )


def _game() -> Game:
    return Game(
        game_id="BAL@CLE",
        season=2026,
        week=6,
        game_date=Date(2026, 10, 18),
        home=TeamGameInfo(name="CLE", abbrev="CLE", is_home=True),
        away=TeamGameInfo(name="BAL", abbrev="BAL", is_home=False),
    )


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026],
            "week": [6],
            "home_team": ["CLE"],
            "away_team": ["BAL"],
            "home_qb_id": ["00-WATSON"],
            "away_qb_id": ["00-0034796"],
        }
    )


def test_the_report_keeps_only_this_weeks_out_and_doubtful(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(injuries.nflverse, "injuries", lambda season: _report())
    rows = injuries.official_report(2026, 6)
    assert [r.player for r in rows] == ["Lamar Jackson", "Cam Jurgens"]
    assert rows[0].label() == "QB Lamar Jackson (out, ankle)"


def test_a_missing_report_marks_no_one(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(injuries.nflverse, "injuries", lambda season: pd.DataFrame())
    assert injuries.official_report(2026, 6) == []


def test_a_ruled_out_scheduled_starter_is_flagged(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(books_mod.nflverse, "games", _schedule)
    monkeypatch.setattr(injuries.nflverse, "injuries", lambda season: _report())
    games = [_game()]
    assert books_mod.attach_qbs(games, 2026, 6) == 2
    assert games[0].away_qb_out == "Lamar Jackson"
    # Watson's Out is on last week's report, not this one.
    assert games[0].home_qb_out is None


def _book() -> qb_mod.StarterBook:
    return qb_mod.StarterBook(prior={(2026, "BAL"): "00-0034796", (2026, "CLE"): "00-WATSON"})


def test_a_ruled_out_incumbent_is_charged_as_a_fill_in():
    delta, notes = qb_mod.margin_delta(
        _book(),
        season=2026,
        week=6,
        home="CLE",
        away="BAL",
        home_qb="00-WATSON",
        away_qb="00-0034796",
        away_qb_out="Lamar Jackson",
    )
    assert delta == -qb_mod.FILL_IN_MARGIN_POINTS
    assert "Lamar Jackson out on the official report" in notes[0]


def test_without_the_report_the_incumbent_charges_nothing():
    delta, notes = qb_mod.margin_delta(
        _book(),
        season=2026,
        week=6,
        home="CLE",
        away="BAL",
        home_qb="00-WATSON",
        away_qb="00-0034796",
    )
    assert (delta, notes) == (0.0, ())


def test_the_out_line_leads_with_the_report_and_keeps_espns_extras():
    away = TeamBrief(code="BAL", name="Baltimore Ravens", qb="Lamar Jackson")
    away.out = ["C Cam Jurgens (Out)", "WR Elijah Sarratt (Out)"]
    rows = [
        injuries.ReportRow("BAL", "00-JURGENS", "Cam Jurgens", "C", "Out", "Concussion"),
        injuries.ReportRow("BAL", "00-0034796", "Lamar Jackson", "QB", "Out", "Ankle"),
    ]
    _apply_report(away, injuries.report_by_team(rows)["BAL"])
    assert away.out == [
        "QB Lamar Jackson (out, ankle)",
        "C Cam Jurgens (out, concussion)",
        "WR Elijah Sarratt (Out)",
    ]
    assert away.qb == "Lamar Jackson (out)"
    brief = GameBrief(
        matchup="BAL @ CLE", home=TeamBrief(code="CLE", name="Cleveland Browns"), away=away
    )
    assert "QB Lamar Jackson (out, ankle)" in _out_line(brief, "")


def _pbp() -> pd.DataFrame:
    """AAA throws well and runs badly; BBB the reverse. Week 3 and a blowout play are out."""
    rows = []
    for off, dfn, kind, epa in (
        ("AAA", "BBB", "pass", 0.4),
        ("AAA", "BBB", "rush", -0.3),
        ("BBB", "AAA", "pass", -0.2),
        ("BBB", "AAA", "rush", 0.2),
    ):
        rows.append(
            {
                "season": 2026,
                "season_type": "REG",
                "week": 1,
                "posteam": off,
                "defteam": dfn,
                "pass": int(kind == "pass"),
                "rush": int(kind == "rush"),
                "qb_dropback": int(kind == "pass"),
                "sack": int(off == "BBB" and kind == "pass"),
                "epa": epa,
                "wp": 0.5,
            }
        )
    late = dict(rows[2], week=3, epa=5.0)
    blowout = dict(rows[2], wp=0.97, epa=5.0)
    return pd.DataFrame([*rows, late, blowout])


def test_unit_ranks_use_prior_weeks_and_drop_garbage_time():
    from nfl_engine.output.brief import unit_ranks

    ranks = unit_ranks(_pbp(), 2026, 3)
    assert ranks["AAA"] == {
        "pass_off": 1,
        "rush_off": 2,
        "pass_def": 1,
        "rush_def": 2,
        "pass_rush": 1,
    }
    assert ranks["BBB"]["pass_off"] == 2 and ranks["BBB"]["rush_off"] == 1
    assert unit_ranks(pd.DataFrame(), 2026, 3) == {}


def test_the_units_line_reads_like_the_cfb_card():
    from nfl_engine.output.card import _units_line

    home = TeamBrief(code="CLE", name="Cleveland Browns")
    away = TeamBrief(code="BAL", name="Baltimore Ravens")
    away.units = {"pass_off": 3, "rush_off": 20, "pass_def": 10, "rush_def": 4, "pass_rush": 7}
    home.units = {"pass_off": 30, "rush_off": 15, "pass_def": 25, "rush_def": 18, "pass_rush": 2}
    line = _units_line(GameBrief(matchup="BAL @ CLE", home=home, away=away))
    assert (
        "Baltimore Ravens&#x27; elite pass offense (#3) meets a poor Cleveland Browns pass" in line
    )
    assert "defense (#25) — edge Baltimore Ravens" in line
    assert "run offense (#20) meets a middling Cleveland Browns run defense (#18) — even" in line
    assert "Pass rush (sacks per dropback): Baltimore Ravens #7, Cleveland Browns #2" in line
    assert (
        _units_line(GameBrief(matchup="X @ Y", home=TeamBrief("Y", "Y"), away=TeamBrief("X", "X")))
        == ""
    )
