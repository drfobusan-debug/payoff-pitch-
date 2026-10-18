"""RotoWire lineups feed: parser, goalie ladder, injuries, thin-state fallback."""

from __future__ import annotations

import json
from datetime import date as Date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests

from nhl_engine.data import availability as avail
from nhl_engine.data import rotowire
from nhl_engine.features import lineup_feed, starters
from nhl_engine.pipeline import _apply_thin_fallback

FIX = Path(__file__).parent / "fixtures" / "rotowire_lineups.html"
SLATE = Date(2026, 9, 30)


def _slate() -> rotowire.RotoSlate:
    return rotowire.parse_page(FIX.read_text(encoding="utf-8"), SLATE, "2026-09-30T12:00:00Z")


def test_parse_games_teams_goalies_and_sections():
    s = _slate()
    assert [g.matchup for g in s.games] == ["PIT@PHI", "LAK@COL"]
    g = s.games[0]
    assert g.time_et == "7:30 PM ET" and not g.started
    assert (g.away.goalie, g.away.goalie_status) == ("Arturs Silovs", "probable")
    assert (g.home.goalie, g.home.goalie_status) == ("D. Vladar", "confirmed")
    assert g.away.pp_units == {"PP1": ["Sidney Crosby", "Erik Karlsson"], "PP2": ["Rickard Rakell"]}
    assert [(p.name, p.status, p.pos) for p in g.away.injuries] == [
        ("Justin Brazeau", "IR", "RW"),
        ("Tristan Broz", "DTD", "C"),
    ]
    started = s.games[1]
    assert started.started
    assert started.away.goalie == "Darcy Kuemper" and started.away.goalie_status == ""
    assert started.home.goalie == "" and started.home.players == ()


def test_parse_unknown_markup_yields_empty_not_wrong():
    assert rotowire.parse_page("<html><body>maintenance</body></html>", SLATE).games == []


def test_fetch_only_today_or_tomorrow_and_archives_raw(tmp_path):
    sess = MagicMock()
    resp = MagicMock(text=FIX.read_text(encoding="utf-8"))
    resp.raise_for_status.return_value = None
    sess.get.return_value = resp
    out = rotowire.fetch(SLATE, today=SLATE, data_dir=tmp_path, session=sess)
    assert sess.get.call_args.args[0] == rotowire.URL
    assert out.raw_path is not None and out.raw_path.exists()
    rotowire.fetch(SLATE, today=Date(2026, 9, 29), session=sess)
    assert sess.get.call_args.args[0].endswith("?date=tomorrow")
    with pytest.raises(ValueError):
        rotowire.fetch(SLATE, today=Date(2026, 9, 27), session=sess)
    sess.get.side_effect = requests.ConnectionError("down")
    with pytest.raises(requests.ConnectionError):
        rotowire.fetch(SLATE, today=SLATE, session=sess)


def _mp() -> MagicMock:
    goalies = pd.DataFrame(
        {
            "playerId": [8481668, 8478435, 8475311, 1],
            "name": ["Arturs Silovs", "Dan Vladar", "Darcy Kuemper", "Dylan Vladar"],
            "team": ["PIT", "PHI", "L.A", "PHI"],
            "situation": ["all"] * 4,
        }
    )
    skaters = pd.DataFrame(
        {
            "playerId": [8470000, 8470001],
            "name": ["Justin Brazeau", "Sidney Crosby"],
            "team": ["PIT", "PIT"],
            "situation": ["all", "all"],
        }
    )
    mp = MagicMock()
    mp.season_goalies.side_effect = lambda s: goalies if s == 2026 else pd.DataFrame()
    mp.season_skaters.side_effect = lambda s: skaters if s == 2026 else pd.DataFrame()
    return mp


def test_resolver_full_name_and_initial_fallback_within_team():
    r = lineup_feed.NameResolver(_mp(), 2026)
    assert r.resolve("Arturs Silovs", "PIT", "G") == (8481668, "Arturs Silovs")
    assert r.resolve("D. Kuemper", "LAK", "G") == (8475311, "Darcy Kuemper")
    assert r.resolve("D. Vladar", "PHI", "G") is None  # two D. Vladars -> ambiguous
    assert r.resolve("Nobody Here", "PIT", "S") is None


def test_apply_slate_writes_overrides_and_availability(tmp_path):
    s = _slate()
    summary = lineup_feed.apply_slate(s, mp=_mp(), data_dir=tmp_path, season=2026)
    ov = starters.load_overrides(starters.overrides_path(tmp_path, SLATE))
    assert ov["PIT"] == {
        "player_id": 8481668,
        "status": "probable",
        "source": "rotowire",
        "name": "Arturs Silovs",
    }
    assert "PHI" not in ov and any("PHI D. Vladar" in u for u in summary.goalies_unresolved)
    assert "LAK" not in ov  # no status dot -> nothing written
    recs = avail.read_log(tmp_path, SLATE)
    by_name = {r.name: r for r in recs}
    assert by_name["Justin Brazeau"].status == "out"
    assert by_name["Justin Brazeau"].player_id == 8470000 and by_name["Justin Brazeau"].role == "F"
    assert by_name["Tristan Broz"].status == "questionable"
    assert (
        by_name["Tristan Broz"].player_id == 0 and by_name["Tristan Broz"].tag == avail.NOT_SCORED
    )
    assert summary.injuries_logged == 2 and summary.injuries_unresolved == ["PIT Tristan Broz"]
    again = lineup_feed.apply_slate(s, mp=_mp(), data_dir=tmp_path, season=2026)
    assert again.injuries_logged == 0 and again.goalies_kept and not again.goalies_set


def test_manual_confirmed_override_beats_rotowire_expected(tmp_path):
    path = starters.overrides_path(tmp_path, SLATE)
    starters.save_override(path, "PIT", 999, "confirmed", "manual")
    summary = lineup_feed.apply_slate(_slate(), mp=_mp(), data_dir=tmp_path, season=2026)
    ov = starters.load_overrides(path)
    assert ov["PIT"]["player_id"] == 999 and ov["PIT"]["source"] == "manual"
    assert summary.goalies_kept and not summary.goalies_set
    # a stale probable pointing at another goalie is replaced by an equal-rank read
    starters.save_override(path, "PIT", 999, "probable", "manual")
    lineup_feed.apply_slate(_slate(), mp=_mp(), data_dir=tmp_path, season=2026)
    assert starters.load_overrides(path)["PIT"]["player_id"] == 8481668


def test_rotowire_status_clears_goalie_gate():
    for status in ("confirmed", "probable"):
        assert starters.Starter("PIT", 1, "x", status, "rotowire", None).confirmed
    assert not starters.Starter(
        "PIT", 1, "x", "projected", "season ice-time leader", None
    ).confirmed


def test_thin_fallback_keeps_pp_pk_prior_touches_only_sh_states():
    rates = {"pp_xgf60": 8.5, "pk_xga60": 6.5, "pp_xga60": 0.9, "sh_xgf60": 1.1}
    rel = dict.fromkeys(rates, 0.0)
    league = {"pp_xgf60": 7.4, "pk_xga60": 7.4, "pp_xga60": 0.6, "sh_xgf60": 0.6}
    out = _apply_thin_fallback(rates, rel, league, 0.2)
    assert out["pp_xgf60"] == 8.5 and out["pk_xga60"] == 6.5
    assert out["pp_xga60"] == 0.6 and out["sh_xgf60"] == 0.6


def test_apply_slate_appends_every_goalie_read_to_history(tmp_path):
    lineup_feed.apply_slate(_slate(), mp=_mp(), data_dir=tmp_path, season=2026, seen_at="t1")
    lineup_feed.apply_slate(_slate(), mp=_mp(), data_dir=tmp_path, season=2026, seen_at="t2")
    lines = starters.history_path(tmp_path, SLATE).read_text().splitlines()
    reads = [json.loads(x) for x in lines]
    assert [r["seen_at"] for r in reads if r["team"] == "PIT"] == ["t1", "t2"]
    assert all(r["player_id"] == 8481668 and r["status"] == "probable" for r in reads)
    assert not any(r["team"] in ("PHI", "LAK") for r in reads)  # unresolved / no status
    assert starters.load_overrides(starters.overrides_path(tmp_path, SLATE))["PIT"]
