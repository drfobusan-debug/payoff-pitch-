"""Daily Faceoff goalie board: parser, status map, fetch URL/archive, ladder precedence."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from nhl_engine.data import dailyfaceoff, rotowire
from nhl_engine.features import lineup_feed, starters

FIX = Path(__file__).parent / "fixtures" / "dailyfaceoff_goalies.html"
SLATE = Date(2026, 10, 26)


def _slate(fetched: str = "2026-10-26T20:00:00Z") -> rotowire.RotoSlate:
    return dailyfaceoff.parse_page(FIX.read_text(encoding="utf-8"), SLATE, fetched)


def test_parse_goalies_and_status_ladder():
    s = _slate()
    assert [g.matchup for g in s.games] == ["LAK@NYR", "TOR@CGY", "UTA@SJS"]
    g = s.games[0]
    assert (g.away.goalie, g.away.goalie_status) == ("Darcy Kuemper", "confirmed")
    assert (g.home.goalie, g.home.goalie_status) == ("James Reimer", "confirmed")
    assert g.time_et == "19:00" and not g.started
    likely, none = s.games[1].away, s.games[1].home
    assert likely.goalie_status == "probable"
    assert (none.goalie, none.goalie_status) == ("Devin Cooley", "")
    unconfirmed, expected = s.games[2].away, s.games[2].home
    assert unconfirmed.goalie_status == ""
    assert expected.goalie_status == "probable"
    assert not any(t.players for g in s.games for t in (g.away, g.home))


def test_started_flag_reads_the_gmt_tip():
    s = _slate("2026-10-27T00:00:00Z")
    assert [g.started for g in s.games] == [True, False, False]


def test_unknown_markup_or_other_date_yields_empty_not_wrong():
    assert dailyfaceoff.parse_page("<html>nothing</html>", SLATE).games == []
    assert dailyfaceoff.parse_page(FIX.read_text(encoding="utf-8"), Date(2026, 10, 27)).games == []


def test_fetch_url_by_date_and_archives_raw(tmp_path):
    assert dailyfaceoff.url_for(SLATE, SLATE) == dailyfaceoff.URL
    assert dailyfaceoff.url_for(Date(2026, 10, 27), SLATE) == dailyfaceoff.URL + "2026-10-27"
    sess = MagicMock()
    sess.get.return_value = MagicMock(
        text=FIX.read_text(encoding="utf-8"), raise_for_status=lambda: None
    )
    s = dailyfaceoff.fetch(SLATE, today=SLATE, data_dir=tmp_path, session=sess)
    assert len(s.games) == 3
    assert s.raw_path is not None and s.raw_path.parent == tmp_path / "dailyfaceoff" / "2026-10-26"
    assert s.raw_path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def _mp() -> MagicMock:
    mp = MagicMock()
    mp.season_goalies.return_value = pd.DataFrame(
        {
            "playerId": [8475311, 8473503],
            "name": ["Darcy Kuemper", "James Reimer"],
            "team": ["LAK", "NYR"],
            "situation": ["all", "all"],
        }
    )
    mp.season_skaters.return_value = pd.DataFrame(columns=["playerId", "name", "team", "situation"])
    return mp


def test_dailyfaceoff_confirmed_beats_rotowire_probable_and_is_tagged(tmp_path):
    path = starters.overrides_path(tmp_path, SLATE)
    starters.save_override(path, "LAK", 8475311, "probable", rotowire.SOURCE, name="Darcy Kuemper")
    starters.save_override(path, "NYR", 1, "confirmed", "manual")
    summary = lineup_feed.apply_slate(
        _slate(), mp=_mp(), data_dir=tmp_path, season=2026, source=dailyfaceoff.SOURCE
    )
    ov = starters.load_overrides(path)
    assert ov["LAK"] == {
        "player_id": 8475311,
        "status": "confirmed",
        "source": "dailyfaceoff",
        "name": "Darcy Kuemper",
    }
    # an equal-rank read naming another goalie is the newer word and replaces the stale one
    assert ov["NYR"]["player_id"] == 8473503 and ov["NYR"]["source"] == "dailyfaceoff"
    assert summary.goalies_set == ["LAK Darcy Kuemper confirmed", "NYR James Reimer confirmed"]
    assert not summary.goalies_kept
    hist = starters.history_path(tmp_path, SLATE)
    assert "dailyfaceoff" in hist.read_text(encoding="utf-8")
