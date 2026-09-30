"""The card's context box: records, ranks, ESPN colour, and the market line under the title."""

from __future__ import annotations

from datetime import date
from html import escape
from typing import NamedTuple

import pytest

from cfb_engine.data import cfbd
from cfb_engine.data.cfbd import CFBDClient, GameResult
from cfb_engine.data.espn import GameColor, TeamColor, _tags, parse_summary
from cfb_engine.market.tiers import Tier
from cfb_engine.output.brief import GameBrief, TeamBrief, _apply_color, _apply_sp, _form
from cfb_engine.output.card import _game_section, _market_line
from cfb_engine.recommendations import Recommendation, load_json, save_json

DAY = date(2026, 9, 12)


class _Resp(NamedTuple):
    payload: object

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


def _client(monkeypatch: pytest.MonkeyPatch, payload: object) -> CFBDClient:
    monkeypatch.setattr(cfbd.http, "get", lambda url, **kw: _Resp(payload))
    return CFBDClient("key", cache_dir=None)


# --- CFBD parsing ---------------------------------------------------------


def test_sp_table_reads_offense_and_defense_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "team": "Kansas",
            "rating": 8.1,
            "ranking": 41,
            "offense": {"rating": 31.2, "ranking": 38},
            "defense": {"rating": 23.1, "ranking": 24},
        },
        {"team": "nationalAverages", "rating": "n/a"},
    ]
    sp = _client(monkeypatch, rows).fetch_sp_table(2026)
    assert set(sp) == {"kansas"}
    line = sp["kansas"]
    assert (line.rank, line.off_rank, line.def_rank) == (41, 38, 24)
    assert line.off_rating == 31.2 and line.def_rating == 23.1


def test_ap_poll_takes_the_latest_week_only(monkeypatch: pytest.MonkeyPatch) -> None:
    weeks = [
        {"week": 1, "polls": [{"poll": "AP Top 25", "ranks": [{"school": "Texas", "rank": 1}]}]},
        {
            "week": 3,
            "polls": [
                {"poll": "Coaches Poll", "ranks": [{"school": "Ohio State", "rank": 1}]},
                {"poll": "AP Top 25", "ranks": [{"school": "Ohio State", "rank": 1}, {"school": "Texas", "rank": 4}]},
            ],
        },
    ]
    assert _client(monkeypatch, weeks).fetch_ap_poll(2026) == {"ohio state": 1, "texas": 4}


def test_player_ppa_rows_skip_incomplete_players(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"name": "A. Beck", "position": "QB", "team": "Georgia", "totalPPA": {"all": 41.2}},
        {"name": "No Total", "position": "RB", "team": "Georgia", "totalPPA": {}},
        "garbage",
    ]
    assert _client(monkeypatch, rows).fetch_player_ppa_rows(2026) == [("A. Beck", "QB", "Georgia", 41.2)]


def test_unavailable_client_returns_empty_context() -> None:
    c = CFBDClient("", cache_dir=None)
    assert c.fetch_sp_table(2026) == {}
    assert c.fetch_ap_poll(2026) == {}
    assert c.fetch_player_ppa_rows(2026) == []


# --- records and streaks --------------------------------------------------


def _g(home: str, away: str, hp: int, ap: int, stamp: str) -> GameResult:
    return GameResult(home=home, away=away, home_points=hp, away_points=ap, start_date=stamp)


def test_form_counts_record_streak_and_last_result() -> None:
    results = [
        _g("Kansas", "Fresno State", 31, 7, "2026-08-29T23:00:00.000Z"),
        _g("Wagner", "Kansas", 3, 45, "2026-09-05T20:00:00.000Z"),
        _g("Kansas", "Missouri", 20, 24, "2026-09-12T23:30:00.000Z"),  # tonight: not yet played
    ]
    t = _form(results, "Kansas", DAY)
    assert t.record == "2-0"
    assert t.streak == "W2"
    assert t.last == "beat Wagner 45-3"


def test_form_uses_the_eastern_kickoff_day_not_utc() -> None:
    # 8pm ET Friday is 00:00Z Saturday; a Saturday slate must still count it.
    late = _g("Kansas", "Wagner", 45, 3, "2026-09-12T00:00:00.000Z")
    assert _form([late], "Kansas", DAY).record == "1-0"
    assert _form([late], "Kansas", date(2026, 9, 11)).record == "0-0"


def test_form_losing_streak_and_no_games() -> None:
    results = [_g("A", "B", 10, 20, "2026-08-30T20:00:00Z"), _g("C", "A", 30, 0, "2026-09-05T20:00:00Z")]
    t = _form(results, "A", DAY)
    assert (t.record, t.streak, t.last) == ("0-2", "L2", "lost to C 0-30")
    empty = _form(results, "Z", DAY)
    assert (empty.record, empty.streak, empty.last) == ("0-0", None, None)


def test_apply_sp_fills_every_rating_field() -> None:
    t = TeamBrief(name="Kansas")
    _apply_sp(t, {"kansas": cfbd.SPLine(8.1, 41, 31.2, 38, 23.1, 24)})
    assert (t.sp_rank, t.sp_rating, t.off_rank, t.def_rank) == (41, 8.1, 38, 24)
    untouched = TeamBrief(name="Nobody")
    _apply_sp(untouched, {})
    assert untouched.sp_rank is None and not untouched.rated


# --- ESPN summary ---------------------------------------------------------


def _summary() -> dict[str, object]:
    return {
        "header": {
            "competitions": [
                {
                    "neutralSite": False,
                    "conferenceCompetition": True,
                    "notes": [{"headline": "Border War"}],
                    "competitors": [
                        {
                            "team": {"id": "1", "location": "Kansas", "displayName": "Kansas Jayhawks"},
                            "record": [{"type": "total", "displayValue": "1-0"}],
                        },
                        {
                            "team": {"id": "257", "location": "Richmond", "displayName": "Richmond Spiders"},
                            "record": [{"type": "total", "displayValue": "0-1"}],
                        },
                    ],
                }
            ]
        },
        "againstTheSpread": [
            {"team": {"id": "1", "displayName": "Kansas Jayhawks"}, "records": [{"displayValue": "1-0-0"}]},
        ],
        "leaders": [
            {
                "team": {"id": "257", "displayName": "Richmond Spiders"},  # mascot form only, as ESPN sends
                "leaders": [
                    {
                        "name": "passingYards",
                        "leaders": [
                            {
                                "displayValue": "18/27, 210 YDS, 2 TD",
                                "athlete": {"displayName": "Q. Back", "position": {"abbreviation": "QB"}},
                            }
                        ],
                    },
                    {"name": "tackles", "leaders": [{"displayValue": "9", "athlete": {"displayName": "X"}}]},
                ],
            }
        ],
        "gameInfo": {
            "venue": {"fullName": "David Booth Stadium", "grass": False, "address": {"city": "Lawrence", "state": "KS"}},
            "weather": {"temperature": 71, "precipitation": 10, "gust": 14},
        },
        "article": {
            "type": "Preview",
            "headline": "Rivalry renewed: <b>Missouri</b> visits Kansas",
            "story": "LAWRENCE, Kan. -- The rivals meet again. Kansas must win to stay alive. A third sentence.",
        },
        "predictor": {"homeTeam": {"gameProjection": "38.8"}},
    }


def test_parse_summary_reads_colour_and_resolves_mascot_names_by_id() -> None:
    gc = parse_summary(_summary(), "Kansas", "Richmond")
    assert gc.home.record == "1-0" and gc.away.record == "0-1"
    assert gc.home.ats == "1-0-0" and gc.away.ats is None
    assert gc.away.leaders == ["QB Q. Back — 18/27, 210 YDS, 2 TD"]
    assert gc.venue == "David Booth Stadium" and gc.city == "Lawrence, KS" and gc.grass is False
    assert (gc.temperature_f, gc.precip_pct, gc.gust_mph) == (71.0, 10.0, 14.0)
    assert gc.headline == "Rivalry renewed: Missouri visits Kansas"
    assert gc.story == "The rivals meet again. Kansas must win to stay alive."
    assert gc.fpi_home == 38.8
    assert gc.conference_game is True
    assert "rivalry" in gc.tags and "must-win" in gc.tags


def test_parse_summary_ignores_recaps_and_empty_payloads() -> None:
    p = _summary()
    p["article"] = {"type": "Recap", "headline": "Kansas rolls", "story": "Kansas won the rivalry game."}
    gc = parse_summary(p, "Kansas", "Richmond")
    assert gc.headline is None and gc.story is None
    assert gc.tags == ["rivalry"]  # only from the header note, not the recap
    empty = parse_summary({}, "Kansas", "Richmond")
    assert empty == GameColor()


def test_tags_come_only_from_source_text() -> None:
    assert _tags("Ohio State hosts Michigan") == []
    assert _tags("No. 3 Ohio State hosts No. 12 Michigan in a rivalry game") == ["rivalry", "ranked"]
    assert _tags(None, "") == []


def test_apply_color_only_fills_a_record_cfbd_did_not_have() -> None:
    b = GameBrief(home=TeamBrief(name="Kansas", wins=2), away=TeamBrief(name="Richmond"))
    gc = GameColor(
        home=TeamColor(record="9-9", ats="1-0-0"),
        away=TeamColor(record="0-1", leaders=["QB Q. Back — 210 YDS"]),
        venue="David Booth Stadium",
        neutral_site=True,
        temperature_f=71.0,
    )
    _apply_color(b, gc)
    assert b.home.record == "2-0"  # CFBD's finals win
    assert b.away.record == "0-1"  # ESPN fills the blank
    assert b.away.leaders == ["QB Q. Back — 210 YDS"]
    assert b.neutral_site and b.hfa_pts == 0.0
    assert b.temperature_f == 71.0


# --- persistence ----------------------------------------------------------


def _brief() -> GameBrief:
    return GameBrief(
        home=TeamBrief(name="Kansas", wins=1, streak="W1", sp_rank=41, sp_rating=8.1, off_rating=31.2, off_rank=38, def_rating=23.1, def_rank=24, tr_rank=52),
        away=TeamBrief(name="Missouri", wins=1, streak="W1", sp_rank=22, poll_rank=25, key_players=["QB B. Pribula"]),
        hfa_pts=2.4,
        venue="David Booth Stadium",
        city="Lawrence, KS",
        temperature_f=71.0,
        wind_mph=8.0,
        headline="Rivalry renewed",
        tags=["rivalry"],
    )


def test_game_brief_round_trips_through_dict() -> None:
    b = _brief()
    again = GameBrief.from_dict(b.to_dict())
    assert again == b
    assert GameBrief.from_dict({"home": {"name": "A"}, "away": {"name": "B"}}).home.name == "A"


def _rec(market: str, selection: str, model: float, fair: float, brief: GameBrief | None) -> Recommendation:
    return Recommendation(
        game_date=DAY,
        game_id="g1",
        matchup="Missouri @ Kansas",
        market=market,
        selection=selection,
        model_prob=model,
        market_american=-110,
        ev=0.0,
        edge=model - fair,
        tier=Tier.PASS,
        fair_prob=fair,
        home_abbrev="Kansas",
        away_abbrev="Missouri",
        team_side="home",
        side="win",
        brief=brief,
    )


def test_recommendation_json_carries_the_brief_and_loads_without_one(tmp_path) -> None:
    path = tmp_path / "p.json"
    save_json([_rec("game_ml", "Kansas ML", 0.388, 0.345, _brief())], path)
    (loaded,) = load_json(path)
    assert loaded.brief == _brief()

    legacy = tmp_path / "old.json"
    save_json([_rec("game_ml", "Kansas ML", 0.388, 0.345, None)], legacy)
    text = legacy.read_text().replace('"brief": null', '"tier": "Pass"')
    legacy.write_text(text)
    (old,) = load_json(legacy)
    assert old.brief is None


# --- card rendering -------------------------------------------------------


def _slate_recs(brief: GameBrief | None) -> list[Recommendation]:
    return [
        _rec("game_ml", "Kansas ML", 0.388, 0.345, brief),
        _rec("game_ats", "Kansas +5.5", 0.52, 0.502, brief),
        _rec("game_total", "Over 51.5", 0.48, 0.505, brief),
    ]


def test_market_line_is_italic_and_shows_every_market_once() -> None:
    line = _market_line(_slate_recs(None))
    assert line.startswith("<p class='mkt'><i>") and line.endswith("</i></p>")
    assert "ML" in line and "34.5%" in line and "38.8%" in line
    assert "ATS" in line and "50.2%" in line
    assert "Total" in line and "50.5%" in line
    assert _market_line([]) == ""


def test_game_section_puts_the_market_line_under_the_title_and_renders_context() -> None:
    html = _game_section(_slate_recs(_brief()))
    title = html.index("<h2>")
    mkt = html.index("<p class='mkt'>")
    assert title < mkt < html.index("<div class='shape'>")
    for needle in ("1-0", "W1", "#41", "#22", "#38", "#24", "TR #52", "No. 25", "B. Pribula", "David Booth Stadium", "2.4"):
        assert needle in html, needle
    assert "Rivalry renewed" in html


def test_game_section_without_a_brief_still_renders() -> None:
    html = _game_section(_slate_recs(None))
    assert "<h2>" in html and "<p class='mkt'>" in html
    assert "<table class='teams'>" not in html


def test_external_text_is_escaped_in_the_card() -> None:
    b = _brief()
    b.headline = "<script>alert(1)</script> & co"
    b.venue = "Stadium <i>x</i>"
    b.away.leaders = ["QB <b>Bad</b> — 1 YD"]
    html = _game_section(_slate_recs(b))
    assert "<script>" not in html and escape("<script>alert(1)</script>") in html
    assert "<i>x</i>" not in html
    assert "<b>Bad</b>" not in html
