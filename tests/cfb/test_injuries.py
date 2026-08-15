"""Availability feed, usage book, and the latency log they exist to write."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from cfb_engine.config import Config
from cfb_engine.data import injuries as inj
from cfb_engine.data.starters import PasserGame, build_starter_book, starter_absent

ROW = {
    "ID": "41413",
    "player": "Jabari Bates",
    "team": "San Jose State",
    "IR": "Out",
    "position": "QB",
    "injury_type": "Lower Leg",
    "ReturnDate": "Aug 29th",
    "game_datetime": "2026-08-29 15:00:00",
}


class _Resp:
    def __init__(self, payload: object = None, text: str = "") -> None:
        self._payload = payload
        self.text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


def test_report_keys_by_school_and_reads_the_designation(monkeypatch):
    monkeypatch.setattr(inj.http, "get", lambda *a, **k: _Resp([ROW, {"player": "no team"}]))
    book = inj.fetch_injury_report()
    assert list(book) == ["san jose state"]
    row = book["san jose state"][0]
    assert row.player == "Jabari Bates" and row.position == "QB"
    assert row.unavailable


@pytest.mark.parametrize(
    ("designation", "unavailable"),
    [("Out", True), ("OFS", True), ("SUSP", True), ("Doubtful", True),
     ("Questionable", False), ("Probable", False), ("", False)],
)
def test_only_real_absences_count(designation, unavailable):
    row = inj.InjuryRow("A", "1", "iowa", "QB", designation, "", "", "")
    assert row.unavailable is unavailable


def test_feed_failure_is_soft(monkeypatch):
    def boom(*a, **k):
        raise OSError("blocked")

    monkeypatch.setattr(inj.http, "get", boom)
    assert inj.fetch_injury_report() == {}
    assert inj.fetch_news() == {}


def test_non_list_payload_is_soft(monkeypatch):
    monkeypatch.setattr(inj.http, "get", lambda *a, **k: _Resp({"message": "nope"}))
    assert inj.fetch_injury_report() == {}


RSS = """<?xml version="1.0"?><rss><channel>
 <item><title>Jabari Bates: Will not play</title>
  <link>https://www.rotowire.com//cfootball/player/jabari-bates-41413</link>
  <pubDate>Fri, 14 Aug 2026 4:25:00 PM PDT</pubDate></item>
 <item><title>no id here</title><link>/cfootball/player/nobody</link>
  <pubDate>Fri, 14 Aug 2026 4:25:00 PM PDT</pubDate></item>
</channel></rss>"""


def test_stamps_accumulate_across_runs(tmp_path, monkeypatch):
    """The feed is a five-item window; Saturday's absence was Thursday's news."""
    cache = tmp_path / "injury_news.json"
    monkeypatch.setattr(inj.http, "get", lambda *a, **k: _Resp(text=RSS))
    first = inj.fetch_news(cache=cache)

    empty = "<?xml version='1.0'?><rss><channel></channel></rss>"
    monkeypatch.setattr(inj.http, "get", lambda *a, **k: _Resp(text=empty))
    later = inj.fetch_news(cache=cache)
    assert later["41413"].posted == first["41413"].posted
    assert later["41413"].headline == first["41413"].headline


def test_a_fresher_item_replaces_the_cached_one(tmp_path, monkeypatch):
    cache = tmp_path / "injury_news.json"
    monkeypatch.setattr(inj.http, "get", lambda *a, **k: _Resp(text=RSS))
    inj.fetch_news(cache=cache)
    newer = RSS.replace("4:25:00 PM PDT", "6:00:00 PM PDT").replace(
        "Will not play", "Will play after all"
    )
    monkeypatch.setattr(inj.http, "get", lambda *a, **k: _Resp(text=newer))
    items = inj.fetch_news(cache=cache)
    assert items["41413"].posted == datetime(2026, 8, 15, 1, 0, tzinfo=timezone.utc)
    assert items["41413"].headline.endswith("Will play after all")


def test_news_dates_the_designation(monkeypatch):
    monkeypatch.setattr(inj.http, "get", lambda *a, **k: _Resp(text=RSS))
    news = inj.fetch_news()
    assert list(news) == ["41413"]
    assert news["41413"].posted == datetime(2026, 8, 14, 23, 25, tzinfo=timezone.utc)


def test_note_names_who_is_out_and_says_it_is_not_scored():
    book = {"iowa": [inj.InjuryRow("Cade McNamara", "9", "iowa", "QB", "Out", "", "", "")]}
    note = inj.injury_note(book, "Iowa", "Nebraska")
    assert note is not None
    assert "Cade McNamara" in note and "not scored" in note
    assert inj.injury_note({}, "Iowa", "Nebraska") is None


def test_log_stamps_the_line_and_the_lead_time(tmp_path):
    seen = datetime(2026, 9, 5, 15, 0, tzinfo=timezone.utc)
    news = {"9": inj.NewsItem("9", "Out Saturday", seen - timedelta(hours=3))}
    path = tmp_path / "availability.jsonl"
    inj.log_availability(
        path,
        home="Iowa",
        away="Nebraska",
        rows=[inj.InjuryRow("Cade McNamara", "9", "iowa", "QB", "Out", "Knee", "", "")],
        spread=-6.5,
        news=news,
        observed=seen,
    )
    entry = json.loads(path.read_text().strip())
    assert entry["spread_home"] == -6.5
    assert entry["lead_time_s"] == 10800.0
    assert entry["observed_at"].startswith("2026-09-05T15:00")


def test_log_survives_an_unwritable_path(tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    inj.log_availability(
        blocked / "nested.jsonl", home="a", away="b",
        rows=[inj.InjuryRow("x", "1", "a", "QB", "Out", "", "", "")],
        spread=None, news={},
    )  # must not raise


# -- usage book ----------------------------------------------------------------

def _games(*specs: tuple[int, str, str, int]) -> list[PasserGame]:
    return [PasserGame(week=w, team=t, player_id=p, name=p.upper(), attempts=a)
            for w, t, p, a in specs]


def test_starter_is_the_prior_weeks_primary_passer():
    book = build_starter_book(
        _games((1, "Iowa", "cade", 30), (2, "Iowa", "cade", 28), (2, "Iowa", "deacon", 4),
               (3, "Iowa", "cade", 25)),
        through_week=3,
    )
    starter = book["iowa"]
    assert starter.player_id == "cade"
    # Week 3 is excluded, so 58 of 62 attempts.
    assert starter.attempts == 58
    assert starter.share == pytest.approx(58 / 62)
    assert starter.established and not starter.missed_last_week


def test_a_split_backfield_is_not_an_established_starter():
    book = build_starter_book(
        _games((1, "Iowa", "a", 20), (1, "Iowa", "b", 18), (2, "Iowa", "a", 20),
               (2, "Iowa", "b", 18)),
        through_week=3,
    )
    assert not book["iowa"].established  # 52.6% share, under the 75% threshold


def test_thin_usage_is_not_established():
    book = build_starter_book(_games((1, "Iowa", "a", 12), (2, "Iowa", "a", 10)),
                              through_week=3)
    assert book["iowa"].share == 1.0
    assert not book["iowa"].established  # 22 attempts, under the 30 floor


def test_missed_last_week_marks_stale_news():
    book = build_starter_book(
        _games((1, "Iowa", "cade", 40), (2, "Iowa", "cade", 35), (3, "Iowa", "deacon", 30)),
        through_week=4,
    )
    assert book["iowa"].player_id == "cade"
    assert book["iowa"].missed_last_week


def test_absence_matches_on_name_not_id():
    book = build_starter_book(
        _games((1, "Iowa", "cade", 40), (2, "Iowa", "cade", 35)), through_week=3
    )
    starter = book["iowa"]
    assert starter_absent(starter, ["CADE"])
    assert starter_absent(starter, ["Somebody Else"]) is False
    assert starter_absent(None, ["CADE"]) is False


def test_pricing_is_off_by_default():
    cfg = Config()
    assert cfg.injury_feed is True
    assert cfg.injury_qb_pts == 0.0


# -- the gate the pipeline applies ---------------------------------------------

def _pipeline(tmp_path, pts: float, monkeypatch):
    from cfb_engine.data.cfbd import CFBDClient
    from cfb_engine.pipeline import Pipeline

    monkeypatch.setenv("CFBE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CFBE_INJURY_QB_PTS", str(pts))
    cfg = Config()
    return Pipeline(cfg, cfbd=CFBDClient(None)), cfg


def _game():
    from cfb_engine.schemas import Game, TeamGameInfo

    return Game(
        game_id="1",
        game_date=datetime(2026, 9, 5).date(),
        home=TeamGameInfo(name="Iowa", abbrev="IOWA", is_home=True),
        away=TeamGameInfo(name="Nebraska", abbrev="NEB", is_home=False),
    )


def _odds():
    from cfb_engine.market.board import GameOdds
    from cfb_engine.market.ev import MarketQuote

    odds = GameOdds(matchup="NEB @ IOWA")
    odds.add_spread(-6.5, "IOWA", MarketQuote(book="b", american=-110, opposite_american=-110))
    return odds


def _book():
    # The name is what joins the feed to the usage book; ``_games`` names its
    # passers by upper-cased id, so "cade" is the same man as "Cade".
    return {"iowa": [inj.InjuryRow("Cade", "9", "iowa", "QB", "Out", "Knee", "", "")]}


def _starters(*, stale: bool):
    games = _games((1, "Iowa", "cade", 40), (2, "Iowa", "cade", 35))
    if stale:
        games += _games((3, "Iowa", "deacon", 10))
    return build_starter_book(games, through_week=4 if stale else 3)


def test_absence_is_logged_with_the_line_and_priced_at_nothing(tmp_path, monkeypatch):
    from cfb_engine.features.adjustments import Adjustment

    pipe, cfg = _pipeline(tmp_path, 0.0, monkeypatch)
    adj = Adjustment()
    pipe._record_absences(adj, _game(), _odds(), _book(), _starters(stale=False))

    assert adj.margin_delta == 0.0
    assert any("not scored" in r for r in adj.reasons)
    entry = json.loads(cfg.availability_file.read_text().strip())
    assert entry["spread_home"] == -6.5 and entry["player"] == "Cade"


def test_enabling_the_points_moves_the_line_away_from_the_short_handed_team(
    tmp_path, monkeypatch
):
    from cfb_engine.features.adjustments import Adjustment

    pipe, _ = _pipeline(tmp_path, 2.2, monkeypatch)
    adj = Adjustment()
    pipe._record_absences(adj, _game(), _odds(), _book(), _starters(stale=False))
    assert adj.margin_delta == pytest.approx(-2.2)  # home team is the one missing him


def test_stale_news_is_never_charged(tmp_path, monkeypatch):
    """Once the backup is common knowledge the market has it: holdout 47.1%."""
    from cfb_engine.features.adjustments import Adjustment

    pipe, _ = _pipeline(tmp_path, 2.2, monkeypatch)
    adj = Adjustment()
    pipe._record_absences(adj, _game(), _odds(), _book(), _starters(stale=True))
    assert adj.margin_delta == 0.0
    assert any("already priced in" in r for r in adj.reasons)


def test_a_rotation_player_is_not_a_starter_absence(tmp_path, monkeypatch):
    """Losing under 75% of the attempts measured *plus* 1.5 pts, so it is not a fade."""
    from cfb_engine.features.adjustments import Adjustment

    pipe, cfg = _pipeline(tmp_path, 2.2, monkeypatch)
    split = build_starter_book(
        _games((1, "Iowa", "cade", 20), (1, "Iowa", "deacon", 18)), through_week=2
    )
    adj = Adjustment()
    pipe._record_absences(adj, _game(), _odds(), _book(), split)
    assert adj.margin_delta == 0.0
    assert not any("without QB" in r for r in adj.reasons)
    # Still logged: the log is how the threshold gets re-measured.
    assert cfg.availability_file.exists()
