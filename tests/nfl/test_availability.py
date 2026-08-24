"""The injury watcher: who is out, when we heard it, and what the number did.

The property under test is the same one the benchmark holds to: an observation is
recorded and displayed, and no probability, screen, tier or price moves because of
it. The rest is the measurement -- the archive bracketing a posting -- and the ways
a free feed fails.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pytest

from nfl_engine import cli
from nfl_engine.audit import availability
from nfl_engine.audit.availability import AHEAD, BEHIND, UNMEASURED
from nfl_engine.audit.ledger import ENGINE, PAPER, LedgerEntry
from nfl_engine.data import capture, injuries
from nfl_engine.data.injuries import InjuryRow, NewsItem
from nfl_engine.output.card import build_card, render_html, render_markdown

SEASON, WEEK = 2026, 1
MATCHUP = "BUF @ KC"
POSTED = datetime(2026, 9, 9, 18, 0, tzinfo=timezone.utc)


def quote(captured_at: str, home_point: float, total: float, book: str = "dk") -> list:
    home = MATCHUP.split(" @ ")[-1]
    away = MATCHUP.split(" @ ")[0]
    common = dict(
        captured_at=captured_at,
        season=SEASON,
        week=WEEK,
        game_date="2026-09-13",
        matchup=MATCHUP,
        book=book,
    )
    return [
        capture.QuoteRow(
            market=capture.SPREAD,
            side=home,
            line=home_point,
            american=-110,
            opposite_american=-110,
            **common,
        ),
        capture.QuoteRow(
            market=capture.SPREAD,
            side=away,
            line=-home_point,
            american=-110,
            opposite_american=-110,
            **common,
        ),
        capture.QuoteRow(
            market=capture.TOTAL,
            side="over",
            line=total,
            american=-105,
            opposite_american=-115,
            **common,
        ),
    ]


def archive(root, snapshots: list[tuple[str, float, float]]) -> None:
    for taken, home_point, total in snapshots:
        capture.write_snapshot(quote(taken, home_point, total), season=SEASON, week=WEEK, root=root)


def injury(position: str = "QB", designation: str = "Out", player: str = "Josh Allen") -> InjuryRow:
    return InjuryRow(
        player=player,
        player_id="12345",
        team="BUF",
        position=position,
        designation=designation,
        injury="Elbow",
    )


def news(posted: datetime = POSTED) -> NewsItem:
    return NewsItem(player_id="12345", headline="Josh Allen: Will not play", posted=posted)


# -- what is watched -----------------------------------------------------
def test_the_watcher_covers_quarterbacks_skill_positions_and_the_line() -> None:
    assert injuries.group_of("QB") == injuries.QB
    assert {injuries.group_of(p) for p in ("RB", "WR", "TE", "FB")} == {injuries.SKILL}
    assert {injuries.group_of(p) for p in ("LT", "RT", "C", "G", "OL")} == {injuries.LINE}


def test_a_fourth_safety_is_read_but_not_watched() -> None:
    """Recorded groups stay narrow deliberately: a nickel corner moves no number."""
    row = injury(position="CB")
    assert row.group == injuries.OTHER
    assert row.unavailable and not row.watched


def test_questionable_is_not_an_absence() -> None:
    assert not injury(designation="Questionable").unavailable
    assert injury(designation="Doubtful").unavailable
    assert injury(designation="IR").unavailable


def test_the_designation_is_kept_as_published() -> None:
    row = injury(designation="Injured Reserve")
    assert row.designation == "Injured Reserve"
    assert row.source == injuries.ROTOWIRE


def test_the_watched_list_is_ordered_by_group() -> None:
    book = {
        "BUF": [
            injury(position="WR", player="Khalil Shakir"),
            injury(position="LT", player="Dion Dawkins"),
            injury(position="QB", player="Josh Allen"),
            injury(position="CB", player="Christian Benford"),
        ]
    }
    assert [r.player for r in injuries.watched_for(book, "BUF")] == [
        "Josh Allen",
        "Khalil Shakir",
        "Dion Dawkins",
    ]


# -- the feeds -----------------------------------------------------------
def test_the_report_reads_the_table_and_keys_it_by_team(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [
        {
            "ID": "1",
            "player": "Josh Allen",
            "team": "BUF",
            "position": "QB",
            "injury": "Elbow",
            "status": "Out",
        },
        {
            "ID": "2",
            "player": "Puka Nacua",
            "team": "LAR",
            "position": "WR",
            "injury": "Knee",
            "status": "Questionable",
        },
        {"ID": "3", "player": "Nobody", "team": "XXX", "position": "QB", "status": "Out"},
        "not a row",
    ]
    monkeypatch.setattr(injuries.http, "get", lambda *a, **k: _resp(payload))
    book = injuries.fetch_report()
    # LAR is spelled LA in the game file, and an unknown code is dropped rather
    # than keyed on a team that does not exist.
    assert sorted(book) == ["BUF", "LA"]
    assert book["BUF"][0].designation == "Out"


def test_a_dead_report_is_an_empty_book(monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(*_a: object, **_k: object) -> object:
        raise OSError("connection reset by peer")

    monkeypatch.setattr(injuries.http, "get", refused)
    assert injuries.fetch_report() == {}


def test_a_report_that_is_not_a_list_is_an_empty_book(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(injuries.http, "get", lambda *a, **k: _resp({"error": "nope"}))
    assert injuries.fetch_report() == {}


def test_a_pacific_afternoon_stamp_is_not_read_as_morning_utc() -> None:
    """The feed's pubDate is not RFC 822; read naively it is 19 hours out."""
    read = injuries.posted_at("Sun, 23 Aug 2026 9:07:00 PM PDT")
    assert read == datetime(2026, 8, 24, 4, 7, tzinfo=timezone.utc)


def test_an_unreadable_stamp_is_none() -> None:
    assert injuries.posted_at("") is None
    assert injuries.posted_at("whenever") is None


def test_news_stamps_accumulate_across_runs(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A Wednesday item has to be able to date a Sunday absence."""
    cache = tmp_path / "injury_news.json"
    monkeypatch.setattr(
        injuries.http,
        "get",
        lambda *a, **k: _resp_text(_rss("12345", "Wed, 09 Sep 2026 11:00:00 AM PDT")),
    )
    first = injuries.fetch_news(cache=cache)
    assert "12345" in first

    def gone(*_a: object, **_k: object) -> object:
        raise OSError("feed down")

    monkeypatch.setattr(injuries.http, "get", gone)
    again = injuries.fetch_news(cache=cache)
    assert again["12345"].posted == first["12345"].posted


def test_a_corrupt_stamp_cache_is_not_fatal(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    cache = tmp_path / "injury_news.json"
    cache.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(injuries.http, "get", lambda *a, **k: _resp_text("<rss></rss>"))
    assert injuries.fetch_news(cache=cache) == {}


# -- the measurement -----------------------------------------------------
def test_the_move_is_bracketed_by_the_captures_either_side(tmp_path) -> None:
    archive(
        tmp_path,
        [
            ("2026-09-09T12:00:00Z", -2.5, 47.5),
            ("2026-09-09T20:00:00Z", -1.0, 46.5),
            ("2026-09-10T12:00:00Z", -0.5, 46.0),
        ],
    )
    move = availability.movement(MATCHUP, POSTED, season=SEASON, week=WEEK, root=tmp_path)
    assert move.before is not None and move.before.captured_at == "2026-09-09T12:00:00Z"
    assert move.after is not None and move.after.captured_at == "2026-09-09T20:00:00Z"
    assert move.spread_move == 1.5
    assert move.total_move == -1.0
    assert move.timing == AHEAD


def test_a_still_number_after_the_news_is_behind_it(tmp_path) -> None:
    """Either the market knew or it does not care; one game cannot say which."""
    archive(tmp_path, [("2026-09-09T12:00:00Z", -2.5, 47.5), ("2026-09-09T20:00:00Z", -2.5, 48.5)])
    move = availability.movement(MATCHUP, POSTED, season=SEASON, week=WEEK, root=tmp_path)
    assert move.spread_move == 0.0
    assert move.timing == BEHIND


def test_no_capture_before_the_news_is_unmeasured_not_zero(tmp_path) -> None:
    archive(tmp_path, [("2026-09-09T20:00:00Z", -1.0, 46.5)])
    move = availability.movement(MATCHUP, POSTED, season=SEASON, week=WEEK, root=tmp_path)
    assert move.before is None
    assert move.spread_move is None
    assert move.timing == UNMEASURED


def test_no_capture_after_the_news_is_unmeasured(tmp_path) -> None:
    archive(tmp_path, [("2026-09-09T12:00:00Z", -2.5, 47.5)])
    move = availability.movement(MATCHUP, POSTED, season=SEASON, week=WEEK, root=tmp_path)
    assert move.after is None
    assert move.timing == UNMEASURED


def test_an_empty_archive_is_unmeasured(tmp_path) -> None:
    move = availability.movement(MATCHUP, POSTED, season=SEASON, week=WEEK, root=tmp_path)
    assert (move.before, move.after, move.timing) == (None, None, UNMEASURED)


def test_the_reading_is_the_consensus_not_one_book(tmp_path) -> None:
    rows = (
        quote("2026-09-09T12:00:00Z", -2.5, 47.5, book="dk")
        + quote("2026-09-09T12:00:00Z", -3.0, 47.5, book="fd")
        + quote("2026-09-09T12:00:00Z", -3.0, 48.0, book="mgm")
    )
    read = availability.reading(rows, MATCHUP)
    assert read is not None
    assert read.home_spread == -3.0
    assert read.total == 47.5
    assert read.books == 3


def test_another_game_in_the_snapshot_is_not_read(tmp_path) -> None:
    assert availability.reading(quote("2026-09-09T12:00:00Z", -2.5, 47.5), "NYJ @ MIA") is None


def test_an_undated_absence_is_recorded_with_no_assumed_lead(tmp_path) -> None:
    archive(tmp_path, [("2026-09-09T12:00:00Z", -2.5, 47.5), ("2026-09-09T20:00:00Z", -1.0, 46.5)])
    obs = availability.observe(
        injury(),
        season=SEASON,
        week=WEEK,
        matchup=MATCHUP,
        news=None,
        observed=POSTED,
        root=tmp_path,
    )
    assert obs.posted_at is None and obs.lead_time_s is None
    assert obs.spread_move is None
    assert obs.timing == UNMEASURED


def test_an_observation_keeps_the_lead_time_and_both_readings(tmp_path) -> None:
    archive(tmp_path, [("2026-09-09T12:00:00Z", -2.5, 47.5), ("2026-09-09T20:00:00Z", -1.0, 46.5)])
    seen = datetime(2026, 9, 9, 18, 30, tzinfo=timezone.utc)
    obs = availability.observe(
        injury(),
        season=SEASON,
        week=WEEK,
        matchup=MATCHUP,
        news=news(),
        observed=seen,
        root=tmp_path,
    )
    assert obs.lead_time_s == 1800.0
    assert (obs.spread_before, obs.spread_after, obs.spread_move) == (-2.5, -1.0, 1.5)
    assert (obs.capture_before, obs.capture_after) == (
        "2026-09-09T12:00:00Z",
        "2026-09-09T20:00:00Z",
    )
    assert obs.group == injuries.QB and obs.source == injuries.ROTOWIRE


# -- the log -------------------------------------------------------------
def observation(**overrides: object) -> availability.Observation:
    base = dict(
        observed_at="2026-09-09T18:30:00Z",
        season=SEASON,
        week=WEEK,
        matchup=MATCHUP,
        team="BUF",
        player="Josh Allen",
        player_id="12345",
        position="QB",
        group=injuries.QB,
        designation="Out",
        injury="Elbow",
        source=injuries.ROTOWIRE,
        posted_at=POSTED.isoformat(),
        headline="Josh Allen: Will not play",
        lead_time_s=1800.0,
        spread_before=-2.5,
        spread_after=-1.0,
        spread_move=1.5,
        total_before=47.5,
        total_after=46.5,
        capture_before="2026-09-09T12:00:00Z",
        capture_after="2026-09-09T20:00:00Z",
        timing=AHEAD,
    )
    base.update(overrides)
    return availability.Observation(**base)  # type: ignore[arg-type]


def test_the_log_appends_and_never_rewrites(tmp_path) -> None:
    path = tmp_path / "availability.jsonl"
    assert availability.append(path, [observation()]) == 1
    assert availability.append(path, [observation(player="Dion Dawkins", position="LT")]) == 1
    held = availability.read_log(path)
    assert [obs.player for obs in held] == ["Josh Allen", "Dion Dawkins"]


def test_a_corrupt_line_costs_that_row_only(tmp_path) -> None:
    path = tmp_path / "availability.jsonl"
    availability.append(path, [observation()])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{truncated\n")
        handle.write(json.dumps({"season": SEASON}) + "\n")
    availability.append(path, [observation(player="Dion Dawkins")])
    assert [obs.player for obs in availability.read_log(path)] == ["Josh Allen", "Dion Dawkins"]


def test_an_unwritable_log_is_reported_not_raised(tmp_path) -> None:
    path = tmp_path / "readonly"
    path.mkdir()
    path.chmod(0o500)
    try:
        assert availability.append(path / "sub" / "availability.jsonl", [observation()]) == 0
    finally:
        path.chmod(0o700)


def test_the_log_reads_back_only_the_week_asked_for(tmp_path) -> None:
    path = tmp_path / "availability.jsonl"
    availability.append(path, [observation(), observation(week=2, player="Rasheed Walker")])
    assert [obs.player for obs in availability.read_log(path, season=SEASON, week=WEEK)] == [
        "Josh Allen"
    ]


def test_a_missing_log_is_empty_not_an_error(tmp_path) -> None:
    assert availability.read_log(tmp_path / "nothing.jsonl") == []


def test_timing_counts_split_by_group(tmp_path) -> None:
    counts = availability.timing_counts(
        [
            observation(),
            observation(timing=BEHIND),
            observation(group=injuries.LINE, timing=BEHIND),
        ]
    )
    assert counts[AHEAD] == 1
    assert counts[BEHIND] == 2
    assert counts[f"{injuries.LINE}/{BEHIND}"] == 1


# -- the card ------------------------------------------------------------
def engine_row(**overrides: object) -> LedgerEntry:
    base = dict(
        date="2026-09-13",
        season=SEASON,
        week=WEEK,
        matchup=MATCHUP,
        market="moneyline",
        side="KC",
        line=None,
        book="dk",
        odds=-130.0,
        opposite_odds=110.0,
        model_prob=0.60,
        fair_prob=0.55,
        ev_model=0.04,
        ev_fair=0.05,
        tier="Moderate",
        screens="",
        paired_books=3,
        captured_at="2026-09-09T18:00:00Z",
        source=ENGINE,
        mode=PAPER,
    )
    base.update(overrides)
    return LedgerEntry(**base)  # type: ignore[arg-type]


def test_the_card_names_who_is_out_and_says_it_is_not_priced() -> None:
    card = build_card([engine_row()], season=SEASON, week=WEEK, absences=[observation()])
    text = render_markdown(card)
    assert "BUF: QB Josh Allen" in text
    assert "reported, not priced" in text
    assert "1 ahead of an archived move" in text
    assert "BUF: QB Josh Allen" in render_html(card)


def test_absences_on_another_game_are_not_shown() -> None:
    card = build_card(
        [engine_row()],
        season=SEASON,
        week=WEEK,
        absences=[observation(matchup="NYJ @ MIA")],
    )
    assert card.games[0].absences == ""


def test_a_long_list_is_summarised_not_truncated_silently() -> None:
    rows = [observation(player=f"Player {n}", position="WR") for n in range(5)]
    assert "+2" in availability.note(rows, MATCHUP)


def test_the_play_is_the_same_with_and_without_the_absence() -> None:
    """The whole point: an absence is read, and it moves nothing."""
    without = build_card([engine_row()], season=SEASON, week=WEEK)
    with_it = build_card([engine_row()], season=SEASON, week=WEEK, absences=[observation()])
    assert [p.__dict__ for p in without.plays()] == [p.__dict__ for p in with_it.plays()]
    assert without.record == with_it.record


# -- the command ---------------------------------------------------------
def _args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {"season": SEASON, "week": WEEK, "write": True}
    base.update(overrides)
    return argparse.Namespace(**base)


def test_the_command_records_the_watched_groups_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
) -> None:
    path = tmp_path / "availability.jsonl"
    archive(tmp_path, [("2026-09-09T12:00:00Z", -2.5, 47.5), ("2026-09-09T20:00:00Z", -1.0, 46.5)])
    monkeypatch.setattr(availability, "log_path", lambda: path)
    monkeypatch.setattr(cli.availability, "log_path", lambda: path)
    monkeypatch.setattr(cli, "_week_matchups", lambda season, week: [(MATCHUP, "BUF", "KC")])
    monkeypatch.setattr(
        cli.injuries,
        "fetch_report",
        lambda: {"BUF": [injury(), injury(position="CB", player="Christian Benford")]},
    )
    monkeypatch.setattr(cli.injuries, "fetch_news", lambda **_k: {"12345": news()})
    monkeypatch.setattr(
        cli.capture, "now_utc", lambda: datetime(2026, 9, 9, 18, 30, tzinfo=timezone.utc)
    )
    monkeypatch.setattr(cli.capture, "capture_dir", lambda root=None: tmp_path / "captures")
    assert cli.cmd_injuries(_args()) == 0
    held = availability.read_log(path)
    assert [obs.player for obs in held] == ["Josh Allen"]
    assert (held[0].spread_move, held[0].timing) == (1.5, AHEAD)
    assert "reported, never priced" in capsys.readouterr().out


def test_a_dead_feed_records_nothing_and_still_exits_clean(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
) -> None:
    path = tmp_path / "availability.jsonl"
    monkeypatch.setattr(cli.availability, "log_path", lambda: path)
    monkeypatch.setattr(cli.injuries, "fetch_report", lambda: {})
    monkeypatch.setattr(cli.injuries, "fetch_news", lambda **_k: pytest.fail("dated nothing"))
    assert cli.cmd_injuries(_args()) == 0
    assert not path.exists()
    assert "no injury report available" in capsys.readouterr().out


def test_the_feed_failing_costs_the_watcher_not_the_week(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    def refused() -> dict[str, list[InjuryRow]]:
        raise OSError("connection reset by peer")

    monkeypatch.setattr(cli.injuries, "fetch_report", refused)
    assert cli._injury_step(_args()) == 0
    assert "injury report unavailable" in capsys.readouterr().out


def _resp(payload: object) -> object:
    class Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> object:
            return payload

    return Resp()


def _resp_text(text: str) -> object:
    class Resp:
        def __init__(self) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    return Resp()


def _rss(player_id: str, pub_date: str) -> str:
    return (
        "<rss><channel><item>"
        f"<link>https://www.rotowire.com/football/player/josh-allen-{player_id}</link>"
        f"<pubDate>{pub_date}</pubDate><title>Josh Allen: Out</title>"
        "</item></channel></rss>"
    )
