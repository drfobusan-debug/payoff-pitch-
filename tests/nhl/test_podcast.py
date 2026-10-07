"""Podcast read: feed/title dating, transcript segmentation, summary cache, PDF block,
and that none of it reaches a price or gate."""

from __future__ import annotations

import json
from datetime import date as Date
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nhl_engine import report
from nhl_engine.data import podcast
from nhl_engine.features import podcast_read as pr
from tests.nhl.test_report import _card

FIX = Path(__file__).parent / "fixtures"
PUB = datetime(2026, 10, 6, 3, 6, tzinfo=timezone.utc)
MATCHUPS = [("CAR", "MTL"), ("NSH", "TOR"), ("UTA", "NJD"), ("MIN", "BUF"), ("OTT", "DET")]


def _lines() -> list[podcast.Line]:
    return podcast.read_transcript(FIX / "podcast_ep640_excerpt.txt")


def _episode(**kw) -> podcast.Episode:
    base = dict(
        guid="g1",
        title="NHL Picks & Best Bets: Tuesday, October 6 (Ep. 640)",
        published=PUB.isoformat(),
        audio_url="https://x/ep.mp3",
        duration="00:54:29",
        slate_dates=("2026-10-06",),
    )
    base.update(kw)
    return podcast.Episode(**base)


# ---------------------------------------------------------------- feed / dates


@pytest.mark.parametrize(
    ("title", "pub", "expect"),
    [
        ("NHL Picks & Best Bets: Tuesday, October 6 (Ep. 640)", PUB, ("2026-10-06",)),
        ("NHL Picks & Best Bets: Saturday, Oct. 3 (Ep. 639)", PUB, ("2026-10-03",)),
        (
            "NHL Picks & Best Bets: Thursday, Oct. 1 & Friday, Oct. 2 (Ep. 638)",
            datetime(2026, 10, 1, 4, tzinfo=timezone.utc),
            ("2026-10-01", "2026-10-02"),
        ),
        (
            "NHL Opening Night Picks: NHL Best Bets for 9/29 + 9/30 (Ep. 637)",
            datetime(2026, 9, 29, tzinfo=timezone.utc),
            ("2026-09-29", "2026-09-30"),
        ),
        ("NHL Atlantic Division Preview: Sabres, Canadiens (Ep. 636)", PUB, ()),
        # Year rolls over: Dec 31 publish, Jan 1 slate.
        (
            "NHL Picks: Thursday, Jan. 1",
            datetime(2026, 12, 31, 23, tzinfo=timezone.utc),
            ("2027-01-01",),
        ),
        # A date months away from publish is not this episode's slate.
        ("NHL Picks: March 5 look-ahead", PUB, ()),
    ],
)
def test_slate_dates_from_title(title, pub, expect):
    assert podcast.slate_dates_from_title(title, pub) == expect


def test_parse_feed_and_select():
    eps = podcast.parse_feed((FIX / "podcast_feed.xml").read_text())
    assert len(eps) == 6 and eps[0].audio_url.startswith("https://")
    assert eps[0].slate_dates == ("2026-10-06",)
    assert eps[2].slate_dates == ("2026-10-01", "2026-10-02")
    assert eps[4].slate_dates == ()  # division preview: never attached to a slate
    hit = podcast.episodes_for(eps, Date(2026, 10, 2))
    assert [e.title[-9:] for e in hit] == ["(Ep. 638)"]
    assert podcast.episodes_for(eps, Date(2026, 10, 7)) == []


def test_transcript_round_trip(tmp_path: Path):
    lines = [podcast.Line(5, "hello"), podcast.Line(3725, "late line")]
    p = podcast.write_transcript(tmp_path / "t.txt", lines)
    assert p.read_text() == "[00:05] hello\n[62:05] late line\n"
    assert podcast.read_transcript(p) == lines


def test_ensure_transcript_cached_and_no_transcribe(tmp_path: Path):
    ep = _episode()
    assert podcast.ensure_transcript(ep, tmp_path, transcribe_missing=False) is None
    assert json.loads(podcast.meta_path(tmp_path, ep).read_text())["title"] == ep.title
    podcast.write_transcript(podcast.transcript_path(tmp_path, ep), [podcast.Line(0, "x")])
    with patch.object(podcast, "download_audio", side_effect=AssertionError("no net")):
        assert podcast.ensure_transcript(ep, tmp_path) == podcast.transcript_path(tmp_path, ep)


def test_transcribe_without_whisper_is_unavailable(tmp_path: Path):
    with (
        patch.dict("sys.modules", {"faster_whisper": None}),
        pytest.raises(podcast.PodcastUnavailable),
    ):
        podcast.transcribe(tmp_path / "a.mp3", tmp_path / "a.txt")


def test_download_failure_is_unavailable(tmp_path: Path):
    sess = MagicMock()
    sess.get.side_effect = podcast.requests.ConnectionError("down")
    with pytest.raises(podcast.PodcastUnavailable):
        podcast.download_audio(_episode(), tmp_path, session=sess)


# ---------------------------------------------------------------- segmentation


def test_teams_in_exact_and_fuzzy():
    codes = {"BUF", "MIN", "TOR", "NSH", "LAK", "DET"}
    assert set(pr.teams_in("minnesota while on minus 112 at the buffalo savers", codes)) == {
        "BUF",
        "MIN",
    }
    assert set(pr.teams_in("the nastro predators at the Toronto maple leaves", codes)) == {
        "NSH",
        "TOR",
    }
    # kings/wings are exact-only: a near miss never crosses over.
    assert pr.teams_in("the kings looked old", codes) == ["LAK"]
    assert pr.teams_in("the wings were booed", codes) == ["DET"]
    assert pr.teams_in("wins and kinks", codes) == []


def test_segment_anchors_quotes_and_mentions():
    games, others = pr.segment(_lines(), MATCHUPS)
    assert list(games) == ["CAR @ MTL", "NSH @ TOR", "UTA @ NJD", "MIN @ BUF"]
    car = games["CAR @ MTL"]
    assert car.anchor == "16:46"
    assert car.quoted_ml == {"CAR": -125, "MTL": 105} and car.quoted_total == 6.5
    tor = games["NSH @ TOR"]
    assert tor.quoted_ml == {"NSH": 130, "TOR": -155} and tor.quoted_total == 5.5
    assert any("25 bucks bucks" in m.text for m in tor.mentions)
    uta = games["UTA @ NJD"]
    assert any("puck bucks" in m.text for m in uta.mentions)
    assert all(len(g.mentions) <= pr.MAX_MENTIONS for g in games.values())
    assert "OTT @ DET" not in games  # not in the excerpt -> not invented
    assert others == []
    # The segment text is bounded by the next intro.
    assert "[19:23]" not in car.excerpt and "[19:23]" in tor.excerpt


def test_segment_flags_intros_for_games_not_on_slate():
    games, others = pr.segment(_lines(), [("CAR", "MTL")])
    assert list(games) == ["CAR @ MTL"]
    assert len(others) == 3 and others[0].startswith(("NSH/TOR", "TOR/NSH"))


def test_segment_ignores_casual_two_team_lines():
    lines = [
        podcast.Line(10, "the habs beat the penguins and then carolina lost"),
        podcast.Line(20, "I think montreal is good"),
    ]
    games, _ = pr.segment(lines, [("CAR", "MTL")])
    assert games == {}


# ---------------------------------------------------------------- read json / summaries


def _read() -> pr.PodcastRead:
    return pr.build_read(_episode(), _lines(), Date(2026, 10, 6), MATCHUPS)


def test_read_json_round_trip(tmp_path: Path):
    read = _read()
    pr.save_read(read, tmp_path)
    back = pr.load_read(tmp_path, Date(2026, 10, 6))
    assert back == read
    assert pr.load_read(tmp_path, Date(2026, 10, 7)) is None


def test_add_summaries_no_key_is_noop(tmp_path: Path):
    read = _read()
    assert pr.add_summaries(read, tmp_path, api_key=None) == 0
    assert all(g.summary == [] and g.summary_source == "" for g in read.games.values())


def test_add_summaries_caches_and_survives_failure(tmp_path: Path):
    read = _read()
    sess = MagicMock()
    resp = MagicMock()
    resp.json.return_value = {
        "choices": [
            {"message": {"content": "- MTL ML +105 — 20 puck bucks — home opener\n- Over 6.5"}}
        ]
    }
    sess.post.return_value = resp
    n = pr.add_summaries(read, tmp_path, api_key="k", session=sess)
    assert n == 4 and sess.post.call_count == 4
    car = read.games["CAR @ MTL"]
    assert car.summary == ["MTL ML +105 — 20 puck bucks — home opener", "Over 6.5"]
    assert car.summary_source == f"openai:{pr.OPENAI_MODEL}"
    body = sess.post.call_args.kwargs["json"]
    assert body["model"] == pr.OPENAI_MODEL and "Transcript:" in body["messages"][1]["content"]
    # Second pass: all cached, no calls.
    read2 = _read()
    sess.post.reset_mock()
    assert pr.add_summaries(read2, tmp_path, api_key="k", session=sess) == 0
    assert sess.post.call_count == 0 and read2.games["CAR @ MTL"].summary == car.summary
    # A failing call leaves that game deterministic-only.
    read3 = pr.build_read(_episode(), _lines(), Date(2026, 10, 6), [("OTT", "DET")] + MATCHUPS)
    for g in read3.games.values():
        g.excerpt += " changed"
    sess.post.side_effect = pr.requests.ConnectionError("down")
    assert pr.add_summaries(read3, tmp_path, api_key="k", session=sess) == 0
    assert all(g.summary == [] for g in read3.games.values())


# ---------------------------------------------------------------- report


def test_pdf_block_per_game_and_no_episode(tmp_path: Path):
    card, ctx = _card(tmp_path)
    before = [(r.model_prob, r.edge, r.ev, r.is_buy, tuple(r.gates)) for r in card.rows]
    assert "Pod read" not in report.render_html(card, ctx)

    ctx.podcast = pr.PodcastRead(
        "2026-10-10", "no_episode", detail="no episode titled for this slate"
    )
    html = report.render_html(card, ctx)
    assert "no episode titled for this slate" in html and "Pod read" not in html

    ctx.podcast = pr.PodcastRead(
        "2026-10-10", "ok", episode_title="Ep. 1", published="2026-10-10T03:00"
    )
    html = report.render_html(card, ctx)
    assert "game not discussed on the episode" in html

    ctx.podcast.games["A @ H"] = pr.GameRead(
        "A @ H",
        "16:46",
        {"A": 105, "H": -125},
        6.5,
        [pr.Mention("17:38", "20 puck bucks on <them>")],
    )
    html = report.render_html(card, ctx)
    assert "Pod read" in html and "A +105 · H -125 · total 6.5" in html
    assert "Transcript bet mentions" not in html and "20 puck bucks on" not in html
    assert "not an input to the model" in html

    ctx.podcast.games["A @ H"].summary = ["H ML -125 — 20 — home opener"]
    ctx.podcast.games["A @ H"].summary_source = "openai:gpt-4o-mini"
    html = report.render_html(card, ctx)
    assert "home opener" not in html and "Verbatim" not in html
    assert "summary openai:gpt-4o-mini" not in html

    after = [(r.model_prob, r.edge, r.ev, r.is_buy, tuple(r.gates)) for r in card.rows]
    assert before == after


def test_cli_podcast_no_episode_and_cached_transcript(tmp_path: Path, monkeypatch):
    from nhl_engine import cli
    from nhl_engine.data import capture

    monkeypatch.setenv("NHLE_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    eps = podcast.parse_feed((FIX / "podcast_feed.xml").read_text())
    with patch.object(podcast, "fetch_feed", return_value=eps):
        assert cli.main(["podcast", "--date", "2026-10-07"]) == 0
        assert pr.load_read(tmp_path, Date(2026, 10, 7)).status == "no_episode"

        # Episode exists but no transcript and transcription disabled -> no_transcript.
        assert cli.main(["podcast", "--date", "2026-10-06", "--no-transcribe"]) == 1
        assert pr.load_read(tmp_path, Date(2026, 10, 6)).status == "no_transcript"

        # Cached transcript + archived board -> read with anchors.
        ep = podcast.episodes_for(eps, Date(2026, 10, 6))[0]
        podcast.write_transcript(podcast.transcript_path(tmp_path, ep), _lines())
        quotes = [
            capture.QuoteRow(
                "t", "2026-10-06", f"{a} @ {h}", f"e{a}", "h2h", "home", h, None, "dk", -110, 100
            )
            for a, h in MATCHUPS
        ]
        with patch.object(capture, "read_day", return_value=quotes):
            assert cli.main(["podcast", "--date", "2026-10-06", "--no-transcribe"]) == 0
        read = pr.load_read(tmp_path, Date(2026, 10, 6))
        assert read.status == "ok" and set(read.games) == {
            "CAR @ MTL",
            "NSH @ TOR",
            "UTA @ NJD",
            "MIN @ BUF",
        }
        assert read.games["CAR @ MTL"].summary == []
