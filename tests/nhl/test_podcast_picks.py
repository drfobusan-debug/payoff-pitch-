"""Podcast picks: extraction from the game read, host hand-off attribution,
self-reported records, grading/CLV, PDF record lines, and model neutrality."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path
from unittest.mock import patch

from nhl_engine import report
from nhl_engine.audit import podcast_picks as pp
from nhl_engine.data import capture, podcast
from nhl_engine.features import podcast_read as pr
from nhl_engine.schemas import GameResult, PeriodScore
from tests.nhl.test_podcast import MATCHUPS, _episode, _lines
from tests.nhl.test_report import _card

SLATE = Date(2026, 10, 6)


def _read() -> pr.PodcastRead:
    return pr.build_read(_episode(), _lines(), SLATE, MATCHUPS)


def _result(away: str, home: str, a: int, h: int, decided: str = "REG") -> GameResult:
    ps = (PeriodScore(1, "REG", a, h), PeriodScore(2, "REG", 0, 0), PeriodScore(3, "REG", 0, 0))
    if decided != "REG":
        ps = ps + (PeriodScore(4, decided, 0, 1),)
    return GameResult(1, SLATE, away, home, "OFF", ps, decided)


def test_extract_picks_sides_prices_stakes_and_declines():
    picks = pp.extract_picks(_read())
    by = {(p.matchup, p.market, p.side): p for p in picks}
    habs = by[("CAR @ MTL", "game_ml", "MTL")]
    assert habs.american == 105 and habs.stamp == "18:14"  # price from the intro line
    under = by[("NSH @ TOR", "game_total", "under")]
    assert under.line == 5.5 and under.american == 140 and under.stake == 20.0
    assert ("MIN @ BUF", "game_ml", "BUF") not in by  # "can't trust Buffalo's goalie" is scenery
    assert all(not pp._DECLINE.search(p.text) for p in picks)


def test_total_line_rejects_stakes_and_implausible_lines():
    assert pp._total_line("give me the over 25 each") is None
    assert pp._total_line("over six and a half") == ("over", 6.5)
    assert pp._total_line("under 5.5 at plus 140") == ("under", 5.5)
    assert pp._total_line("over 2") == ("over", None)


def test_handoff_attribution_only():
    lines = [
        podcast.Line(600, "down to seven oclock the devils minus 135 at the mammoth plus 114"),
        podcast.Line(605, "chase what you got"),
        podcast.Line(610, "give me the devils 20 puck bucks"),
        podcast.Line(620, "I agree with archer on the mammoth, give me the mammoth"),
    ]
    read = pr.build_read(_episode(), lines, SLATE, [("NJD", "UTA")])
    picks = pp.extract_picks(read)
    assert [(p.side, p.host) for p in picks] == [
        ("NJD", "Chase Saul"),
        ("UTA", "Chase Saul"),  # a name on the pick line is not the speaker; hand-off rules
    ]
    read2 = pr.build_read(_episode(), lines[:1] + lines[2:3], SLATE, [("NJD", "UTA")])
    assert [p.host for p in pp.extract_picks(read2)] == [pp.UNATTRIBUTED]


def test_stated_records_parse_words_and_digits():
    recs = pp.stated_records(
        [
            ("05:09", "I was eight and nine down 25 puck bucks jol six and ten down 78"),
            ("52:19", "Joel 21 22 down with 39 archer 17 and 14 up 62"),
        ]
    )
    by = {r.host: r for r in recs}
    assert by["Joel Meyer"].wins == 21 and by["Joel Meyer"].units == -39.0  # latest wins
    assert by["Sean Marciano"].wins == 17 and by["Sean Marciano"].units == 62.0
    assert "Ryan Gilbert" not in by  # first-person record is not attributable


def test_grade_picks_pnl_clv_and_unpriced(tmp_path: Path):
    picks = pp.extract_picks(_read())
    results = {
        "CAR @ MTL": _result("CAR", "MTL", 2, 5),
        "NSH @ TOR": _result("NSH", "TOR", 2, 3, "OT"),
    }
    quotes = [
        capture.QuoteRow(
            "2026-10-06T22:00:00Z",
            "2026-10-06",
            "CAR @ MTL",
            "e",
            "game_ml",
            "MTL",
            "",
            None,
            b,
            -105,
            -115,
        )
        for b in ("dk", "fd")
    ]
    starts = {"CAR @ MTL": "2026-10-06T23:00:00Z", "NSH @ TOR": "2026-10-06T23:00:00Z"}
    pp.grade_picks(picks, results, quotes, graded_at="g", starts=starts)
    by = {(p.matchup, p.market, p.side): p for p in picks}
    habs = by[("CAR @ MTL", "game_ml", "MTL")]
    assert habs.outcome == "win" and habs.pnl_flat == 1.05 and habs.clv is not None and habs.clv > 0
    over = by[("CAR @ MTL", "game_total", "over")]
    assert (
        over.outcome == "win" and over.american is None and over.pnl_flat is None
    )  # no spoken price
    under = by[("NSH @ TOR", "game_total", "under")]
    assert under.outcome == "loss" and under.pnl_flat == -1.0 and under.pnl_stake == -20.0
    assert all(p.outcome is None for p in picks if p.matchup.startswith("UTA"))

    path = pp.save_picks(picks, pp.picks_path(tmp_path, SLATE))
    assert pp.load_picks(path) == picks and pp.load_all_picks(tmp_path) == picks
    text = pp.render(picks)
    assert "game_ml" in text and "no spoken price" in text
    assert pp.tally(picks, "market")["game_total"].record() == "1-1"


def test_pdf_shows_picks_and_records_without_touching_rows(tmp_path: Path):
    card, ctx = _card(tmp_path)
    before = [(r.model_prob, r.american, r.is_buy, tuple(r.gates)) for r in card.rows]
    card.games[0].matchup = "CAR @ MTL"
    ctx.podcast = _read()
    picks = pp.extract_picks(ctx.podcast)
    pp.grade_picks(picks, {"CAR @ MTL": _result("CAR", "MTL", 2, 5)}, [], graded_at="g")
    ctx.pod_picks = ctx.pod_history = picks
    ctx.pod_stated = [pp.StatedRecord("Joel Meyer", 21, 22, -39.0, "52:19")]
    html = report.render_html(card, ctx)
    assert "Picks logged" in html and "MTL ML +105" in html and "under 5.5 +140" not in html
    assert "self-reported 21-22 -39 pb" in html and "graded by us" in html
    assert "(no reason given)" in html and "pb self-reported)" not in html
    habs = next(p for p in picks if p.side == "MTL")
    assert habs.text not in html

    ctx.podcast.games["CAR @ MTL"].summary = [
        "MTL ML +105 — Joel, 20 puck bucks — home opener, Carolina on a back-to-back",
        "Over 6.5 — 20",
    ]
    html = report.render_html(card, ctx)
    assert "MTL ML +105</b> (home opener, Carolina on a back-to-back)" in html
    assert [(r.model_prob, r.american, r.is_buy, tuple(r.gates)) for r in card.rows] == before


def test_cli_podcast_logs_picks_and_audit_grades_them(tmp_path: Path, monkeypatch):
    from nhl_engine import cli

    monkeypatch.setenv("NHLE_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    eps = podcast.parse_feed((Path(__file__).parent / "fixtures" / "podcast_feed.xml").read_text())
    ep = podcast.episodes_for(eps, SLATE)[0]
    podcast.write_transcript(podcast.transcript_path(tmp_path, ep), _lines())
    quotes = [
        capture.QuoteRow(
            "t", "2026-10-06", f"{a} @ {h}", f"e{a}", "h2h", "home", h, None, "dk", -110, 100
        )
        for a, h in MATCHUPS
    ]
    with (
        patch.object(podcast, "fetch_feed", return_value=eps),
        patch.object(capture, "read_day", return_value=quotes),
    ):
        assert cli.main(["podcast", "--date", "2026-10-06", "--no-transcribe"]) == 0
    picks = pp.load_picks(pp.picks_path(tmp_path, SLATE))
    assert picks and all(p.outcome is None for p in picks)
    results = {"CAR @ MTL": _result("CAR", "MTL", 2, 5)}
    with (
        patch.object(cli, "_results_for", return_value=(results, {})),
        patch.object(capture, "read_day", return_value=[]),
    ):
        assert cli.main(["audit", "--date", "2026-10-06"]) == 0
    graded = pp.load_picks(pp.picks_path(tmp_path, SLATE))
    assert {p.outcome for p in graded if p.matchup == "CAR @ MTL"} == {"win"}
