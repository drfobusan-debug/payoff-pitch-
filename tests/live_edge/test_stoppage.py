from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from live_edge.espn import LiveGame
from live_edge.oddsapi import EventQuotes, Quote
from live_edge.roller import LiveState
from live_edge.stoppage import RULES, Study, fair, lead_adj, median_line, settle, stoppage

START = datetime(2026, 11, 5, 0, 0, tzinfo=timezone.utc)


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def game(
    status: str,
    home: int,
    away: int,
    period: int = 2,
    clock: int = 0,
    detail: str = "STATUS_HALFTIME",
    gid: str = "1",
) -> LiveGame:
    return LiveGame(
        event_id=gid,
        home="Boston Celtics",
        away="New York Knicks",
        status=status,
        state=LiveState(home, away, period=period, clock_secs=clock),
        kickoff=_iso(START),
        home_aliases=("bostonceltics",),
        away_aliases=("newyorkknicks",),
        detail=detail,
    )


def board(lines: dict[str, float], at: datetime, gid: str = "o1") -> list[EventQuotes]:
    qs = []
    for bk, line in lines.items():
        qs += [
            Quote(bk, "total", "over", line, -110, _iso(at)),
            Quote(bk, "total", "under", line, -115 if bk == "betmgm" else -110, _iso(at)),
        ]
    return [EventQuotes(gid, "Boston Celtics", "New York Knicks", _iso(START), tuple(qs))]


def _pregame(s: Study) -> None:
    t = START - timedelta(minutes=30)
    s.tick(
        [game("pre", 0, 0, 1, 720, "STATUS_SCHEDULED")],
        board({"draftkings": 220, "betmgm": 220, "fanduel": 220}, t),
        t,
    )


def test_the_locked_formula():
    surprise, f = fair(RULES["cbb"], 140.0, 75)
    assert round(surprise, 6) == round(75 - 0.47 * 140, 6)
    assert round(f, 6) == round(140 + 1.12 * surprise, 6)


def test_only_the_half_or_end_of_p2_is_primary():
    nba, nhl, cbb = RULES["nba"], RULES["nhl"], RULES["cbb"]
    assert stoppage(nba, game("in", 60, 60)) == ("half", True)
    assert stoppage(nba, game("in", 30, 30, 1, 0, "STATUS_END_PERIOD")) == ("end of Q1", False)
    assert stoppage(nba, game("in", 30, 30, 1, 300, "STATUS_IN_PROGRESS")) is None
    assert stoppage(nhl, game("in", 2, 1, 2, 0, "STATUS_END_PERIOD")) == ("end of P2", True)
    assert stoppage(nhl, game("in", 1, 1, 1, 0, "STATUS_END_PERIOD")) == ("end of P1", False)
    assert stoppage(cbb, game("in", 40, 35, 1)) == ("half", True)


def test_a_confirmed_gap_makes_one_ticket_at_the_better_exec_book_and_grades(tmp_path: Path):
    s = Study("nba", tmp_path)
    _pregame(s)
    lines = {"draftkings": 236, "betmgm": 236.5, "fanduel": 235.5}
    t1, t2 = START + timedelta(minutes=75), START + timedelta(minutes=77)
    # 62-58 at the half on a 220 total: fair 230, median live 236, gap +6 past the 5 screen.
    assert s.tick([game("in", 62, 58)], board(lines, t1), t1)["tickets"] == 0
    assert s.tick([game("in", 62, 58)], board(lines, t2), t2)["tickets"] == 1
    t3 = t2 + timedelta(minutes=2)
    assert s.tick([game("in", 62, 58)], board(lines, t3), t3)["tickets"] == 0  # one a game
    (t,) = s.read("tickets.jsonl")
    assert (t["side"], t["book"], t["line"], t["price"]) == ("under", "betmgm", 236.5, -115)
    assert t["fair"] == 230 and t["gap"] == 6 and t["naive_side"] == "under"
    s.tick([game("post", 115, 113, 4, 0, "STATUS_FINAL")], None, t3 + timedelta(hours=1))
    (g,) = s.read("grades.jsonl")
    assert g["final"] == 228 and g["result"] == "win" and g["units"] == round(100 / 115, 4)
    assert "1-0-0" in s.report() and "under its count (1 of 8" in s.report()


def test_a_lead_at_the_margin_is_a_pass(tmp_path: Path):
    s = Study("nba", tmp_path)
    _pregame(s)
    lines = {"draftkings": 236, "betmgm": 236, "fanduel": 236}
    for m in (75, 77):
        t = START + timedelta(minutes=m)
        s.tick([game("in", 70, 50)], board(lines, t), t)  # lead of 20
    assert s.read("tickets.jsonl") == []
    assert "passed on the margin: 1" in s.report()


def test_a_score_change_or_a_single_poll_does_not_count(tmp_path: Path):
    s = Study("nba", tmp_path)
    _pregame(s)
    lines = {"draftkings": 236, "betmgm": 236, "fanduel": 236}
    t1, t2 = START + timedelta(minutes=75), START + timedelta(minutes=77)
    s.tick([game("in", 62, 58)], board(lines, t1), t1)
    s.tick([game("in", 64, 58)], board({"draftkings": 238, "betmgm": 238, "fanduel": 238}, t2), t2)
    assert s.read("tickets.jsonl") == []


def test_no_exec_book_leaves_the_price_missing(tmp_path: Path):
    s = Study("nba", tmp_path)
    _pregame(s)
    for m in (75, 77):
        t = START + timedelta(minutes=m)
        s.tick([game("in", 62, 58)], board({"fanduel": 236, "caesars": 236, "espnbet": 236}, t), t)
    (t,) = s.read("tickets.jsonl")
    assert t["book"] is None and t["price"] is None and t["line"] == 236
    s.tick([game("post", 115, 113, 4, 0, "STATUS_FINAL")], None, START + timedelta(hours=3))
    assert "1 with no DK/BetMGM price" in s.report() and "ROI n/a on 0 priced" in s.report()


def test_settle_pushes_and_unpriced_wins():
    assert settle("under", 230.0, 230, -110) == ("push", 0.0)
    assert settle("over", 229.5, 230, None) == ("win", None)
    assert settle("over", 230.5, 230, 100) == ("loss", -1.0)


def test_a_median_needs_three_fresh_two_sided_books():
    row = {"line": 140.5, "over": -110, "under": -110, "age": 30}
    two = {"draftkings": row, "betmgm": row}
    assert median_line(two) == (None, 2)
    assert median_line({**two, "fanduel": {**row, "line": 141.5}}) == (140.5, 3)
    assert median_line({**two, "fanduel": {**row, "under": None}}) == (None, 2)
    assert median_line({**two, "fanduel": {**row, "age": 900}}) == (None, 2)


def test_two_books_at_the_half_make_no_ticket(tmp_path: Path):
    s = Study("nba", tmp_path)
    _pregame(s)
    for m in (75, 77):
        t = START + timedelta(minutes=m)
        s.tick([game("in", 62, 58)], board({"draftkings": 236, "betmgm": 236}, t), t)
    assert s.read("tickets.jsonl") == []
    snap = s.read("tape.jsonl")[-1]
    assert snap["live"] is None and snap["n_books"] == 2


def test_cbb_lead_column_is_graded_but_never_makes_the_ticket(tmp_path: Path):
    assert (lead_adj("cbb", 3), lead_adj("cbb", 6), lead_adj("cbb", 14)) == (0.4, -0.2, -0.7)
    assert lead_adj("nba", 3) is None
    s = Study("cbb", tmp_path)
    t0 = START - timedelta(minutes=30)
    pre = {"draftkings": 140, "betmgm": 140, "fanduel": 140}
    s.tick([game("pre", 0, 0, 1, 1200, "STATUS_SCHEDULED")], board(pre, t0), t0)
    # 40-32 at the half on 140: surprise 72 - 65.8 = 6.2, fair 146.944; lead 8 adds -0.2.
    lines = {"draftkings": 151, "betmgm": 151, "fanduel": 151}
    for m in (60, 62):
        t = START + timedelta(minutes=m)
        s.tick([game("in", 40, 32, 1)], board(lines, t), t)
    (t,) = s.read("tickets.jsonl")
    assert t["side"] == "under" and t["gap"] == 4.056
    assert t["fair_lead"] == 146.744 and t["gap_lead"] == 4.256 and t["lead_side"] == "under"
    s.tick([game("post", 75, 70, 2, 0, "STATUS_FINAL")], None, START + timedelta(hours=3))
    (g,) = s.read("grades.jsonl")
    assert g["result"] == g["lead_result"] == "win"
    assert "lead-adjusted" in s.report() and "lead-adjusted" not in Study("nba", tmp_path).report()


def test_cbb_lead_side_is_blank_when_the_adjustment_lands_inside_the_screen(tmp_path: Path):
    s = Study("cbb", tmp_path)
    t0 = START - timedelta(minutes=30)
    pre = {"draftkings": 140, "betmgm": 140, "fanduel": 140}
    s.tick([game("pre", 0, 0, 1, 1200, "STATUS_SCHEDULED")], board(pre, t0), t0)
    # Lead 2: fair 146.944 + 0.4; a live 151 is 4.056 past the frozen fair, 3.656 past the lead one.
    lines = {"draftkings": 151, "betmgm": 151, "fanduel": 151}
    for m in (60, 62):
        t = START + timedelta(minutes=m)
        s.tick([game("in", 37, 35, 1)], board(lines, t), t)
    (t,) = s.read("tickets.jsonl")
    assert t["side"] == "under" and t["lead_side"] is None and t["lead_line"] is None
    s.tick([game("post", 75, 70, 2, 0, "STATUS_FINAL")], None, START + timedelta(hours=3))
    (g,) = s.read("grades.jsonl")
    assert g["lead_result"] is None and g["result"] == "win"
