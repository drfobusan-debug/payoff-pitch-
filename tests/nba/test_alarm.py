from datetime import date

from nba_engine import alarm
from nba_engine.config import AlarmParams
from nba_engine.data.capture import QuoteRow
from nba_engine.data.espn_injuries import FeedRow
from nba_engine.data.injuries import InjuryRow
from nba_engine.schemas import Game

GAME = Game(date(2025, 12, 10), "LAL", "SAS", "2025-12-11T03:00:00Z", "ev1")
GAMES = {"ev1": GAME}
PARAMS = AlarmParams(ml_prob=0.03, spread_pts=1.0, total_pts=1.5)


def _q(
    market: str, side: str, line: float | None, am: float, opp: float | None, book: str = "dk"
) -> QuoteRow:
    return QuoteRow(
        "t", "2025-12-10", "SAS @ LAL", "ev1", market, side, "", line, book, am, opp, "oddsapi"
    )


def _board(ml: float, spread: float, total: float) -> list[QuoteRow]:
    return [
        _q("game_ml", "LAL", None, ml, -ml if ml > 0 else 100 - ml - 100),
        _q("game_ats", "LAL", spread, -110, -110),
        _q("game_ats", "SAS", -spread, -110, -110),
        _q("game_total", "over", total, -110, -110),
        _q("game_total", "under", total, -110, -110),
    ]


def test_consensus_uses_paired_quotes_only():
    rows = _board(-200, -6.5, 230.5) + [_q("game_ats", "LAL", -20, -110, None, book="x")]
    c = alarm.consensus(rows)["ev1"]
    assert c["spread"] == -6.5 and c["total"] == 230.5
    assert 0.6 < c["ml"] < 0.7


def test_board_alerts_fire_at_threshold_and_not_below():
    before = _board(-200, -6.5, 230.5)
    small = alarm.board_alerts(before, _board(-205, -7.0, 231.5), GAMES, PARAMS, "t1")
    assert small == []
    big = alarm.board_alerts(before, _board(-120, -3.0, 228.5), GAMES, PARAMS, "t1")
    assert {a.kind for a in big} == {"ml_move", "spread_move", "total_move"}
    assert all(a.event_id == "ev1" for a in big)


def test_feed_alerts_on_status_change_only_for_teams_playing():
    def row(team: str, status: str, comment: str = "") -> FeedRow:
        return FeedRow(team, "Player", "1", status, "", comment)

    games = [GAME]
    assert (
        alarm.feed_alerts(
            [row("LAL", "Questionable")], [row("LAL", "Questionable", "x")], games, "t"
        )
        == []
    )
    out = alarm.feed_alerts([row("LAL", "Questionable")], [row("LAL", "Out")], games, "t")
    assert len(out) == 1 and out[0].detail == "LAL Player: Questionable -> Out"
    assert alarm.feed_alerts([], [row("BOS", "Out")], games, "t") == []
    cleared = alarm.feed_alerts([row("SAS", "Out")], [], games, "t")
    assert cleared[0].after == ""


def test_official_report_alerts_map_full_team_names():
    def row(status: str) -> InjuryRow:
        return InjuryRow(
            "r",
            "12/10/2025",
            "10:00",
            "SAS@LAL",
            "San Antonio Spurs",
            "Wembanyama, Victor",
            status,
            "",
        )

    out = alarm.official_alerts([row("Questionable")], [row("Out")], [GAME], "t")
    assert [(a.kind, a.event_id) for a in out] == [("injury_official", "ev1")]


def _alert(at: str, kind: str = "ml_move") -> alarm.Alert:
    return alarm.Alert(at, "2025-12-10", "ev1", "SAS @ LAL", kind, "", "", "")


def test_alerts_round_trip_through_the_archive(tmp_path):
    a1, a2 = _alert("2025-12-10T22:00:00Z", "injury"), _alert("2025-12-10T23:00:00Z")
    alarm.write_alerts([a1], tmp_path, date(2025, 12, 10), a1.detected_at)
    alarm.write_alerts([a2], tmp_path, date(2025, 12, 10), a2.detected_at)
    assert alarm.read_alerts(tmp_path, date(2025, 12, 10)) == [a1, a2]


def test_pending_until_settled_and_then_priced_after_the_settle():
    news = [_alert("2025-12-10T22:00:00Z", "injury"), _alert("2025-12-10T23:00:00Z")]
    assert alarm.pending(news, {}) == {"ev1"}
    assert alarm.pending(news, {"ev1": "2025-12-10T23:05:00Z"}) == {"ev1"}  # never settled
    assert alarm.unsettled(news) == {"ev1"}
    calm = [*news, _alert("2025-12-10T23:04:00Z", alarm.SETTLED)]
    assert alarm.unsettled(calm) == set()
    assert alarm.pending(calm, {"ev1": "2025-12-10T23:02:00Z"}) == {"ev1"}  # priced pre-settle
    assert alarm.pending(calm, {"ev1": "2025-12-10T23:04:00Z"}) == {"ev1"}  # not strictly newer
    assert alarm.pending(calm, {"ev1": "2025-12-10T23:05:00Z"}) == set()
    again = [*calm, _alert("2025-12-10T23:10:00Z", "injury")]
    assert alarm.pending(again, {"ev1": "2025-12-10T23:05:00Z"}) == {"ev1"}
    assert alarm.unsettled(again) == {"ev1"}


def test_reference_lets_a_slow_creep_add_up():
    boards = [_board(-200, -6.5, 230.5), _board(-200, -6.5, 231.0), _board(-200, -6.5, 231.5)]
    for i, rows in enumerate(boards):
        boards[i] = [r.__class__(**{**r.__dict__, "captured_at": f"t{i}"}) for r in rows]
    now = alarm.consensus(_board(-200, -6.5, 232.0))
    tick = alarm.board_alerts(boards[-1], _board(-200, -6.5, 232.0), GAMES, PARAMS, "t3")
    assert tick == []  # each look moved only 0.5
    crept = alarm.move_alerts(alarm.reference(boards, []), now, GAMES, PARAMS, "t3")
    assert [a.kind for a in crept] == ["total_move"]
    anchored = alarm.reference(boards, [_alert("t1", "injury")])
    assert anchored["ev1"]["total"] == 231.0
    assert alarm.move_alerts(anchored, now, GAMES, PARAMS, "t3") == []


def _cons(spread: float, total: float, ml: float = 0.66) -> dict[str, float]:
    return {"ml": ml, "spread": spread, "total": total}


def _poller(path: list[dict[str, float]]):
    looks = iter(path)
    return lambda events: {"ev1": next(looks)}


def test_settle_waits_for_two_quiet_polls():
    start = {"ev1": _cons(-6.5, 230.5)}
    slept: list[float] = []
    stepping = [_cons(-3.0, 228.5), _cons(-3.5, 228.5), _cons(-3.5, 228.0), _cons(-4.0, 228.0)]
    assert alarm.settle({"ev1"}, start, _poller(stepping), PARAMS, slept.append) == {}
    assert slept == [PARAMS.settle_interval_s] * PARAMS.settle_polls
    steady = [_cons(-3.0, 228.5), _cons(-3.25, 228.5), _cons(-3.25, 228.5)]
    done = alarm.settle({"ev1"}, start, _poller(steady), PARAMS, lambda s: None)
    assert list(done) == ["ev1"] and len(done["ev1"]) == 4
    calm = alarm.settled_alerts(done, GAMES, "2025-12-10T23:04:00Z")
    assert [(a.kind, a.event_id) for a in calm] == [(alarm.SETTLED, "ev1")]


def test_settle_keeps_a_vanished_market_pending():
    vanished = {"ev1": {"ml": 0.66}}
    gone = alarm.settle(
        {"ev1"}, {"ev1": _cons(-6.5, 230.5)}, lambda ev: vanished, PARAMS, lambda s: None
    )
    assert gone == {}
