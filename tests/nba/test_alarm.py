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


def test_pending_until_priced_after_the_latest_alert(tmp_path):
    a1 = alarm.Alert("2025-12-10T22:00:00Z", "2025-12-10", "ev1", "SAS @ LAL", "injury", "", "", "")
    a2 = alarm.Alert(
        "2025-12-10T23:00:00Z", "2025-12-10", "ev1", "SAS @ LAL", "ml_move", "", "", ""
    )
    alarm.write_alerts([a1], tmp_path, date(2025, 12, 10), a1.detected_at)
    alarm.write_alerts([a2], tmp_path, date(2025, 12, 10), a2.detected_at)
    held = alarm.read_alerts(tmp_path, date(2025, 12, 10))
    assert held == [a1, a2]
    assert alarm.pending(held, {}) == {"ev1"}
    assert alarm.pending(held, {"ev1": "2025-12-10T22:30:00Z"}) == {"ev1"}
    assert alarm.pending(held, {"ev1": "2025-12-10T23:05:00Z"}) == set()
