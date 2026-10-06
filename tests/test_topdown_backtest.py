"""Top-down backtest: Pinnacle fair price, first-trigger bets, CLV and grading."""

from __future__ import annotations

import gzip
import json
import math
from datetime import date as Date
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import topdown_backtest as td
from scripts.oos_odds_history import Budget

HOME, AWAY = "Los Angeles Dodgers", "Atlanta Braves"
COMMENCE = "2025-07-22T23:10:00Z"
DAY = Date(2025, 7, 22)


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _bk(key: str, h2h=None, rl=None, tot=None) -> dict:
    m = []
    if h2h:
        m.append(
            {
                "key": "h2h",
                "outcomes": [{"name": HOME, "price": h2h[0]}, {"name": AWAY, "price": h2h[1]}],
            }
        )
    if rl:
        pt, hp, ap = rl
        m.append(
            {
                "key": "spreads",
                "outcomes": [
                    {"name": HOME, "price": hp, "point": pt},
                    {"name": AWAY, "price": ap, "point": -pt},
                ],
            }
        )
    if tot:
        pt, o, u = tot
        m.append(
            {
                "key": "totals",
                "outcomes": [
                    {"name": "Over", "price": o, "point": pt},
                    {"name": "Under", "price": u, "point": pt},
                ],
            }
        )
    return {"key": key, "markets": m}


def _write(cache: Path, ts: str, books: list[dict], commence: str = COMMENCE) -> None:
    raw = {
        "timestamp": ts,
        "data": [
            {"home_team": HOME, "away_team": AWAY, "commence_time": commence, "bookmakers": books}
        ],
    }
    path = td.snap_path(cache, DAY, _utc(ts))
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        json.dump(raw, fh)


def _game(home_runs: int = 5, away_runs: int = 3) -> td.Game:
    return td.Game(1, DAY, "LAD", "ATL", _utc(COMMENCE), "Final", "Final", home_runs, away_runs)


def _paths(cache: Path) -> list[Path]:
    return sorted(cache.rglob("*.json.gz"))


def test_day_grid_runs_from_lead_to_last_first_pitch() -> None:
    grid = td.day_grid([_utc("2025-07-22T17:05:00Z"), _utc("2025-07-22T23:10:00Z")], 2.0, 10)
    assert grid[0] == _utc("2025-07-22T15:10:00Z")
    assert grid[-1] == _utc("2025-07-22T23:10:00Z")
    assert len(grid) == 49


def test_postponed_games_are_not_planned() -> None:
    ppd = td.Game(
        2, DAY, "NYY", "BOS", _utc("2025-07-22T12:00:00Z"), "Final", "Postponed", None, None
    )
    snaps = td.plan([_game(), ppd], 2.0, 10)
    assert snaps[DAY][0] == _utc("2025-07-22T21:10:00Z")
    assert not ppd.final


def test_a_soft_price_over_pinnacle_fair_is_bet_once_and_scored_at_the_close(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "2025-07-22T22:00:00Z",
        [
            _bk("pinnacle", h2h=(-110, -110)),
            _bk("draftkings", h2h=(105, -125)),
            _bk("betmgm", h2h=(102, -125)),
        ],
    )
    _write(
        tmp_path,
        "2025-07-22T22:10:00Z",
        [_bk("pinnacle", h2h=(-120, 100)), _bk("draftkings", h2h=(-110, -110))],
    )
    sc = td.scan(_paths(tmp_path), 0.005)
    df = td.bets(sc, [_game()], 0.01, 10)
    assert len(df) == 1
    bet = df.iloc[0]
    assert (bet.market, bet.side, bet.book, bet.price) == ("h2h", "home", "draftkings", 105)
    assert bet.ev_bet == pytest.approx(0.5 * 2.05 - 1)
    close = (120 / 220) / (120 / 220 + 0.5)
    assert bet.close_fair == pytest.approx(close)
    assert bet.clv_ev == pytest.approx(close * 2.05 - 1)
    assert bet.persist_min == 0
    assert (bet.result, bet.pnl) == ("win", pytest.approx(1.05))
    assert td.bets(sc, [_game()], 0.03, 10).empty


def test_a_gap_that_stays_open_is_measured(tmp_path: Path) -> None:
    for ts in ("2025-07-22T22:00:00Z", "2025-07-22T22:10:00Z", "2025-07-22T22:40:00Z"):
        _write(
            tmp_path, ts, [_bk("pinnacle", h2h=(-110, -110)), _bk("draftkings", h2h=(105, -125))]
        )
    df = td.bets(td.scan(_paths(tmp_path), 0.005), [_game()], 0.01, 10)
    assert df.iloc[0].persist_min == 10


def test_points_must_match_and_a_moved_total_has_no_close(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "2025-07-22T22:00:00Z",
        [
            _bk("pinnacle", tot=(8.5, -105, -105)),
            _bk("draftkings", tot=(8.5, 110, -130)),
            _bk("betmgm", tot=(8.0, 150, -180)),
        ],
    )
    _write(tmp_path, "2025-07-22T22:10:00Z", [_bk("pinnacle", tot=(9.0, -110, -110))])
    df = td.bets(td.scan(_paths(tmp_path), 0.005), [_game(5, 3)], 0.01, 10)
    assert list(zip(df.side, df.line, df.book, strict=True)) == [("over", 8.5, "draftkings")]
    assert math.isnan(df.iloc[0].clv_ev)
    assert df.iloc[0].result == "loss"


def test_started_games_are_not_priced() -> None:
    raw = {
        "timestamp": "2025-07-22T23:15:00Z",
        "data": [
            {
                "home_team": HOME,
                "away_team": AWAY,
                "commence_time": COMMENCE,
                "bookmakers": [_bk("pinnacle", h2h=(-110, -110))],
            }
        ],
    }
    assert td.parse_snapshot(raw)[1] == {}


@pytest.mark.parametrize(
    ("market", "side", "line", "hr", "ar", "want"),
    [
        ("h2h", "away", None, 5, 3, "loss"),
        ("spreads", "home", -1.5, 5, 3, "win"),
        ("spreads", "away", -1.5, 4, 3, "win"),
        ("spreads", "home", 1.5, 2, 3, "win"),
        ("totals", "over", 8.0, 5, 3, "push"),
        ("totals", "under", 8.5, 5, 3, "win"),
    ],
)
def test_grade(market: str, side: str, line: float | None, hr: int, ar: int, want: str) -> None:
    assert td.grade(market, side, line, hr, ar) == want


def test_stats_api_oakland_matches_the_odds_name() -> None:
    games = td.parse_schedule(
        [
            {
                "date": "2024-07-23",
                "games": [
                    {
                        "gamePk": 9,
                        "gameDate": "2024-07-23T23:07:00Z",
                        "status": {"abstractGameState": "Final", "detailedState": "Final"},
                        "teams": {
                            "home": {"team": {"abbreviation": "OAK"}, "score": 2},
                            "away": {"team": {"abbreviation": "AZ"}, "score": 1},
                        },
                    }
                ],
            }
        ]
    )
    assert (games[0].home, games[0].away) == ("ATH", "AZ")
    assert td.team_abbr("Oakland Athletics") == "ATH"


class _Resp:
    status_code = 200

    def __init__(self, at: str) -> None:
        self.headers = {"x-requests-last": "30", "x-requests-remaining": "4000000"}
        self._at = at

    def json(self) -> dict:
        return {"timestamp": self._at, "data": []}


class _Session:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get(self, url: str, params: dict, timeout: int) -> _Resp:
        self.calls.append(params)
        return _Resp(params["date"])


def test_pull_stops_at_the_cap_and_resumes_from_cache(tmp_path: Path) -> None:
    snaps = td.plan([_game()], 0.5, 10)  # 22:40 .. 23:10 -> 4 snapshots
    sess = _Session()
    assert td.pull(snaps, tmp_path, "k", Budget(60, 0), sess, td.DEFAULT_BOOKS) == 1
    assert len(sess.calls) == 2
    assert sess.calls[0]["bookmakers"].split(",")[0] == "pinnacle"
    assert "regions" not in sess.calls[0]
    assert td.pull(snaps, tmp_path, "k", Budget(1000, 0), sess, td.DEFAULT_BOOKS) == 0
    assert len(sess.calls) == 4


def test_pull_respects_the_floor_left_for_other_engines(tmp_path: Path) -> None:
    sess = _Session()
    budget = Budget(10_000, 3_999_990)
    assert td.pull(td.plan([_game()], 0.5, 10), tmp_path, "k", budget, sess, td.DEFAULT_BOOKS) == 1
    assert len(sess.calls) == 1


def test_more_than_ten_books_is_refused_before_any_call() -> None:
    books = ",".join([*td.DEFAULT_BOOKS, "lowvig"])
    assert td.main(["plan", "--books", books]) == 2
    assert td.main(["plan", "--books", "draftkings,betmgm"]) == 2


def test_snapshot_paths_keep_late_utc_games_on_their_slate_day() -> None:
    p = td.snap_path(Path("/c"), DAY, datetime(2025, 7, 23, 2, 10, tzinfo=timezone.utc))
    assert p == Path("/c/2025-07-22/20250723T0210Z.json.gz")
