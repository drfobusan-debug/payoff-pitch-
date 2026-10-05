import gzip
import json
from datetime import date, datetime, timedelta
from pathlib import Path

from nba_engine.data import history
from nba_engine.data.oddsapi import Snapshot

FIX = Path(__file__).parent / "fixtures"
DAY = date(2025, 12, 10)


def _event() -> dict:
    return json.loads((FIX / "oddsapi_hist_event_20251210.json").read_text())["data"]


class FakeClient:
    """Serves the fixture event; charges like the API (1 to list, 10 per market)."""

    def __init__(self, remaining: int = 1_000_000, fail: set[str] | None = None) -> None:
        self.credits_remaining: int | None = remaining
        self.credits_last: int | None = None
        self.last_status: int | None = None
        self.fail = fail or set()
        self.calls: list[tuple[str, datetime]] = []

    def _charge(self, cost: int) -> None:
        self.credits_last = cost
        assert self.credits_remaining is not None
        self.credits_remaining -= cost

    def historical_events(self, at: datetime) -> Snapshot | None:
        self._charge(1)
        event = _event()
        listing = {k: event[k] for k in ("id", "commence_time", "home_team", "away_team")}
        return Snapshot(timestamp=at.isoformat(), data=[listing])

    def historical_event_odds(self, event_id: str, at: datetime, markets: tuple[str, ...]):
        self.calls.append((event_id, at))
        if "404" in self.fail:
            self.credits_last, self.last_status = 0, 404
            return None
        if "net" in self.fail:
            self.credits_last, self.last_status = None, None
            return None
        self._charge(10 * len(markets))
        self.last_status = 200
        return Snapshot(timestamp="2025-12-11T02:50:38Z", data=_event())


def _pull(client, tmp_path, **kw):
    kw.setdefault("budget", 10_000)
    kw.setdefault("min_credits", 0)
    return history.pull(client, tmp_path, [DAY], **kw)  # type: ignore[arg-type]


def test_pull_archives_once_and_a_rerun_spends_nothing(tmp_path):
    client = FakeClient()
    first = _pull(client, tmp_path, anchors=("close", "t90"))
    assert (first.fetched, first.credits) == (2, 1 + 2 * 110)
    tip = datetime.fromisoformat("2025-12-11T03:00:00+00:00")
    assert sorted(at for _, at in client.calls) == [
        tip - timedelta(minutes=90),
        tip - timedelta(minutes=5),
    ]
    again = _pull(client, tmp_path, anchors=("close", "t90"))
    assert (again.fetched, again.skipped, again.credits) == (0, 2, 0)


def test_pull_stops_at_budget_and_at_the_shared_floor(tmp_path):
    assert _pull(FakeClient(), tmp_path, budget=50).stopped == "budget"
    floor = _pull(FakeClient(remaining=1_000), tmp_path / "b", min_credits=950)
    assert floor.stopped == "floor" and floor.fetched == 0


def test_not_posted_is_recorded_but_a_transport_failure_is_retried(tmp_path):
    _pull(FakeClient(fail={"404"}), tmp_path)
    eid = _event()["id"]
    assert history.raw_path(tmp_path, DAY, eid, "close").exists()
    net = tmp_path / "net"
    _pull(FakeClient(fail={"net"}), net)
    assert not history.raw_path(net, DAY, eid, "close").exists()
    assert _pull(FakeClient(), net).fetched == 1


def test_history_rows_refuse_a_snapshot_at_or_after_tip(tmp_path):
    _pull(FakeClient(), tmp_path)
    assert history_rows_count(tmp_path) > 0
    path = history.raw_path(tmp_path, DAY, _event()["id"], "close")
    body = json.loads(gzip.decompress(path.read_bytes()))
    body["timestamp"] = "2025-12-11T03:00:00Z"
    path.write_bytes(gzip.compress(json.dumps(body).encode()))
    assert history_rows_count(tmp_path) == 0


def history_rows_count(root: Path) -> int:
    return len(history.history_rows(root, DAY))


def test_coverage_counts_held_and_not_posted(tmp_path):
    _pull(FakeClient(), tmp_path)
    _pull(FakeClient(fail={"404"}), tmp_path, anchors=("t3h",))
    cov = history.coverage(tmp_path, [DAY])
    assert cov["events_listed"] == 1
    held, not_posted = cov["held"], cov["not_posted"]
    assert isinstance(held, dict) and isinstance(not_posted, dict)
    assert held["close"] == 1 and not_posted["t3h"] == 1


def test_season_days_runs_october_to_june():
    days = history.season_days("2024-25")
    assert days[0] == date(2024, 10, 1) and days[-1] == date(2025, 6, 30)
