"""Pull genuine historical NBA quotes from The Odds API, resumably and within budget.

For every slate date in a season the pull lists the events posted at 10:00 ET
(1 credit, cached so a rerun never pays twice), then fetches each event's
markets at a set of *anchors* -- minutes before the scheduled tip:

    close  T-5    the closing line the ledger grades CLV against (plan §2)
    t30    T-30   \
    t90    T-90    > the send-time study (plan §10) and the line-movement filter
    t3h    T-180  /
    t6h    T-360   the morning read
    t12h   T-720   the earliest snapshot most props are posted for

Each response is stored verbatim, gzipped, at
``<data>/history/raw/<YYYY-MM-DD>/<event_id>_<anchor>.json.gz`` -- one file per
(event, anchor), immutable, so a rerun skips what it already holds and a merge
between machines is a set union. The API serves the last snapshot *at or before*
the requested time, so a close is never a post-tip price; ``history_rows`` still
drops any snapshot stamped at or after the tip.

The key is shared with every engine's live capture: the pull stops before the
balance falls under ``min_credits`` and at its own ``budget``.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime, time, timedelta
from pathlib import Path

from nba_engine.data.capture import ALL_MARKETS, QuoteRow
from nba_engine.data.oddsapi import SLATE_TZ, OddsAPIClient, event_rows, games_from, parse_utc
from nba_engine.schemas import Game

log = logging.getLogger(__name__)

ANCHORS: dict[str, int] = {
    "close": 5,
    "t30": 30,
    "t90": 90,
    "t3h": 180,
    "t6h": 360,
    "t12h": 720,
}
LIST_AT = time(10, 0)  # ET; every NBA tip is later than this
CREDITS_PER_MARKET = 10
NOT_POSTED = frozenset({404, 422})


def history_dir(data_dir: Path) -> Path:
    return data_dir / "history"


def raw_path(data_dir: Path, day: Date, event_id: str, anchor: str) -> Path:
    return history_dir(data_dir) / "raw" / day.isoformat() / f"{event_id}_{anchor}.json.gz"


def events_path(data_dir: Path, day: Date) -> Path:
    return history_dir(data_dir) / "events" / f"{day.isoformat()}.json.gz"


def season_days(season: str) -> list[Date]:
    """Every calendar day from 1 October to 30 June of a season named ``2025-26``."""
    start_year = int(season.split("-")[0])
    day, end = Date(start_year, 10, 1), Date(start_year + 1, 6, 30)
    out: list[Date] = []
    while day <= end:
        out.append(day)
        day += timedelta(days=1)
    return out


def _write_gz(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(payload, separators=(",", ":")).encode()))
    tmp.replace(path)


def _read_gz(path: Path) -> object:
    try:
        return json.loads(gzip.decompress(path.read_bytes()))
    except (OSError, ValueError):
        return None


@dataclass
class PullReport:
    days: int = 0
    events: int = 0
    fetched: int = 0
    skipped: int = 0
    missing: int = 0
    credits: int = 0
    remaining: int | None = None
    stopped: str = ""
    per_anchor: dict[str, int] = field(default_factory=dict)


class Budget:
    """Stop before the run's own budget or the shared floor is crossed."""

    def __init__(self, budget: int, min_credits: int) -> None:
        self.budget = budget
        self.min_credits = min_credits
        self.spent = 0

    def allows(self, cost: int, remaining: int | None) -> str:
        if self.spent + cost > self.budget:
            return "budget"
        if remaining is not None and remaining - cost < self.min_credits:
            return "floor"
        return ""

    def charge(self, cost: int | None) -> None:
        self.spent += cost or 0


def day_games(
    client: OddsAPIClient, data_dir: Path, day: Date, budget: Budget, report: PullReport
) -> list[Game] | None:
    """The day's games from the cached listing, fetching it once if absent."""
    path = events_path(data_dir, day)
    payload = _read_gz(path) if path.exists() else None
    if payload is None:
        if stop := budget.allows(1, client.credits_remaining):
            report.stopped = stop
            return None
        at = datetime.combine(day, LIST_AT, tzinfo=SLATE_TZ)
        snap = client.historical_events(at)
        budget.charge(client.credits_last)
        report.credits += client.credits_last or 0
        if snap is None or not isinstance(snap.data, list):
            return []
        payload = {"timestamp": snap.timestamp, "data": snap.data}
        _write_gz(path, payload)
    data = payload.get("data") if isinstance(payload, dict) else None
    return games_from(data, day) if isinstance(data, list) else []


def pull(
    client: OddsAPIClient,
    data_dir: Path,
    days: list[Date],
    *,
    anchors: tuple[str, ...] = ("close",),
    markets: tuple[str, ...] = ALL_MARKETS,
    budget: int,
    min_credits: int,
) -> PullReport:
    """Fetch every (event, anchor) not already archived, newest day first."""
    report = PullReport()
    guard = Budget(budget, min_credits)
    est = CREDITS_PER_MARKET * len(markets)
    now = datetime.now(SLATE_TZ)
    for day in sorted(days, reverse=True):
        if datetime.combine(day, LIST_AT, tzinfo=SLATE_TZ) > now:
            continue
        games = day_games(client, data_dir, day, guard, report)
        if games is None:
            break
        report.days += 1
        report.events += len(games)
        for game in games:
            tip = parse_utc(game.start_utc)
            if tip is None:
                continue
            for anchor in anchors:
                path = raw_path(data_dir, day, game.event_id, anchor)
                if path.exists():
                    report.skipped += 1
                    continue
                if stop := guard.allows(est, client.credits_remaining):
                    report.stopped = stop
                    report.remaining = client.credits_remaining
                    report.credits = guard.spent
                    return report
                at = tip - timedelta(minutes=ANCHORS[anchor])
                snap = client.historical_event_odds(game.event_id, at, markets)
                guard.charge(client.credits_last)
                if snap is None or not isinstance(snap.data, dict):
                    report.missing += 1
                    if client.last_status in NOT_POSTED:
                        # The event was not posted yet at that time; record it so a rerun
                        # does not ask again. Transport failures are retried next run.
                        _write_gz(
                            path, {"timestamp": "", "status": client.last_status, "data": None}
                        )
                    continue
                _write_gz(path, {"timestamp": snap.timestamp, "anchor": anchor, "data": snap.data})
                report.fetched += 1
                report.per_anchor[anchor] = report.per_anchor.get(anchor, 0) + 1
    report.remaining = client.credits_remaining
    report.credits = guard.spent
    return report


def iter_raw(data_dir: Path, day: Date, anchor: str | None = None) -> Iterator[Path]:
    directory = history_dir(data_dir) / "raw" / day.isoformat()
    if not directory.is_dir():
        return iter(())
    pattern = f"*_{anchor}.json.gz" if anchor else "*.json.gz"
    return iter(sorted(directory.glob(pattern)))


def history_rows(data_dir: Path, day: Date, anchor: str = "close") -> list[QuoteRow]:
    """Archived rows for a day's events at one anchor, stamped with the snapshot time.

    A snapshot taken at or after the event's tip is dropped: it is not a close.
    """
    rows: list[QuoteRow] = []
    for path in iter_raw(data_dir, day, anchor):
        payload = _read_gz(path)
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            continue
        event = payload["data"]
        games = games_from([event], day)
        if not games:
            continue
        game = games[0]
        taken = str(payload.get("timestamp", ""))
        tip, snap = parse_utc(game.start_utc), parse_utc(taken)
        if tip is None or snap is None or snap >= tip:
            continue
        rows.extend(event_rows(event, game, taken))
    return rows


def coverage(data_dir: Path, days: list[Date]) -> dict[str, object]:
    """Events listed vs (event, anchor) files held, per anchor, over ``days``."""
    listed = 0
    held: dict[str, int] = {a: 0 for a in ANCHORS}
    empty: dict[str, int] = {a: 0 for a in ANCHORS}
    for day in days:
        path = events_path(data_dir, day)
        payload = _read_gz(path) if path.exists() else None
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            listed += len(games_from(payload["data"], day))
        for raw in iter_raw(data_dir, day):
            anchor = raw.name.rsplit("_", 1)[-1].removesuffix(".json.gz")
            if anchor not in held:
                continue
            body = _read_gz(raw)
            if isinstance(body, dict) and body.get("data"):
                held[anchor] += 1
            else:
                empty[anchor] += 1
    return {"events_listed": listed, "held": held, "not_posted": empty}


__all__ = [
    "ANCHORS",
    "Budget",
    "PullReport",
    "coverage",
    "day_games",
    "events_path",
    "history_dir",
    "history_rows",
    "iter_raw",
    "pull",
    "raw_path",
    "season_days",
]
