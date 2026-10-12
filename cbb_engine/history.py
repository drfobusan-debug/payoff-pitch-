"""§4.e / decision 10.1: buy 2024-25 game-line closes from the Odds API historical endpoint.

One bulk ``/historical/.../odds`` snapshot per distinct tip cluster at T-5 (floored to the
10-minute snapshot grid) covers every game tipping in that window; each snapshot is written
as its own immutable ``close`` shard under the slate date, so `fit_width.py` and the audit
read it exactly like a live close pass. ``--dense`` adds the 10-minute featured grid from
T-3h on named slates (stale-line frequency / persistence / per-bet CLV SD for the MDEs).

Nothing here spends until the plan has been printed and ``--yes`` given; a rerun skips
snapshots already in the day's manifest, so an interrupted pull resumes.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Protocol

from cbb_engine.config import SPORT
from cbb_engine.data import espn
from cbb_engine.data.capture import rows_from_events, slate_date_of
from cbb_engine.market import consensus
from engine_runtime import quotes
from engine_runtime.odds_client import HIST_MULT

HIST_PREFIX = "hist"
CLOSE_MARKETS: tuple[str, ...] = ("spreads", "totals")
GRID_MIN = 10
CLOSE_BEFORE_MIN = 5
DENSE_WINDOW_H = 3.0
CSV_FIELDS = (
    "game_date", "event_id", "home", "away", "tip_utc", "close_total", "final_total",
    "paired_books", "spread_abs", "close_shard",
)


def _parse(stamp: str) -> datetime | None:
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _floor(moment: datetime, step: int = GRID_MIN) -> datetime:
    return moment.replace(second=0, microsecond=0, minute=(moment.minute // step) * step)


def close_times(tips: Iterable[str]) -> list[str]:
    """One snapshot per distinct T-5 grid cell; the snapshot at or before that time is the close."""
    out: set[datetime] = set()
    for t in tips:
        dt = _parse(t)
        if dt is not None:
            out.add(_floor(dt - timedelta(minutes=CLOSE_BEFORE_MIN)))
    return [_iso(m) for m in sorted(out)]


def dense_times(tips: Iterable[str], *, window_h: float = DENSE_WINDOW_H) -> list[str]:
    """The 10-minute featured grid from T-window to T-5, unioned across the slate's tips."""
    out: set[datetime] = set()
    for t in tips:
        dt = _parse(t)
        if dt is None:
            continue
        m = _floor(dt - timedelta(hours=window_h))
        last = _floor(dt - timedelta(minutes=CLOSE_BEFORE_MIN))
        while m <= last:
            out.add(m)
            m += timedelta(minutes=GRID_MIN)
    return [_iso(m) for m in sorted(out)]


def plan(
    tips_by_day: dict[str, list[str]],
    *,
    dense_days: Sequence[str] = (),
    markets: tuple[str, ...] = CLOSE_MARKETS,
    regions: int = 1,
) -> dict[str, object]:
    """Snapshots and credits the pull will cost, per day and in total, before any spend."""
    per_call = HIST_MULT * len(markets) * regions
    days: dict[str, dict[str, int]] = {}
    for day, tips in sorted(tips_by_day.items()):
        n_close = len(close_times(tips))
        n_dense = len(dense_times(tips)) if day in dense_days else 0
        days[day] = {
            "games": len(tips), "close_snapshots": n_close, "dense_snapshots": n_dense,
            "credits": per_call * (n_close + n_dense),
        }
    return {
        "markets": list(markets), "credits_per_snapshot": per_call, "days": days,
        "snapshots": sum(d["close_snapshots"] + d["dense_snapshots"] for d in days.values()),
        "credits": sum(d["credits"] for d in days.values()),
    }


def _pass_id(kind: str, at: str) -> str:
    return f"{HIST_PREFIX}_{kind}_{at.replace('-', '').replace(':', '')}"


def _in_window(ev: dict[str, object], at: datetime, horizon: timedelta) -> bool:
    tip = _parse(str(ev.get("commence_time", "")))
    return tip is not None and at < tip <= at + horizon


class HistoricalSource(Protocol):
    def historical_board(
        self, markets: tuple[str, ...], *, at: str, pass_id: str = "", **params: str
    ) -> tuple[str, list[dict[str, object]]]: ...


def pull_day(
    client: HistoricalSource,
    root: Path,
    day: str,
    tips: Iterable[str],
    *,
    dense: bool = False,
    markets: tuple[str, ...] = CLOSE_MARKETS,
) -> dict[str, int]:
    """Buy the day's close (and optionally dense) snapshots; skip any already in the manifest."""
    tips = list(tips)
    done = {str(m.get("pass_id", "")) for m in quotes.read_manifest(root, SPORT, day)}
    counts = {"snapshots": 0, "skipped": 0, "rows": 0, "events": 0}
    jobs = [("close", at, timedelta(minutes=GRID_MIN + CLOSE_BEFORE_MIN)) for at in close_times(tips)]
    if dense:
        jobs += [("featured", at, timedelta(hours=DENSE_WINDOW_H)) for at in dense_times(tips)]
    for kind, at, horizon in jobs:
        pid = _pass_id(kind, at)
        if pid in done:
            counts["skipped"] += 1
            continue
        at_dt = _parse(at)
        assert at_dt is not None
        stamp, evs = client.historical_board(markets, at=at, pass_id=pid)
        if not stamp:
            continue
        evs = [e for e in evs if _in_window(e, at_dt, horizon) and slate_date_of(str(e.get("commence_time", ""))) == day]
        rows = rows_from_events(evs, captured_at=stamp, pass_id=pid, pass_kind=kind)
        teams = {str(e.get("id", "")): [str(e.get("home_team", "")), str(e.get("away_team", ""))] for e in evs}
        written = quotes.write_shard(
            root, SPORT, day, rows, pass_id=pid, pass_kind=kind,
            extra={"hist": True, "requested_at": at, "teams": teams},
        )
        counts["snapshots"] += 1
        if written is not None:
            counts["rows"] += len(rows)
            counts["events"] += len(evs)
    return counts


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def latest_scoreboard(data_dir: Path, day: str) -> list[espn.Game]:
    files = sorted((data_dir / "espn" / day).glob("scoreboard_*.json"))
    if not files:
        return []
    payload = json.loads(files[-1].read_text())
    return espn.parse_scoreboard(payload) if isinstance(payload, dict) else []


def export_day(root: Path, data_dir: Path, day: str) -> list[dict[str, object]]:
    """Join the day's hist close shards to ESPN finals: one row per game for `fit_width.py`.

    The close is the last hist close shard captured before the game's tip; ``close_total`` and
    ``spread_abs`` are consensus lines (execution books excluded), ``paired_books`` the total's.
    """
    man = quotes.read_manifest(root, SPORT, day)
    teams: dict[str, list[str]] = {}
    for m in man:
        tm = m.get("teams")
        if m.get("hist") and m.get("pass_kind") == "close" and isinstance(tm, dict):
            teams.update({str(k): [str(x) for x in v] for k, v in tm.items()})
    shards = [q for q in quotes.read_day(root, SPORT, day) if q.pass_kind == "close" and q.pass_id.startswith(HIST_PREFIX)]
    by_event: dict[str, list[quotes.Quote]] = {}
    for q in shards:
        by_event.setdefault(q.event_id, []).append(q)
    finals = {(_norm(g.home), _norm(g.away)): g.final for g in latest_scoreboard(data_dir, day) if g.final}
    out: list[dict[str, object]] = []
    for eid, qs in sorted(by_event.items()):
        tip = _parse(qs[0].tip_utc)
        pre = [q for q in qs if (c := _parse(q.captured_at)) is not None and tip is not None and c < tip]
        if not pre:
            continue
        last = max(q.pass_id for q in pre)
        anchors = consensus.build([q for q in pre if q.pass_id == last], shard=last)
        total = [a for a in anchors if a.market == "game_total" and a.paired_books]
        if not total:
            continue
        close_total = consensus.consensus_line(anchors, eid, "game_total")
        ats = [abs(a.line) for a in anchors if a.market == "game_ats" and a.line is not None and a.paired_books]
        spread = median(ats) if ats else None
        home, away = teams.get(eid, ["", ""])
        final = finals.get((_norm(home), _norm(away)))
        out.append({
            "game_date": day, "event_id": eid, "home": home, "away": away, "tip_utc": qs[0].tip_utc,
            "close_total": close_total,
            "final_total": sum(final) if final else "",
            "paired_books": max(a.paired_books for a in total),
            "spread_abs": abs(spread) if spread is not None else "",
            "close_shard": last,
        })
    return out


def write_csv(rows: Iterable[dict[str, object]], path: Path) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def days_between(start: Date, end: Date) -> list[str]:
    n = (end - start).days
    return [(start + timedelta(days=i)).isoformat() for i in range(max(0, n) + 1)]


def tips_from_games(games: Iterable[espn.Game], day: str) -> list[str]:
    return [g.start_utc for g in games if slate_date_of(g.start_utc) == day]


__all__ = [
    "CLOSE_MARKETS", "CSV_FIELDS", "HIST_PREFIX", "close_times", "days_between", "dense_times",
    "export_day", "latest_scoreboard", "plan", "pull_day", "tips_from_games", "write_csv",
]
