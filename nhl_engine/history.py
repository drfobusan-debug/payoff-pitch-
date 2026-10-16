"""Historical closes for the preregistered close test: `nhl-engine history plan|pull|export`.

One bulk ``/historical/.../odds`` snapshot per distinct puck-drop cluster at T-5 (floored to
the 10-minute grid) covers every game starting in that window. Each snapshot is written
into the day's price archive as a ``hist_*`` snapshot (the same CSV the live ``board``
captures use), so the card pipeline can price a past night exactly like a live one.

Nothing here spends until the plan has been printed and ``--yes`` given; a rerun skips
snapshots already in the day's ``hist_manifest.json``, so an interrupted pull resumes.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

from engine_runtime.odds_client import HIST_MULT
from nhl_engine.data import capture
from nhl_engine.data.capture import QuoteRow
from nhl_engine.data.oddsapi import _to_game, event_rows
from nhl_engine.market import board
from nhl_engine.schemas import Game, GameResult

HIST_LABEL = "hist"
CLOSE_MARKETS: tuple[str, ...] = ("h2h", "totals", "spreads")
GRID_MIN = 10
CLOSE_BEFORE_MIN = 5
MANIFEST = "hist_manifest.json"
CSV_FIELDS = (
    "game_date",
    "event_id",
    "away",
    "home",
    "start_utc",
    "close_at",
    "close_home_ml",
    "close_total",
    "close_over",
    "close_pl_home",
    "final_away",
    "final_home",
    "decided",
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


def close_times(starts: Iterable[str]) -> list[str]:
    """One snapshot per distinct T-5 grid cell; the snapshot at or before it is the close."""
    out: set[datetime] = set()
    for t in starts:
        dt = _parse(t)
        if dt is not None:
            out.add(_floor(dt - timedelta(minutes=CLOSE_BEFORE_MIN)))
    return [_iso(m) for m in sorted(out)]


def plan(
    starts_by_day: dict[str, list[str]],
    *,
    markets: tuple[str, ...] = CLOSE_MARKETS,
    regions: int = 1,
) -> dict[str, object]:
    """Snapshots and credits the pull will cost, per day and in total, before any spend."""
    per_call = HIST_MULT * len(markets) * regions
    days: dict[str, dict[str, int]] = {}
    for day, starts in sorted(starts_by_day.items()):
        n = len(close_times(starts))
        days[day] = {"games": len(starts), "snapshots": n, "credits": per_call * n}
    return {
        "markets": list(markets),
        "credits_per_snapshot": per_call,
        "days": days,
        "games": sum(d["games"] for d in days.values()),
        "snapshots": sum(d["snapshots"] for d in days.values()),
        "credits": sum(d["credits"] for d in days.values()),
    }


def _pass_id(at: str) -> str:
    return f"{HIST_LABEL}_{at.replace('-', '').replace(':', '')}"


def _manifest_path(data_dir: Path, day: Date) -> Path:
    return capture.day_dir(data_dir, day) / MANIFEST


def read_manifest(data_dir: Path, day: Date) -> list[dict[str, object]]:
    p = _manifest_path(data_dir, day)
    if not p.is_file():
        return []
    raw = json.loads(p.read_text())
    return [m for m in raw if isinstance(m, dict)] if isinstance(raw, list) else []


def _append_manifest(data_dir: Path, day: Date, entry: dict[str, object]) -> None:
    p = _manifest_path(data_dir, day)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([*read_manifest(data_dir, day), entry], indent=1))


class HistoricalSource(Protocol):
    def historical_board(
        self, markets: tuple[str, ...], *, at: str, pass_id: str = "", **params: str
    ) -> tuple[str, list[dict[str, object]]]: ...


def pull_day(
    client: HistoricalSource,
    data_dir: Path,
    day: Date,
    starts: Iterable[str],
    *,
    markets: tuple[str, ...] = CLOSE_MARKETS,
) -> dict[str, int]:
    """Buy the day's close snapshots; skip any already in the manifest."""
    done = {str(m.get("pass_id", "")) for m in read_manifest(data_dir, day)}
    counts = {"snapshots": 0, "skipped": 0, "rows": 0, "games": 0}
    horizon = timedelta(minutes=GRID_MIN + CLOSE_BEFORE_MIN)
    for at in close_times(starts):
        pid = _pass_id(at)
        if pid in done:
            counts["skipped"] += 1
            continue
        at_dt = _parse(at)
        assert at_dt is not None
        stamp, evs = client.historical_board(markets, at=at, pass_id=pid)
        if not stamp:
            continue
        rows: list[QuoteRow] = []
        games: list[Game] = []
        for raw in evs:
            start = _parse(str(raw.get("commence_time", "")))
            if start is None or not (at_dt < start <= at_dt + horizon):
                continue
            game = _to_game(raw, day)
            if game is None or game.game_date != day:
                continue
            games.append(game)
            rows.extend(event_rows(raw, game, stamp))
        capture.write_snapshot(rows, data_dir, day, label=HIST_LABEL, captured_at=stamp)
        _append_manifest(
            data_dir,
            day,
            {
                "pass_id": pid,
                "requested_at": at,
                "captured_at": stamp,
                "games": [g.matchup for g in games],
                "rows": len(rows),
            },
        )
        counts["snapshots"] += 1
        counts["rows"] += len(rows)
        counts["games"] += len(games)
    return counts


def hist_rows(data_dir: Path, day: Date) -> list[QuoteRow]:
    out: list[QuoteRow] = []
    for p in capture.snapshot_paths(data_dir, day, label=HIST_LABEL):
        out.extend(capture.read_snapshot(p))
    return out


def _prob(
    sels: list[board.Selection], market: str, side: str, line: float | None = None
) -> float | None:
    hits = [
        s
        for s in sels
        if s.market == market and s.side == side and (line is None or s.line == line)
    ]
    return hits[0].consensus if hits else None


def export_day(
    data_dir: Path,
    day: Date,
    starts: dict[str, str],
    results: dict[str, GameResult],
) -> list[dict[str, object]]:
    """One row per game: no-vig close (home ML, main total + Over, home puck line) and the final.

    The close is the last hist snapshot captured before the game's start; ``starts`` and
    ``results`` are keyed by matchup.
    """
    rows = hist_rows(data_dir, day)
    out: list[dict[str, object]] = []
    for matchup in sorted({r.matchup for r in rows}):
        start = _parse(starts.get(matchup, ""))
        if start is None:
            continue
        pre = [
            r
            for r in rows
            if r.matchup == matchup and (c := _parse(r.captured_at)) is not None and c < start
        ]
        if not pre:
            continue
        close_at = max(r.captured_at for r in pre)
        sels = board.selections([r for r in pre if r.captured_at == close_at])
        totals = [s for s in sels if s.market == "game_total" and s.side == "over"]
        main = min(totals, key=lambda s: abs(s.consensus - 0.5)) if totals else None
        res = results.get(matchup)
        away, home = matchup.split(" @ ")
        out.append(
            {
                "game_date": day.isoformat(),
                "event_id": pre[0].event_id,
                "away": away,
                "home": home,
                "start_utc": starts[matchup],
                "close_at": close_at,
                "close_home_ml": _prob(sels, "game_ml", home),
                "close_total": main.line if main else "",
                "close_over": main.consensus if main else "",
                "close_pl_home": _prob(sels, "game_pl", home, -1.5),
                "final_away": res.final_away if res else "",
                "final_home": res.final_home if res else "",
                "decided": res.decided if res else "",
            }
        )
    return out


def write_csv(rows: Iterable[dict[str, object]], path: Path) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def days_between(start: Date, end: Date) -> list[Date]:
    n = (end - start).days
    return [start + timedelta(days=i) for i in range(max(0, n) + 1)]


__all__ = [
    "CLOSE_MARKETS",
    "CSV_FIELDS",
    "HIST_LABEL",
    "close_times",
    "days_between",
    "export_day",
    "hist_rows",
    "plan",
    "pull_day",
    "read_manifest",
    "write_csv",
]
