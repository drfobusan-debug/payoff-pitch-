"""Availability log (§5.7): every absence is a line on the card and nothing else.

Append-only JSONL under ``<data_dir>/availability/<slate>.jsonl``. A record is
written when a player is first seen as out/doubtful/returning and again when
the status changes; nothing is edited in place. Each record carries the
market's own read of the game (the latest archived home moneyline for that
matchup) at ``seen_at`` so the later study -- did the market already know? --
has both halves of the answer without reconstruction.

Sources this block:

* ``nhl_api_roster`` -- players on the team's previous game roster who are
  not in tonight's gamecenter ``rosterSpots`` (``derive_from_rosters``); the
  pre-drop pass stamps ``minutes_to_drop`` so the audit knows how much lead
  time it had. The boxscore, not this log, is the roster of record.
* ``manual`` -- ``nhl availability add`` from the CLI.

Projected-lines feeds (RotoWire/DailyFaceoff) plug in as further sources with
the same record; none is scraped yet.

No record here changes a price. The lineup rebuild reads ``out`` players from
this log and sizes them by fitted isolated impact; a player without a fitted
impact contributes 0 and the card tags him ``[reported, not scored]``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import date as Date
from datetime import datetime, timezone
from pathlib import Path

from nhl_engine.data.capture import QuoteRow, last_quotes
from nhl_engine.data.nhlapi import RosterSpot

STATUSES = ("out", "doubtful", "questionable", "scratched", "returning", "in")
ROLES = ("G1", "G2", "F", "D", "")
NOT_SCORED = "[reported, not scored]"
PRICED_IN = "[already priced in]"


@dataclass(frozen=True)
class Availability:
    slate: str
    team: str
    player_id: int
    name: str
    status: str
    source: str
    posted_at: str  # when the source published (ISO UTC) -- '' if unknown
    seen_at: str  # when we first saw it
    role: str = ""  # G1 / G2 / F / D
    replacement: str = ""
    minutes_to_drop: float | None = None
    market_home_ml: float | None = None  # latest archived home ML when seen
    market_book: str = ""
    tag: str = NOT_SCORED
    note: str = ""

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.team, self.player_id, self.status)


def availability_dir(data_dir: Path) -> Path:
    return data_dir / "availability"


def log_path(data_dir: Path, slate: Date) -> Path:
    return availability_dir(data_dir) / f"{slate.isoformat()}.jsonl"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_log(data_dir: Path, slate: Date) -> list[Availability]:
    path = log_path(data_dir, slate)
    if not path.exists():
        return []
    out: list[Availability] = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(Availability(**json.loads(line)))
    return out


def append(data_dir: Path, records: Iterable[Availability]) -> list[Availability]:
    """Append records whose (team, player, status) is not already logged for the slate."""
    written: list[Availability] = []
    by_slate: dict[str, list[Availability]] = {}
    for r in records:
        by_slate.setdefault(r.slate, []).append(r)
    for slate, recs in by_slate.items():
        path = log_path(data_dir, Date.fromisoformat(slate))
        seen = {r.key for r in read_log(data_dir, Date.fromisoformat(slate))}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            for r in recs:
                if r.key in seen:
                    continue
                fh.write(json.dumps(asdict(r), sort_keys=True) + "\n")
                seen.add(r.key)
                written.append(r)
    return written


def current_out(records: Sequence[Availability], team: str) -> dict[int, Availability]:
    """Latest status per player for ``team``; only those not expected to play."""
    latest: dict[int, Availability] = {}
    for r in sorted(records, key=lambda r: r.seen_at):
        if r.team == team:
            latest[r.player_id] = r
    return {pid: r for pid, r in latest.items() if r.status in ("out", "doubtful", "scratched")}


def market_snapshot(rows: Sequence[QuoteRow], matchup: str) -> tuple[float | None, str]:
    """Latest archived home moneyline for ``matchup`` (``AWAY@HOME``), any book."""
    best: QuoteRow | None = None
    home = matchup.split("@")[-1]
    for row in last_quotes(list(rows)).values():
        if row.matchup == matchup and row.market == "game_ml" and row.side == home:
            if best is None or row.captured_at > best.captured_at:
                best = row
    return (best.american, best.book) if best else (None, "")


def derive_from_rosters(
    slate: Date,
    team: str,
    previous: Sequence[RosterSpot],
    tonight: Sequence[RosterSpot],
    *,
    minutes_to_drop: float | None,
    market: tuple[float | None, str] = (None, ""),
    seen_at: str | None = None,
) -> list[Availability]:
    """Players dressed last game and absent from tonight's gamecenter roster."""
    now = seen_at or now_utc()
    here = {r.player_id for r in tonight if r.team == team}
    out: list[Availability] = []
    for spot in previous:
        if spot.team != team or spot.player_id in here:
            continue
        role = "G1" if spot.position == "G" else ("D" if spot.position == "D" else "F")
        out.append(
            Availability(
                slate=slate.isoformat(),
                team=team,
                player_id=spot.player_id,
                name=spot.name,
                status="scratched",
                source="nhl_api_roster",
                posted_at="",
                seen_at=now,
                role=role,
                minutes_to_drop=minutes_to_drop,
                market_home_ml=market[0],
                market_book=market[1],
            )
        )
    return out


__all__ = [
    "NOT_SCORED",
    "PRICED_IN",
    "ROLES",
    "STATUSES",
    "Availability",
    "append",
    "availability_dir",
    "current_out",
    "derive_from_rosters",
    "log_path",
    "market_snapshot",
    "read_log",
]
