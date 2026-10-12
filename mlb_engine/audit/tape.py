"""The quote tape: every book's two-sided price the engine already fetched.

The opening board and the closing snapshot each collapse a selection to one
consensus number, so the per-book pair -- and with it the only way to tell a
real close from a bad join -- is thrown away the moment it is recorded. The tape
keeps it: one row per book per selection per capture, append-only, with the
vendor's event id and the hours to first pitch stamped when the row is written.

Nothing here makes a request. The client copies each quote into a
:class:`QuoteCapture` while the event is still in hand (so a doubleheader's two
games stay apart), and :func:`append_tape` freezes those into :class:`TapeRow`
at write time. Hours are stored, never recomputed: recomputed by the overnight
audit, every row would look in play.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date as Date
from datetime import datetime, timezone
from pathlib import Path

from mlb_engine.market import keys
from mlb_engine.market.ev import MarketQuote
from mlb_engine.schemas import Game

log = logging.getLogger(__name__)

TAPE_VERSION = 1
RUN = "run"
CLOSE = "close"


@dataclass(frozen=True)
class QuoteCapture:
    """One book's quote as the client parsed it, before the run knows anything else."""

    event_id: str | None
    commence: datetime | None
    matchup: str
    market: str
    selection: str
    book: str
    american: float
    opposite_american: float | None


@dataclass(frozen=True)
class TapeRow:
    tape_version: int
    captured_at: str
    source: str
    slate_date: str
    event_id: str | None
    game_pk: int | None
    commence_utc: str | None
    matchup: str
    market: str
    selection: str
    canonical: str
    line: float | None
    side: str
    book: str
    american: float
    opposite_american: float | None
    hours_to_first_pitch: float | None

    @property
    def game_key(self) -> str:
        return game_key(self.event_id, self.matchup, self.commence_utc)

    @property
    def dedup_key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.source, self.captured_at, self.game_key, self.market, self.selection, self.book,
        )

    def quote(self) -> MarketQuote:
        return MarketQuote(
            book=self.book, american=self.american, opposite_american=self.opposite_american
        )


def tape_path(audit_dir: Path, slate_date: Date) -> Path:
    return audit_dir / f"tape_{slate_date.isoformat()}.jsonl.gz"


def _iso(ts: datetime | None) -> str | None:
    if ts is None:
        return None
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def game_key(event_id: str | None, matchup: str, commence_utc: str | None) -> str:
    """The vendor's event id, else ``matchup|start``. Bare matchup is never a key."""
    if event_id:
        return event_id
    start = _parse_iso(commence_utc)
    stamp = start.strftime("%Y-%m-%dT%H:%MZ") if start is not None else ""
    return f"{matchup}|{stamp}"


_NUM = r"[+-]?\d+(?:\.\d+)?"
_RL = re.compile(rf"^(?P<team>\S+)(?: F5)? (?P<pt>{_NUM})$")
_TOTAL = re.compile(rf"^(?:F5 )?(?P<side>Over|Under) (?P<pt>{_NUM})$")
_PROP = re.compile(r"\s(?P<side>[ou])(?P<pt>\d+(?:\.\d+)?)$")
_ML = re.compile(r"^(?P<team>\S+)(?: F5)? ML$")


def parse_line(market: str, selection: str) -> float | None:
    """The line inside a selection string; ``None`` for a moneyline."""
    if market in ("game_ml", "f5_ml"):
        return None
    if market in ("game_rl", "f5_rl"):
        m = _RL.match(selection)
        return float(m.group("pt")) if m else None
    if market in ("game_total", "f5_total"):
        m = _TOTAL.match(selection)
        return float(m.group("pt")) if m else None
    m = _PROP.search(selection)
    return float(m.group("pt")) if m else None


def parse_side(market: str, selection: str) -> str:
    """``over``/``under`` for totals and props, the team abbreviation otherwise."""
    if market in ("game_total", "f5_total"):
        m = _TOTAL.match(selection)
        return m.group("side").lower() if m else ""
    if market in ("game_ml", "f5_ml"):
        m = _ML.match(selection)
        return m.group("team") if m else ""
    if market in ("game_rl", "f5_rl"):
        m = _RL.match(selection)
        return m.group("team") if m else ""
    m = _PROP.search(selection)
    if not m:
        return ""
    return "under" if m.group("side") == "u" else "over"


def _matchup(g: Game) -> str:
    return f"{g.away.abbrev} @ {g.home.abbrev}"


def match_game_pks(
    captures: Iterable[QuoteCapture], games: Iterable[Game]
) -> dict[str, int | None]:
    """Each event's stats-api game: same matchup, nearest scheduled first pitch.

    Two games of a doubleheader sit inside one 8-hour window, and the nearest
    start separates them (13:05 vs 19:10). ``None`` only when nothing matches or
    two games are equally near -- that is not guessed.
    """
    from mlb_engine.data.oddsapi import _SAME_GAME_WINDOW

    by_matchup: dict[str, list[tuple[datetime | None, int]]] = {}
    for g in games:
        by_matchup.setdefault(_matchup(g), []).append((_parse_iso(g.game_datetime_utc), g.game_pk))
    out: dict[str, int | None] = {}
    for c in captures:
        if not c.event_id or c.event_id in out:
            continue
        cands = by_matchup.get(c.matchup, [])
        if len(cands) == 1 and (c.commence is None or cands[0][0] is None):
            out[c.event_id] = cands[0][1]
            continue
        if c.commence is None:
            out[c.event_id] = None
            continue
        near = sorted(
            (abs((start - c.commence).total_seconds()), pk)
            for start, pk in cands
            if start is not None and abs(start - c.commence) <= _SAME_GAME_WINDOW
        )
        if not near or (len(near) > 1 and near[0][0] == near[1][0]):
            out[c.event_id] = None
        else:
            out[c.event_id] = near[0][1]
    return out


def vsin_captures(
    quotes: dict[tuple[str, str, str], list[MarketQuote]],
    supplied: set[tuple[str, str, str]],
    games: Iterable[Game],
) -> list[QuoteCapture]:
    """Rows for the keys only VSIN priced, with no event id.

    The start comes from the slate when the matchup has exactly one game, so
    the row's ``game_key`` still carries a time; a doubleheader leaves it blank.
    """
    starts: dict[str, list[datetime | None]] = {}
    for g in games:
        starts.setdefault(_matchup(g), []).append(_parse_iso(g.game_datetime_utc))
    out: list[QuoteCapture] = []
    for key, qs in quotes.items():
        if key in supplied:
            continue
        matchup, market, selection = key
        times = starts.get(matchup, [])
        commence = times[0] if len(times) == 1 else None
        for q in qs:
            out.append(
                QuoteCapture(
                    event_id=None, commence=commence, matchup=matchup, market=market,
                    selection=selection, book=q.book, american=q.american,
                    opposite_american=q.opposite_american,
                )
            )
    return out


def to_rows(
    captures: Iterable[QuoteCapture],
    *,
    slate_date: Date,
    source: str,
    now: datetime,
    game_pks: dict[str, int | None] | None = None,
) -> list[TapeRow]:
    """Freeze captures into tape rows, stamping the capture time and hours to first pitch."""
    pks = game_pks or {}
    captured = _iso(now) or ""
    rows: list[TapeRow] = []
    for c in captures:
        hours = (
            round((c.commence - now).total_seconds() / 3600.0, 4)
            if c.commence is not None else None
        )
        rows.append(
            TapeRow(
                tape_version=TAPE_VERSION,
                captured_at=captured,
                source=source,
                slate_date=slate_date.isoformat(),
                event_id=c.event_id,
                game_pk=pks.get(c.event_id) if c.event_id else None,
                commence_utc=_iso(c.commence),
                matchup=c.matchup,
                market=c.market,
                selection=c.selection,
                canonical=keys.canonical(c.selection),
                line=parse_line(c.market, c.selection),
                side=parse_side(c.market, c.selection),
                book=c.book,
                american=float(c.american),
                opposite_american=(
                    float(c.opposite_american) if c.opposite_american is not None else None
                ),
                hours_to_first_pitch=hours,
            )
        )
    return rows


def load_tape(path: Path) -> list[TapeRow]:
    """Every row on the tape; rows of an unknown version are skipped, not raised on."""
    if not path.exists():
        return []
    rows: list[TapeRow] = []
    skipped = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("tape_version") != TAPE_VERSION:
                skipped += 1
                continue
            rows.append(TapeRow(**{k: raw.get(k) for k in TapeRow.__dataclass_fields__}))
    if skipped:
        log.warning("tape %s: skipped %d row(s) of an unknown tape_version", path.name, skipped)
    return rows


def _write(path: Path, rows: list[TapeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), sort_keys=True) + "\n")
    tmp.replace(path)


def _union(existing: list[TapeRow], fresh: Iterable[TapeRow]) -> tuple[list[TapeRow], int]:
    seen = {r.dedup_key for r in existing}
    merged = list(existing)
    added = 0
    for r in fresh:
        if r.dedup_key in seen:
            continue
        seen.add(r.dedup_key)
        merged.append(r)
        added += 1
    return merged, added


def append_tape(
    audit_dir: Path,
    slate_date: Date,
    captures: Iterable[QuoteCapture],
    source: str,
    now: datetime | None = None,
    game_pks: dict[str, int | None] | None = None,
) -> int:
    """Append this capture to the date's tape; returns the number of new rows."""
    stamp = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    fresh = to_rows(captures, slate_date=slate_date, source=source, now=stamp, game_pks=game_pks)
    if not fresh:
        return 0
    path = tape_path(audit_dir, slate_date)
    merged, added = _union(load_tape(path), fresh)
    if added:
        _write(path, merged)
    return added


def merge_tape_files(remote: Path, local: Path) -> bool:
    """Set union of two machines' tapes on the dedup key: idempotent and order-free."""
    if not remote.exists():
        return False
    merged, added = _union(load_tape(local), load_tape(remote))
    if added or not local.exists():
        _write(local, sorted(merged, key=lambda r: r.dedup_key))
    return True
