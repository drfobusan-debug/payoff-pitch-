"""Durable capture of every NHL quote the engine sees.

Every quote the engine sees is a row in a dated CSV under
``<data>/prices/<YYYY-MM-DD>/``. A snapshot is only written when its content
differs from the previous one (fingerprint excludes ``captured_at``), so a
30-minute cron on a quiet board costs nothing, while a moving board keeps every
change. Nothing is priced here; this is the archive the period and prop ledgers
will be graded against once those models exist.

Row shape (one side per row, opposite side attached when the book quotes it):

    captured_at, game_date, matchup, event_id, market, side, entity, line,
    book, american, opposite_american

``market`` is the engine key (``game_ml``, ``p1_total``, ``sk_sog`` ...), not the
provider's. ``side`` is a team code, ``over``/``under``, ``draw`` or ``yes``.
``entity`` names the team for team totals and the player for props.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from datetime import date as Date
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

ODDSAPI = "oddsapi"

# Provider market key -> (engine market key, ot_rule).
# ``ot_rule`` is how the provider's mainstream books settle the market; per-book
# exceptions live in book_rules.py and gate any future bet, not the archive.
MARKET_MAP: dict[str, tuple[str, str]] = {
    "h2h": ("game_ml", "incl_ot_so"),
    "h2h_3_way": ("game_ml3", "reg_only"),
    "spreads": ("game_pl", "incl_ot_so"),
    "alternate_spreads": ("game_pl_alt", "incl_ot_so"),
    "totals": ("game_total", "incl_ot_so"),
    "alternate_totals": ("game_total_alt", "incl_ot_so"),
    "team_totals": ("team_total", "book_rule"),
    "alternate_team_totals": ("team_total_alt", "book_rule"),
    "h2h_p1": ("p1_ml", "reg_only"),
    "h2h_p2": ("p2_ml", "reg_only"),
    "h2h_p3": ("p3_ml", "reg_only"),
    "h2h_3_way_p1": ("p1_ml3", "reg_only"),
    "h2h_3_way_p2": ("p2_ml3", "reg_only"),
    "h2h_3_way_p3": ("p3_ml3", "reg_only"),
    "spreads_p1": ("p1_pl", "reg_only"),
    "spreads_p2": ("p2_pl", "reg_only"),
    "spreads_p3": ("p3_pl", "reg_only"),
    "totals_p1": ("p1_total", "reg_only"),
    "totals_p2": ("p2_total", "reg_only"),
    "totals_p3": ("p3_total", "reg_only"),
    "team_totals_p1": ("p1_team_total", "reg_only"),
    "player_shots_on_goal": ("sk_sog", "incl_ot"),
    "player_points": ("sk_pts", "incl_ot"),
    "player_goals": ("sk_g", "incl_ot"),
    "player_assists": ("sk_a", "incl_ot"),
    "player_blocked_shots": ("sk_blk", "incl_ot"),
    "player_power_play_points": ("sk_ppp", "incl_ot"),
    "player_total_saves": ("g_saves", "incl_ot"),
    "player_goal_scorer_anytime": ("ags", "incl_ot"),
    "player_goal_scorer_first": ("fgs", "incl_ot"),
    "player_goal_scorer_last": ("lgs", "incl_ot"),
}

GAME_MARKETS: tuple[str, ...] = ("h2h", "spreads", "totals")
EVENT_MARKETS: tuple[str, ...] = tuple(k for k in MARKET_MAP if k not in GAME_MARKETS)
PERIOD_MARKETS: tuple[str, ...] = tuple(k for k in MARKET_MAP if k.endswith(("_p1", "_p2", "_p3")))
PROP_MARKETS: tuple[str, ...] = tuple(k for k in MARKET_MAP if k.startswith("player_"))

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class QuoteRow:
    captured_at: str
    game_date: str
    matchup: str
    event_id: str
    market: str
    side: str
    entity: str
    line: float | None
    book: str
    american: float
    opposite_american: float | None
    source: str = ODDSAPI

    @property
    def key(self) -> tuple[str, str, str, str, float | None, str]:
        return (self.matchup, self.market, self.side, self.entity, self.line, self.book)


FIELDS: tuple[str, ...] = tuple(f.name for f in fields(QuoteRow))


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def prices_dir(data_dir: Path) -> Path:
    return data_dir / "prices"


def day_dir(data_dir: Path, game_date: Date) -> Path:
    return prices_dir(data_dir) / game_date.isoformat()


def _content_key(row: QuoteRow) -> str:
    return "|".join(
        [
            row.matchup,
            row.event_id,
            row.market,
            row.side,
            row.entity,
            "" if row.line is None else f"{row.line:g}",
            row.book,
            f"{row.american:g}",
            "" if row.opposite_american is None else f"{row.opposite_american:g}",
        ]
    )


def fingerprint(rows: list[QuoteRow]) -> str:
    """Content hash independent of order and capture time."""
    h = hashlib.sha256()
    for key in sorted(_content_key(r) for r in rows):
        h.update(key.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def _clean(text: str) -> str:
    return _CONTROL.sub("", text).strip()


def write_snapshot(
    rows: list[QuoteRow],
    data_dir: Path,
    game_date: Date,
    *,
    label: str = "board",
    captured_at: str | None = None,
) -> Path | None:
    """Write one CSV snapshot unless the latest one has identical content.

    Returns the path written, or ``None`` when the board was unchanged or empty.
    """
    if not rows:
        return None
    directory = day_dir(data_dir, game_date)
    directory.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(rows)
    latest = latest_snapshot(data_dir, game_date, label=label)
    if latest is not None and fingerprint(read_snapshot(latest)) == fp:
        return None
    stamp = (captured_at or now_utc()).replace(":", "").replace("-", "")
    path = directory / f"{label}_{stamp}_{fp}.csv"
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDS)
    writer.writeheader()
    for row in rows:
        d = asdict(row)
        d["line"] = "" if row.line is None else f"{row.line:g}"
        d["opposite_american"] = (
            "" if row.opposite_american is None else f"{row.opposite_american:g}"
        )
        d["american"] = f"{row.american:g}"
        writer.writerow(d)
    path.write_text(buf.getvalue(), encoding="utf-8")
    return path


def snapshot_paths(data_dir: Path, game_date: Date, *, label: str | None = None) -> list[Path]:
    directory = day_dir(data_dir, game_date)
    if not directory.is_dir():
        return []
    pattern = f"{label}_*.csv" if label else "*.csv"
    return sorted(directory.glob(pattern))


def latest_snapshot(data_dir: Path, game_date: Date, *, label: str | None = None) -> Path | None:
    paths = snapshot_paths(data_dir, game_date, label=label)
    return paths[-1] if paths else None


def _float_or_none(raw: str) -> float | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def read_snapshot(path: Path) -> list[QuoteRow]:
    """Read a snapshot, skipping rows that no longer parse."""
    rows: list[QuoteRow] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return rows
    for raw in csv.DictReader(io.StringIO(text)):
        try:
            american = _float_or_none(raw.get("american") or "")
            if american is None:
                continue
            rows.append(
                QuoteRow(
                    captured_at=_clean(raw.get("captured_at") or ""),
                    game_date=_clean(raw.get("game_date") or ""),
                    matchup=_clean(raw.get("matchup") or ""),
                    event_id=_clean(raw.get("event_id") or ""),
                    market=_clean(raw.get("market") or ""),
                    side=_clean(raw.get("side") or ""),
                    entity=_clean(raw.get("entity") or ""),
                    line=_float_or_none(raw.get("line") or ""),
                    book=_clean(raw.get("book") or ""),
                    american=american,
                    opposite_american=_float_or_none(raw.get("opposite_american") or ""),
                    source=_clean(raw.get("source") or ODDSAPI),
                )
            )
        except (TypeError, ValueError):
            continue
    return rows


def read_day(data_dir: Path, game_date: Date) -> list[QuoteRow]:
    """Every row archived for a slate date, across all snapshots."""
    out: list[QuoteRow] = []
    for path in snapshot_paths(data_dir, game_date):
        out.extend(read_snapshot(path))
    return out


def last_quotes(
    rows: list[QuoteRow],
) -> dict[tuple[str, str, str, str, float | None, str], QuoteRow]:
    """Latest quote per (matchup, market, side, entity, line, book)."""
    latest: dict[tuple[str, str, str, str, float | None, str], QuoteRow] = {}
    for row in sorted(rows, key=lambda r: r.captured_at):
        latest[row.key] = row
    return latest


def archive_summary(rows: list[QuoteRow]) -> dict[str, object]:
    games = {r.matchup for r in rows}
    books = {r.book for r in rows}
    per_market = Counter(r.market for r in rows)
    per_game: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        per_game[r.matchup].add(r.market)
    return {
        "rows": len(rows),
        "games": len(games),
        "books": sorted(books),
        "markets": dict(sorted(per_market.items())),
        "markets_per_game": {g: len(m) for g, m in sorted(per_game.items())},
    }


__all__ = [
    "EVENT_MARKETS",
    "FIELDS",
    "GAME_MARKETS",
    "MARKET_MAP",
    "ODDSAPI",
    "PERIOD_MARKETS",
    "PROP_MARKETS",
    "QuoteRow",
    "archive_summary",
    "day_dir",
    "fingerprint",
    "last_quotes",
    "latest_snapshot",
    "now_utc",
    "prices_dir",
    "read_day",
    "read_snapshot",
    "snapshot_paths",
    "write_snapshot",
]
