"""Per-pass quote shards: Parquet on disk, a daily manifest of hashes, DuckDB on top.

Every pass writes one immutable shard ``<root>/quotes/<sport>/<YYYY-MM-DD>/<pass_id>.parquet``
stamped with ``pass_kind`` (``board`` | ``featured`` | ``close``) and ``captured_at``.
The manifest (``manifest.jsonl`` beside the shards) carries each shard's sha256,
row count and pass metadata; the manifest, not the shards, is what the small
git side (engine-state) mirrors. ``query()`` opens DuckDB over the shards of a
day so the pricing and the audit read the same rows.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PASS_KINDS = ("board", "featured", "close")


@dataclass(frozen=True)
class Quote:
    captured_at: str
    game_date: str
    event_id: str
    tip_utc: str
    market: str
    side: str
    entity: str
    line: float | None
    book: str
    american: float
    opposite_american: float | None
    pass_id: str
    pass_kind: str
    source: str = "oddsapi"


FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Quote))
_SCHEMA = pa.schema(
    [
        pa.field("captured_at", pa.string()),
        pa.field("game_date", pa.string()),
        pa.field("event_id", pa.string()),
        pa.field("tip_utc", pa.string()),
        pa.field("market", pa.string()),
        pa.field("side", pa.string()),
        pa.field("entity", pa.string()),
        pa.field("line", pa.float64()),
        pa.field("book", pa.string()),
        pa.field("american", pa.float64()),
        pa.field("opposite_american", pa.float64()),
        pa.field("pass_id", pa.string()),
        pa.field("pass_kind", pa.string()),
        pa.field("source", pa.string()),
    ]
)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_pass_id(kind: str, at: str | None = None) -> str:
    if kind not in PASS_KINDS:
        raise ValueError(f"pass_kind must be one of {PASS_KINDS}, got {kind!r}")
    stamp = (at or now_utc()).replace("-", "").replace(":", "")
    return f"{kind}_{stamp}"


def day_dir(root: Path, sport: str, game_date: str) -> Path:
    return root / "quotes" / sport / game_date


def shard_path(root: Path, sport: str, game_date: str, pass_id: str) -> Path:
    return day_dir(root, sport, game_date) / f"{pass_id}.parquet"


def write_shard(
    root: Path,
    sport: str,
    game_date: str,
    rows: Iterable[Quote],
    *,
    pass_id: str,
    pass_kind: str,
    extra: Mapping[str, object] | None = None,
) -> Path | None:
    """Write one immutable shard and append its hash to the day's manifest.

    Returns ``None`` when ``rows`` is empty (no shard, no manifest line) and
    refuses to overwrite an existing ``pass_id``.
    """
    if pass_kind not in PASS_KINDS:
        raise ValueError(f"pass_kind must be one of {PASS_KINDS}, got {pass_kind!r}")
    data = [asdict(r) for r in rows]
    if not data:
        return None
    path = shard_path(root, sport, game_date, pass_id)
    if path.exists():
        raise FileExistsError(f"shard already written: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(data, schema=_SCHEMA)
    pq.write_table(table, path, compression="zstd")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entry = {
        "pass_id": pass_id,
        "pass_kind": pass_kind,
        "shard": path.name,
        "sha256": digest,
        "rows": len(data),
        "captured_at": data[0]["captured_at"],
        **(dict(extra) if extra else {}),
    }
    with (path.parent / "manifest.jsonl").open("a") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return path


def read_manifest(root: Path, sport: str, game_date: str) -> list[dict[str, object]]:
    path = day_dir(root, sport, game_date) / "manifest.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify(root: Path, sport: str, game_date: str) -> list[str]:
    """Shards whose bytes no longer match the manifest (empty = intact)."""
    bad: list[str] = []
    for entry in read_manifest(root, sport, game_date):
        shard = day_dir(root, sport, game_date) / str(entry["shard"])
        if not shard.exists() or hashlib.sha256(shard.read_bytes()).hexdigest() != entry["sha256"]:
            bad.append(str(entry["shard"]))
    return bad


def read_day(root: Path, sport: str, game_date: str) -> list[Quote]:
    out: list[Quote] = []
    for shard in sorted(day_dir(root, sport, game_date).glob("*.parquet")):
        for rec in pq.read_table(shard).to_pylist():
            out.append(Quote(**rec))
    return out


def query(root: Path, sport: str, game_date: str, sql: str) -> list[dict[str, object]]:
    """Run ``sql`` over the day's shards exposed as the view ``quotes``."""
    import duckdb

    pattern = str(day_dir(root, sport, game_date) / "*.parquet")
    con = duckdb.connect()
    try:
        con.execute(f"CREATE VIEW quotes AS SELECT * FROM read_parquet('{pattern}')")
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description or []]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
    finally:
        con.close()


def expected_passes(
    tips: Mapping[str, str],
    manifest: Iterable[Mapping[str, object]],
    *,
    board_minutes: tuple[int, int, int] = (60, 30, 15),
    featured_minutes: int = 10,
    featured_window_h: float = 3.0,
) -> dict[str, dict[str, int]]:
    """Capture-gap count per ``pass_kind`` against that clock's expected interval.

    For each kind, consecutive shard times are compared with the interval the
    cadence table prescribes at that time-to-tip (earliest tip on the day);
    a gap longer than 1.5x the expected interval counts as one missing pass.
    """
    first_tip = min(tips.values()) if tips else None
    times: dict[str, list[datetime]] = {k: [] for k in PASS_KINDS}
    for e in manifest:
        kind = str(e.get("pass_kind", ""))
        if kind in times:
            times[kind].append(datetime.fromisoformat(str(e["captured_at"]).replace("Z", "+00:00")))
    out: dict[str, dict[str, int]] = {}
    tip_dt = datetime.fromisoformat(first_tip.replace("Z", "+00:00")) if first_tip else None
    for kind, stamps in times.items():
        stamps.sort()
        missing = 0
        for a, b in zip(stamps, stamps[1:], strict=False):
            if kind == "featured":
                expect = featured_minutes
            elif kind == "close":
                expect = 5
            else:
                hours_out = (tip_dt - a).total_seconds() / 3600 if tip_dt else 6.0
                expect = (
                    board_minutes[0]
                    if hours_out > 6
                    else board_minutes[1]
                    if hours_out > 3
                    else board_minutes[2]
                )
            gap = (b - a).total_seconds() / 60
            if gap > 1.5 * expect:
                missing += int(gap // expect) - 1
        out[kind] = {"passes": len(stamps), "missing": missing}
    _ = featured_window_h
    return out


__all__ = [
    "FIELDS",
    "PASS_KINDS",
    "Quote",
    "day_dir",
    "expected_passes",
    "new_pass_id",
    "now_utc",
    "query",
    "read_day",
    "read_manifest",
    "shard_path",
    "verify",
    "write_shard",
]
