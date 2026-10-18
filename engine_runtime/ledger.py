"""Write-once decision ledger on the grain ``(event_id, market, side, line, book, pass_id)``.

Every priced *and* refused selection is a row (CBB plan §4.a). Rows are written
once per pass to ``<root>/ledger/<sport>/<YYYY-MM-DD>/<pass_id>.jsonl``, each
line carrying the sha256 of the previous line (hash chain); corrections are new
rows with ``supersedes`` set, never edits. ``status`` says what the row may do
(``live`` | ``shadow`` | ``research``) and ``bet_of_record`` marks the card
pass that was allowed to write -- close passes never create a decision row.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

STATUSES = ("live", "shadow", "research")
GENESIS = "0" * 64


@dataclass
class Row:
    # identity
    slate_date: str
    event_id: str
    tip_utc: str
    market: str
    side: str
    line: float | None
    book: str
    pass_id: str
    pass_kind: str
    bet_of_record: bool
    status: str
    stale_flag: bool = False
    # quotes used (shard ids, not re-read boards)
    quote_shard: str = ""
    american: float | None = None
    opposite_american: float | None = None
    books: int = 0
    paired_books: int = 0
    exec_books: int = 0
    # provenance
    code_sha: str = ""
    env_lock_hash: str = ""
    params_hash: str = ""
    model_version: str = ""
    inputs_asof: str = ""
    quality_bucket: str = ""
    bucket_inputs: dict[str, float] = field(default_factory=dict)
    # distribution
    p_fair: float | None = None
    p_model: float | None = None
    target_median: float | None = None
    sim_median: float | None = None
    paths: int = 0
    se: float | None = None
    push_mass: float | None = None
    parity_mass: float | None = None
    edge: float | None = None
    ev: float | None = None
    tier: str = ""
    edge_by_rung: dict[str, float] = field(default_factory=dict)
    prior_variants: dict[str, float] = field(default_factory=dict)
    avail_points: float | None = None
    avail_estimator: str = ""
    avail_events: int = 0
    # gates
    gates: list[str] = field(default_factory=list)
    # execution (filled by `placed`, never at write)
    placed: str = "unknown"  # Y | N | unknown
    placed_american: float | None = None
    placed_at: str = ""
    placed_limit: float | None = None
    placed_quote_id: str = ""
    # close (filled by grade)
    close_status: str = ""  # pre_tip | post_tip | pulled
    close_consensus: float | None = None
    close_consensus_shard: str = ""
    close_exec_american: float | None = None
    close_exec_shard: str = ""
    clv_consensus: float | None = None
    clv_exec: float | None = None
    outcome: str = ""
    pnl: float | None = None
    supersedes: str = ""
    prev_hash: str = ""
    row_hash: str = ""

    @property
    def key(self) -> tuple[str, str, str, float | None, str, str]:
        return (self.event_id, self.market, self.side, self.line, self.book, self.pass_id)

    @property
    def refused(self) -> bool:
        return bool(self.gates)

    @property
    def is_buy(self) -> bool:
        return not self.gates and self.tier not in ("", "Pass") and self.american is not None


FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Row))


def _digest(payload: dict[str, object]) -> str:
    body = {k: v for k, v in payload.items() if k != "row_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def pass_path(root: Path, sport: str, slate_date: str, pass_id: str) -> Path:
    return root / "ledger" / sport / slate_date / f"{pass_id}.jsonl"


def last_hash(root: Path, sport: str, slate_date: str) -> str:
    """Tail of the day's chain across all passes written so far."""
    day = root / "ledger" / sport / slate_date
    if not day.is_dir():
        return GENESIS
    files = sorted(day.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        return GENESIS
    lines = files[-1].read_text().splitlines()
    return json.loads(lines[-1])["row_hash"] if lines else GENESIS


def write_once(root: Path, sport: str, slate_date: str, pass_id: str, rows: Iterable[Row]) -> Path:
    """Append the pass's rows, chained, refusing a second write for the same pass."""
    path = pass_path(root, sport, slate_date, pass_id)
    if path.exists():
        raise FileExistsError(f"pass already written: {path}")
    for r in rows:
        if r.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}: {r.status!r}")
        if r.pass_kind == "close" and r.bet_of_record:
            raise ValueError("a close pass can never be the bet of record")
    path.parent.mkdir(parents=True, exist_ok=True)
    prev = last_hash(root, sport, slate_date)
    seen: set[tuple[str, str, str, float | None, str, str]] = set()
    with path.open("w") as fh:
        for r in rows:
            if r.key in seen:
                raise ValueError(f"duplicate row in pass: {r.key}")
            seen.add(r.key)
            r.prev_hash = prev
            payload = asdict(r)
            r.row_hash = _digest(payload)
            payload["row_hash"] = r.row_hash
            fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
            prev = r.row_hash
    return path


def read_day(root: Path, sport: str, slate_date: str) -> list[Row]:
    day = root / "ledger" / sport / slate_date
    out: list[Row] = []
    if not day.is_dir():
        return out
    for path in sorted(day.glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
        for line in path.read_text().splitlines():
            if line.strip():
                out.append(Row(**json.loads(line)))
    return out


def verify_chain(rows: Iterable[Row]) -> bool:
    prev = GENESIS
    for r in rows:
        if r.prev_hash != prev or _digest(asdict(r)) != r.row_hash:
            return False
        prev = r.row_hash
    return True


def record(
    root: Path, sport: str, slate_date: str
) -> dict[tuple[str, str, str, float | None, str], Row]:
    """The bet of record per position: the `bet_of_record` row, latest correction wins."""
    out: dict[tuple[str, str, str, float | None, str], Row] = {}
    superseded = {r.supersedes for r in read_day(root, sport, slate_date) if r.supersedes}
    for r in read_day(root, sport, slate_date):
        if r.row_hash in superseded or not r.bet_of_record:
            continue
        out.setdefault((r.event_id, r.market, r.side, r.line, r.book), r)
    return out


__all__ = [
    "FIELDS",
    "GENESIS",
    "STATUSES",
    "Row",
    "last_hash",
    "pass_path",
    "read_day",
    "record",
    "verify_chain",
    "write_once",
]
