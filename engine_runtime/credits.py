"""Shared daily Odds API credit ledger — the one cross-sport change.

``debit(sport, cost)`` is called by every engine before a paid request. It
appends one row per call to ``<root>/credits/<YYYY-MM-DD>.csv`` and answers
whether the call fits the budget in ``<root>/credit_budget.yaml``::

    enforce: false            # observe-only: log every call, never deny
    key_total: 20000          # credits the key has per day
    reserve:                  # held back from everyone else
      cbb_featured: 2400
    sports:
      mlb: 4000
      cbb: 6000

Rules (integration plan §3): a missing or unreadable ledger or budget **fails
open** -- the caller falls through to the per-client ``min_credits`` floor it
already has -- and a refused call is logged, never raised. A broken ledger must
never silence a live capture.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

FIELDS = (
    "ts",
    "sport",
    "endpoint",
    "markets",
    "regions",
    "events",
    "cost",
    "remaining_header",
    "pass_id",
    "allowed",
)
ENV_ROOT = "ENGINE_RUNTIME_DIR"


class LedgerUnavailable(Exception):
    """The ledger or budget cannot be read; callers fail open."""


@dataclass(frozen=True)
class Budget:
    enforce: bool = False
    key_total: int | None = None
    sports: dict[str, int] | None = None
    reserves: dict[str, int] | None = None

    def sport_cap(self, sport: str) -> int | None:
        return (self.sports or {}).get(sport)

    def reserve_total(self, *, exclude: str | None = None) -> int:
        return sum(
            v for k, v in (self.reserves or {}).items() if not (exclude and k.startswith(exclude))
        )


def runtime_dir() -> Path:
    raw = os.getenv(ENV_ROOT)
    return Path(raw).expanduser() if raw else Path.home() / ".engine_runtime"


def ledger_path(root: Path, day: str) -> Path:
    return root / "credits" / f"{day}.csv"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _parse_scalar(raw: str) -> int | bool | str:
    v = raw.strip().strip('"').strip("'")
    if v.lower() in ("true", "yes", "on"):
        return True
    if v.lower() in ("false", "no", "off"):
        return False
    try:
        return int(float(v))
    except ValueError:
        return v


def load_budget(path: Path) -> Budget:
    """Read the two-level YAML subset the budget uses (no PyYAML dependency)."""
    if not path.exists():
        raise LedgerUnavailable(f"budget missing: {path}")
    try:
        text = path.read_text()
    except OSError as exc:
        raise LedgerUnavailable(str(exc)) from exc
    top: dict[str, int | bool | str] = {}
    nested: dict[str, dict[str, int]] = {}
    section: str | None = None
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].rstrip()
        if not stripped.strip():
            continue
        indented = stripped[0] in " \t"
        key, _, val = stripped.strip().partition(":")
        key = key.strip()
        if not indented:
            if val.strip():
                top[key] = _parse_scalar(val)
                section = None
            else:
                section = key
                nested.setdefault(key, {})
        elif section is not None:
            parsed = _parse_scalar(val)
            if isinstance(parsed, int) and not isinstance(parsed, bool):
                nested[section][key] = parsed
    enforce = top.get("enforce", False)
    total = top.get("key_total")
    return Budget(
        enforce=bool(enforce),
        key_total=int(total) if isinstance(total, int) and not isinstance(total, bool) else None,
        sports=nested.get("sports") or {},
        reserves=nested.get("reserve") or {},
    )


def spent(path: Path) -> dict[str, int]:
    """Credits allowed today, per sport (``""`` key = total)."""
    out: dict[str, int] = {"": 0}
    if not path.exists():
        return out
    try:
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("allowed") != "True":
                    continue
                cost = int(float(row.get("cost") or 0))
                out[""] += cost
                out[row.get("sport", "")] = out.get(row.get("sport", ""), 0) + cost
    except (OSError, ValueError) as exc:
        raise LedgerUnavailable(str(exc)) from exc
    return out


def _append(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def debit(
    sport: str,
    cost: int,
    *,
    endpoint: str = "",
    markets: str = "",
    regions: str = "",
    events: int = 0,
    remaining_header: int | None = None,
    pass_id: str = "",
    root: Path | None = None,
    day: str | None = None,
) -> bool:
    """Record a paid call and say whether it fits today's budget.

    ``True`` when allowed, when ``enforce`` is off, or when the ledger cannot
    be read (fail-open). ``False`` only when enforcement is on and the sport
    slice or the key total less the reserves would be exceeded.
    """
    root = root or runtime_dir()
    day = day or _today()
    path = ledger_path(root, day)
    try:
        budget = load_budget(root / "credit_budget.yaml")
        used = spent(path)
    except LedgerUnavailable as exc:
        log.warning("credit ledger unavailable (%s); falling through to client floor", exc)
        return True
    allowed = True
    if budget.enforce:
        cap = budget.sport_cap(sport)
        if cap is not None and used.get(sport, 0) + cost > cap:
            allowed = False
        if budget.key_total is not None:
            headroom = budget.key_total - budget.reserve_total(exclude=sport)
            if used[""] + cost > headroom:
                allowed = False
    try:
        _append(
            path,
            {
                "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sport": sport,
                "endpoint": endpoint,
                "markets": markets,
                "regions": regions,
                "events": events,
                "cost": cost,
                "remaining_header": "" if remaining_header is None else remaining_header,
                "pass_id": pass_id,
                "allowed": allowed,
            },
        )
    except OSError as exc:
        log.warning("credit ledger not writable (%s); falling through to client floor", exc)
        return True
    if not allowed:
        log.info("credit budget: %s refused %d credits (%s)", sport, cost, endpoint)
    return allowed


def debit_for(sport: str, url: str, query: dict[str, str], *, remaining: int | None = None) -> bool:
    """``debit`` from an existing engine's ``_get_json``: cost = markets x regions.

    A free endpoint (no ``markets`` in the query: ``/events``, ``/sports``) is not
    logged; a per-event URL is logged as ``event_odds`` with ``events=1``.
    """
    markets = [m for m in query.get("markets", "").split(",") if m]
    if not markets:
        return True
    regions = [r for r in query.get("regions", "us").split(",") if r]
    per_event = "/events/" in url
    return debit(
        sport,
        len(markets) * max(1, len(regions)),
        endpoint="event_odds" if per_event else url.rstrip("/").rsplit("/", 1)[-1],
        markets=query.get("markets", ""),
        regions=",".join(regions),
        events=1 if per_event else 0,
        remaining_header=remaining,
    )


__all__ = [
    "Budget",
    "FIELDS",
    "LedgerUnavailable",
    "debit",
    "debit_for",
    "ledger_path",
    "load_budget",
    "runtime_dir",
    "spent",
]
