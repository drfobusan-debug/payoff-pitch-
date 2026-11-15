"""Novig's price beside each engine's bets: capture at card time, value at the close.

Novig is a no-vig exchange. Each engine pulls Novig's board once per card run
(a ``bookmakers=novig`` request, one region's cost) and appends its quotes,
keyed in the engine's own vocabulary, to ``novig_<date>.jsonl``. The report pairs
every graded row with the latest Novig quote on the same side **at the same
line** captured before the game started, and values that price against the
engine's own close:

    close EV at Novig = (1 + close EV at the book) x decimal(Novig) / decimal(book) - 1

which holds because both prices are on the same line and settle on the same
outcome. A Novig quote at a different number is not paired (the row is counted
as uncovered). Quotes carry no depth. Nothing here is read by pricing.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

BOOK = "novig"
BAR_BUYS = 150
CALL_SE = 2.0
MAX_PRICE = 2000.0
MAX_BOOK = 1.10


def decimal(american: float) -> float:
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / -american)


def key(*parts: object) -> str:
    return "|".join("" if p is None else str(p) for p in parts)


def _utc(stamp: str) -> datetime | None:
    try:
        t = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def record(
    k: str,
    line: float | None,
    american: float,
    opposite: float | None,
    captured_at: str,
    tip: str,
) -> dict[str, object]:
    return {
        "key": k,
        "line": line,
        "american": american,
        "opposite": opposite,
        "captured_at": captured_at,
        "tip": tip,
    }


def _implied(american: float) -> float:
    return 1.0 / decimal(american)


def tradable(american: object, opposite: object) -> bool:
    """A real two-way quote, not an empty book's placeholder.

    Novig's board carries -9900 / -99900 / -100000 stand-ins where nobody is
    quoting; those, and two-way pairs holding over 10%, are not a price.
    """
    if not isinstance(american, int | float) or abs(american) > MAX_PRICE or abs(american) < 100:
        return False
    if opposite is None:
        return True
    if not isinstance(opposite, int | float) or abs(opposite) > MAX_PRICE or abs(opposite) < 100:
        return False
    return _implied(american) + _implied(opposite) <= MAX_BOOK


def append(path: Path, records: Iterable[Mapping[str, object]]) -> int:
    """Append tradable pregame quotes (seen before ``tip``); returns how many were kept."""
    kept = []
    for r in records:
        seen, tip = _utc(str(r.get("captured_at", ""))), _utc(str(r.get("tip", "")))
        if not tradable(r.get("american"), r.get("opposite")):
            continue
        if seen is not None and (tip is None or seen < tip):
            kept.append(json.dumps(dict(r), sort_keys=True))
    if kept:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.writelines(f"{line}\n" for line in kept)
    return len(kept)


def load(path: Path) -> dict[tuple[str, float | None], list[dict[str, object]]]:
    """Every pregame quote per (key, line), oldest first."""
    out: dict[tuple[str, float | None], list[dict[str, object]]] = defaultdict(list)
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        try:
            r = json.loads(raw)
        except ValueError:
            continue
        line = r.get("line")
        out[(str(r.get("key", "")), None if line is None else float(line))].append(r)
    for quotes in out.values():
        quotes.sort(key=lambda r: str(r.get("captured_at", "")))
    return out


def as_of(
    index: Mapping[tuple[str, float | None], list[dict[str, object]]],
    k: str,
    line: float | None,
    when: str,
) -> dict[str, object] | None:
    """The latest quote on (``k``, ``line``) seen at or before ``when`` (the bet's price time).

    With no ``when`` the latest pregame quote stands in, which flatters Novig (a
    later quote has less movement left), so engines pass the bet's own stamp.
    """
    quotes = index.get((k, line), [])
    at = _utc(when) if when else None
    best = None
    for r in quotes:
        seen = _utc(str(r.get("captured_at", "")))
        if at is None or (seen is not None and seen <= at):
            best = r
    return best


def close_ev_at_novig(book_american: float, book_close_ev: float, novig_american: float) -> float:
    return (1.0 + book_close_ev) * decimal(novig_american) / decimal(book_american) - 1.0


@dataclass(frozen=True)
class Pair:
    """One graded row: its group, game, and EV at the close at the book (and at Novig if quoted)."""

    group: str
    game: str
    buy: bool
    book_ev: float
    novig_ev: float | None


@dataclass(frozen=True)
class Compare:
    label: str
    n: int
    of: int
    book_ev: float | None
    novig_ev: float | None
    gain: float | None
    gain_se: float | None

    @property
    def verdict(self) -> str:
        if self.gain is None or self.gain_se is None:
            return "no Novig quotes yet"
        if self.n < BAR_BUYS:
            return f"needs {BAR_BUYS - self.n} more"
        if abs(self.gain) <= CALL_SE * self.gain_se:
            return f"within {CALL_SE:g} se"
        return f"Novig {'ahead' if self.gain > 0 else 'behind'} by > {CALL_SE:g} se"


def _clustered(rows: list[tuple[str, float]]) -> tuple[float, float | None]:
    vals = [v for _, v in rows]
    mean = sum(vals) / len(vals)
    games: dict[str, float] = defaultdict(float)
    for g, v in rows:
        games[g] += v - mean
    k = len(games)
    if k < 2:
        return mean, None
    return mean, math.sqrt(k / (k - 1) * sum(s * s for s in games.values())) / len(vals)


def _compare(label: str, pairs: list[Pair]) -> Compare:
    quoted = [p for p in pairs if p.novig_ev is not None]
    if not quoted:
        return Compare(label, 0, len(pairs), None, None, None, None)
    gain, se = _clustered([(p.game, (p.novig_ev or 0.0) - p.book_ev) for p in quoted])
    return Compare(
        label,
        len(quoted),
        len(pairs),
        sum(p.book_ev for p in quoted) / len(quoted),
        sum(p.novig_ev or 0.0 for p in quoted) / len(quoted),
        gain,
        se,
    )


def compare(pairs: list[Pair]) -> list[Compare]:
    """Buys (the pre-registered row), buys by group, and every graded row."""
    buys = [p for p in pairs if p.buy]
    out = [_compare("Buys", buys)]
    for g in sorted({p.group for p in buys}):
        out.append(_compare(f"Buys, {g}", [p for p in buys if p.group == g]))
    out.append(_compare("Every graded row", pairs))
    return out


def render(rows: list[Compare]) -> str:
    def pct(v: float | None) -> str:
        return "" if v is None else f"{v:+.2%}"

    lines = [
        f"{'rows':24s} {'with Novig':>12s} {'EV@close book':>13s} {'Novig':>8s} {'gain':>18s}  verdict"
    ]
    for r in rows:
        gain = "" if r.gain is None else f"{r.gain:+.2%} ± {r.gain_se or 0:.2%}"
        lines.append(
            f"{r.label:24s} {f'{r.n} of {r.of}':>12s} {pct(r.book_ev):>13s} {pct(r.novig_ev):>8s} "
            f"{gain:>18s}  {r.verdict}"
        )
    lines.append(
        f"Same side, same line, the Novig quote seen at or before the bet; valued at the close. "
        f"Pre-registered: Novig improves EV at the close if the buys' gain is beyond {CALL_SE:g} SE "
        f"over {BAR_BUYS} buys. Quotes carry no depth; graded only."
    )
    return "\n".join(lines)


__all__ = [
    "BOOK",
    "Compare",
    "Pair",
    "append",
    "as_of",
    "close_ev_at_novig",
    "compare",
    "key",
    "load",
    "record",
    "render",
    "tradable",
]
