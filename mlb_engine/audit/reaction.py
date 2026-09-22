"""How fast a prop price moves after the lineup posts.

The season's ledger prices every prop once per run and grades it against a
single close, so it cannot say whether a book that was slow to re-price a
posted lineup ever left a number to take. This file can. A watcher polls the
slate, and for a game whose lineup has just appeared it snapshots that game's
props on a short clock after the posting; games not yet posted get a slower
baseline snapshot so a before-price exists. The report then measures, per
book and market, how far the line sat from its pre-posting price at each mark
and how often it was pulled -- the size of the stale-price window, if there
is one, before anything is built to bet it.

No scoring reads this file.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from mlb_engine.data.oddsapi import Quotes

BASELINE = "baseline"
LINEUP = "lineup"

#: Minutes after the posting at which a snapshot is wanted, and how far off
#: that mark a capture may land and still be read as it.
MARKS: tuple[int, ...] = (0, 5, 15, 30)
MARK_SLACK = 2.5

FIELDS = (
    "captured_at", "slate_date", "game_pk", "matchup", "event", "side", "event_at",
    "minutes_since", "market", "selection", "book", "american", "opposite_american",
)


@dataclass(frozen=True)
class ReactionRow:
    captured_at: str
    slate_date: str
    game_pk: int
    matchup: str
    event: str  # BASELINE | LINEUP
    side: str  # "away" | "home" | "" for a baseline
    event_at: str  # posting stamp the clock runs from; "" for a baseline
    minutes_since: float | None
    market: str
    selection: str
    book: str
    american: float
    opposite_american: float | None


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _parse(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def rows_from_quotes(
    quotes: Quotes,
    *,
    slate_date: str,
    game_pk: int,
    event: str,
    side: str = "",
    event_at: str = "",
    now: datetime | None = None,
) -> list[ReactionRow]:
    """Flatten one event's quotes into rows stamped ``now``."""
    now = now or datetime.now(timezone.utc)
    since = None
    if event_at:
        since = round((now - _parse(event_at)).total_seconds() / 60, 1)
    out: list[ReactionRow] = []
    for (matchup, market, selection), qs in quotes.items():
        for q in qs:
            out.append(
                ReactionRow(
                    captured_at=_iso(now),
                    slate_date=slate_date,
                    game_pk=game_pk,
                    matchup=matchup,
                    event=event,
                    side=side,
                    event_at=event_at,
                    minutes_since=since,
                    market=market,
                    selection=selection,
                    book=q.book,
                    american=q.american,
                    opposite_american=q.opposite_american,
                )
            )
    return out


def append_rows(path: Path, rows: list[ReactionRow]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def load_rows(path: Path) -> list[ReactionRow]:
    if not path.exists():
        return []
    out: list[ReactionRow] = []
    with path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            kw: dict = {}
            for k in FIELDS:
                v = raw.get(k, "")
                if k == "game_pk":
                    kw[k] = int(v)
                elif k == "american":
                    kw[k] = float(v)
                elif k in ("opposite_american", "minutes_since"):
                    kw[k] = float(v) if v not in ("", "None") else None
                else:
                    kw[k] = v
            out.append(ReactionRow(**kw))
    return out


def implied(american: float) -> float:
    return 100 / (american + 100) if american > 0 else -american / (-american + 100)


@dataclass
class MarkStat:
    n: int = 0
    moved: int = 0  # |move| > threshold
    pulled: int = 0  # priced before, absent at the mark
    abs_move: float = 0.0  # summed |move| in probability points

    def add(self, move_pts: float | None, threshold: float) -> None:
        self.n += 1
        if move_pts is None:
            self.pulled += 1
            return
        self.abs_move += abs(move_pts)
        if abs(move_pts) > threshold:
            self.moved += 1

    @property
    def priced(self) -> int:
        return self.n - self.pulled

    @property
    def mean_abs(self) -> float | None:
        return self.abs_move / self.priced if self.priced else None


Key = tuple[int, str, str, str]  # game_pk, market, selection, book


def summarize(
    rows: list[ReactionRow], *, threshold_pts: float = 1.0
) -> dict[tuple[str, str, int], MarkStat]:
    """Per (book, market, mark): how the price at the mark sits against the last
    baseline taken before the posting.

    A selection counts at a mark only when it had a baseline before ``event_at``
    and the game had *some* capture near that mark (otherwise nothing was looked
    at and its absence means nothing). Priced then and absent now is a pull.
    """
    baselines: dict[Key, list[tuple[datetime, float]]] = defaultdict(list)
    for r in rows:
        if r.event == BASELINE:
            baselines[(r.game_pk, r.market, r.selection, r.book)].append(
                (_parse(r.captured_at), implied(r.american))
            )
    # one posting per (game, side); its captures grouped by the mark they landed on
    postings: dict[tuple[int, str, str], dict[int, dict[Key, float]]] = defaultdict(dict)
    for r in rows:
        if r.event != LINEUP or r.minutes_since is None:
            continue
        mark = _nearest_mark(r.minutes_since)
        if mark is None:
            continue
        slot = postings[(r.game_pk, r.side, r.event_at)].setdefault(mark, {})
        slot[(r.game_pk, r.market, r.selection, r.book)] = implied(r.american)

    out: dict[tuple[str, str, int], MarkStat] = defaultdict(MarkStat)
    for (game_pk, _side, event_at), by_mark in postings.items():
        posted = _parse(event_at)
        before: dict[Key, float] = {}
        for key, seen in baselines.items():
            if key[0] != game_pk:
                continue
            prior = [(t, p) for t, p in seen if t < posted]
            if prior:
                before[key] = max(prior)[1]
        for mark, priced in by_mark.items():
            for key, base in before.items():
                _, market, _, book = key
                now = priced.get(key)
                out[(book, market, mark)].add(
                    None if now is None else 100 * (now - base), threshold_pts
                )
    return dict(out)


def _nearest_mark(minutes: float) -> int | None:
    best = min(MARKS, key=lambda m: abs(m - minutes))
    return best if abs(best - minutes) <= MARK_SLACK else None


def render_summary(stats: dict[tuple[str, str, int], MarkStat], *, threshold_pts: float) -> str:
    if not stats:
        return "No lineup-posting captures with a prior baseline yet."
    lines = [
        f"Price reaction to a posted lineup (move vs last pre-posting baseline, "
        f"'moved' = |move| > {threshold_pts:.1f} pt):",
        f"{'book':16s}{'market':12s}{'mark':>6s}{'n':>7s}{'pulled':>8s}{'moved':>7s}{'mean|move|':>11s}",
    ]
    for (book, market, mark), s in sorted(stats.items()):
        mean = f"{s.mean_abs:.2f}" if s.mean_abs is not None else "-"
        lines.append(
            f"{book:16s}{market:12s}{mark:>5d}m{s.n:>7d}{s.pulled:>8d}{s.moved:>7d}{mean:>11s}"
        )
    by_mark: dict[int, MarkStat] = defaultdict(MarkStat)
    for (_, _, mark), s in stats.items():
        t = by_mark[mark]
        t.n += s.n
        t.moved += s.moved
        t.pulled += s.pulled
        t.abs_move += s.abs_move
    lines.append("")
    for mark, s in sorted(by_mark.items()):
        share = f"{100 * s.moved / s.priced:.0f}%" if s.priced else "-"
        mean = f"{s.mean_abs:.2f}" if s.mean_abs is not None else "-"
        lines.append(
            f"all books +{mark}m: {s.n} lines, {s.pulled} pulled, {share} moved, mean |move| {mean} pt"
        )
    return "\n".join(lines)
