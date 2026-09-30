"""Did we hear it before the number moved?

The injury feed says who is out. This says what the market did about it, by
reading our own price archive: for a posting at time *t*, the last snapshot taken
strictly before *t* and the first one taken after it. The difference between those
two readings is the movement the news is measured against.

Three outcomes are possible and all three are worth recording:

* **ahead** -- we captured the news before any archived move. That is the only
  case in which an absence could ever be worth a price, and it has to be measured
  prospectively because no historical prop-free archive exists to backtest it.
* **behind** -- the number had already moved when the item was posted. The
  information is in the price and re-charging for it would be paying twice.
* **unmeasured** -- no capture on one side of the posting. Recorded as such
  rather than silently scored as zero movement, because a missing observation and
  a still number are not the same thing.

Nothing here forms or adjusts a price. It appends observations to a log and
summarises them for the card; the day the log says we are consistently ahead of
the move on some group, that is the evidence a model input would need, and it
would be a separate, argued change.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

from nfl_engine.config import data_dir
from nfl_engine.data import capture
from nfl_engine.data.injuries import InjuryRow, NewsItem
from nfl_engine.market.board import OVER

log = logging.getLogger(__name__)

AHEAD = "ahead"
BEHIND = "behind"
UNMEASURED = "unmeasured"

LOG_NAME = "availability.jsonl"


def log_path(root: Path | None = None) -> Path:
    return (root or data_dir()) / "audit" / LOG_NAME


@dataclass(frozen=True)
class Reading:
    """The market on one game at one archived moment.

    The consensus number, not a book's: the median across books of the home
    spread and of the total. A single book's hook moving is not the market
    moving, and the point of this file is to avoid crediting the feed with one.
    """

    captured_at: str
    home_spread: float | None
    total: float | None
    books: int


@dataclass(frozen=True)
class Movement:
    """What the number did around one posting."""

    matchup: str
    posted_at: str
    before: Reading | None
    after: Reading | None

    @property
    def spread_move(self) -> float | None:
        return _delta(
            None if self.before is None else self.before.home_spread,
            None if self.after is None else self.after.home_spread,
        )

    @property
    def total_move(self) -> float | None:
        return _delta(
            None if self.before is None else self.before.total,
            None if self.after is None else self.after.total,
        )

    @property
    def timing(self) -> str:
        """Whether the archive can say we were ahead of the move.

        ``ahead`` requires both a capture before the posting and a measured move
        after it. A posting followed by a still number is ``behind``: either the
        market already knew, or it does not care -- and this log deliberately does
        not guess which, since the two are indistinguishable from one game.
        """
        if self.before is None or self.after is None:
            return UNMEASURED
        move = self.spread_move
        if move is None:
            return UNMEASURED
        return AHEAD if abs(move) >= 0.5 else BEHIND


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return round(after - before, 2)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def reading(rows: list[capture.QuoteRow], matchup: str) -> Reading | None:
    """Collapse one snapshot's rows for one game into a consensus reading."""
    mine = [row for row in rows if row.matchup == matchup]
    if not mine:
        return None
    home = matchup.split(" @ ")[-1]
    spreads = [
        row.line
        for row in mine
        if row.market == capture.SPREAD and row.side == home and row.line is not None
    ]
    totals = [
        row.line
        for row in mine
        if row.market == capture.TOTAL and row.side == OVER and row.line is not None
    ]
    return Reading(
        captured_at=mine[0].captured_at,
        home_spread=_median(spreads),
        total=_median(totals),
        books=len({row.book for row in mine}),
    )


def movement(
    matchup: str,
    posted: datetime,
    *,
    season: int,
    week: int,
    root: Path | None = None,
) -> Movement:
    """Read the archive either side of ``posted`` for one game.

    Snapshots are only written when the board changed (see
    :func:`nfl_engine.data.capture.write_snapshot`), so consecutive files are
    genuine moves and the pair around a posting is the tightest bracket the
    archive can offer.
    """
    stamp = capture.stamp(posted)
    before: Reading | None = None
    after: Reading | None = None
    for path in capture.snapshot_paths(season, week, capture.GAME_KIND, root=root):
        rows = capture.read_snapshot(path)
        read = reading(rows, matchup)
        if read is None or not read.captured_at:
            continue
        if read.captured_at < stamp:
            before = read
        elif after is None:
            after = read
    return Movement(matchup=matchup, posted_at=stamp, before=before, after=after)


@dataclass(frozen=True)
class Observation:
    """One absence, as we knew it, with the market either side of the news.

    Written as a flat JSON line: the row a later study reads, so every field it
    would need is present including the ones that are ``None`` today.
    """

    observed_at: str
    season: int
    week: int
    matchup: str
    team: str
    player: str
    player_id: str
    position: str
    group: str
    designation: str
    injury: str
    source: str
    posted_at: str | None
    headline: str | None
    lead_time_s: float | None
    spread_before: float | None
    spread_after: float | None
    spread_move: float | None
    total_before: float | None
    total_after: float | None
    capture_before: str | None
    capture_after: str | None
    timing: str


def observe(
    row: InjuryRow,
    *,
    season: int,
    week: int,
    matchup: str,
    news: NewsItem | None,
    observed: datetime,
    root: Path | None = None,
) -> Observation:
    """Build one observation, measuring the archive when the news is dated.

    An undated designation still produces a row -- we know he is out, we simply
    cannot say when we learned it -- and its timing is ``unmeasured`` rather than
    an assumed lead of zero.
    """
    move = (
        movement(matchup, news.posted, season=season, week=week, root=root)
        if news is not None
        else None
    )
    before = move.before if move is not None else None
    after = move.after if move is not None else None
    return Observation(
        observed_at=capture.stamp(observed),
        season=season,
        week=week,
        matchup=matchup,
        team=row.team,
        player=row.player,
        player_id=row.player_id,
        position=row.position,
        group=row.group,
        designation=row.designation,
        injury=row.injury,
        source=row.source,
        posted_at=news.posted.isoformat() if news is not None else None,
        headline=news.headline if news is not None else None,
        lead_time_s=(observed - news.posted).total_seconds() if news is not None else None,
        spread_before=before.home_spread if before is not None else None,
        spread_after=after.home_spread if after is not None else None,
        spread_move=move.spread_move if move is not None else None,
        total_before=before.total if before is not None else None,
        total_after=after.total if after is not None else None,
        capture_before=before.captured_at if before is not None else None,
        capture_after=after.captured_at if after is not None else None,
        timing=move.timing if move is not None else UNMEASURED,
    )


def append(path: Path, observations: list[Observation]) -> int:
    """Append observations, returning how many were written.

    Appended, never rewritten: what we knew on Wednesday is evidence even after
    Sunday proves it wrong, and an overwrite would destroy the only record of the
    lead time. A dead disk costs the log line, not the week.
    """
    if not observations:
        return 0
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for obs in observations:
                handle.write(json.dumps(asdict(obs)) + "\n")
    except OSError as exc:
        log.warning("could not append the availability log: %s", exc)
        return 0
    return len(observations)


def read_log(
    path: Path, *, season: int | None = None, week: int | None = None
) -> list[Observation]:
    """Every observation on file, newest last; a corrupt line is skipped.

    One truncated write cannot cost the rest of the log, the same rule the ledger
    holds to (#271).
    """
    if not path.exists():
        return []
    named = {field.name for field in fields(Observation)}
    out: list[Observation] = []
    skipped = 0
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        log.warning("could not read the availability log: %s", exc)
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except ValueError:
            skipped += 1
            continue
        if not isinstance(raw, dict) or not named.issubset(raw):
            skipped += 1
            continue
        obs = Observation(**{name: raw[name] for name in named})
        if season is not None and obs.season != season:
            continue
        if week is not None and obs.week != week:
            continue
        out.append(obs)
    if skipped:
        log.warning("skipped %d unreadable availability rows", skipped)
    return out


def note(observations: list[Observation], matchup: str) -> str:
    """One line for the card naming who is out, and whether we were early.

    Display only, and labelled as such: no probability, screen or tier has seen
    any of this.
    """
    mine = [obs for obs in observations if obs.matchup == matchup]
    if not mine:
        return ""
    by_team: dict[str, list[Observation]] = {}
    for obs in mine:
        by_team.setdefault(obs.team, []).append(obs)
    parts: list[str] = []
    for team, rows in sorted(by_team.items()):
        who = ", ".join(f"{obs.position} {obs.player}" for obs in rows[:3])
        extra = f" +{len(rows) - 3}" if len(rows) > 3 else ""
        parts.append(f"{team}: {who}{extra}")
    ahead = sum(1 for obs in mine if obs.timing == AHEAD)
    timing = f"; {ahead} ahead of an archived move" if ahead else ""
    return "Out -- " + "; ".join(parts) + timing + " [reported, not priced]"


def timing_counts(observations: list[Observation]) -> dict[str, int]:
    """How the log's postings sit against the archive, by group.

    The summary the decision will eventually be made on: a group whose news
    lands ahead of the move often enough, over enough weeks, is a candidate for
    an input. Nothing is a candidate yet.
    """
    counts: dict[str, int] = {}
    for obs in observations:
        counts[obs.timing] = counts.get(obs.timing, 0) + 1
        counts[f"{obs.group}/{obs.timing}"] = counts.get(f"{obs.group}/{obs.timing}", 0) + 1
    return counts
