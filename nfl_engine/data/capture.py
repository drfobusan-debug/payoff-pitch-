"""The price archive: every quote we ever saw, timestamped, on disk.

This exists because of an asymmetry that decides what the engine can ever prove.
Historical **game** prices are free -- nflverse ships the closing spread, total
and moneyline for every game back to 1999 -- but historical **player-prop** prices
do not exist anywhere at any price. A prop model can therefore never be
backtested against the number it would have had to beat; the archive has to be
built forward, starting before the first bet, or the phase-4 gate can never be
run on anything but the model's own opinion.

So capture is deliberately separated from pricing. ``capture`` fetches and
writes; it forms no probability, screens nothing and stakes nothing. It can run
on a schedule from now until the props layer is unblocked and the only thing that
accrues is evidence.

Two properties matter more than the format:

**Idempotence.** A snapshot is written only if its contents differ from the last
snapshot of the same kind for the same week. Running the capture every fifteen
minutes through a Sunday morning yields one file per *move*, not ninety-six
copies of a board that never changed -- and a re-run after a crash adds nothing.

**One row, one price.** Each row carries the side, the rung, the book, the
American price *and the same book's opposite side*, because a price without its
pair cannot be de-vigged and a rung without its book cannot be graded. Nothing is
consensus-collapsed on the way in; that is the EV layer's job and it can only be
redone later if the raw ladder survived.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from nfl_engine.audit.ledger import CONTROL_CHARS
from nfl_engine.config import data_dir
from nfl_engine.market.board import OVER, GameOdds, MarketQuote

log = logging.getLogger(__name__)

GAME_KIND = "game"
PROP_KIND = "prop"
CLOSE_KIND = "close"
ODDSAPI = "oddsapi"

MONEYLINE, SPREAD, TOTAL = "moneyline", "spread", "total"


@dataclass(frozen=True)
class QuoteRow:
    """One book's price on one side of one line, as it stood at ``captured_at``.

    Game and prop quotes share this schema. They stay distinguishable without a
    second table because a game row's ``market`` is one of moneyline/spread/total
    and its ``player`` is empty, while a prop row carries the Odds API market key
    (``player_reception_yds``) and the player it belongs to.
    """

    captured_at: str  # UTC, second resolution
    season: int
    week: int
    game_date: str
    matchup: str  # "AWAY @ HOME"
    market: str
    side: str  # team code, or over/under
    line: float | None  # the side's own handicap, or the total/prop line
    book: str
    american: float
    opposite_american: float | None
    player: str = ""
    event_id: str = ""
    source: str = ODDSAPI


FIELDS = [f.name for f in fields(QuoteRow)]
# captured_at is what makes two otherwise identical snapshots differ, so it is
# excluded from the fingerprint that decides whether a snapshot is new.
_FINGERPRINT_FIELDS = [name for name in FIELDS if name != "captured_at"]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def stamp(moment: datetime | None = None) -> str:
    return (moment or now_utc()).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def capture_dir(root: Path | None = None) -> Path:
    return (root or data_dir()) / "captures"


def week_dir(season: int, week: int, *, root: Path | None = None) -> Path:
    return capture_dir(root) / str(season) / f"wk{week:02d}"


# -- board -> rows --------------------------------------------------------
def _quote_row(
    market: str,
    side: str,
    line: float | None,
    quote: MarketQuote,
    *,
    captured_at: str,
    season: int,
    week: int,
    game_date: str,
    matchup: str,
    event_id: str,
    source: str,
) -> QuoteRow:
    return QuoteRow(
        captured_at=captured_at,
        season=season,
        week=week,
        game_date=game_date,
        matchup=matchup,
        market=market,
        side=side,
        line=line,
        book=quote.book,
        american=quote.american,
        opposite_american=quote.opposite_american,
        event_id=event_id,
        source=source,
    )


def rows_from_board(
    board: dict[str, GameOdds],
    *,
    season: int,
    week: int,
    captured_at: str,
    dates: dict[str, str] | None = None,
    event_ids: dict[str, str] | None = None,
    source: str = ODDSAPI,
) -> list[QuoteRow]:
    """Flatten a board into archive rows, keeping the whole ladder."""
    out: list[QuoteRow] = []
    for matchup, odds in board.items():
        home = matchup.split(" @ ")[-1]
        row = partial(
            _quote_row,
            captured_at=captured_at,
            season=season,
            week=week,
            game_date=(dates or {}).get(matchup, ""),
            matchup=matchup,
            event_id=(event_ids or {}).get(matchup, ""),
            source=source,
        )
        for side, quotes in odds.ml.items():
            out.extend(row(MONEYLINE, side, None, quote) for quote in quotes)
        for home_point, sides in odds.spreads.items():
            for side, quotes in sides.items():
                # Stored on the side's own handicap, matching the ledger, so a
                # graded row and its archived quote key the same way.
                own = home_point if side == home else -home_point
                out.extend(row(SPREAD, side, own, quote) for quote in quotes)
        for line, sides in odds.totals.items():
            for side, quotes in sides.items():
                out.extend(row(TOTAL, side, line, quote) for quote in quotes)
    return sorted(out, key=_sort_key)


def board_from_rows(rows: list[QuoteRow]) -> dict[str, GameOdds]:
    """Rebuild a board from archived rows, for pricing a snapshot offline.

    The archive is the only board available when the API is down, when a capture
    is being re-priced after a model change, or when a past snapshot has to be
    re-graded -- so the round trip has to be exact.
    """
    board: dict[str, GameOdds] = {}
    for r in rows:
        odds = board.setdefault(r.matchup, GameOdds(matchup=r.matchup))
        quote = MarketQuote(
            book=r.book, american=r.american, opposite_american=r.opposite_american
        )
        if r.market == MONEYLINE:
            odds.add_ml(r.side, quote)
        elif r.market == SPREAD and r.line is not None:
            home = r.matchup.split(" @ ")[-1]
            odds.add_spread(r.line if r.side == home else -r.line, r.side, quote)
        elif r.market == TOTAL and r.line is not None:
            odds.add_total(r.line, r.side == OVER, quote)
    return board


def _sort_key(row: QuoteRow) -> tuple[str, str, str, str, float, str]:
    return (
        row.matchup,
        row.market,
        row.player,
        row.side,
        row.line if row.line is not None else 0.0,
        row.book,
    )


# -- persistence ---------------------------------------------------------
def _canonical(value: object) -> str:
    """One spelling per value, so a CSV round trip cannot change a fingerprint.

    -150 and -150.0 are the same price. Without this the comparison against the
    previous snapshot always differs -- every capture would look like a move and
    the archive would fill with identical files.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):g}"
    return str(value)


def fingerprint(rows: list[QuoteRow]) -> str:
    """Content hash of a snapshot, ignoring when it was taken."""
    digest = hashlib.sha256()
    for row in sorted(rows, key=_sort_key):
        values = asdict(row)
        digest.update(
            "|".join(f"{name}={_canonical(values[name])}" for name in _FINGERPRINT_FIELDS).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()[:20]


def snapshot_paths(
    season: int, week: int, kind: str = GAME_KIND, *, root: Path | None = None
) -> list[Path]:
    folder = week_dir(season, week, root=root)
    if not folder.exists():
        return []
    return sorted(folder.glob(f"{kind}-*.csv"))


def latest_snapshot(
    season: int, week: int, kind: str = GAME_KIND, *, root: Path | None = None
) -> Path | None:
    paths = snapshot_paths(season, week, kind, root=root)
    return paths[-1] if paths else None


def read_snapshot(path: Path) -> list[QuoteRow]:
    if not path.exists():
        return []
    out: list[QuoteRow] = []
    # Read whole and strip control characters, rather than streaming the file: one
    # NUL byte anywhere makes ``csv`` refuse the entire read, and an archive is the
    # only copy of prices that can never be fetched again.
    text = CONTROL_CHARS.sub("", path.read_text(encoding="utf-8", errors="replace"))
    skipped = 0
    for raw in csv.DictReader(io.StringIO(text, newline="")):
        american = _float(raw.get("american"))
        if american is None:
            skipped += 1
            continue
        try:
            season = int(raw.get("season") or 0)
            week = int(raw.get("week") or 0)
        except ValueError:
            # A row that cannot say which week it belongs to is unreadable, not
            # fatal: skip it and keep the rest of the snapshot.
            skipped += 1
            continue
        out.append(
            QuoteRow(
                captured_at=raw.get("captured_at", ""),
                season=season,
                week=week,
                game_date=raw.get("game_date", ""),
                matchup=raw.get("matchup", ""),
                market=raw.get("market", ""),
                side=raw.get("side", ""),
                line=_float(raw.get("line")),
                book=raw.get("book", ""),
                american=american,
                opposite_american=_float(raw.get("opposite_american")),
                player=raw.get("player", ""),
                event_id=raw.get("event_id", ""),
                source=raw.get("source", ODDSAPI),
            )
        )
    if skipped:
        log.warning("skipped %d unreadable quote row(s) in %s", skipped, path.name)
    return out


def write_snapshot(
    rows: list[QuoteRow],
    *,
    season: int,
    week: int,
    kind: str = GAME_KIND,
    root: Path | None = None,
) -> Path | None:
    """Archive a snapshot unless it repeats the previous one.

    Returns the file written, or ``None`` when the board had not moved -- so a
    caller can report "no change" instead of implying a fresh capture.
    """
    if not rows:
        return None
    previous = latest_snapshot(season, week, kind, root=root)
    if previous is not None and fingerprint(read_snapshot(previous)) == fingerprint(rows):
        log.info("board unchanged since %s; nothing archived", previous.name)
        return None
    folder = week_dir(season, week, root=root)
    folder.mkdir(parents=True, exist_ok=True)
    taken = rows[0].captured_at or stamp()
    path = _free_path(folder, kind, taken)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=_sort_key):
            writer.writerow(asdict(row))
    return path


def _free_path(folder: Path, kind: str, taken: str) -> Path:
    """A path that does not already exist, so two moves a second apart both keep."""
    base = f"{kind}-{taken.replace(':', '')}"
    path = folder / f"{base}.csv"
    suffix = 1
    while path.exists():
        path = folder / f"{base}-{suffix}.csv"
        suffix += 1
    return path


def archive_summary(rows: list[QuoteRow]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row.market] += 1
    games = len({row.matchup for row in rows})
    books = len({row.book for row in rows})
    parts = ", ".join(f"{market} {n}" for market, n in sorted(counts.items()))
    return f"{len(rows)} quotes on {games} games from {books} books ({parts})"


def _float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


__all__ = [
    "CLOSE_KIND",
    "GAME_KIND",
    "PROP_KIND",
    "QuoteRow",
    "archive_summary",
    "board_from_rows",
    "capture_dir",
    "fingerprint",
    "latest_snapshot",
    "read_snapshot",
    "rows_from_board",
    "snapshot_paths",
    "stamp",
    "week_dir",
    "write_snapshot",
]
