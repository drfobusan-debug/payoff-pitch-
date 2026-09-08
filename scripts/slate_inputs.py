"""Where a slate script's three inputs come from: the day, the predictions and
the Statcast frame.

Every one of these was a constant in the article scripts, pinned to the slate
they were written on, so a script kept working for exactly one date and then
raised ``FileNotFoundError`` forever. The day is now resolved from the state
directory and the frame from the cache, so the scripts follow the engine instead
of a filename.
"""

from __future__ import annotations

import re
from datetime import date as Date
from pathlib import Path

PREVIEWS = re.compile(r"previews_(\d{4}-\d{2}-\d{2})\.json")
CACHE = re.compile(r"statcast_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.pkl")


def predictions_path(audit_dir: Path, day: Date) -> Path:
    """The graded predictions for ``day``, or the pregame capture of them.

    A day whose slate has not been graded yet has only the pregame file, which
    carries the same recommendations; the article reads the same fields either
    way.
    """
    final = audit_dir / f"predictions_{day.isoformat()}.json"
    return final if final.exists() else audit_dir / f"predictions_{day.isoformat()}.pregame.json"


def slate_days(audit_dir: Path) -> list[Date]:
    """Every day the state directory can build an article for, oldest first."""
    days = []
    for path in audit_dir.glob("previews_*.json"):
        m = PREVIEWS.fullmatch(path.name)
        if m is None:
            continue
        day = Date.fromisoformat(m.group(1))
        if predictions_path(audit_dir, day).exists():
            days.append(day)
    return sorted(days)


def resolve_day(audit_dir: Path, arg: str | None = None) -> Date:
    """``arg`` as a date, or the most recent slate the state directory holds.

    A day the state cannot build is refused by name rather than a page into the
    run, where it used to surface as a bare ``FileNotFoundError`` on whichever
    of the two files happened to be read first.
    """
    days = slate_days(audit_dir)
    if arg is not None:
        day = Date.fromisoformat(arg)
        if day not in days:
            have = ", ".join(d.isoformat() for d in days[-7:]) or "none"
            raise FileNotFoundError(
                f"{audit_dir} has no complete slate for {day.isoformat()} "
                f"(needs previews_<date>.json and predictions_<date>[.pregame].json); "
                f"latest available: {have}"
            )
        return day
    if not days:
        raise FileNotFoundError(
            f"no slate in {audit_dir}: need a previews_<date>.json and its predictions"
        )
    return days[-1]


def statcast_frame(cache_dir: Path, day: Date, arg: str | None = None) -> Path:
    """The cached frame to read the slate's form off.

    Prefer the widest window that ends on or before ``day`` -- a window running
    past the slate would let the article describe a hitter with games he had not
    played when the bet was priced. Where the cache has nothing that old the
    widest frame available stands in, because a stale window is still a window
    and no frame at all is no article.
    """
    if arg is not None:
        path = Path(arg).expanduser()
        return path if path.is_absolute() or path.exists() else cache_dir / arg
    before: tuple[int, Path] | None = None
    widest: tuple[int, Path] | None = None
    for path in cache_dir.glob("statcast_*.pkl"):
        m = CACHE.fullmatch(path.name)
        if m is None:
            continue
        start, end = Date.fromisoformat(m.group(1)), Date.fromisoformat(m.group(2))
        span = (end - start).days
        if widest is None or span > widest[0]:
            widest = (span, path)
        if end <= day and (before is None or span > before[0]):
            before = (span, path)
    best = before or widest
    if best is None:
        raise FileNotFoundError(f"no statcast_<start>_<end>.pkl frame in {cache_dir}")
    return best[1]
