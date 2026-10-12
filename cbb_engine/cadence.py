"""§5 cadence table as code: which pass is due for a game at a given time-to-tip.

The clocks are the ones ``engine_runtime.quotes.expected_passes`` counts gaps
against; both read the same ``Cadence``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone

from cbb_engine.config import Cadence

UTC = timezone.utc


def parse_utc(raw: str) -> datetime | None:
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def board_interval(tip: datetime, now: datetime, cad: Cadence) -> int:
    """Minutes between board passes for this game right now (60 / 30 / 15)."""
    hours = (tip - now).total_seconds() / 3600
    if hours > cad.mid_window_h:
        return cad.board_far
    if hours > cad.featured_window_h:
        return cad.board_mid
    return cad.board_near


def slate_board_interval(tips: Iterable[datetime], now: datetime, cad: Cadence) -> int:
    """The board is one request for every game, so it runs on the fastest clock any game needs."""
    future = [t for t in tips if t > now]
    return min((board_interval(t, now, cad) for t in future), default=cad.board_far)


def featured_due(tips: Iterable[datetime], now: datetime, cad: Cadence) -> bool:
    return any(0 < (t - now).total_seconds() <= cad.featured_window_h * 3600 for t in tips)


def close_due(tips: Mapping[str, datetime], now: datetime, cad: Cadence) -> list[str]:
    """Event ids inside the T-60 -> T-5 close window."""
    lo, hi = timedelta(minutes=cad.close_last_min), timedelta(minutes=cad.close_first_min)
    return sorted(e for e, t in tips.items() if lo <= t - now <= hi)


def due(last: datetime | None, interval_min: int, now: datetime) -> bool:
    return last is None or now - last >= timedelta(minutes=interval_min)


def daily_budget(
    n_games: int, cad: Cadence, *, featured_cost: int = 3, event_cost: int = 0, regions: int = 1
) -> dict[str, int]:
    """Credits one slate day costs under the cadence table, by pass_kind.

    A typical game is posted ~30h out: 24 board passes at 60 min, 6 at 30 min,
    12 at 15 min; 18 featured passes inside T-3h; 12 close passes per game.
    Board/featured passes are one request for the whole slate; per-event
    markets and close passes are per game.
    """
    board_passes = (
        (cad.horizon_hours - cad.mid_window_h) * 60 / cad.board_far
        + (cad.mid_window_h - cad.featured_window_h) * 60 / cad.board_mid
        + cad.featured_window_h * 60 / cad.board_near
    )
    featured_passes = cad.featured_window_h * 60 / cad.featured
    close_passes = (cad.close_first_min - cad.close_last_min) / cad.close_last_min + 1
    out = {
        "board": int(round(board_passes * featured_cost * regions)),
        "featured": int(round(featured_passes * featured_cost * regions)),
        "close": int(round(close_passes * featured_cost * regions)) * n_games,
        "event_markets": int(event_cost * regions * n_games * board_passes) if event_cost else 0,
    }
    out["total"] = sum(out.values())
    return out


__all__ = [
    "UTC",
    "board_interval",
    "close_due",
    "daily_budget",
    "due",
    "featured_due",
    "parse_utc",
    "slate_board_interval",
]
