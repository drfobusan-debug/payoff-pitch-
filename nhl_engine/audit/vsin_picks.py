"""VSiN's written NHL best bets, placed on the slate and graded like the podcast picks.

The articles are read by :mod:`engine_common.podcasts.vsin`; this module turns
each game bet (moneyline, puck line, total) into the NHL :class:`Pick` so the
same settle/P&L/CLV code grades it. The writer is the host. Props, parlays and
period bets are listed on the card as written and never graded. Nothing here
feeds a price, a gate or a stake.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date as Date
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from engine_common.podcasts import vsin
from engine_common.podcasts.extract import Pick as ArticlePick
from engine_common.podcasts.shows import NHL
from nhl_engine.audit.podcast_picks import Pick, Tally, load_picks, save_picks, tally
from nhl_engine.features.podcast_read import teams_in

DIR = "vsin"
_ET = ZoneInfo("America/New_York")
_MARKET = {"game_ml": "game_ml", "game_ats": "game_pl", "game_total": "game_total"}


def picks_path(data_dir: Path, slate: Date) -> Path:
    return data_dir / DIR / f"picks_{slate.isoformat()}.json"


def load_all(data_dir: Path) -> list[Pick]:
    return [p for f in sorted((data_dir / DIR).glob("picks_*.json")) for p in load_picks(f)]


def window(slate: Date) -> tuple[datetime, datetime]:
    """The slate's ET calendar day, in UTC: the day's article is posted that morning."""
    start = datetime.combine(slate, time(0), _ET).astimezone(timezone.utc)
    return start, start + timedelta(days=1)


def to_pick(a: ArticlePick, slate: Date, games: dict[str, tuple[str, str]]) -> Pick | None:
    """One article bet as a gradeable NHL pick; ``None`` if it is not a game bet on the slate."""
    market = _MARKET.get(a.market)
    if market is None or not a.team:
        return None
    named = " ".join(x for x in (a.team, a.opponent or "") if x)
    for matchup, codes in games.items():
        if market == "game_total":
            if not teams_in(named, set(codes)):
                continue
            if a.side not in ("over", "under") or a.line is None:
                return None
            side = a.side
        else:
            hits = teams_in(a.team, set(codes))
            if not hits:
                continue
            if market == "game_pl" and a.line is None:
                return None
            side = hits[0]
        return Pick(
            slate_date=slate.isoformat(),
            matchup=matchup,
            market=market,
            side=side,
            line=a.line if market != "game_ml" else None,
            american=int(a.price) if a.price is not None else None,
            stake=a.units,
            host=a.host or vsin.SHOW_NAME,
            stamp=a.published,
            text=a.description,
        )
    return None


def unplaced(articles: Iterable[ArticlePick]) -> list[ArticlePick]:
    """The day's bets that are not game bets: listed as written, never graded."""
    return [a for a in articles if a.market not in _MARKET]


def read_day(
    data_dir: Path, slate: Date, games: dict[str, tuple[str, str]]
) -> tuple[list[Pick], list[ArticlePick]]:
    """Fetch the day's articles; merge the game bets into the slate's file (graded ones kept)."""
    since, until = window(slate)
    articles = [p for post in vsin.fetch_posts(NHL, since, until) for p in vsin.picks_in(post)]
    path = picks_path(data_dir, slate)
    kept = {p.key: p for p in load_picks(path)}
    for a in articles:
        pick = to_pick(a, slate, games)
        if pick is not None:
            kept.setdefault(pick.key, pick)
    picks = list(kept.values())
    if picks:
        save_picks(picks, path)
    return picks, unplaced(articles)


def render(picks: list[Pick]) -> str:
    total = Tally()
    for p in picks:
        total.add(p)
    lines = [f"VSiN best bets (graded by us, incl. OT/SO): {total.line()}"]
    for k, t in sorted(tally(picks, "market").items()):
        lines.append(f"  {k:<11} {t.line()}")
    for k, t in sorted(tally(picks, "host").items(), key=lambda kv: -kv[1].n):
        lines.append(f"  {k:<15} {t.line()}")
    return "\n".join(lines) + "\n"


__all__ = ["DIR", "load_all", "picks_path", "read_day", "render", "to_pick", "unplaced", "window"]
