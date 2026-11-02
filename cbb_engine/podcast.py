"""The CBB adapter for the shared podcast picks (:mod:`engine_common.podcasts`).

Supplies what is CBB about a pick: matching a spoken school to the card's Odds
API team names, the slate's games from the card's own ledger rows, the engine's
view of a side, and grading on ESPN finals. Display and audit only: a podcast
pick is never written to the engine's ledger and never reaches a price, a tier
or a gate.

A spread or total with no number said is not graded. A price is the one the host
said or, failing that, the engine's own execution price at the host's number
(``board``); with neither, the pick counts in the won-lost record but carries no
units.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cbb_engine.data import espn
from cbb_engine.history import _FOLD, match_final
from engine_common.podcasts import picks as core
from engine_common.podcasts import render
from engine_common.podcasts.extract import PICKS_DIR, Extraction, load_extractions
from engine_common.podcasts.picks import Game, League, ledger_path
from engine_common.podcasts.shows import CBB
from engine_runtime.ledger import Row

Rows = list[Row]
Placed = core.Placed[Rows]
SlatePicks = core.SlatePicks[Rows]
PodcastView = render.PodcastView[Rows]
Teams = Mapping[str, tuple[str, str]]  # event_id -> (home, away), Odds API names

# A pick for tonight is said within the two days before tip.
HORIZON = timedelta(days=2)

# How hosts say a school the Odds API spells out.
_SPOKEN: dict[str, str] = {
    "uconn": "connecticut",
    "unc": "n carolina",
    "ole miss": "mississippi",
    "pitt": "pittsburgh",
    "umass": "massachusetts",
    "vcu": "virginia commonwealth",
    "smu": "southern methodist",
    "tcu": "texas christian",
    "lsu": "louisiana st",
    "byu": "brigham young",
    "ucf": "ucf",
    "usc": "usc",
}
# A word after the school that makes it another school: "Iowa" is not Iowa State.
_QUALIFIERS = frozenset(
    {
        "st",
        "tech",
        "am",
        "at",
        "n",
        "s",
        "e",
        "w",
        "c",
        "city",
        "christian",
        "baptist",
        "international",
        "atlantic",
        "gulf",
        "coast",
        "coastal",
        "poly",
        "pacific",
        "valley",
        "oh",
        "fl",
        "chicago",
        "pine",
        "little",
        "upstate",
        "wilmington",
        "greensboro",
        "charlotte",
        "asheville",
        "kearney",
        "omaha",
        "martin",
        "monroe",
        "lafayette",
        "commerce",
        "arlington",
        "rio",
        "el",
        "san",
        "dominguez",
        "fullerton",
        "northridge",
        "bakersfield",
        "riverside",
        "irvine",
        "davis",
        "santa",
        "corpus",
        "maryland",
        "md",
    }
)


def _words(name: str) -> list[str]:
    out: list[str] = []
    for raw in name.lower().replace("-", " ").replace("(", " ").replace(")", " ").split():
        tok = _FOLD.get(raw, raw)
        tok = "".join(ch for ch in tok if ch.isalnum())
        if tok:
            out.append(tok)
    return out


def same_team(card_name: str, spoken: str) -> bool:
    """Whether the card's Odds API name ("Duke Blue Devils") is the school a host said.

    The spoken school must be the start of the card's name, and the next word must
    not make it another school, so "Iowa" never answers to Iowa State Cyclones.
    """
    said = spoken.strip().lower().removeprefix("the ")
    c = _words(card_name)
    for name in {said, _SPOKEN.get(said, said)}:
        s = _words(name)
        if not s or len(s) > len(c) or c[: len(s)] != s:
            continue
        rest = c[len(s) :]
        if not rest or (rest[0] not in _QUALIFIERS and len(rest) <= 2):
            return True
    return False


def _tip(stamp: str) -> datetime | None:
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")) if stamp else None
    except ValueError:
        return None


def to_games(rows: Iterable[Row], teams: Teams, day: Date) -> list[Game[Rows]]:
    """The slate's games, one per event the card priced, named as the card names them."""
    by_event: dict[str, Rows] = {}
    for r in rows:
        by_event.setdefault(r.event_id, []).append(r)
    out: list[Game[Rows]] = []
    for eid, mine in by_event.items():
        home, away = teams.get(eid, ("", ""))
        if not home or not away:
            continue
        spread = next((r.line for r in mine if r.market == "game_ats" and r.side == "home"), None)
        out.append(Game(f"{away} @ {home}", home, away, _tip(mine[0].tip_utc), day, spread, mine))
    return out


def espn_games(games: Iterable[espn.Game], day: Date) -> list[Game[Rows]]:
    """ESPN's games of ``day``, unpriced: only to tell which game a pick was about."""
    return [Game(f"{g.away} @ {g.home}", g.home, g.away, _tip(g.start_utc), day) for g in games]


def _same_side(r: Row, pl: Placed) -> bool:
    if r.market != pl.pick.market:
        return False
    if pl.pick.market == "game_total":
        return r.side == pl.pick.side
    return r.side == pl.team_side


def engine_view(pl: Placed) -> str:
    """Whether the engine bought this side, the other side, or neither."""
    buys = [r for r in pl.game.rows or [] if r.is_buy and r.market == pl.pick.market]
    if not buys:
        return ""
    return "engine agrees" if any(_same_side(r, pl) for r in buys) else "engine disagrees"


LEAGUE: League[Rows] = League(CBB, same_team, engine_view, horizon=HORIZON)


def slate_picks(
    extractions: Iterable[Extraction],
    games: list[Game[Rows]],
    day: Date,
    earlier: Sequence[Game[Rows]] = (),
) -> SlatePicks:
    """CBB picks published in the two days before the slate's tips, placed on its games."""
    tips = [g.kickoff for g in games if g.kickoff is not None]
    start = datetime.combine(day, datetime.min.time(), timezone.utc)
    first, last = (min(tips), max(tips)) if tips else (start, start + timedelta(days=1))
    return core.slate_picks(
        extractions,
        games,
        LEAGUE,
        since=first - HORIZON,
        until=last,
        unmatched_since=start - timedelta(hours=12),
        earlier=earlier,
    )


def board_price(pl: Placed) -> float | None:
    """The engine's own execution price on this side at the host's number, if it priced it."""
    for r in pl.game.rows or []:
        if r.american is None or not _same_side(r, pl):
            continue
        if pl.pick.market == "game_ml" or (r.line is not None and r.line == pl.line):
            return r.american
    return None


def grade_row(pl: Placed, finals: Sequence[espn.Game]) -> dict[str, str]:
    p = pl.pick
    final = match_final(pl.game.home, pl.game.away, finals)
    # ESPN's final is (away, home).
    outcome = None if final is None else core.settle(pl, float(final[1]), float(final[0]))
    price, source = (p.price, "stated") if p.price is not None else (board_price(pl), "board")
    return core.ledger_row(
        pl,
        league=CBB,
        day=pl.game.day,
        outcome=outcome,
        price=price,
        price_source=source if price is not None else "",
        close_line=None,
        clv_pts=None,
        engine=engine_view(pl),
    )


def grade_day(
    store: Path,
    day: Date,
    rows: Iterable[Row],
    teams: Teams,
    finals: Sequence[espn.Game],
    *,
    ledger: Path | None = None,
    earlier: Sequence[Game[Rows]] = (),
) -> list[dict[str, str]]:
    """Grade ``day``'s placed picks on its ESPN finals; the rows written.

    A regrade replaces the day's rows, so a pick no longer placed leaves the ledger.
    """
    if not (store / PICKS_DIR).exists():
        return []
    games = to_games(rows, teams, day)
    if not games or not finals:
        return []
    slate = slate_picks(load_extractions(store), games, day, earlier)
    out = [grade_row(pl, finals) for pl in slate.placed]
    path = ledger or ledger_path(store, CBB)
    if out or any(r.get("date") == day.isoformat() for r in core.load_ledger(path)):
        core.update_ledger(path, out, {day})
    return out


def view(
    store: Path,
    games: list[Game[Rows]],
    day: Date,
    *,
    ledger: Path | None = None,
    earlier: Sequence[Game[Rows]] = (),
) -> PodcastView | None:
    """Everything the card prints about podcasts today; ``None`` if none were read."""
    if not (store / PICKS_DIR).exists():
        return None
    extractions = load_extractions(store)
    if not extractions:
        return None
    return render.PodcastView(
        slate_picks(extractions, games, day, earlier),
        core.records(core.load_ledger(ledger or ledger_path(store, CBB))),
    )


__all__ = [
    "HORIZON",
    "Game",
    "Rows",
    "LEAGUE",
    "PodcastView",
    "board_price",
    "engine_view",
    "espn_games",
    "grade_day",
    "grade_row",
    "same_team",
    "slate_picks",
    "to_games",
    "view",
]
