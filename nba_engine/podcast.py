"""The NBA adapter for the shared podcast picks (:mod:`engine_common.podcasts`).

Supplies what is NBA about a pick: team names, the slate's games from the
card's own ledger rows, the engine's edge on a side, and grading against the
finals the graded ledger carries. Display and audit only: a podcast pick is
never written to the engine's ledger and never reaches a price, a tier or a gate.

A spread or total with no number said is not graded. A price is the one the
host said or, failing that, the engine's own execution price at the host's
number (``board``); with neither, the pick counts in the won-lost record but
carries no units.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from pathlib import Path

from engine_common.podcasts import picks as core
from engine_common.podcasts import render
from engine_common.podcasts.extract import PICKS_DIR, Extraction, load_extractions
from engine_common.podcasts.picks import Game, League, ledger_path
from engine_common.podcasts.shows import NBA
from nba_engine.audit.ledger import LedgerRow
from nba_engine.data.teamnames import BY_NAME, CODES, canonical, code_for
from nba_engine.output.card import SlateCard, build_card

Rows = list[LedgerRow]
Placed = core.Placed[Rows]
SlatePicks = core.SlatePicks[Rows]
PodcastView = render.PodcastView[Rows]

# A pick for tonight is said within the two days before tip.
HORIZON = timedelta(days=2)

_NICKNAMES: dict[str, str] = {full.split()[-1]: code for full, code in BY_NAME.items()}
_CITIES: dict[str, str] = {
    "atlanta": "ATL",
    "boston": "BOS",
    "brooklyn": "BKN",
    "charlotte": "CHA",
    "chicago": "CHI",
    "cleveland": "CLE",
    "dallas": "DAL",
    "denver": "DEN",
    "detroit": "DET",
    "golden state": "GSW",
    "houston": "HOU",
    "indiana": "IND",
    "memphis": "MEM",
    "miami": "MIA",
    "milwaukee": "MIL",
    "minnesota": "MIN",
    "new orleans": "NOP",
    "new york": "NYK",
    "oklahoma city": "OKC",
    "okc": "OKC",
    "orlando": "ORL",
    "philadelphia": "PHI",
    "philly": "PHI",
    "phoenix": "PHX",
    "portland": "POR",
    "sacramento": "SAC",
    "san antonio": "SAS",
    "toronto": "TOR",
    "utah": "UTA",
    "washington": "WAS",
}
# How hosts say a team that the board's names do not cover.
_SPOKEN: dict[str, str] = {
    "sixers": "PHI",
    "trail blazers": "POR",
    "cavs": "CLE",
    "mavs": "DAL",
    "wolves": "MIN",
    "t-wolves": "MIN",
    "pels": "NOP",
    "nugs": "DEN",
    "grizz": "MEM",
    "dubs": "GSW",
    "clips": "LAC",
}


def team_code(spoken: str) -> str | None:
    """The NBA tricode for a spoken team, or ``None`` if it is not exactly one team.

    "Los Angeles" alone is two teams, so it places nowhere.
    """
    name = spoken.strip().lower().removeprefix("the ")
    if not name:
        return None
    if name.upper() in CODES or canonical(name) in CODES:
        return canonical(name)
    for table in (_SPOKEN, _NICKNAMES, _CITIES):
        if name in table:
            return table[name]
    code = code_for(name)
    if code is not None:
        return code
    last = name.split()[-1]
    if last != name:
        return _SPOKEN.get(last) or _NICKNAMES.get(last)
    return None


def same_team(card_code: str, spoken: str) -> bool:
    code = team_code(spoken)
    return code is not None and canonical(code) == canonical(card_code)


def _tip(stamp: str) -> datetime | None:
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def to_games(card: SlateCard) -> list[Game[Rows]]:
    """The slate's games as the card prints them, from the card's own rows."""
    out: list[Game[Rows]] = []
    for g in card.games:
        spread = next((r.line for r in g.main("game_ats") if r.side == g.home), None)
        out.append(
            Game(
                matchup=g.matchup,
                home=g.home,
                away=g.away,
                kickoff=_tip(g.tip_utc),
                day=card.day,
                home_spread=spread,
                rows=list(g.rows),
            )
        )
    return out


def _same_side(r: LedgerRow, pl: Placed) -> bool:
    if r.entity or r.market != pl.pick.market:
        return False
    if pl.pick.market == "game_total":
        return r.side == pl.pick.side
    return canonical(r.side) == canonical(pl.team)


def engine_view(pl: Placed) -> str:
    """Whether the engine bought this side, the other side, or neither."""
    buys = [r for r in pl.game.rows or [] if r.is_buy and r.market == pl.pick.market]
    buys = [r for r in buys if not r.entity]
    if not buys:
        return ""
    return "engine agrees" if any(_same_side(r, pl) for r in buys) else "engine disagrees"


def engine_edge(pl: Placed) -> render.EngineEdge | None:
    """The engine's model-minus-fair on this side, at the host's number or the nearest."""
    mine = [
        (r.line, r.edge) for r in pl.game.rows or [] if _same_side(r, pl) and r.edge is not None
    ]
    return render.nearest_edge(pl.line, mine)


LEAGUE: League[Rows] = League(NBA, same_team, engine_view, horizon=HORIZON)


def slate_picks(extractions: Iterable[Extraction], card: SlateCard) -> SlatePicks:
    """NBA picks published in the two days before the slate's tips, placed on its games."""
    games = to_games(card)
    tips = [g.kickoff for g in games if g.kickoff is not None]
    start = datetime.combine(card.day, datetime.min.time(), timezone.utc)
    first, last = (min(tips), max(tips)) if tips else (start, start + timedelta(days=1))
    return core.slate_picks(
        extractions,
        games,
        LEAGUE,
        since=first - HORIZON,
        until=last,
        unmatched_since=start - timedelta(hours=12),
    )


def board_price(pl: Placed) -> float | None:
    """The engine's own execution price on this side at the host's number, if it priced it."""
    for r in pl.game.rows or []:
        if r.american is None or not _same_side(r, pl):
            continue
        if pl.pick.market == "game_ml" or (r.line is not None and r.line == pl.line):
            return r.american
    return None


def _final(rows: Sequence[LedgerRow]) -> tuple[float, float] | None:
    for r in rows:
        if r.home_score is not None and r.away_score is not None:
            return float(r.home_score), float(r.away_score)
    return None


def grade_row(pl: Placed) -> dict[str, str]:
    p = pl.pick
    final = _final(pl.game.rows or [])
    outcome = None if final is None else core.settle(pl, final[0], final[1])
    price, source = (p.price, "stated") if p.price is not None else (board_price(pl), "board")
    return core.ledger_row(
        pl,
        league=NBA,
        day=pl.game.day,
        outcome=outcome,
        price=price,
        price_source=source if price is not None else "",
        close_line=None,
        clv_pts=None,
        engine=engine_view(pl),
    )


def grade_day(
    store: Path, day: Date, graded: Iterable[LedgerRow], *, ledger: Path | None = None
) -> list[dict[str, str]]:
    """Grade ``day``'s placed picks off its graded ledger rows; the rows written."""
    if not (store / PICKS_DIR).exists():
        return []
    rows = list(graded)
    if not rows:
        return []
    slate = slate_picks(load_extractions(store), build_card(rows, day=day))
    if not slate.placed:
        return []
    out = [grade_row(pl) for pl in slate.placed]
    core.update_ledger(ledger or ledger_path(store, NBA), out, {day})
    return out


def view(store: Path, card: SlateCard, *, ledger: Path | None = None) -> PodcastView | None:
    """Everything the card prints about podcasts tonight; ``None`` if none were read."""
    if not (store / PICKS_DIR).exists():
        return None
    extractions = load_extractions(store)
    if not extractions:
        return None
    return render.PodcastView(
        slate_picks(extractions, card),
        core.records(core.load_ledger(ledger or ledger_path(store, NBA))),
        engine_edge,
    )


__all__ = [
    "LEAGUE",
    "PodcastView",
    "board_price",
    "engine_edge",
    "engine_view",
    "grade_day",
    "grade_row",
    "same_team",
    "slate_picks",
    "team_code",
    "to_games",
    "view",
]
