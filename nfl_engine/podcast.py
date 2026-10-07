"""The NFL adapter for the shared podcast picks (:mod:`engine_common.podcasts`).

Supplies what is NFL about a pick: team names, the week's games from the
engine's own ledger rows, the engine's view of a side, and grading against
nflverse's final scores and closing lines. Display and audit only: a podcast pick
is never written to the engine's ledger and never reaches a price, a tier or a
screen.

A spread or total with no number said is not graded. A price is the one the
host said or, failing that, the engine's own board price at the host's number
(``board``); with neither, the pick counts in the won-lost record but carries no
units. CLV is in points against nflverse's closing line, which is the
pre-kickoff number.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date as Date
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from engine_common.podcasts import picks as core
from engine_common.podcasts import render
from engine_common.podcasts.extract import PICKS_DIR, Extraction, load_extractions
from engine_common.podcasts.picks import HORIZON, Game, League, ledger_path
from engine_common.podcasts.shows import NFL
from nfl_engine.audit.ledger import ENGINE, LedgerEntry
from nfl_engine.data.teamnames import BY_NAME, canonical, code_for, is_team
from nfl_engine.output.card import market_reads

Rows = list[LedgerEntry]
Placed = core.Placed[Rows]
SlatePicks = core.SlatePicks[Rows]
PodcastView = render.PodcastView[Rows]

# Pick market -> the engine's ledger market.
MARKET = {"game_ml": "moneyline", "game_ats": "spread", "game_total": "total"}
# How hosts say a team that the board's names do not cover.
_SPOKEN: dict[str, str] = {
    "niners": "SF",
    "bucs": "TB",
    "pats": "NE",
    "jags": "JAX",
    "commanders": "WAS",
    "skins": "WAS",
    "philly": "PHI",
    "big blue": "NYG",
    "bolts": "LAC",
}


def team_code(spoken: str) -> str | None:
    """The nflverse code for a spoken team, or ``None`` if it is not exactly one team.

    "New York" or "Los Angeles" alone is two teams, so it places nowhere.
    """
    name = spoken.strip().lower().removeprefix("the ")
    if not name:
        return None
    if name in _SPOKEN:
        return _SPOKEN[name]
    if is_team(name) and len(name) <= 3:
        # "LA" alone is the Rams' code but either Los Angeles club when spoken.
        return None if name == "la" else canonical(name)
    code = code_for(name)
    if code is not None:
        return code
    cities = {c for full, c in BY_NAME.items() if full.startswith(f"{name} ")}
    if len(cities) == 1:
        return cities.pop()
    last = name.split()[-1]
    if last != name:
        return _SPOKEN.get(last) or code_for(last)
    return None


def same_team(card_code: str, spoken: str) -> bool:
    code = team_code(spoken)
    return code is not None and canonical(code) == canonical(card_code)


def _kickoff(rows: Rows) -> datetime | None:
    stamp = next((e.kickoff_utc for e in rows if e.kickoff_utc), "")
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def to_games(entries: Iterable[LedgerEntry], *, season: int, week: int) -> list[Game[Rows]]:
    """The week's games as the card prints them, from the engine's own rows."""
    by_game: dict[str, Rows] = {}
    for e in entries:
        if e.season == season and e.week == week and e.source == ENGINE:
            by_game.setdefault(e.matchup, []).append(e)
    out: list[Game[Rows]] = []
    for matchup, rows in by_game.items():
        away, _, home = matchup.partition(" @ ")
        spread = next((r.line for r in market_reads(rows) if r.market == "spread"), None)
        out.append(
            Game(
                matchup=matchup,
                home=home.strip(),
                away=away.strip(),
                kickoff=_kickoff(rows),
                day=Date.fromisoformat(rows[0].date[:10]),
                home_spread=spread,
                rows=rows,
            )
        )
    return out


def _side(pl: Placed) -> str:
    return (pl.pick.side or "") if pl.pick.market == "game_total" else canonical(pl.team)


def _same_side(e: LedgerEntry, pl: Placed) -> bool:
    side = _side(pl)
    return e.side == side if pl.pick.market == "game_total" else canonical(e.side) == side


def engine_view(pl: Placed) -> str:
    """Whether the engine played this side, the other side, or neither."""
    market = MARKET.get(pl.pick.market)
    plays = [e for e in pl.game.rows or [] if e.market == market and not e.screens]
    if not plays:
        return ""
    return "engine agrees" if any(_same_side(e, pl) for e in plays) else "engine disagrees"


def engine_edge(pl: Placed) -> render.EngineEdge | None:
    """The engine's model-minus-fair on this side, at the host's number or the nearest."""
    market = MARKET.get(pl.pick.market)
    mine = [
        (e.line, e.model_prob - e.fair_prob)
        for e in pl.game.rows or []
        if e.market == market
        and _same_side(e, pl)
        and e.model_prob is not None
        and e.fair_prob is not None
    ]
    return render.nearest_edge(pl.line, mine)


LEAGUE: League[Rows] = League(NFL, same_team, engine_view, horizon=HORIZON)


def slate_picks(
    extractions: Iterable[Extraction], entries: Iterable[LedgerEntry], *, season: int, week: int
) -> SlatePicks:
    """NFL picks published in the eight days before each of the week's kickoffs."""
    games = to_games(entries, season=season, week=week)
    kicks = [g.kickoff for g in games if g.kickoff is not None]
    if not kicks:
        return SlatePicks([], [])
    first, last = min(kicks), max(kicks)
    return core.slate_picks(
        extractions,
        games,
        LEAGUE,
        since=first - HORIZON,
        until=last,
        unmatched_since=first - timedelta(days=7),
    )


def board_price(pl: Placed) -> float | None:
    """The engine's own price on this side at the host's number, if it priced that rung."""
    market = MARKET.get(pl.pick.market)
    for e in pl.game.rows or []:
        if e.market != market or e.odds is None or not _same_side(e, pl):
            continue
        if market == "moneyline" or (e.line is not None and e.line == pl.line):
            return e.odds
    return None


Finals = dict[tuple[int, int, str, str], tuple[float, float, float | None, float | None]]


def finals(frame: pd.DataFrame) -> Finals:
    """(season, week, away, home) -> (home score, away score, close spread, close total).

    nflverse's ``spread_line`` is the home side's expected margin, so the home
    handicap at the close is its negative.
    """
    out: Finals = {}
    if frame.empty:
        return out
    for r in frame.itertuples(index=False):
        if pd.isna(r.home_score) or pd.isna(r.away_score):
            continue
        spread = None if pd.isna(r.spread_line) else float(r.spread_line)
        total = None if pd.isna(r.total_line) else float(r.total_line)
        key = (int(r.season), int(r.week), canonical(str(r.away_team)), canonical(str(r.home_team)))
        out[key] = (float(r.home_score), float(r.away_score), spread, total)
    return out


def grade_row(
    pl: Placed, result: tuple[float, float, float | None, float | None] | None
) -> dict[str, str]:
    p = pl.pick
    outcome = None if result is None else core.settle(pl, result[0], result[1])
    close: float | None = None
    if result is not None:
        if p.market == "game_ats" and result[2] is not None:
            close = -result[2] if pl.team_side == "home" else result[2]
        elif p.market == "game_total":
            close = result[3]
    price, source = (p.price, "stated") if p.price is not None else (board_price(pl), "board")
    return core.ledger_row(
        pl,
        league=NFL,
        day=pl.game.day,
        outcome=outcome,
        price=price,
        price_source=source,
        close_line=close,
        clv_pts=core.line_clv(pl, close),
        engine=engine_view(pl),
    )


def grade_week(
    store: Path,
    entries: list[LedgerEntry],
    *,
    season: int,
    week: int,
    frame: pd.DataFrame,
    ledger: Path | None = None,
) -> list[dict[str, str]]:
    """Grade the week's placed picks into the NFL podcast ledger; the rows written."""
    if not (store / PICKS_DIR).exists():
        return []
    slate = slate_picks(load_extractions(store), entries, season=season, week=week)
    if not slate.placed:
        return []
    done = finals(frame)
    rows = []
    for pl in slate.placed:
        key = (season, week, canonical(pl.game.away), canonical(pl.game.home))
        rows.append(grade_row(pl, done.get(key)))
    days = {pl.game.day for pl in slate.placed}
    core.update_ledger(ledger or ledger_path(store, NFL), rows, days)
    return rows


def view(
    store: Path, entries: list[LedgerEntry], *, season: int, week: int, ledger: Path | None = None
) -> PodcastView | None:
    """Everything the card prints about podcasts this week; ``None`` if none were read."""
    if not (store / PICKS_DIR).exists():
        return None
    extractions = load_extractions(store)
    if not extractions:
        return None
    return render.PodcastView(
        slate_picks(extractions, entries, season=season, week=week),
        core.records(core.load_ledger(ledger or ledger_path(store, NFL))),
        engine_edge,
    )


__all__ = [
    "LEAGUE",
    "MARKET",
    "PodcastView",
    "board_price",
    "engine_edge",
    "engine_view",
    "finals",
    "grade_row",
    "grade_week",
    "same_team",
    "slate_picks",
    "team_code",
    "to_games",
    "view",
]
