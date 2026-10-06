"""Put podcast picks on a slate's games, keep their ledger, and count their record.

League-free. An engine describes its slate as :class:`Game` objects and supplies
a :class:`League` adapter -- how a spoken team name matches its card's label,
and what the engine itself thought of the side -- and everything else is here:

* A pick belongs to a game when its team plays in it, the game kicks off after
  the episode was published and within the adapter's horizon of it. One that
  fits no game, or more than one, is listed as unmatched rather than forced.
* A show repeating an opinion is one pick; an official bet supersedes a lean.
* Each league keeps its own graded ledger. A price is the one the host said or
  the board's at the host's number (``price_source``); without one a pick counts
  in the won-lost record but carries no units, so ROI is over priced picks only.
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generic, TypeVar

from engine_common.podcasts.extract import Extraction, Pick

R = TypeVar("R")

HORIZON = timedelta(days=8)
BREAK_EVEN = 110 / 210
WIN, LOSS, PUSH = "win", "loss", "push"
MARKET_LABEL = {"game_ml": "ML", "game_ats": "ATS", "game_total": "Total"}


@dataclass(frozen=True)
class Game(Generic[R]):
    """One game on a card, as the card labels it. ``rows`` is the engine's own."""

    matchup: str
    home: str
    away: str
    kickoff: datetime | None
    day: Date
    home_spread: float | None = None  # the card's handicap on the home side
    rows: R | None = field(default=None, hash=False, compare=False)


@dataclass(frozen=True)
class Placed(Generic[R]):
    """A pick on one of the card's games, from the picked team's side."""

    pick: Pick
    game: Game[R] = field(hash=False, compare=False)
    team_side: str | None  # "home" | "away" for ML/ATS; None for totals
    line: float | None  # the side's own handicap (spread sign fixed to the card)

    @property
    def matchup(self) -> str:
        return self.game.matchup

    @property
    def team(self) -> str:
        return self.game.home if self.team_side == "home" else self.game.away

    @property
    def label(self) -> str:
        p = self.pick
        if p.market == "game_total":
            num = f" {self.line:g}" if self.line is not None else ""
            return f"{(p.side or '').title()}{num}"
        if p.market == "game_ml":
            return f"{self.team} ML"
        num = f" {self.line:+g}" if self.line is not None else ""
        return f"{self.team}{num}"

    @property
    def direction(self) -> str:
        """The side taken, independent of the number: one opinion per game and market."""
        return self.pick.side or "" if self.pick.market == "game_total" else self.team_side or ""


def _no_view(_: Placed[R]) -> str:
    return ""


@dataclass(frozen=True)
class League(Generic[R]):
    """What one engine supplies: its key, team matching and its own view of a side."""

    key: str
    same_team: Callable[[str, str], bool]
    engine_view: Callable[[Placed[R]], str] = _no_view
    horizon: timedelta = HORIZON
    # Picks read before picks carried a league were this league's (CFB).
    legacy: bool = False

    def owns(self, p: Pick) -> bool:
        return p.league == self.key or (self.legacy and not p.league)


def _team_side(game: Game[R], p: Pick, league: League[R]) -> str | None:
    home = league.same_team(game.home, p.team or "")
    away = league.same_team(game.away, p.team or "")
    if home == away:
        return None
    return "home" if home else "away"


def place(p: Pick, games: Iterable[Game[R]], league: League[R]) -> Placed[R] | None:
    """The one game ``p`` is about, or ``None`` if none or more than one fits."""
    if p.market not in MARKET_LABEL or not p.team or not league.owns(p):
        return None
    published = datetime.fromisoformat(p.published)
    found: list[tuple[Game[R], str]] = []
    for game in games:
        kick = game.kickoff
        if kick is None or not published < kick <= published + league.horizon:
            continue
        side = _team_side(game, p, league)
        if side is not None:
            found.append((game, side))
    if len(found) > 1 and p.opponent:
        # Only to choose between games: a spoken opponent is misheard as often
        # as a team, so it never vetoes the one game the team fits.
        opp = p.opponent
        found = [
            (g, sd) for g, sd in found if league.same_team(g.away if sd == "home" else g.home, opp)
        ]
    if len(found) != 1:
        return None
    game, side = found[0]
    if p.market == "game_total":
        return Placed(p, game, None, p.line)
    line = p.line
    if p.market == "game_ats" and line is not None and game.home_spread is not None:
        card = game.home_spread if side == "home" else -game.home_spread
        # The number is checked against the audio; its sign is the model's
        # reading of who is favoured, which the card knows better.
        if abs(card) >= 3 and (card > 0) != (line > 0) and line != 0:
            line = -line
    return Placed(p, game, side, line)


def counted(placed: Iterable[Placed[R]]) -> list[Placed[R]]:
    """One pick per host, game, market and side: the earliest official, else the earliest lean.

    A show repeating a pick across its Tuesday and Friday episodes is one bet.
    """
    best: dict[tuple[str, str, str, str, str], Placed[R]] = {}
    for pl in sorted(placed, key=lambda x: (x.pick.published, x.pick.seconds)):
        p = pl.pick
        k = (p.show, p.host or "", pl.matchup, p.market, pl.direction)
        held = best.get(k)
        if held is None or (held.pick.kind == "lean" and p.kind == "official"):
            best[k] = pl
    return sorted(best.values(), key=lambda x: (x.pick.published, x.pick.seconds))


@dataclass
class SlatePicks(Generic[R]):
    """Every pick for one slate: on a game, or not on this card."""

    placed: list[Placed[R]]
    unmatched: list[Pick]

    def for_game(self, matchup: str) -> list[Placed[R]]:
        return [pl for pl in self.placed if pl.matchup == matchup]


def slate_picks(
    extractions: Iterable[Extraction],
    games: list[Game[R]],
    league: League[R],
    *,
    since: datetime,
    until: datetime,
    unmatched_since: datetime,
) -> SlatePicks[R]:
    """The league's picks published in ``[since, until)``, placed on ``games``.

    A pick that fits no game is listed only if published after ``unmatched_since``,
    so last week's other-slate picks don't crowd this week's card.
    """
    placed: list[Placed[R]] = []
    unmatched: list[Pick] = []
    for ex in extractions:
        for p in ex.picks:
            if not league.owns(p):
                continue
            published = datetime.fromisoformat(p.published)
            if not since <= published < until:
                continue
            pl = place(p, games, league)
            if pl is not None:
                placed.append(pl)
            elif published >= unmatched_since:
                unmatched.append(p)
    return SlatePicks(counted(placed), unmatched)


# --------------------------------------------------------------- grading


def settle(pl: Placed[R], home_score: float, away_score: float) -> str | None:
    """Win, loss or push at the host's number; ``None`` where no number was said."""
    p = pl.pick
    if p.market == "game_total":
        if pl.line is None or p.side not in ("over", "under"):
            return None
        diff = home_score + away_score - pl.line
        diff = diff if p.side == "over" else -diff
    else:
        margin = home_score - away_score if pl.team_side == "home" else away_score - home_score
        if p.market == "game_ml":
            diff = margin
        elif p.market == "game_ats" and pl.line is not None:
            diff = margin + pl.line
        else:
            return None
    return WIN if diff > 0 else LOSS if diff < 0 else PUSH


def american_to_decimal(price: float) -> float:
    return 1.0 + (price / 100.0 if price > 0 else 100.0 / -price)


def stake_of(p: Pick) -> float:
    return p.units if p.units is not None and p.units > 0 else 1.0


def units_won(outcome: str | None, price: float | None, stake: float) -> float | None:
    if outcome is None or price is None:
        return None
    return {WIN: stake * (american_to_decimal(price) - 1.0), LOSS: -stake, PUSH: 0.0}.get(outcome)


def line_clv(pl: Placed[R], close_line: float | None) -> float | None:
    """Points of line value against the close, on a spread or a total."""
    if close_line is None or pl.line is None:
        return None
    if pl.pick.market == "game_ats":
        return pl.line - close_line
    if pl.pick.market == "game_total":
        return close_line - pl.line if pl.pick.side == "over" else pl.line - close_line
    return None


LEDGER_FIELDS = (
    "date",
    "league",
    "pick_id",
    "show",
    "show_name",
    "host",
    "kind",
    "market",
    "matchup",
    "selection",
    "line",
    "price",
    "price_source",
    "stake",
    "result",
    "units",
    "close_line",
    "clv_pts",
    "engine",
    "published",
    "episode_title",
    "stamp",
    "quote",
    "edge",
    "fair_line",
    "reason",
)


def ledger_row(
    pl: Placed[R],
    *,
    league: str,
    day: Date,
    outcome: str | None,
    price: float | None,
    price_source: str,
    close_line: float | None,
    clv_pts: float | None,
    engine: str,
) -> dict[str, str]:
    """One graded pick as a ledger row."""
    p = pl.pick
    stake = stake_of(p)
    units = units_won(outcome, price, stake)
    return {
        "date": day.isoformat(),
        "league": league,
        "pick_id": p.pick_id,
        "show": p.show,
        "show_name": p.show_name,
        "host": p.host or "",
        "kind": p.kind,
        "market": p.market,
        "matchup": pl.matchup,
        "selection": pl.label,
        "line": "" if pl.line is None else f"{pl.line:g}",
        "price": "" if price is None else f"{price:+.0f}",
        "price_source": price_source if price is not None else "",
        "stake": f"{stake:g}",
        "result": outcome or "",
        "units": "" if units is None else f"{units:.3f}",
        "close_line": "" if close_line is None else f"{close_line:g}",
        "clv_pts": "" if clv_pts is None else f"{clv_pts:g}",
        "engine": engine,
        "published": p.published,
        "episode_title": p.episode_title,
        "stamp": p.stamp,
        "quote": p.quote,
        "edge": p.edge,
        "fair_line": "" if p.fair_line is None else f"{p.fair_line:g}",
        "reason": p.reason,
    }


def load_ledger(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def update_ledger(
    path: Path, rows: list[dict[str, str]], dates: Iterable[Date]
) -> list[dict[str, str]]:
    """Replace the rows on ``dates`` (a re-audit is authoritative for them), keep the rest."""
    drop = {d.isoformat() for d in dates}
    kept = [r for r in load_ledger(path) if r.get("date") not in drop]
    merged = sorted([*kept, *rows], key=lambda r: (r["date"], r["published"], r["pick_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(LEDGER_FIELDS), extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)
    return merged


def ledger_path(root: Path, league: str) -> Path:
    """One league's graded podcast picks, in the shared store."""
    return root / "ledgers" / f"{league}.csv"


# --------------------------------------------------------------- records


@dataclass(frozen=True)
class Record:
    label: str
    wins: int
    losses: int
    pushes: int
    priced: int
    units: float
    staked: float
    clv_n: int
    clv_mean: float | None

    @property
    def decided(self) -> int:
        return self.wins + self.losses

    @property
    def n(self) -> int:
        return self.decided + self.pushes

    @property
    def win_pct(self) -> float | None:
        return self.wins / self.decided if self.decided else None

    @property
    def se(self) -> float | None:
        return math.sqrt(0.25 / self.decided) if self.decided else None

    @property
    def roi(self) -> float | None:
        return self.units / self.staked if self.staked else None

    @property
    def underpowered(self) -> bool:
        """Too few results to tell from a coin flip at -110 (within 2 SE of break-even)."""
        if self.win_pct is None or self.se is None:
            return True
        return abs(self.win_pct - BREAK_EVEN) < 2 * self.se

    @property
    def wlp(self) -> str:
        return f"{self.wins}-{self.losses}" + (f"-{self.pushes}" if self.pushes else "")


def record(label: str, rows: Iterable[dict[str, str]]) -> Record:
    w = l_ = pu = priced = clv_n = 0
    units = staked = clv_sum = 0.0
    for r in rows:
        res = r.get("result", "")
        if res == WIN:
            w += 1
        elif res == LOSS:
            l_ += 1
        elif res == PUSH:
            pu += 1
        if res and r.get("units"):
            priced += 1
            units += float(r["units"])
            staked += float(r.get("stake") or 1.0)
        if r.get("clv_pts"):
            clv_n += 1
            clv_sum += float(r["clv_pts"])
    return Record(
        label, w, l_, pu, priced, units, staked, clv_n, clv_sum / clv_n if clv_n else None
    )


_NUMBER = re.compile(r"\s+(ML|[+-]?\d+(\.\d+)?)$")


def show_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """A show's record counts each opinion once, however many hosts shared it."""
    seen: dict[tuple[str, ...], dict[str, str]] = {}
    for r in sorted(rows, key=lambda r: r["published"]):
        side = _NUMBER.sub("", r["selection"])
        k = (r.get("league", ""), r["show"], r["kind"], r["date"], r["matchup"], r["market"], side)
        seen.setdefault(k, r)
    return list(seen.values())


@dataclass
class Records:
    shows: list[tuple[Record, Record]]  # (official, lean) per show
    hosts: list[tuple[Record, Record]]  # per show/host
    markets: list[Record]  # official picks by show and market

    def official_for(self, show: str, host: str | None) -> Record | None:
        for off, _ in self.hosts if host else self.shows:
            if off.label == (f"{show} · {host}" if host else show):
                return off
        return None


def records(ledger: list[dict[str, str]]) -> Records:
    """One league's records: by show, by host the episode named, and by market."""
    graded = [r for r in ledger if r.get("result")]
    shows: list[tuple[Record, Record]] = []
    hosts: list[tuple[Record, Record]] = []
    markets: list[Record] = []
    for name in sorted({r["show_name"] for r in graded}):
        mine = [r for r in graded if r["show_name"] == name]
        uniq = show_rows(mine)
        shows.append(
            (
                record(name, [r for r in uniq if r["kind"] == "official"]),
                record(name, [r for r in uniq if r["kind"] == "lean"]),
            )
        )
        for host in sorted({r["host"] for r in mine if r["host"]}):
            hr = [r for r in mine if r["host"] == host]
            hosts.append(
                (
                    record(f"{name} · {host}", [r for r in hr if r["kind"] == "official"]),
                    record(f"{name} · {host}", [r for r in hr if r["kind"] == "lean"]),
                )
            )
        for market, short in MARKET_LABEL.items():
            mr = [r for r in uniq if r["kind"] == "official" and r["market"] == market]
            if mr:
                markets.append(record(f"{name} · {short}", mr))
    return Records(shows, hosts, markets)


__all__ = [
    "BREAK_EVEN",
    "Game",
    "HORIZON",
    "LEDGER_FIELDS",
    "LOSS",
    "League",
    "MARKET_LABEL",
    "PUSH",
    "Placed",
    "Record",
    "Records",
    "SlatePicks",
    "WIN",
    "counted",
    "ledger_path",
    "ledger_row",
    "line_clv",
    "load_ledger",
    "place",
    "record",
    "records",
    "settle",
    "show_rows",
    "slate_picks",
    "stake_of",
    "units_won",
    "update_ledger",
]
