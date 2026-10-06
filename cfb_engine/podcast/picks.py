"""Put podcast picks on the slate's games, grade them, and keep their record.

A pick belongs to a game when its team plays in it, the game kicks off after
the episode was published and within :data:`HORIZON` of it. Matching is by the
card's own team labels, so a pick lands only on a game the card prints; one
that fits no game, or more than one, is listed as unmatched rather than forced.

Grading reuses the engine's settlement (:func:`cfb_engine.audit.grade.grade`) at
the host's own number. A spread or total with no number said is not graded. A
price is the one the host said or, failing that, the day-of board's price at
the host's number (labelled ``board``); with neither, the pick counts in the
won-lost record but carries no units, so ROI is over priced picks only.
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cfb_engine.audit import snapshot
from cfb_engine.audit.clv import ClosingQuote, compute_clv
from cfb_engine.audit.grade import LOSS, PUSH, WIN, ResultIndex, grade, result_for
from cfb_engine.data.teamnames import label_matches, school_key
from cfb_engine.market.odds import american_to_decimal
from cfb_engine.market.tiers import Tier
from cfb_engine.podcast.extract import Extraction, Pick
from cfb_engine.recommendations import Recommendation

HORIZON = timedelta(days=8)
BREAK_EVEN = 110 / 210
MARKET_LABEL = {"game_ml": "ML", "game_ats": "ATS", "game_total": "Total"}


def same_school(card_label: str, spoken: str) -> bool:
    """Whether the card's label and a spoken school name are one school.

    Exact on the canonical school key; the card's prefix rule only for a label
    the card cut at 14 characters, so "Georgia" never answers to Georgia State.
    """
    if not card_label or not spoken:
        return False
    if school_key(card_label) == school_key(spoken):
        return True
    return len(card_label.strip()) >= 14 and label_matches(card_label, spoken)


def _kickoff(recs: list[Recommendation]) -> datetime | None:
    stamp = recs[0].kickoff_utc
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass(frozen=True)
class Placed:
    """A pick on one of the card's games, from the picked team's side."""

    pick: Pick
    game: list[Recommendation] = field(hash=False, compare=False)
    team_side: str | None  # "home" | "away" for ML/ATS; None for totals
    line: float | None  # the side's own handicap (spread sign fixed to the card)

    @property
    def matchup(self) -> str:
        return self.game[0].matchup

    @property
    def label(self) -> str:
        p = self.pick
        r = self.game[0]
        if p.market == "game_total":
            num = f" {self.line:g}" if self.line is not None else ""
            return f"{(p.side or '').title()}{num}"
        team = r.home_abbrev if self.team_side == "home" else r.away_abbrev
        if p.market == "game_ml":
            return f"{team} ML"
        num = f" {self.line:+g}" if self.line is not None else ""
        return f"{team}{num}"

    @property
    def direction(self) -> str:
        """The side taken, independent of the number: one opinion per game and market."""
        return self.pick.side or "" if self.pick.market == "game_total" else self.team_side or ""


def _team_side(game: list[Recommendation], p: Pick) -> str | None:
    r = game[0]
    home = same_school(r.home_abbrev or "", p.team or "")
    away = same_school(r.away_abbrev or "", p.team or "")
    if home == away:
        return None
    return "home" if home else "away"


def _opponent_fits(game: list[Recommendation], side: str, opponent: str) -> bool:
    r = game[0]
    return same_school((r.away_abbrev if side == "home" else r.home_abbrev) or "", opponent)


def _card_line(game: list[Recommendation], team_side: str) -> float | None:
    for r in game:
        if r.market == "game_ats" and r.team_side == team_side:
            return r.line
    return None


def place(p: Pick, games: dict[str, list[Recommendation]]) -> Placed | None:
    """The one game ``p`` is about, or ``None`` if none or more than one fits."""
    if p.market not in MARKET_LABEL or not p.team:
        return None
    published = datetime.fromisoformat(p.published)
    found: list[tuple[list[Recommendation], str]] = []
    for game in games.values():
        kick = _kickoff(game)
        if kick is None or not published < kick <= published + HORIZON:
            continue
        side = _team_side(game, p)
        if side is not None:
            found.append((game, side))
    if len(found) > 1 and p.opponent:
        # Only to choose between games: a spoken opponent is misheard as often
        # as a team, so it never vetoes the one game the team fits.
        found = [(g, sd) for g, sd in found if _opponent_fits(g, sd, p.opponent)]
    if len(found) != 1:
        return None
    game, side = found[0]
    if p.market == "game_total":
        return Placed(p, game, None, p.line)
    line = p.line
    if p.market == "game_ats" and line is not None:
        card = _card_line(game, side)
        # The number is checked against the audio; its sign is the model's
        # reading of who is favoured, which the card knows better.
        if card is not None and abs(card) >= 3 and (card > 0) != (line > 0) and line != 0:
            line = -line
    return Placed(p, game, side if p.market != "game_total" else None, line)


def by_game(recs: list[Recommendation]) -> dict[str, list[Recommendation]]:
    out: dict[str, list[Recommendation]] = {}
    for r in recs:
        out.setdefault(r.matchup, []).append(r)
    return out


def counted(placed: Iterable[Placed]) -> list[Placed]:
    """One pick per host, game, market and side: the earliest official, else the earliest lean.

    A show repeating a pick across its Tuesday and Friday episodes is one bet.
    """
    best: dict[tuple[str, str, str, str, str], Placed] = {}
    for pl in sorted(placed, key=lambda x: (x.pick.published, x.pick.seconds)):
        p = pl.pick
        k = (p.show, p.host or "", pl.matchup, p.market, pl.direction)
        held = best.get(k)
        if held is None or (held.pick.kind == "lean" and p.kind == "official"):
            best[k] = pl
    return sorted(best.values(), key=lambda x: (x.pick.published, x.pick.seconds))


@dataclass
class SlatePicks:
    """Every pick for one slate: on a game, or not on this card."""

    placed: list[Placed]
    unmatched: list[Pick]

    def for_game(self, matchup: str) -> list[Placed]:
        return [pl for pl in self.placed if pl.matchup == matchup]


def slate_picks(
    extractions: Iterable[Extraction], recs: list[Recommendation], day: Date
) -> SlatePicks:
    """Picks published in the week before ``day``'s slate, placed on its games."""
    games = by_game(recs)
    lo = datetime.combine(day, datetime.min.time(), timezone.utc) - HORIZON
    hi = datetime.combine(day + timedelta(days=1), datetime.min.time(), timezone.utc) + timedelta(
        hours=12
    )
    placed: list[Placed] = []
    unmatched: list[Pick] = []
    for ex in extractions:
        for p in ex.picks:
            if not lo <= datetime.fromisoformat(p.published) < hi:
                continue
            pl = place(p, games)
            if pl is not None:
                placed.append(pl)
            elif datetime.fromisoformat(p.published) >= lo + timedelta(days=1):
                unmatched.append(p)
    return SlatePicks(counted(placed), unmatched)


# --------------------------------------------------------------- grading

LEDGER_FIELDS = (
    "date",
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
    "reason",
)


def _as_rec(pl: Placed) -> Recommendation:
    r = pl.game[0]
    p = pl.pick
    side = {"game_ml": "win", "game_ats": "cover"}.get(p.market, p.side)
    return Recommendation(
        game_date=r.game_date,
        game_id=r.game_id,
        matchup=r.matchup,
        market=p.market,
        selection=pl.label,
        model_prob=0.0,
        line=pl.line,
        tier=Tier.PASS,
        team_side=pl.team_side,
        side=side,
        home_abbrev=r.home_abbrev,
        away_abbrev=r.away_abbrev,
        kickoff_utc=r.kickoff_utc,
    )


def _board_price(pl: Placed, board: dict[str, ClosingQuote]) -> float | None:
    """The day-of board's price on this side, if it was at the host's number."""
    q = board.get(snapshot.key(pl.matchup, pl.pick.market, pl.label))
    if q is None:
        return None
    if pl.pick.market != "game_ml" and (q.line is None or pl.line is None or q.line != pl.line):
        return None
    return q.american


def engine_view(pl: Placed) -> str:
    """Whether the engine bought this side, the other side, or neither."""
    buys = [r for r in pl.game if r.market == pl.pick.market and r.tier != Tier.PASS]
    for r in buys:
        mine = (
            r.side == pl.pick.side
            if pl.pick.market == "game_total"
            else r.team_side == pl.team_side
        )
        return "engine agrees" if mine else "engine disagrees"
    return ""


def grade_row(
    pl: Placed,
    index: ResultIndex,
    board: dict[str, ClosingQuote],
    closing: dict[str, ClosingQuote],
) -> dict[str, str]:
    p = pl.pick
    rec = _as_rec(pl)
    res = result_for(rec, index)
    outcome = None
    if res is not None and (p.market == "game_ml" or pl.line is not None):
        outcome = grade(rec, res)
    price, source = (p.price, "stated") if p.price is not None else (None, "")
    if price is None:
        price = _board_price(pl, board)
        source = "board" if price is not None else ""
    stake = p.units if p.units is not None and p.units > 0 else 1.0
    units: float | None = None
    if outcome is not None and price is not None:
        units = {WIN: stake * (american_to_decimal(price) - 1.0), LOSS: -stake, PUSH: 0.0}[outcome]
    clv = compute_clv(
        pl.matchup, p.market, pl.label, price, None, closing, bet_line=pl.line, side=p.side
    )
    cq = closing.get(snapshot.key(pl.matchup, p.market, pl.label))
    return {
        "date": rec.game_date.isoformat(),
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
        "price_source": source,
        "stake": f"{stake:g}",
        "result": outcome or "",
        "units": "" if units is None else f"{units:.3f}",
        "close_line": "" if cq is None or cq.line is None else f"{cq.line:g}",
        "clv_pts": "" if clv.clv_pts is None else f"{clv.clv_pts:g}",
        "engine": engine_view(pl),
        "published": p.published,
        "episode_title": p.episode_title,
        "stamp": p.stamp,
        "quote": p.quote,
        "reason": p.reason,
    }


def load_ledger(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def update_ledger(path: Path, rows: list[dict[str, str]], day: Date) -> list[dict[str, str]]:
    """Replace ``day``'s rows (a re-audit is authoritative for its date), keep the rest."""
    iso = day.isoformat()
    kept = [r for r in load_ledger(path) if r.get("date") != iso]
    merged = sorted([*kept, *rows], key=lambda r: (r["date"], r["published"], r["pick_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(LEDGER_FIELDS), extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)
    return merged


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


def _show_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """A show's record counts each opinion once, however many hosts shared it."""
    seen: dict[tuple[str, ...], dict[str, str]] = {}
    for r in sorted(rows, key=lambda r: r["published"]):
        side = _NUMBER.sub("", r["selection"])
        k = (r["show"], r["kind"], r["date"], r["matchup"], r["market"], side)
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
    graded = [r for r in ledger if r.get("result")]
    shows: list[tuple[Record, Record]] = []
    hosts: list[tuple[Record, Record]] = []
    markets: list[Record] = []
    for name in sorted({r["show_name"] for r in graded}):
        mine = [r for r in graded if r["show_name"] == name]
        uniq = _show_rows(mine)
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
    "HORIZON",
    "LEDGER_FIELDS",
    "Placed",
    "Record",
    "Records",
    "SlatePicks",
    "by_game",
    "counted",
    "engine_view",
    "grade_row",
    "load_ledger",
    "place",
    "record",
    "records",
    "same_school",
    "slate_picks",
    "update_ledger",
]
