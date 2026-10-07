"""Put podcast picks on the CFB slate's games and grade them.

Placement, counting, the ledger and the records are shared
(:mod:`engine_common.podcasts.picks`); this is the CFB adapter. Matching is by
the card's own school labels, so a pick lands only on a game the card prints.

Grading reuses the engine's settlement (:func:`cfb_engine.audit.grade.grade`) at
the host's own number. A spread or total with no number said is not graded. A
price is the one the host said or, failing that, the day-of board's price at
the host's number (labelled ``board``); with neither, the pick counts in the
won-lost record but carries no units, so ROI is over priced picks only.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cfb_engine.audit import snapshot
from cfb_engine.audit.clv import ClosingQuote, compute_clv
from cfb_engine.audit.grade import ResultIndex, grade, result_for
from cfb_engine.data.teamnames import label_matches, school_key
from cfb_engine.market.tiers import Tier
from cfb_engine.recommendations import Recommendation
from engine_common.podcasts import picks as core
from engine_common.podcasts.extract import Extraction, Pick
from engine_common.podcasts.picks import (
    BREAK_EVEN,
    HORIZON,
    LEDGER_FIELDS,
    MARKET_LABEL,
    Game,
    League,
    Record,
    Records,
    counted,
    load_ledger,
    record,
    records,
)
from engine_common.podcasts.shows import CFB

Rows = list[Recommendation]
Placed = core.Placed[Rows]
SlatePicks = core.SlatePicks[Rows]


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


def _kickoff(recs: Rows) -> datetime | None:
    stamp = recs[0].kickoff_utc
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def by_game(recs: Rows) -> dict[str, Rows]:
    out: dict[str, Rows] = {}
    for r in recs:
        out.setdefault(r.matchup, []).append(r)
    return out


def to_games(games: dict[str, Rows]) -> list[Game[Rows]]:
    out: list[Game[Rows]] = []
    for recs in games.values():
        r = recs[0]
        home_spread = next(
            (x.line for x in recs if x.market == "game_ats" and x.team_side == "home"), None
        )
        out.append(
            Game(
                matchup=r.matchup,
                home=r.home_abbrev or "",
                away=r.away_abbrev or "",
                kickoff=_kickoff(recs),
                day=r.game_date,
                home_spread=home_spread,
                rows=recs,
            )
        )
    return out


def _rows(pl: Placed) -> Rows:
    return pl.game.rows or []


def engine_view(pl: Placed) -> str:
    """Whether the engine bought this side, the other side, or neither."""
    buys = [r for r in _rows(pl) if r.market == pl.pick.market and r.tier != Tier.PASS]
    for r in buys:
        mine = (
            r.side == pl.pick.side
            if pl.pick.market == "game_total"
            else r.team_side == pl.team_side
        )
        return "engine agrees" if mine else "engine disagrees"
    return ""


LEAGUE: League[Rows] = League(CFB, same_school, engine_view, legacy=True)


def place(p: Pick, games: dict[str, Rows]) -> Placed | None:
    """The one game ``p`` is about, or ``None`` if none or more than one fits."""
    return core.place(p, to_games(games), LEAGUE)


def slate_picks(extractions: Iterable[Extraction], recs: Rows, day: Date) -> SlatePicks:
    """CFB picks published in the week before ``day``'s slate, placed on its games."""
    lo = datetime.combine(day, datetime.min.time(), timezone.utc) - HORIZON
    hi = datetime.combine(day + timedelta(days=1), datetime.min.time(), timezone.utc) + timedelta(
        hours=12
    )
    return core.slate_picks(
        extractions,
        to_games(by_game(recs)),
        LEAGUE,
        since=lo,
        until=hi,
        unmatched_since=lo + timedelta(days=1),
    )


def _as_rec(pl: Placed) -> Recommendation:
    r = _rows(pl)[0]
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
    clv = compute_clv(
        pl.matchup, p.market, pl.label, price, None, closing, bet_line=pl.line, side=p.side
    )
    cq = closing.get(snapshot.key(pl.matchup, p.market, pl.label))
    return core.ledger_row(
        pl,
        league=CFB,
        day=rec.game_date,
        outcome=outcome,
        price=price,
        price_source=source,
        close_line=None if cq is None else cq.line,
        clv_pts=clv.clv_pts,
        engine=engine_view(pl),
    )


def update_ledger(path: Path, rows: list[dict[str, str]], day: Date) -> list[dict[str, str]]:
    """Replace ``day``'s rows (a re-audit is authoritative for its date), keep the rest."""
    return core.update_ledger(path, rows, [day])


__all__ = [
    "BREAK_EVEN",
    "HORIZON",
    "LEAGUE",
    "LEDGER_FIELDS",
    "MARKET_LABEL",
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
    "to_games",
    "update_ledger",
]
