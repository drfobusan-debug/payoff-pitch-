"""Closing-line value: snapshot the closing market and score each bet against it.

``cfb-engine close`` captures the board near kickoff, and the audit compares each
bet's entry number to it:

    clv     = closing no-vig prob of *our* number - the no-vig prob we bet at
    clv_ev  = EV of our bet price under the closing probability

Positive CLV is the durable signal that a model is beating the market before
enough games have been graded to say so from results -- which in college football
is the only signal available, at ~800 games a season.

The subtlety is "our number". A football spread moves in points, not price: the
market decides Alabama is half a point better and -7 becomes -7.5 while the price
stays -110. So the closing quote for a side is generally a quote for a *different
handicap* than the one that was bet, and the two have to be brought onto the same
footing before they can be subtracted. Two steps:

1. Look the side up by a key that does not contain the handicap. Filing closing
   quotes under the full selection (``"ALA -7.0"``) meant a bet placed at -7 and a
   close at -7.5 never met, and ``compute_clv`` returned ``None`` -- so CLV was
   recorded only for bets whose number had not moved, i.e. precisely the rows with
   no line movement in them. That was the sample this engine was going to judge
   itself by.
2. Convert the difference in handicap into probability at the local slope of the
   scoring distribution (:mod:`cfb_engine.market.linevalue`) and add it to the
   closing price's probability. ``clv_pts`` keeps the raw points as well, because
   points are the unit a football bettor actually shops in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cfb_engine.audit import snapshot
from cfb_engine.audit.snapshot import SideQuote as ClosingQuote
from cfb_engine.market.board import GameOdds
from cfb_engine.market.ev import ev_per_dollar
from cfb_engine.market.linevalue import prob_per_point, value_points
from cfb_engine.schemas import Slate

__all__ = [
    "ClosingQuote",
    "ClvResult",
    "ClvSummary",
    "closing_quotes",
    "clv_summary",
    "compute_clv",
    "load_closing",
    "merge_closing",
    "save_closing",
]


def closing_quotes(slate: Slate, board: dict[str, GameOdds]) -> dict[str, ClosingQuote]:
    """Map ``"<market>|<side>"`` -> closing quote (with its line) for every priced side."""
    return snapshot.board_quotes(slate, board)


def merge_closing(
    existing: dict[str, ClosingQuote], fresh: dict[str, ClosingQuote]
) -> dict[str, ClosingQuote]:
    """Later capture wins per side, but nothing already captured is dropped.

    A Saturday runs from noon to past midnight, so any single capture sees the
    close of one kickoff window and the long-stale opener of the rest. Repeat
    captures therefore merge instead of replacing: an in-progress game has left
    the pre-match board, and its captured close must survive later snapshots.
    """
    return snapshot.merge_last_wins(existing, fresh)


def save_closing(quotes: dict[str, ClosingQuote], path: Path) -> None:
    snapshot.save(quotes, path)


def load_closing(path: Path) -> dict[str, ClosingQuote]:
    return snapshot.load(path)


@dataclass(frozen=True)
class ClvResult:
    """What the close said about the number we took."""

    close_odds: float | None = None
    close_prob: float | None = None  # closing no-vig prob *at our line*
    clv: float | None = None
    clv_ev: float | None = None
    clv_pts: float | None = None  # points of line value (ATS/totals only)

    def as_tuple(self) -> tuple[float | None, float | None, float | None, float | None]:
        return self.close_odds, self.close_prob, self.clv, self.clv_ev


def compute_clv(
    matchup: str,
    market: str,
    selection: str,
    bet_american: float | None,
    bet_fair_prob: float | None,
    closing: dict[str, ClosingQuote],
    *,
    bet_line: float | None = None,
    side: str | None = None,
    margin_sd: float = 16.0,
    total_sd: float = 13.0,
) -> ClvResult:
    """Score one bet against the close, correcting for a line that moved.

    Also tries the legacy selection-keyed lookup, so an audit directory written by
    an older build still reads -- exactly, since a legacy key only matched when the
    number had not moved.
    """
    cq = closing.get(snapshot.key(matchup, market, selection))
    if cq is None:
        cq = closing.get(f"{market}|{selection}")
    if cq is None:
        return ClvResult()

    pts = value_points(market, side, bet_line, cq.line)
    close_prob = cq.no_vig_prob
    if pts:
        sd = margin_sd if market == "game_ats" else total_sd
        close_prob = min(max(close_prob + pts * prob_per_point(sd), 0.0), 1.0)
    close_prob = round(close_prob, 4)
    clv = None if bet_fair_prob is None else round(close_prob - bet_fair_prob, 4)
    clv_ev = None if bet_american is None else round(ev_per_dollar(close_prob, bet_american), 4)
    return ClvResult(cq.american, close_prob, clv, clv_ev, pts)


@dataclass
class ClvSummary:
    label: str
    n: int
    mean_clv: float
    beat_close_pct: float
    mean_clv_ev: float

    @property
    def positive(self) -> bool:
        return self.mean_clv >= 0


def clv_summary(rows: list[tuple[str, float | None, float | None]]) -> list[ClvSummary]:
    """Summarize ``(category, clv, clv_ev)`` rows by category plus an ALL row."""
    by_cat: dict[str, list[tuple[float, float]]] = {}
    allrows: list[tuple[float, float]] = []
    for cat, clv, clv_ev in rows:
        if clv is None:
            continue
        pair = (clv, clv_ev if clv_ev is not None else 0.0)
        by_cat.setdefault(cat, []).append(pair)
        allrows.append(pair)

    def summarize(label: str, pairs: list[tuple[float, float]]) -> ClvSummary:
        n = len(pairs)
        mean_clv = sum(c for c, _ in pairs) / n if n else 0.0
        beat = sum(1 for c, _ in pairs if c > 0) / n if n else 0.0
        mean_ev = sum(e for _, e in pairs) / n if n else 0.0
        return ClvSummary(label, n, round(mean_clv, 4), round(beat, 4), round(mean_ev, 4))

    out = [summarize(cat, by_cat[cat]) for cat in sorted(by_cat)]
    if allrows:
        out.append(summarize("ALL", allrows))
    return out
