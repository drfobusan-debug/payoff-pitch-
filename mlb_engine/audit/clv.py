"""Closing line value: did we beat the price, not did the pick win.

Win rate is a slow and noisy way to find out whether a model has an edge. Nine
retro-priced slates put the card at -5.4% ROI with a 95% interval of [-15.6%,
+6.1%] -- hundreds of bets and still no verdict. Closing line value answers the
same question in dozens of bets, because it compares our price against the
sharpest estimate available (the closing market) instead of against a single
binary outcome.

Two numbers per bet:

    clv      = closing no-vig probability - the no-vig probability we bet
    clv_ev   = EV of our price under the closing no-vig probability

``clv`` is in probability points and says which direction the market moved after
we bet. ``clv_ev`` converts that into money at our actual price, so a bet at
-150 and a bet at +200 are comparable. Positive means we bought the side the
market later agreed with, at a price it no longer offers.

Capturing the close costs three credits a slate (one bulk request, three game
markets) plus one per event for props, against ~250 for a historical re-price.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

from mlb_engine.audit.ledger import LedgerEntry
from mlb_engine.market import keys
from mlb_engine.market.ev import MarketQuote, evaluate
from mlb_engine.market.odds import american_to_decimal

log = logging.getLogger(__name__)

_KEY_SEP = "|"


def quote_key(matchup: str, market: str, selection: str) -> str:
    """The key a snapshotted quote is stored under."""
    return _KEY_SEP.join((matchup, market, selection))


@dataclass(frozen=True)
class ClosingQuote:
    """The market's last word on one selection before first pitch."""

    matchup: str
    market: str
    selection: str
    american: float  # best available price at the close
    no_vig_prob: float  # book-weighted consensus, vig removed where possible

    @property
    def key(self) -> str:
        return quote_key(self.matchup, self.market, self.selection)


def closing_quotes(
    quotes: dict[tuple[str, str, str], list[MarketQuote]],
) -> list[ClosingQuote]:
    """Collapse a fetched odds board into one closing quote per selection.

    Uses the same consensus math as the live EV screen (``evaluate``) so the
    closing probability is directly comparable to the one we bet against.
    """
    out: list[ClosingQuote] = []
    for (matchup, market, selection), qs in quotes.items():
        if not qs:
            continue
        res = evaluate(0.5, qs)
        out.append(
            ClosingQuote(
                matchup=matchup,
                market=market,
                selection=selection,
                american=res.best_quote.american,
                no_vig_prob=round(res.fair_prob, 6),
            )
        )
    return out


def merge_closing(
    existing: dict[str, ClosingQuote], fresh: list[ClosingQuote]
) -> list[ClosingQuote]:
    """Later capture wins per selection, but nothing already captured is dropped.

    A slate rarely closes at one moment: by the time the 7pm games are near
    first pitch the afternoon games are already under way and have left the
    pre-match board entirely. Capturing twice and overwriting would therefore
    trade the day games' close for the night games', so the last price seen for
    each selection is kept instead.
    """
    merged = dict(existing)
    for q in fresh:
        merged[q.key] = q
    return sorted(merged.values(), key=lambda q: q.key)


def merge_board(
    existing: dict[str, ClosingQuote], fresh: list[ClosingQuote]
) -> list[ClosingQuote]:
    """The mirror of ``merge_closing``: the *first* price seen per selection wins.

    Same file format, opposite end of the day. Held across the slate's runs this
    accumulates the opening board, which is what a pre-bet CLV check needs to
    measure the day's drift against (see ``features.drift_gate``).
    """
    merged = dict(existing)
    for q in fresh:
        merged.setdefault(q.key, q)
    return sorted(merged.values(), key=lambda q: q.key)


def save_closing(path: Path, quotes: list[ClosingQuote]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "matchup": q.matchup,
            "market": q.market,
            "selection": q.selection,
            "american": q.american,
            "no_vig_prob": q.no_vig_prob,
        }
        for q in quotes
    ]
    path.write_text(json.dumps(payload, indent=2))


def load_closing(path: Path) -> dict[str, ClosingQuote]:
    """Closing quotes keyed by ``matchup|market|selection``; empty if not captured."""
    if not path.exists():
        return {}
    out: dict[str, ClosingQuote] = {}
    for row in json.loads(path.read_text()):
        q = ClosingQuote(
            matchup=str(row["matchup"]),
            market=str(row["market"]),
            selection=str(row["selection"]),
            american=float(row["american"]),
            no_vig_prob=float(row["no_vig_prob"]),
        )
        out[q.key] = q
    return out


def board_path(audit_dir: Path, slate_date: Date) -> Path:
    """Where the slate's opening board lives, alongside its closing snapshot."""
    return audit_dir / f"board_{slate_date.isoformat()}.json"


# Team markets: both sides are a whole team, so neither is ever a longshot. The
# lopsided ones are the props, where a weak hitter to homer is honestly +2600.
_TEAM_MARKETS = frozenset(
    {"game_ml", "game_rl", "game_total", "f5_ml", "f5_rl", "f5_total"}
)

# A favourite this heavy is not a pre-game price on anything the engine bets: -1000 is
# a 91% certainty, and across 7,894 captured closes the most negative was -375. A team
# up 6-0 in the seventh is -2000.
IMPLAUSIBLE_FAVOURITE = -1000.0

# The dog side has to be judged per market, because a symmetric bound is impossible.
# The most positive legitimate close on a team market is +375 (an F5 moneyline); on a
# home-run prop it is +2600, which a symmetric rule would throw away -- 34 real
# ``batter_hr o0.5`` closes sit between +1000 and +2600, all with CLV of about 0.000.
IMPLAUSIBLE_TEAM_DOG = 1000.0


def is_plausible_close(american: float, market: str) -> bool:
    """Whether a captured price can be a pre-game close at all.

    Cheap and one-sided on purpose. It cannot detect an in-play price that merely
    looks normal -- a hitter who already has two hits prices his over at a perfectly
    ordinary number -- so a pass here is not a certificate. Only a fail is meaningful,
    and it is meant to fire on the case that is wrong by a factor of four rather than
    to police the margins.
    """
    if american <= IMPLAUSIBLE_FAVOURITE:
        return False
    return not (market in _TEAM_MARKETS and american >= IMPLAUSIBLE_TEAM_DOG)


def clv_points(bet_prob: float, close_prob: float) -> float:
    """Probability points the market moved our way after we bet."""
    return round(close_prob - bet_prob, 6)


def clv_ev(bet_american: float, close_prob: float) -> float:
    """EV per unit staked at our price, judged by the closing no-vig probability."""
    dec = american_to_decimal(bet_american)
    return round(close_prob * (dec - 1.0) - (1.0 - close_prob), 6)


def attach_clv(entries: list[LedgerEntry], closing: dict[str, ClosingQuote]) -> int:
    """Fill the CLV columns on every entry we have both a bet price and a close for.

    Returns the number of entries priced. Rows without a captured close, or that
    were never priced at bet time, are left as None rather than defaulted -- a
    missing close is missing information, not zero closing line value.
    """
    # The close is keyed on the book's spelling of a name and the ledger on the
    # lineup feed's, so props need the same accent/suffix-insensitive fallback
    # the live pricing uses.
    aliases = keys.canonical_index(
        {(c.matchup, c.market, c.selection): c for c in closing.values()}
    )
    n = 0
    for e in entries:
        if e.odds is None:
            continue
        close = closing.get(_KEY_SEP.join((e.matchup, e.market, e.selection)))
        if close is None:
            close = aliases.get((e.matchup, e.market, keys.canonical(e.selection)))
        if close is None:
            continue
        if not is_plausible_close(close.american, e.market):
            log.warning(
                "CLV: refusing %s %s %s -- a close of %+.0f is an in-play price and "
                "not a close; the row stays unpriced",
                e.matchup,
                e.market,
                e.selection,
                close.american,
            )
            continue
        e.close_odds = close.american
        e.close_prob = close.no_vig_prob
        # Measured against the price we bet, not the probability we predicted:
        # CLV asks whether the market came to our side, not whether we were right.
        e.clv = clv_points(_no_vig_at_bet(e), close.no_vig_prob)
        e.clv_ev = clv_ev(e.odds, close.no_vig_prob)
        n += 1
    return n


def _no_vig_at_bet(e: LedgerEntry) -> float:
    """The devigged market probability for the side we took, at bet time."""
    if e.fair_prob is not None:
        return e.fair_prob
    # Pre-devig ledger rows: fall back to the raw implied probability of our
    # price, which overstates the market by about half the hold and therefore
    # makes CLV look worse than it was. Flagged here rather than silently mixed.
    return 1.0 / american_to_decimal(e.odds) if e.odds is not None else 0.5


@dataclass
class ClvSummary:
    """Closing-line-value rollup for one market (or 'ALL')."""

    label: str
    n: int
    mean_clv: float  # probability points, our side
    beat_close_pct: float  # share of bets where the close came to us
    mean_clv_ev: float  # EV per unit at our price, closing probabilities

    @property
    def positive(self) -> bool:
        return self.mean_clv_ev > 0


def clv_rows(entries: list[LedgerEntry]) -> list[tuple[str, float, float]]:
    """``(market, clv, clv_ev)`` for every entry with closing line value attached."""
    return [
        (e.market, e.clv, e.clv_ev)
        for e in entries
        if e.clv is not None and e.clv_ev is not None
    ]


def summarize(rows: list[tuple[str, float, float]]) -> list[ClvSummary]:
    """Roll ``(market, clv, clv_ev)`` triples up per market, plus an ALL row.

    Only bets with a captured close appear, so a short list means the closing
    snapshot was missing or partial, not that the engine had no edge.
    """
    by_market: dict[str, list[tuple[float, float]]] = {}
    for market, clv, ev in rows:
        by_market.setdefault(market, []).append((clv, ev))
    out: list[ClvSummary] = []
    for label in sorted(by_market):
        out.append(_summary(label, by_market[label]))
    if by_market:
        out.append(_summary("ALL", [v for vals in by_market.values() for v in vals]))
    return out


def _summary(label: str, vals: list[tuple[float, float]]) -> ClvSummary:
    n = len(vals)
    return ClvSummary(
        label=label,
        n=n,
        mean_clv=round(sum(c for c, _ in vals) / n, 5),
        beat_close_pct=round(sum(1 for c, _ in vals if c > 0) / n, 4),
        mean_clv_ev=round(sum(e for _, e in vals) / n, 5),
    )
