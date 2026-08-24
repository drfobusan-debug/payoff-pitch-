"""Grade an outside forecast into the same ledger, on its own rows.

FPI calls the same games we do, so its call can be graded against the same final
score and written beside ours -- one row per game, ``source=fpi`` -- and read side
by side, week by week.

Three rules hold this together, and the first two are the same ones the MLB
benchmark needed:

1. **Their number, not ours.** ``model_prob`` is FPI's published probability and
   ``tier`` is its predicted margin as text (``FPI -3.2``), never a translation
   into Strong/Moderate buy. A benchmark rethresholded into our tiers is
   measuring our thresholds.
2. **Their rows are never our rows.** Everything measuring the engine filters on
   ``source == ENGINE`` first. A benchmark leaking into our ROI, CLV or
   calibration would corrupt the exact numbers it exists to check.
3. **It is a forecast, not a bet.** FPI publishes no price, so its rows are
   staked at nothing: ``pnl`` stays 0 and no ROI is ever claimed for it. What is
   comparable is who called the game right, and at what confidence.

Display beside our plays is the point, so :func:`head_to_head` pairs the two on
the moneyline and says whether it agrees, disagrees, or called a game we passed.
"""

from __future__ import annotations

from dataclasses import dataclass

from nfl_engine.audit.ledger import (
    ENGINE,
    LOSS,
    PAPER,
    PUSH,
    WIN,
    LedgerEntry,
    Metrics,
    metrics,
)
from nfl_engine.data.espn import FPI, FpiGame
from nfl_engine.market.ev import MONEYLINE
from nfl_engine.market.screens import Tier

BUY_TIERS = frozenset({Tier.STRONG.value, Tier.MODERATE.value})


def entries_from_fpi(
    games: list[FpiGame],
    finals: dict[str, tuple[int, int]] | None = None,
    *,
    captured_at: str = "",
) -> list[LedgerEntry]:
    """Ledger rows for one week of FPI calls, graded where the score is known.

    ``finals`` maps our matchup string to ``(home_score, away_score)``; a game
    missing from it is written ungraded rather than dropped, because a benchmark
    that only appears after the fact cannot be checked against what it said
    beforehand.

    A game FPI calls even carries no side and is skipped: there is nothing to
    grade in a 50/50, and inventing a side for it would credit or blame FPI for a
    call it never made.
    """
    scores = finals or {}
    out: list[LedgerEntry] = []
    for game in games:
        side = game.pick
        if not side:
            continue
        entry = LedgerEntry(
            season=game.season,
            week=game.week,
            date=game.date,
            matchup=game.matchup,
            market=MONEYLINE,
            side=side,
            line=None,
            book=FPI,
            # FPI publishes no price. Left null rather than filled with a market
            # number, which would make its record depend on a price it never
            # quoted -- and on our shopping.
            odds=None,
            opposite_odds=None,
            tier=f"FPI {game.home_margin:+.1f}",
            model_prob=game.pick_prob,
            fair_prob=None,
            ev_model=None,
            ev_fair=None,
            paired_books=0,
            source=FPI,
            mode=PAPER,
            captured_at=captured_at,
        )
        final = scores.get(game.matchup)
        if final is not None:
            entry.home_score, entry.away_score = final
            margin = final[0] - final[1]
            own = margin if side == game.home else -margin
            entry.result = WIN if own > 0 else PUSH if own == 0 else LOSS
            # Staked at nothing: see rule 3 above.
            entry.pnl = 0.0
        out.append(entry)
    return out


def benchmark_metrics(entries: list[LedgerEntry], source: str = FPI) -> Metrics | None:
    """One record row for an outside source, over the games it called.

    ``None`` when it has nothing graded yet, so an empty benchmark shows as absent
    rather than as a 0-0 record that reads like a failure.
    """
    rows = [e for e in entries if e.source == source and e.result]
    if not rows:
        return None
    return metrics(rows, lambda e: True, f"{source.upper()} (benchmark)")


@dataclass(frozen=True)
class HeadToHead:
    """One game, our moneyline call beside theirs."""

    matchup: str
    ours: str
    our_tier: str
    our_result: str
    theirs: str
    their_prob: float | None
    their_margin: str
    their_result: str

    @property
    def agree(self) -> bool:
        return bool(self.ours) and self.ours == self.theirs

    @property
    def contested(self) -> bool:
        """Both of us backed something, and not the same thing."""
        return bool(self.ours) and bool(self.theirs) and not self.agree

    def mark(self) -> str:
        """How the benchmark reads next to our play, in one word."""
        if not self.theirs:
            return ""
        if not self.ours:
            return f"{self.theirs} (we passed)"
        return "agrees" if self.agree else f"fade: {self.theirs}"


def head_to_head(entries: list[LedgerEntry], *, season: int, week: int) -> list[HeadToHead]:
    """Pair our moneyline plays with the benchmark's calls for one week.

    A game either of us passed on shows as an empty side rather than being
    dropped: declining a game the benchmark liked is a call, and the whole point
    of a benchmark is that it can be right where we said nothing.
    """
    scope = [e for e in entries if e.season == season and e.week == week]
    ours: dict[str, LedgerEntry] = {}
    for entry in scope:
        if entry.source != ENGINE or entry.market != MONEYLINE:
            continue
        if entry.screens or entry.tier not in BUY_TIERS:
            continue
        ours.setdefault(entry.matchup, entry)
    theirs = {e.matchup: e for e in scope if e.source == FPI and e.market == MONEYLINE}
    out: list[HeadToHead] = []
    for matchup in sorted(set(ours) | set(theirs)):
        us, them = ours.get(matchup), theirs.get(matchup)
        out.append(
            HeadToHead(
                matchup=matchup,
                ours=us.side if us else "",
                our_tier=us.tier if us else "",
                our_result=us.result if us else "",
                theirs=them.side if them else "",
                their_prob=them.model_prob if them else None,
                their_margin=them.tier if them else "",
                their_result=them.result if them else "",
            )
        )
    return out
