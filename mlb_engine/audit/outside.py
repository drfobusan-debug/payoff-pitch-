"""Grade an outside model's picks into the same ledger, on their own rows.

TeamRankings calls the same three game markets we do, so its picks can be
graded against the same box score and written beside ours -- one row per pick,
``source=teamrankings`` -- and read side by side, day by day.

Two rules hold this together and both matter more than they look:

1. **Their rating, not ours.** The ``tier`` column carries the star rating as
   published (``2 stars``), never a translation into Strong/Moderate buy. A
   benchmark rethresholded into our own tiers is measuring our thresholds.
2. **Their rows are never our rows.** Everything that measures the engine runs
   through :func:`~mlb_engine.audit.ledger.engine_rows` first. A benchmark that
   leaked into our PPV, ROI, CLV or calibration would corrupt exactly the
   numbers it exists to check.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date

from mlb_engine.audit.grade import LOSS, PUSH, WIN
from mlb_engine.audit.ledger import LedgerEntry, pnl_units
from mlb_engine.data.results import GameResult
from mlb_engine.data.teamrankings import TRPick
from mlb_engine.market.tiers import Tier

TEAMRANKINGS = "teamrankings"
# Their projected winner is a forecast, not a bet: it is graded and kept for the
# hit rate, but staked at nothing, so it cannot move the benchmark's P&L.
_NO_BET_MARKETS = frozenset({"game_winner"})


def star_tier(stars: int) -> str:
    """Their confidence as published. ``0`` means the grid showed no stars."""
    if stars <= 0:
        return "unrated"
    return f"{stars} star" if stars == 1 else f"{stars} stars"


def grade_pick(pick: TRPick, res: GameResult) -> str | None:
    """``win``/``loss``/``push``, or ``None`` when the pick is not gradeable."""
    if pick.market == "game_total" and pick.line is not None:
        total = res.home_runs + res.away_runs
        if total == pick.line:
            return PUSH
        return WIN if (total > pick.line) == (pick.side == "over") else LOSS
    if not pick.team_side:
        return None
    team = res.home_runs if pick.team_side == "home" else res.away_runs
    opp = res.away_runs if pick.team_side == "home" else res.home_runs
    if pick.market in ("game_ml", "game_winner"):
        if team == opp:
            return PUSH
        return WIN if team > opp else LOSS
    if pick.market == "game_rl" and pick.line is not None:
        adj = (team - opp) + pick.line
        if adj == 0:
            return PUSH
        return WIN if adj > 0 else LOSS
    return None


def entries_from_picks(
    picks: list[TRPick],
    results: dict[int, GameResult],
    game_pks: dict[str, int],
    date: Date,
) -> list[LedgerEntry]:
    """Ledger rows for one slate of outside picks.

    ``game_pks`` maps our matchup string to the game it is, which is how a pick
    finds its box score: the two feeds agree on ``AWAY @ HOME`` in engine team
    codes, and a pick whose game is not in ``results`` is dropped rather than
    guessed at.
    """
    iso = date.isoformat()
    out: list[LedgerEntry] = []
    for pick in picks:
        if pick.date != iso:
            continue
        pk = game_pks.get(pick.matchup)
        res = results.get(pk) if pk is not None else None
        if res is None or not res.final:
            continue
        result = grade_pick(pick, res)
        if result is None:
            continue
        staked = pick.market not in _NO_BET_MARKETS
        out.append(
            LedgerEntry(
                date=iso,
                matchup=pick.matchup,
                category="game",
                market=pick.market,
                selection=pick.selection,
                line=pick.line,
                book="teamrankings",
                odds=pick.american,
                tier=star_tier(pick.stars),
                # Their own published numbers, never ours: the winner and total
                # columns carry a win probability, the two value columns the
                # edge they see in the price.
                model_prob=pick.win_prob or 0.0,
                ev=pick.value,
                result=result,
                # Their totals column publishes no price, so an unpriced win is
                # paid at the standard -110 (`pnl_units`' default) rather than
                # invented: a total is quoted near that number by every book.
                pnl=pnl_units(result, pick.american) if staked else 0.0,
                margin=_margin(pick, res),
                source=TEAMRANKINGS,
            )
        )
    return out


@dataclass(frozen=True)
class HeadToHead:
    """One game market, our call beside theirs."""

    matchup: str
    market: str
    ours: str
    our_tier: str
    our_result: str
    theirs: str
    their_tier: str
    their_result: str

    @property
    def agree(self) -> bool:
        return self.ours == self.theirs

    @property
    def contested(self) -> bool:
        """Both of us backed something, and not the same thing."""
        return bool(self.ours) and bool(self.theirs) and not self.agree


def head_to_head(
    ours: list[LedgerEntry], theirs: list[LedgerEntry]
) -> list[HeadToHead]:
    """Pair the two ledgers on the game markets both of us bet.

    A market either of us passed on shows as an empty side rather than being
    dropped: declining a game the other model liked is a call, and the whole
    point of a benchmark is that it can be right where we said nothing.
    """
    markets = ("game_ml", "game_rl", "game_total")
    ours_by: dict[tuple[str, str], LedgerEntry] = {}
    for e in ours:
        if e.market in markets and e.tier != Tier.PASS.value:
            ours_by.setdefault((e.matchup, e.market), e)
    theirs_by = {(e.matchup, e.market): e for e in theirs if e.market in markets}
    out: list[HeadToHead] = []
    for key in sorted(set(ours_by) | set(theirs_by)):
        us, them = ours_by.get(key), theirs_by.get(key)
        out.append(
            HeadToHead(
                matchup=key[0],
                market=key[1],
                ours=us.selection if us else "",
                our_tier=us.tier if us else "",
                our_result=us.result if us else "",
                theirs=them.selection if them else "",
                their_tier=them.tier if them else "",
                their_result=them.result if them else "",
            )
        )
    return out


def _margin(pick: TRPick, res: GameResult) -> float | None:
    """Final margin from the backed side, for the run-line miss matrix."""
    if pick.market != "game_rl" or not pick.team_side:
        return None
    team = res.home_runs if pick.team_side == "home" else res.away_runs
    opp = res.away_runs if pick.team_side == "home" else res.home_runs
    return float(team - opp)
