"""Price the screen's survivors off the card's own board.

The screen answers a question the engine does not -- who to hunt, before lineups
are posted -- and then stops, because a matchup rating is not a bet. This module
supplies the missing half without fetching anything: the nightly run has already
priced every market it could, devigged it two-sided, and persisted the result to
``predictions_<date>.json``, so the note reads that file and shows the survivors'
own rows. No Odds API credit is spent here and no probability is recomputed --
whatever the card bet is what the note prints, which is the point. A number that
disagrees with the card would be a second opinion nobody graded.

Two things the join makes visible, and both are information rather than noise:

* A survivor with **no rows at all**. The pipeline skips a game whose lineup is
  not posted, and the screen deliberately runs before that, so the hitter the
  screen likes most is often the one the engine never priced. That is a
  timing fact about the board, not a fault in either.
* A survivor priced **against** the matchup. The screen reads form and exposure;
  the market reads everything. Where the card's edge is negative on a hitter the
  screen rates a buy, the disagreement is the interesting cell on the page.

The arms get the same treatment. The screen keeps a starter because his lineup
is the one to hunt, and that read has a second half -- the starter's own props --
which the card prices every night and the note used to print only when a side
cleared the buy tiers, which on a soft arm is almost never. So the board now
holds one position per stat on every arm the screen kept: the side of his
strikeouts, walks, hits, earned runs and outs the card gave the best expected
value, at the card's own price, with the gate that refused it where one did.
Those rows are recorded and graded like the hitters', so the pitcher half of the
screen's thesis gets a receipt rather than a projection table.

Ratings stay where they are. Nothing here feeds :func:`power_report._rating`,
which is scored on the matchup alone -- a price belongs next to a rating, not
inside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from pathlib import Path

from mlb_engine.market.ranking import price_rank
from mlb_engine.output.power_screen import ScreenResult
from mlb_engine.recommendations import Recommendation

# Rows per hitter. The board carries every side of every line the books hang, so
# a hitter can own twenty rows; the note wants the ones worth acting on.
ROWS_PER_BATTER = 4

# The two markets this note is read for, printed for every hitter who has them
# priced. Sorting on EV alone buries H+R+RBI behind the homer on nearly every
# name -- a long HR price inflates EV by construction -- so the instrument the
# reader is comparing would be the one missing from the comparison.
ANCHOR_MARKETS = ("HR", "HRR")

# Stats the arm board holds, one position apiece, in the order the card reads them.
PITCHER_STATS: tuple[str, ...] = ("K", "BB", "H", "ER", "outs")

# Pretty names for the market keys the pipeline writes.
MARKET_LABEL = {
    "H": "H",
    "1B": "1B",
    "2B": "2B",
    "HR": "HR",
    "R": "R",
    "RBI": "RBI",
    "HRR": "H+R+RBI",
    "TB": "TB",
}

BUY_TIERS = ("Strong buy", "Moderate buy")

# Markets the note shows without holding a position in them. A home run is a ~7%
# event for the best bat the screen can find against the softest arm on the
# board, so books quote it one-way at +400 and up: there is no second side, the
# edge measured against that number is mostly the hold, and graded the screen's
# HR rows went 2-13 for -7.32u while every other market together was -3.06u over
# the same two boards. Across 2288 graded HR overs the book loses 34.5% above
# +300 and worsens monotonically with price, so no price filter rescues it. The
# arsenal work is still the reason to watch a hitter, so the row stays on the
# board -- it just is not quoted as a bet and is not recorded as a position.
DISPLAY_ONLY = frozenset({"HR"})


def market_label(stat: str, category: str = "batter") -> str:
    """The bucket a stat rolls up into: a starter's hits are not a hitter's."""
    label = MARKET_LABEL.get(stat, stat)
    return f"SP {label}" if category == "pitcher" else label


@dataclass(frozen=True)
class BoardRow:
    """One priced market on one screened hitter, exactly as the card had it."""

    batter: str
    stat: str
    line: float | None
    side: str
    model_prob: float
    book: str | None
    american: float | None
    fair_prob: float | None
    edge: float | None
    ev: float | None
    tier: str
    devigged: bool
    # Identity, carried so the row can be graded later off a box score rather
    # than re-matched by name against it (see audit.power_ledger).
    player_id: int | None = None
    game_pk: int | None = None
    # The probability the card's own screens bet on: the model pulled toward the
    # devigged price by ``Config.anchor_for``. ``None`` on a row priced before
    # the anchor existed, or one the card never anchored.
    bet_prob: float | None = None
    # ``batter`` or ``pitcher``. The name field is ``batter`` for every row because
    # the ledger column is; a starter's rows carry his name there and this flag.
    category: str = "batter"
    # The card's screen that refused the row, when a side was passed rather than
    # bought: the note prints why a row is not a bet, not only that it is not,
    # and the ledger can then grade the gate.
    gate: str = ""

    @property
    def shown_prob(self) -> float:
        """The number the note prints, which is the one ``edge`` and ``ev`` use.

        The board's edge and EV always came from the card's anchored evaluation
        while the probability column showed the unanchored model, so the three
        did not describe one bet. Graded over the screen's own 55 two-sided rows
        the anchored probability scores better than the model both in and out of
        sample (Brier .2269 vs .2304 on 8/18, .2364 vs .2534 on 8/19-20), so
        printing it is both the consistent and the more accurate choice.
        """
        return self.model_prob if self.bet_prob is None else self.bet_prob

    @property
    def label(self) -> str:
        stat = market_label(self.stat, self.category)
        point = "" if self.line is None else f" {'o' if self.side == 'over' else 'u'}{self.line}"
        return f"{stat}{point}"

    @property
    def is_buy(self) -> bool:
        """Did the card hold it? A display-only market is never held.

        The tier is assigned by the pricer, which knows nothing about which
        markets the note only watches, so the market has to be checked here too
        or a buy tier on an HR row counts as a position in every consumer that
        did not remember to filter it itself.
        """
        return self.tier in BUY_TIERS and self.stat not in DISPLAY_ONLY


@dataclass
class Board:
    """The screened pool's rows on the card's board, and who had none."""

    rows: list[BoardRow] = field(default_factory=list)
    unpriced: list[str] = field(default_factory=list)
    dropped: int = 0  # rows trimmed by ROWS_PER_BATTER, for the caption
    source: str | None = None
    # The arms' half: one position per stat on each kept starter, and the
    # starters the card had no priced prop on.
    arm_rows: list[BoardRow] = field(default_factory=list)
    arms_unpriced: list[str] = field(default_factory=list)

    @property
    def priced(self) -> list[str]:
        seen: list[str] = []
        for row in self.rows:
            if row.batter not in seen:
                seen.append(row.batter)
        return seen

    @property
    def arms_priced(self) -> list[str]:
        seen: list[str] = []
        for row in self.arm_rows:
            if row.batter not in seen:
                seen.append(row.batter)
        return seen

    @property
    def positions(self) -> list[BoardRow]:
        """Every row the screen holds, hitters then arms."""
        return [*self.rows, *self.arm_rows]

    @property
    def buys(self) -> list[BoardRow]:
        return [r for r in self.positions if r.is_buy]

    def for_batter(self, name: str) -> list[BoardRow]:
        return [r for r in self.rows if r.batter == name]

    def for_pitcher(self, name: str) -> list[BoardRow]:
        return [r for r in self.arm_rows if r.batter == name]

    def best_for_batter(self, name: str) -> BoardRow | None:
        """His best held row -- what the note quotes beside his rating.

        "Best" is the devigged price rather than our EV against it, for the
        reason in :mod:`mlb_engine.market.ranking`: EV ordered these rows by
        price length, which is how a longshot won the column. A display-only
        market is skipped however good it looks, because its EV is computed
        against a price nobody stripped the hold out of.
        """
        rows = [
            r for r in self.for_batter(name) if r.ev is not None and r.stat not in DISPLAY_ONLY
        ]
        return min(rows, key=lambda r: price_rank(r.american, r.fair_prob, r.ev)) if rows else None


def default_predictions_path(audit_dir: Path, as_of: Date) -> Path:
    return audit_dir / f"predictions_{as_of.isoformat()}.json"


def _gate(rec: Recommendation) -> str:
    """Which of the card's screens refused the row, if one did.

    A veto is recorded ahead of a pass: a vetoed row was refused on the matchup
    and a passed one on the price, and when both fired the first is the reason
    the second was never reached.
    """
    return rec.veto_gate or rec.pass_gate or ""


def _row(rec: Recommendation, name: str) -> BoardRow:
    return BoardRow(
        batter=name,
        stat=rec.stat or "",
        line=rec.line,
        side=rec.side or "over",
        model_prob=rec.model_prob,
        book=rec.book,
        american=rec.market_american,
        fair_prob=rec.fair_prob,
        edge=rec.edge,
        ev=rec.ev,
        tier=rec.tier.value,
        devigged=rec.opposite_american is not None,
        player_id=rec.player_id,
        game_pk=rec.game_pk,
        bet_prob=rec.bet_prob,
        category=rec.category,
        gate=_gate(rec),
    )


def _matches(rec: Recommendation, name: str, mlbam_id: int) -> bool:
    """Is this row about this hitter?

    The id is authoritative and the name is the fallback: a prop row carries the
    book's spelling of a name in ``selection``, and two players share a surname
    often enough that a substring test alone would cross-price them.
    """
    if rec.player_id is not None:
        return rec.player_id == mlbam_id
    return rec.selection.startswith(name)


def _ev(row: BoardRow) -> float:
    return row.ev if row.ev is not None else -9.9


def _best_per_quote(rows: list[BoardRow]) -> list[BoardRow]:
    """One row per market, line and side -- the best quote the card found for it.

    The board can hold the same bet from several books; two prices on one bet is
    a line-shopping question, and this page is not that page.
    """
    best: dict[tuple[str, float | None, str], BoardRow] = {}
    for r in rows:
        key = (r.stat, r.line, r.side)
        if key not in best or _ev(r) > _ev(best[key]):
            best[key] = r
    return list(best.values())


def _select(rows: list[BoardRow], limit: int, anchors: tuple[str, ...]) -> list[BoardRow]:
    """The hitter's rows for the note: both anchor markets, then the best rest."""
    kept: list[BoardRow] = []
    for stat in anchors:
        of_stat = [r for r in rows if r.stat == stat]
        if of_stat:
            kept.append(max(of_stat, key=_ev))
    rest = sorted((r for r in rows if r not in kept), key=_ev, reverse=True)
    kept.extend(rest[: max(limit - len(kept), 0)])
    return sorted(kept, key=_ev, reverse=True)


def _select_arm(rows: list[BoardRow], stats: tuple[str, ...]) -> list[BoardRow]:
    """One position per stat: the side the card gave the best expected value.

    Over against under is the whole question on a starter's line -- the screen's
    case is that the arm is soft, and the card's number says whether the board
    already knows it -- so the two sides of one stat are one decision, not two
    rows. A stat the book never hung is simply absent.
    """
    kept: list[BoardRow] = []
    for stat in stats:
        of_stat = [r for r in rows if r.stat == stat]
        if of_stat:
            kept.append(max(of_stat, key=_ev))
    return kept


def build(
    result: ScreenResult,
    recs: list[Recommendation],
    *,
    rows_per_batter: int = ROWS_PER_BATTER,
    source: str | None = None,
    anchors: tuple[str, ...] = ANCHOR_MARKETS,
    arm_stats: tuple[str, ...] = PITCHER_STATS,
) -> Board:
    """The screened hitters' priced rows, best EV first, and the ones with none.

    Only priced rows survive: a market the pipeline modelled but never got a
    quote for has no bet in it, and the note already carries the model's view of
    the matchup in every other table.

    Every hitter shows his homer and his H+R+RBI where both were quoted, even
    when a third market prices better, because those two are what the page is
    for; ``rows_per_batter`` governs how much else comes with them.

    Every arm the screen kept holds one side per stat in ``arm_stats``, chosen on
    the card's expected value, whatever tier the card gave it: the position is
    the screen's, the price and the verdict on it are the card's, and the gate
    that refused it is carried so the two can be told apart when graded.
    """
    batters = [(v.line.name, v.line.mlbam_id) for s in result.sections for v in s.hitters]
    priced = [r for r in recs if r.category == "batter" and r.market_american is not None]
    board = Board(source=source)
    for name, mlbam_id in batters:
        mine = _best_per_quote([_row(r, name) for r in priced if _matches(r, name, mlbam_id)])
        if not mine:
            board.unpriced.append(name)
            continue
        kept = _select(mine, rows_per_batter, anchors)
        board.rows.extend(kept)
        board.dropped += len(mine) - len(kept)
    arms = [(s.starter.name, s.starter.mlbam_id) for s in result.sections]
    priced_arms = [r for r in recs if r.category == "pitcher" and r.market_american is not None]
    for name, mlbam_id in arms:
        mine = _best_per_quote([_row(r, name) for r in priced_arms if _matches(r, name, mlbam_id)])
        if not mine:
            board.arms_unpriced.append(name)
            continue
        board.arm_rows.extend(_select_arm(mine, arm_stats))
    return board
