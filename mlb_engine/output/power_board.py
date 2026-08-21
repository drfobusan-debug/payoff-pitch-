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

Ratings stay where they are. Nothing here feeds :func:`power_report._rating`,
which is scored on the matchup alone -- a price belongs next to a rating, not
inside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from pathlib import Path

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

    @property
    def label(self) -> str:
        stat = MARKET_LABEL.get(self.stat, self.stat)
        point = "" if self.line is None else f" {'o' if self.side == 'over' else 'u'}{self.line}"
        return f"{stat}{point}"

    @property
    def is_buy(self) -> bool:
        return self.tier in BUY_TIERS


@dataclass
class Board:
    """The screened pool's rows on the card's board, and who had none."""

    rows: list[BoardRow] = field(default_factory=list)
    unpriced: list[str] = field(default_factory=list)
    dropped: int = 0  # rows trimmed by ROWS_PER_BATTER, for the caption
    source: str | None = None

    @property
    def priced(self) -> list[str]:
        seen: list[str] = []
        for row in self.rows:
            if row.batter not in seen:
                seen.append(row.batter)
        return seen

    @property
    def buys(self) -> list[BoardRow]:
        return [r for r in self.rows if r.is_buy]

    def for_batter(self, name: str) -> list[BoardRow]:
        return [r for r in self.rows if r.batter == name]

    def best_for_batter(self, name: str) -> BoardRow | None:
        """His best held row by EV -- what the note quotes beside his rating.

        A display-only market is skipped however good its EV looks, because EV on
        a one-way longshot is computed against a price nobody stripped the hold
        out of, and it would otherwise win this column on most of the board.
        """
        rows = [
            r for r in self.for_batter(name) if r.ev is not None and r.stat not in DISPLAY_ONLY
        ]
        return max(rows, key=lambda r: r.ev or 0.0) if rows else None


def default_predictions_path(audit_dir: Path, as_of: Date) -> Path:
    return audit_dir / f"predictions_{as_of.isoformat()}.json"


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


def build(
    result: ScreenResult,
    recs: list[Recommendation],
    *,
    rows_per_batter: int = ROWS_PER_BATTER,
    source: str | None = None,
    anchors: tuple[str, ...] = ANCHOR_MARKETS,
) -> Board:
    """The screened hitters' priced rows, best EV first, and the ones with none.

    Only priced rows survive: a market the pipeline modelled but never got a
    quote for has no bet in it, and the note already carries the model's view of
    the matchup in every other table.

    Every hitter shows his homer and his H+R+RBI where both were quoted, even
    when a third market prices better, because those two are what the page is
    for; ``rows_per_batter`` governs how much else comes with them.
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
    return board
