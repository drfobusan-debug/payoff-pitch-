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
        """His best row by EV -- what the note quotes beside his rating."""
        rows = [r for r in self.for_batter(name) if r.ev is not None]
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


def build(
    result: ScreenResult,
    recs: list[Recommendation],
    *,
    rows_per_batter: int = ROWS_PER_BATTER,
    source: str | None = None,
) -> Board:
    """The screened hitters' priced rows, best EV first, and the ones with none.

    Only priced rows survive: a market the pipeline modelled but never got a
    quote for has no bet in it, and the note already carries the model's view of
    the matchup in every other table.
    """
    batters = [(v.line.name, v.line.mlbam_id) for s in result.sections for v in s.hitters]
    priced = [r for r in recs if r.category == "batter" and r.market_american is not None]
    board = Board(source=source)
    for name, mlbam_id in batters:
        mine = [_row(r, name) for r in priced if _matches(r, name, mlbam_id)]
        if not mine:
            board.unpriced.append(name)
            continue
        mine.sort(key=lambda r: (r.ev if r.ev is not None else -9.9), reverse=True)
        board.rows.extend(mine[:rows_per_batter])
        board.dropped += max(len(mine) - rows_per_batter, 0)
    return board
