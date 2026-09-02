"""Keep a ledger of the power screen's own positions, and grade it the next day.

The note rates hitters and prints the card's prices beside them. Until now none
of that was written down, so the screen could be wrong every morning and read
exactly the same the next one. This module is the receipt: every row the note
showed is appended to ``power_screen_ledger.csv`` with the price it was shown at,
and the following run grades yesterday's rows off the box score and prints the
scorecard in the note.

Three questions, kept apart on purpose, because they have different answers:

* **Did the positions win?** Wins, losses and units at the recorded price. A
  hitter who never batted is voided rather than lost -- a book would void it, and
  a late scratch is not a bad read.
* **Was the model better than the price?** Brier score of the model probability
  against the two-sided no-vig mark on the same rows. This is the only question
  that says whether the engine knows anything the board did not, and it is scored
  only on rows where the vig could actually be stripped.
* **Did the matchup rating discriminate?** The screen's own BUY/HOLD/AVOID
  against realized results. The rating reads no price, so it is allowed to be
  right about the matchup and wrong about the bet, and the two are reported
  separately rather than blended into one number.

Nothing here feeds a price, a probability or a rating. It is a record.
"""

from __future__ import annotations

import csv
import logging
import unicodedata
from dataclasses import asdict, dataclass, field, replace
from datetime import date as Date
from pathlib import Path

from mlb_engine.audit.grade import LOSS, PUSH, WIN, batter_actual, grade_batter
from mlb_engine.audit.ledger import pnl_units
from mlb_engine.data.results import GameResult
from mlb_engine.output.power_board import DISPLAY_ONLY, MARKET_LABEL, Board, BoardRow

log = logging.getLogger(__name__)

LEDGER_NAME = "power_screen_ledger.csv"

FIELDS = (
    "date",
    "batter",
    "player_id",
    "game_pk",
    "stat",
    "line",
    "side",
    "book",
    "odds",
    "model_prob",
    "bet_prob",
    "fair_prob",
    "edge",
    "ev",
    "tier",
    "rating",
    "devigged",
    "delivery",
    "run_id",
    "rank",
    "points",
    "fit_pts",
    "fit_rv",
)


def name_key(name: str) -> str:
    """One hitter's one key, whatever the source spelled his name.

    ``Eugenio Suárez`` and ``Eugenio Suarez`` are both in the recorded ledger --
    the lineup feed accents him and the box score does not -- so every per-hitter
    cut counted him twice and neither half had his record. Compared on the
    accent-stripped, case-folded name, which is what the two sources disagree
    about; anything more aggressive would merge a father and a son.
    """
    stripped = unicodedata.normalize("NFKD", name)
    return "".join(c for c in stripped if not unicodedata.combining(c)).casefold().strip()


@dataclass(frozen=True)
class Composite:
    """Where the screen ranked one hitter, and what carried him there.

    The ledger recorded the tier and the rating and never the ordering, so after
    fifteen days and 275 graded rows the screen's central claim -- that its
    composite picks the right bat -- was the one thing the receipts could not
    grade. These four numbers are what the note's own table prints, carried down
    to every row of that hitter so an audit can sort the results by the order the
    screen put them in.
    """

    rank: int
    #: The composite less the two halves' floors: the part that discriminates.
    points: int
    #: The arsenal-fit points, and the run value per 100 on the pitches he will
    #: actually see. The fit is the term the note argues hardest from, so it is
    #: gradeable on its own rather than only inside the total.
    fit_pts: int
    fit_rv: float | None = None


@dataclass(frozen=True)
class Position:
    """One row of the note, as it was shown, before anything was known."""

    date: str
    batter: str
    player_id: int | None
    game_pk: int | None
    stat: str
    line: float | None
    side: str
    book: str
    odds: float | None
    model_prob: float
    fair_prob: float | None
    edge: float | None
    ev: float | None
    tier: str
    rating: str
    devigged: bool
    # The anchored probability the note printed and the card's screens bet on.
    # Absent on rows recorded before the board carried it, which fall back to the
    # model so an old row still grades against the number it showed.
    bet_prob: float | None = None
    #: What the opposing starter's delivery said about the batted-ball read the
    #: note faded him on: ``confirmed``, ``contradicted`` or ``unmeasured`` (see
    #: :mod:`mlb_engine.features.arm`). Empty on a row recorded before the field
    #: existed, or one whose arm the note could not attribute -- an unknown read
    #: is left unknown rather than filled with the common case.
    #:
    #: Recorded and never gated. Two graded slates have pointed the right way,
    #: which is four arms and not a sample, so the flag has to sit in the ledger
    #: before an audit can say whether it discriminates.
    delivery: str = ""
    #: Which run of the screen wrote this row. The screen runs more than once a
    #: day on purpose -- a second look once lineups post is the point of the
    #: morning job -- and until now the later run replaced the earlier one, here
    #: and on the state branch, so the file could not say which board the reader
    #: was shown and two machines could each delete the other's day. Empty on a
    #: row recorded before the field existed.
    run_id: str = ""
    #: The composite's ordering, carried per row (see :class:`Composite`).
    rank: int | None = None
    points: int | None = None
    fit_pts: int | None = None
    fit_rv: float | None = None

    @property
    def key(self) -> str:
        """Who this row is about: the id where there is one, the name otherwise."""
        return str(self.player_id) if self.player_id is not None else name_key(self.batter)

    @property
    def shown_prob(self) -> float:
        """The probability the note put in front of the reader."""
        return self.model_prob if self.bet_prob is None else self.bet_prob

    @property
    def label(self) -> str:
        stat = MARKET_LABEL.get(self.stat, self.stat)
        point = "" if self.line is None else f" {'o' if self.side == 'over' else 'u'}{self.line}"
        return f"{stat}{point}"

    @property
    def is_buy(self) -> bool:
        """Did the card bet it, as opposed to modelling it and passing?

        A display-only market is shown and never held, so it is not a buy however
        the pricer tiered it. New rows in one of those markets are not written at
        all (``positions_from_board``), but rows recorded before that held do sit
        in the ledger and must not roll up as tickets the card took.
        """
        return self.tier in ("Strong buy", "Moderate buy") and self.stat not in DISPLAY_ONLY


def positions_from_board(
    board: Board,
    as_of: Date,
    ratings: dict[str, str] | None = None,
    deliveries: dict[str, str] | None = None,
    composites: dict[str, Composite] | None = None,
    run_id: str = "",
) -> list[Position]:
    """The board's rows as ledger positions, tagged with what the note said.

    ``ratings`` maps batter name to the note's BUY/HOLD/AVOID, ``deliveries`` to
    the opposing starter's delivery verdict and ``composites`` to where the
    screen ranked him; all are supplied by the caller rather than computed here
    so this module never imports the report that renders them, and all are looked
    up on :func:`name_key` so an accent cannot lose a hitter his rating. Rows in a
    ``DISPLAY_ONLY`` market are shown by the note but held by nobody, so they are
    not positions.
    """
    rated = {name_key(k): v for k, v in (ratings or {}).items()}
    arms = {name_key(k): v for k, v in (deliveries or {}).items()}
    ranked = {name_key(k): v for k, v in (composites or {}).items()}
    return [
        _position(
            row,
            as_of,
            rated.get(name_key(row.batter), ""),
            arms.get(name_key(row.batter), ""),
            ranked.get(name_key(row.batter)),
            run_id,
        )
        for row in board.rows
        if row.stat not in DISPLAY_ONLY
    ]


def _position(
    row: BoardRow,
    as_of: Date,
    rating: str,
    delivery: str = "",
    composite: Composite | None = None,
    run_id: str = "",
) -> Position:
    return Position(
        date=as_of.isoformat(),
        batter=row.batter,
        player_id=row.player_id,
        game_pk=row.game_pk,
        stat=row.stat,
        line=row.line,
        side=row.side,
        book=row.book or "",
        odds=row.american,
        model_prob=round(row.model_prob, 4),
        bet_prob=round(row.bet_prob, 4) if row.bet_prob is not None else None,
        fair_prob=round(row.fair_prob, 4) if row.fair_prob is not None else None,
        edge=round(row.edge, 4) if row.edge is not None else None,
        ev=round(row.ev, 4) if row.ev is not None else None,
        tier=row.tier,
        rating=rating,
        devigged=row.devigged,
        delivery=delivery,
        run_id=run_id,
        rank=None if composite is None else composite.rank,
        points=None if composite is None else composite.points,
        fit_pts=None if composite is None else composite.fit_pts,
        fit_rv=(
            None
            if composite is None or composite.fit_rv is None
            else round(composite.fit_rv, 2)
        ),
    )


def _to_float(v: str) -> float | None:
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _to_int(v: str) -> int | None:
    f = _to_float(v)
    return int(f) if f is not None else None


def load(path: Path) -> list[Position]:
    """Every position ever recorded, oldest first; an absent file is empty."""
    if not path.exists():
        return []
    out: list[Position] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            model = _to_float(r.get("model_prob", ""))
            if model is None:
                continue
            out.append(
                Position(
                    date=r.get("date", ""),
                    batter=r.get("batter", ""),
                    player_id=_to_int(r.get("player_id", "")),
                    game_pk=_to_int(r.get("game_pk", "")),
                    stat=r.get("stat", ""),
                    line=_to_float(r.get("line", "")),
                    side=r.get("side", "over") or "over",
                    book=r.get("book", ""),
                    odds=_to_float(r.get("odds", "")),
                    model_prob=model,
                    bet_prob=_to_float(r.get("bet_prob", "")),
                    fair_prob=_to_float(r.get("fair_prob", "")),
                    edge=_to_float(r.get("edge", "")),
                    ev=_to_float(r.get("ev", "")),
                    tier=r.get("tier", ""),
                    rating=r.get("rating", ""),
                    devigged=str(r.get("devigged", "")).lower() in ("true", "1", "yes"),
                    delivery=r.get("delivery", "") or "",
                    run_id=r.get("run_id", "") or "",
                    rank=_to_int(r.get("rank", "")),
                    points=_to_int(r.get("points", "")),
                    fit_pts=_to_int(r.get("fit_pts", "")),
                    fit_rv=_to_float(r.get("fit_rv", "")),
                )
            )
    return out


def record(
    path: Path, positions: list[Position], as_of: Date, run_id: str = ""
) -> list[Position]:
    """Append this run's positions, replacing only what this run wrote before.

    Re-running the screen is normal -- a second look once lineups post is the
    point of the morning job -- and used to overwrite the whole day, which cost
    more than it saved: the day's rows then depended on which run wrote last, a
    re-run months later silently rewrote graded history, and two machines each
    deleted the other's board (8/26 held one screen's hitters here and a
    different screen's on the state branch, and the same fifteen days graded
    -3.9% or -8.6% depending on which copy was read).

    So a run is the unit: rows are keyed by ``run_id`` and only the same run's
    rows are replaced, which keeps re-running idempotent without making it
    destructive. With no ``run_id`` the old whole-day replacement stands, so a
    caller that has no notion of a run cannot accumulate duplicates.
    """
    day = as_of.isoformat()
    kept = [
        p
        for p in load(path)
        if p.date != day or (run_id != "" and p.run_id != run_id)
    ]
    written = [p if p.run_id == run_id else replace(p, run_id=run_id) for p in positions]
    rows = kept + written
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(FIELDS))
        w.writeheader()
        for p in rows:
            w.writerow({k: "" if v is None else v for k, v in asdict(p).items()})
    return rows



def runs_for(path: Path, day: Date) -> list[str]:
    """Every run of the screen recorded for that day, earliest identifier first."""
    return sorted({p.run_id for p in load(path) if p.date == day.isoformat()})


def positions_for(path: Path, day: Date, run_id: str | None = None) -> list[Position]:
    """One run's positions for that day, less the markets the note only displays.

    The file keeps every run, so a day can hold more than one board and the
    scorecard has to say which it graded. Default is the day's last run -- the
    board the reader ended the day with -- and ``run_id`` pins an earlier one,
    which is what makes the grade reproducible: the rows behind a printed
    scorecard can be named rather than inferred from whatever wrote last.

    Markets are filtered on the way out as well as in, so the boards already
    written with HR rows grade the same way as the ones written after: a
    scorecard whose meaning changed on the day of a code change is worse than no
    scorecard.
    """
    rows = [p for p in load(path) if p.date == day.isoformat() and p.stat not in DISPLAY_ONLY]
    if not rows:
        return []
    want = max({p.run_id for p in rows}) if run_id is None else run_id
    return [p for p in rows if p.run_id == want]


@dataclass(frozen=True)
class GradedPosition:
    position: Position
    result: str  # win / loss / push
    actual: int
    units: float


def grade_positions(
    positions: list[Position], results: dict[int, GameResult]
) -> tuple[list[GradedPosition], int]:
    """Grade what can be graded; return the rows and the count that could not be.

    Ungradeable means the game never finished or the hitter never batted, and it
    is returned as a count rather than folded into the record: a voided prop is
    not evidence about the screen either way.
    """
    graded: list[GradedPosition] = []
    voided = 0
    for p in positions:
        res = results.get(p.game_pk) if p.game_pk is not None else None
        if res is None or not res.final or p.player_id is None or p.line is None:
            voided += 1
            continue
        outcome = grade_batter(res, p.player_id, p.stat, p.line, p.side)
        if outcome is None:
            voided += 1
            continue
        graded.append(
            GradedPosition(
                position=p,
                result=outcome,
                actual=batter_actual(res, p.player_id, p.stat),
                units=pnl_units(outcome, p.odds),
            )
        )
    return graded, voided


@dataclass(frozen=True)
class Record:
    """A win/loss record over some subset, with the units it would have paid."""

    label: str
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    units: float = 0.0

    @property
    def n(self) -> int:
        return self.wins + self.losses + self.pushes

    @property
    def decided(self) -> int:
        return self.wins + self.losses

    @property
    def win_pct(self) -> float | None:
        return self.wins / self.decided if self.decided else None

    @property
    def roi(self) -> float | None:
        return self.units / self.n if self.n else None


def _record(label: str, graded: list[GradedPosition]) -> Record:
    return Record(
        label=label,
        wins=sum(1 for g in graded if g.result == WIN),
        losses=sum(1 for g in graded if g.result == LOSS),
        pushes=sum(1 for g in graded if g.result == PUSH),
        units=round(sum(g.units for g in graded), 4),
    )


def _brier(pairs: list[tuple[float, int]]) -> float | None:
    if not pairs:
        return None
    return round(sum((p - o) ** 2 for p, o in pairs) / len(pairs), 4)


@dataclass
class Scorecard:
    """Yesterday's board, graded three ways."""

    day: str
    overall: Record
    voided: int = 0
    #: Which recorded run of the screen these rows came from, so the printed
    #: scorecard names its own evidence instead of meaning whatever wrote last.
    run_id: str = ""
    by_tier: list[Record] = field(default_factory=list)
    by_rating: list[Record] = field(default_factory=list)
    by_market: list[Record] = field(default_factory=list)
    model_brier: float | None = None
    market_brier: float | None = None
    scored_probs: int = 0
    mean_model_prob: float | None = None
    mean_market_prob: float | None = None
    # The printed (anchored) probability, scored on the same rows. Kept apart
    # from ``model_brier`` on purpose: the model number is what a calibration
    # refit has to measure, and the shown number is what the reader was told.
    shown_brier: float | None = None
    mean_shown_prob: float | None = None

    @property
    def graded(self) -> int:
        return self.overall.n

    @property
    def model_beat_market(self) -> bool | None:
        """Did the model's probabilities score better than the no-vig line's?

        ``None`` when nothing was gradeable two-sided, which is common on a small
        board and must not read as a tie.
        """
        if self.model_brier is None or self.market_brier is None:
            return None
        return self.model_brier < self.market_brier

    @property
    def shown_beat_market(self) -> bool | None:
        """Did the number the note printed score better than the no-vig line's?"""
        if self.shown_brier is None or self.market_brier is None:
            return None
        return self.shown_brier < self.market_brier


def scorecard(day: Date, graded: list[GradedPosition], voided: int = 0) -> Scorecard:
    """Roll the graded rows up into the note's scorecard."""
    decided = [g for g in graded if g.result != PUSH]
    # Only a two-sided quote has a real no-vig mark; a one-way price would hand
    # the market a probability nobody stripped the hold out of, which is exactly
    # the comparison this scorecard exists to make honestly.
    pairs = [
        (g, 1 if g.result == WIN else 0)
        for g in decided
        if g.position.devigged and g.position.fair_prob is not None
    ]
    model = [(g.position.model_prob, o) for g, o in pairs]
    shown = [(g.position.shown_prob, o) for g, o in pairs]
    market = [(g.position.fair_prob or 0.0, o) for g, o in pairs]
    tiers = sorted({g.position.tier for g in graded})
    ratings = sorted({g.position.rating for g in graded if g.position.rating})
    markets = sorted({g.position.stat for g in graded})
    runs = sorted({g.position.run_id for g in graded})
    return Scorecard(
        day=day.isoformat(),
        overall=_record("all", graded),
        voided=voided,
        run_id=runs[-1] if len(runs) == 1 else "",
        by_tier=[_record(t, [g for g in graded if g.position.tier == t]) for t in tiers],
        by_rating=[_record(r, [g for g in graded if g.position.rating == r]) for r in ratings],
        by_market=[
            _record(MARKET_LABEL.get(m, m), [g for g in graded if g.position.stat == m])
            for m in markets
        ],
        model_brier=_brier(model),
        market_brier=_brier(market),
        scored_probs=len(pairs),
        mean_model_prob=(round(sum(p for p, _ in model) / len(model), 4) if model else None),
        mean_market_prob=(round(sum(p for p, _ in market) / len(market), 4) if market else None),
        shown_brier=_brier(shown),
        mean_shown_prob=(round(sum(p for p, _ in shown) / len(shown), 4) if shown else None),
    )
