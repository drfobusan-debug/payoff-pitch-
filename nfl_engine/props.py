"""Pricing the archived prop board, and why nothing it produces is a bet.

The capture layer has been archiving player-prop prices since the preseason
(:mod:`nfl_engine.data.capture`) precisely because that history exists nowhere
else. This module is the other half: it turns those archived quotes into priced
rows, with the de-vig, the pairing rule and the price screens the game markets
already use, plus the guards the MLB engine had to learn on its own prop card.

**Everything here is research, and the code enforces it.** Every row is stamped
``research_only`` -- a veto like any other, on the ledger like any other -- so no
path in this repository can stake one. That is not caution for its own sake:

*Almost nothing here has a measured edge at the numbers a book posts.* Inside the
usage range where lines exist, out of time on 2022-2025, only receptions clears the
base rate of its own rows (Brier 0.2416 against 0.2444); targets and carries are
ties; and the yardage and quarterback markets are *worse* than the base rate, so
they are retired per market -- see :mod:`nfl_engine.models.player` for the table.

*And a pseudo-line is not a price.* The Brier above is measured at lines drawn on
our own projection, which is the friendliest possible test; the real question is
whether the number a book hung can be beaten, and until the forward archive holds
graded prop closes there is no way to ask it. The MLB engine answered that
question the expensive way, by betting first.

What the layer does do, so that the day the archive is deep enough the answer is
one screen away rather than a rebuild: it prices every archived quote, de-vigs the
same book's two sides and refuses to invent a partner for an unpaired one, caps the
price band, refuses a projection under the usage floor, and removes correlated
legs -- one per player, and one per team and direction, because a quarterback's
passing yards and his receiver's receiving yards are one opinion sold twice.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from nfl_engine.config import data_dir
from nfl_engine.data.capture import QuoteRow
from nfl_engine.features.usage import normalise
from nfl_engine.market.board import OVER, UNDER, MarketQuote
from nfl_engine.market.ev import ev
from nfl_engine.market.fair import DEFAULT_METHOD, FairPrice, fair_from_quotes
from nfl_engine.market.odds import american_to_decimal
from nfl_engine.models.player import (
    ATTEMPTS,
    CARRIES,
    COMPLETIONS,
    PASSING_YARDS,
    RECEIVING_YARDS,
    RECEPTIONS,
    RUSHING_YARDS,
    TARGETS,
    Projection,
    prob_over,
)

log = logging.getLogger(__name__)

RESEARCH = "research"
# What the projections were fitted on, stamped on every row the same way the
# calibration basis is: a row measured under one basis cannot be quoted as
# evidence for another.
BASIS = "usage-shrunk-2016-2021"

# Odds API market key -> the weekly stat it is a line on.
MARKET_STATS = {
    "player_pass_attempts": ATTEMPTS,
    "player_pass_completions": COMPLETIONS,
    "player_pass_yds": PASSING_YARDS,
    "player_rush_attempts": CARRIES,
    "player_rush_yds": RUSHING_YARDS,
    "player_receptions": RECEPTIONS,
    "player_reception_yds": RECEIVING_YARDS,
}
# Targets are projected and reported, but no book in the feed quotes them; the
# mapping exists so a feed that starts to is priced rather than dropped.
MARKET_STATS["player_targets"] = TARGETS

# Markets whose projection measured *worse* than the base rate out of time: every
# quarterback market (attempts 0.2537 vs 0.2500, completions 0.2551 vs 0.2498,
# passing yards 0.2575 vs 0.2500) and both yardage markets (receiving 0.2452 vs
# 0.2430, rushing 0.2463 vs 0.2442). Retired per market rather than as a family, so
# a later refit can revive one on its own evidence.
RETIRED_MARKETS = frozenset(
    {
        "player_pass_attempts",
        "player_pass_completions",
        "player_pass_yds",
        "player_reception_yds",
        "player_rush_yds",
    }
)

# Prop holds run 6-8% against 2.5% on a game moneyline, so the execution bar is
# double the game layer's: inside 3c of a prop consensus is inside the noise of
# which books were in the sample.
MIN_EXECUTION_EV = 0.030
# Beyond this the projection's departure from the consensus reads as our error
# rather than the market's. Wider than the game layer's 0.060 because a prop
# consensus is thinner, and still far short of the 20pp gaps a usage model
# produces when it has missed a role change.
MAX_DISAGREEMENT = 0.100
# Price ceiling. A 2pp probability error costs five times as much at +400 as at
# -150, and a prop board's long side is where the hold is concentrated.
MAX_AMERICAN = 200.0
MIN_PAIRED_BOOKS = 2
# One leg per player, and one per team and direction: an over on a passing market
# and an over on that team's receiving market are the same opinion twice, which is
# how the MLB card ended up double-counting total bases and RBIs.
MAX_LEGS_PER_TEAM = 1

# Vetoes.
RESEARCH_ONLY = "research_only"
NO_PROJECTION = "no_projection"
RETIRED = "retired_market"
BELOW_FLOOR = "below_usage_floor"
UNPAIRED = "unpaired"
THIN = "thin_market"
MODEL_ONLY = "model_only"
NO_EDGE = "no_execution_edge"
MODEL_NEGATIVE = "model_negative"
DISAGREES = "model_disagrees"
LONGSHOT = "longshot"
DUPLICATE_PLAYER = "duplicate_player_leg"
CORRELATED = "correlated_leg"


@dataclass(frozen=True)
class PricedProp:
    """One archived prop quote, priced, screened and marked as research."""

    captured_at: str
    season: int
    week: int
    matchup: str
    market: str
    player: str
    side: str  # over | under
    line: float | None
    book: str
    american: float
    opposite_american: float | None
    stat: str
    projection: float | None
    projection_games: int
    model_prob: float
    push_prob: float
    fair_prob: float | None
    paired_books: int
    ev_model: float | None
    ev_fair: float | None
    edge_vs_fair: float | None
    screens: tuple[str, ...]
    basis: str = BASIS
    mode: str = RESEARCH

    @property
    def decimal(self) -> float:
        return american_to_decimal(self.american)

    def label(self) -> str:
        line = "" if self.line is None else f" {self.line:g}"
        return f"{self.player} {self.side.upper()}{line} ({self.stat})"


FIELDS = [f.name for f in fields(PricedProp)]


def stat_for(market: str) -> str | None:
    return MARKET_STATS.get(market)


def _quotes_by_line(
    rows: list[QuoteRow],
) -> dict[tuple[str, str, str, float | None], list[QuoteRow]]:
    """Group archived rows by the position they are all quotes on."""
    grouped: dict[tuple[str, str, str, float | None], list[QuoteRow]] = {}
    for row in rows:
        if row.side not in (OVER, UNDER):
            continue
        grouped.setdefault((row.matchup, row.market, normalise(row.player), row.line), []).append(
            row
        )
    return grouped


def _fair(rows: list[QuoteRow], method: str) -> FairPrice | None:
    return fair_from_quotes(
        [MarketQuote(row.book, row.american, row.opposite_american) for row in rows], method=method
    )


def screen_prop(prop: PricedProp, *, below_floor: bool = False) -> tuple[str, ...]:
    """Every veto this row trips, research veto first.

    Reported in full rather than short-circuited: a row that only trips
    ``research_only`` is the interesting one, and it is invisible if the first
    veto ends the list.
    """
    reasons: list[str] = [RESEARCH_ONLY]

    if prop.market in RETIRED_MARKETS:
        reasons.append(RETIRED)
    if prop.projection is None:
        reasons.append(NO_PROJECTION)
    elif below_floor:
        reasons.append(BELOW_FLOOR)

    if prop.paired_books == 0:
        reasons.append(UNPAIRED)
    elif prop.paired_books < MIN_PAIRED_BOOKS:
        reasons.append(THIN)

    if prop.fair_prob is None:
        reasons.append(MODEL_ONLY)
    elif prop.ev_fair is not None and prop.ev_fair <= MIN_EXECUTION_EV:
        reasons.append(NO_EDGE)

    if prop.ev_model is not None and prop.ev_model <= 0.0:
        reasons.append(MODEL_NEGATIVE)
    if prop.edge_vs_fair is not None and abs(prop.edge_vs_fair) > MAX_DISAGREEMENT:
        reasons.append(DISAGREES)
    if prop.american > MAX_AMERICAN:
        reasons.append(LONGSHOT)

    return tuple(dict.fromkeys(reasons))


def _price_one(
    row: QuoteRow,
    *,
    stat: str,
    projection: Projection | None,
    fair: FairPrice | None,
) -> PricedProp:
    line = row.line
    if projection is None or line is None:
        model_prob, push_prob = 0.0, 0.0
    else:
        prob = prob_over(stat, projection.mean, line)
        push_prob = prob.push
        model_prob = prob.conditional if row.side == OVER else 1.0 - prob.conditional
    decimal = american_to_decimal(row.american)
    fair_prob = fair.prob if fair is not None and fair.is_trustworthy(min_paired=1) else None
    below_floor = projection is not None and not projection.clears_floor()
    prop = PricedProp(
        captured_at=row.captured_at,
        season=row.season,
        week=row.week,
        matchup=row.matchup,
        market=row.market,
        player=row.player,
        side=row.side,
        line=line,
        book=row.book,
        american=row.american,
        opposite_american=row.opposite_american,
        stat=stat,
        projection=None if projection is None else round(projection.mean, 3),
        projection_games=0 if projection is None else projection.games,
        model_prob=round(model_prob, 6),
        push_prob=round(push_prob, 6),
        fair_prob=None if fair_prob is None else round(fair_prob, 6),
        paired_books=0 if fair is None else fair.paired_books,
        ev_model=None if projection is None else round(ev(model_prob, decimal), 6),
        ev_fair=None if fair_prob is None else round(ev(fair_prob, decimal), 6),
        edge_vs_fair=(
            None if fair_prob is None or projection is None else round(model_prob - fair_prob, 6)
        ),
        screens=(),
        mode=RESEARCH,
    )
    return replace(prop, screens=screen_prop(prop, below_floor=below_floor))


def price_props(
    rows: list[QuoteRow],
    projections: dict[tuple[str, str], Projection],
    *,
    method: str = DEFAULT_METHOD,
    best_price_only: bool = True,
) -> list[PricedProp]:
    """Price every archived prop quote, screens and correlation guards applied."""
    priced: list[PricedProp] = []
    for (_, market, name, _), quotes in _quotes_by_line(rows).items():
        stat = stat_for(market)
        if stat is None:
            continue
        projection = projections.get((name, stat))
        by_side: dict[str, list[QuoteRow]] = {}
        for row in quotes:
            by_side.setdefault(row.side, []).append(row)
        for side_rows in by_side.values():
            fair = _fair(side_rows, method)
            priced.extend(
                _price_one(row, stat=stat, projection=projection, fair=fair) for row in side_rows
            )
    if best_price_only:
        priced = best_by_position(priced)
    return decorrelate(priced)


def best_by_position(props: list[PricedProp]) -> list[PricedProp]:
    """One row per player/market/line/side: the same line at two books is one bet."""
    best: dict[tuple[str, str, str, float | None, str], PricedProp] = {}
    for prop in props:
        key = (prop.matchup, prop.market, normalise(prop.player), prop.line, prop.side)
        current = best.get(key)
        if current is None or prop.decimal > current.decimal:
            best[key] = prop
    return list(best.values())


def _edge(prop: PricedProp) -> float:
    return prop.ev_fair if prop.ev_fair is not None else float("-inf")


def decorrelate(props: list[PricedProp]) -> list[PricedProp]:
    """Veto the legs that repeat an opinion already counted.

    Rows are never dropped -- a correlated leg is a row with a reason, so the
    guard itself can be graded. Only rows that survived every other screen
    compete: vetoing a second leg on behalf of a first that is itself a Pass would
    hide the guard behind an unrelated refusal.
    """
    ordered = sorted(props, key=lambda p: -_edge(p))
    seen_players: set[str] = set()
    per_team: dict[tuple[str, str], int] = {}
    out: list[PricedProp] = []
    for prop in ordered:
        blocking = [r for r in prop.screens if r != RESEARCH_ONLY]
        if blocking:
            out.append(prop)
            continue
        player = normalise(prop.player)
        team_key = (prop.matchup, prop.side)
        if player in seen_players:
            out.append(replace(prop, screens=(*prop.screens, DUPLICATE_PLAYER)))
            continue
        if per_team.get(team_key, 0) >= MAX_LEGS_PER_TEAM:
            out.append(replace(prop, screens=(*prop.screens, CORRELATED)))
            continue
        seen_players.add(player)
        per_team[team_key] = per_team.get(team_key, 0) + 1
        out.append(prop)
    return out


def research_path(season: int, week: int, *, root: Path | None = None) -> Path:
    """Where the research rows live: never the ledger the engine's record is cut from."""
    return (root or data_dir()) / "props" / f"research_{season}_wk{week:02d}.csv"


def write_research(
    props: list[PricedProp], *, season: int, week: int, root: Path | None = None
) -> Path | None:
    """Append the priced rows to the week's research file, or report the failure.

    Deliberately a separate file from ``ledger.csv``. Research rows in the ledger
    would be one ``source`` filter away from being counted as a record, and the
    filter is the kind of thing that survives a refactor by accident.
    """
    path = research_path(season, week, root=root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if not exists:
                writer.writeheader()
            for prop in props:
                row = asdict(prop)
                row["screens"] = ";".join(prop.screens)
                writer.writerow(row)
    except OSError as exc:
        log.warning("could not write prop research to %s: %s", path, exc)
        return None
    return path


def summary(props: list[PricedProp]) -> list[str]:
    """What the archive priced, and what stopped it -- the only output that ships."""
    if not props:
        return ["props: nothing priced (no archived quotes for this week)"]
    lines = [
        f"props: {len(props)} rows priced, basis {BASIS}, mode {RESEARCH} -- nothing bettable",
    ]
    counts: dict[str, int] = {}
    for prop in props:
        for reason in prop.screens:
            counts[reason] = counts.get(reason, 0) + 1
    for reason, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {reason}: {n}")
    would_bet = [p for p in props if tuple(r for r in p.screens if r != RESEARCH_ONLY) == ()]
    lines.append(
        f"  rows that only research_only stops: {len(would_bet)}"
        " (these are what the forward archive has to grade before any of it is a bet)"
    )
    return lines
