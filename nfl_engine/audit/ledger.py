"""One row per selection, kept forever, and what the rows are allowed to prove.

Three decisions carried over from the MLB ledger, each of which was learned the
expensive way:

**Rejections are rows.** A screen that only leaves fingerprints on the bets it
approves cannot be graded. Every veto is persisted in ``screens`` on a
``Tier.PASS`` row, so "did ``model_disagrees`` remove winners?" is a query.

**PPV and NPV are scored against the unconditional base rate, not against zero.**
A 52% win rate is not a skill on a market whose base rate is 50%. Every metrics
row therefore carries ``base_rate`` and the lift over it, and ``required_win_pct``
-- the mean 1/decimal of the bets counted -- because a win rate quoted without the
bar it had to clear reads as a success while losing money.

**One position per line.** The same line at two books is one bet; pooling both
double-counts it in ROI and in every confusion matrix. The market layer collapses
to the best price before anything reaches here.

CLV is stored in probability points against the closing number, on the side
actually taken, and it is the primary measurement for this engine rather than a
footnote: with the mean pinned to the market by phase 3, what is being tested is
whether the *execution* is any good, and beating the close is the only clean read
on that available before hundreds of graded bets exist.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from nfl_engine.market.ev import MONEYLINE, SPREAD, TOTAL, PricedBet, ev
from nfl_engine.market.fair import DEFAULT_METHOD, devig
from nfl_engine.market.odds import american_to_decimal, american_to_prob
from nfl_engine.market.screens import Tier, tier_of

WIN, LOSS, PUSH, VOID = "win", "loss", "push", "void"
OVER, UNDER = "over", "under"
ENGINE = "engine"
# Every row records whether money was at risk. Nothing in this repository writes
# LIVE: the dry run has no staking path at all, and the column exists so that the
# day one is added, a paper record cannot be quoted as a real one by accident.
PAPER, LIVE = "paper", "live"


@dataclass
class LedgerEntry:
    season: int
    week: int
    date: str
    matchup: str
    market: str  # moneyline | spread | total
    side: str  # team abbrev, or over/under
    line: float | None
    book: str
    odds: float | None  # american, at the price actually taken
    opposite_odds: float | None  # the other side at the same book, for the fade
    tier: str
    model_prob: float  # the simulator's conditional probability
    fair_prob: float | None  # de-vigged consensus on the same line
    ev_model: float | None
    ev_fair: float | None  # the execution edge this row was bought for
    paired_books: int
    # Which vetoes fired, semicolon-joined. Empty on a bought row.
    screens: str = ""
    result: str = ""  # win | loss | push | void
    pnl: float = 0.0  # net units on a 1u stake
    home_score: int | None = None
    away_score: int | None = None
    # Closing line value on the side taken: the closing price, its no-vig
    # probability, the probability points the market moved our way, and the EV of
    # our price under the closing probability.
    close_odds: float | None = None
    close_prob: float | None = None
    clv: float | None = None
    clv_ev: float | None = None
    # Whose call this row is. An outside benchmark writes its own name so it can
    # sit in the same ledger, graded identically, and be excluded from every
    # measurement of us.
    source: str = ENGINE
    # paper | live. See PAPER above.
    mode: str = PAPER
    # When the price on this row was seen. Distinct from ``date`` (kickoff): an
    # execution edge is a claim about a price that existed at a moment, and
    # without the moment the claim cannot be checked against the archive.
    captured_at: str = ""


LEDGER_FIELDS = [f.name for f in fields(LedgerEntry)]


def entry_from_bet(
    bet: PricedBet,
    *,
    season: int,
    week: int,
    date: str,
    captured_at: str = "",
    mode: str = PAPER,
) -> LedgerEntry:
    return LedgerEntry(
        season=season,
        week=week,
        date=date,
        matchup=bet.matchup,
        market=bet.market,
        side=bet.side,
        line=bet.line,
        book=bet.book,
        odds=bet.american,
        opposite_odds=bet.opposite_american,
        tier=tier_of(bet).value,
        model_prob=round(bet.model_prob, 6),
        fair_prob=None if bet.fair_prob is None else round(bet.fair_prob, 6),
        ev_model=round(bet.ev_model, 6),
        ev_fair=None if bet.ev_fair is None else round(bet.ev_fair, 6),
        paired_books=0 if bet.fair is None else bet.fair.paired_books,
        screens=";".join(bet.screens),
        mode=mode,
        captured_at=captured_at,
    )


# -- grading --------------------------------------------------------------
def grade(entry: LedgerEntry, home_score: int, away_score: int, *, home: str) -> LedgerEntry:
    """Settle one row against a final score, on the side that was actually bet."""
    margin = home_score - away_score
    total = home_score + away_score
    if entry.market == MONEYLINE:
        own = float(margin if entry.side == home else -margin)
        result = WIN if own > 0 else PUSH if own == 0 else LOSS
    elif entry.market == SPREAD:
        if entry.line is None:
            result = VOID
        else:
            own = float(margin if entry.side == home else -margin) + entry.line
            result = WIN if own > 0 else PUSH if own == 0 else LOSS
    elif entry.market == TOTAL:
        if entry.line is None:
            result = VOID
        elif total == entry.line:
            result = PUSH
        else:
            over_hit = total > entry.line
            result = WIN if over_hit == (entry.side == OVER) else LOSS
    else:
        result = VOID
    entry.result = result
    entry.home_score = home_score
    entry.away_score = away_score
    entry.pnl = pnl_units(result, entry.odds)
    return entry


def pnl_units(result: str, odds: float | None) -> float:
    if result == WIN:
        decimal = american_to_decimal(odds) if odds is not None else 1.91
        return round(decimal - 1.0, 4)
    if result == LOSS:
        return -1.0
    return 0.0


# -- closing line value ---------------------------------------------------
def apply_close(
    entry: LedgerEntry,
    close_american: float,
    close_opposite: float | None,
    *,
    method: str = DEFAULT_METHOD,
) -> LedgerEntry:
    """Score the row against the closing number on the same side.

    Without the closing price's *opposite* side there is no hold to remove, so
    the raw implied probability is used and flagged by leaving ``close_prob``
    equal to it: an unpaired close overstates the closing probability by roughly
    half the hold, which would flatter CLV by about 1.2pp on a typical NFL
    two-way market. Better to record the number that exists than to invent one.
    """
    entry.close_odds = close_american
    implied = american_to_prob(close_american)
    if close_opposite is None:
        close_prob = implied
    else:
        close_prob = devig([implied, american_to_prob(close_opposite)], method)[0]
    entry.close_prob = round(close_prob, 6)
    entry.clv = round(close_prob - taken_prob(entry, method=method), 6)
    if entry.odds is not None:
        entry.clv_ev = round(ev(close_prob, american_to_decimal(entry.odds)), 6)
    return entry


def taken_prob(entry: LedgerEntry, *, method: str = DEFAULT_METHOD) -> float:
    """No-vig probability of the price we actually struck, on our side.

    Deliberately *not* the consensus ``fair_prob``: CLV asks whether the number we
    took beat the number the market settled on, so both ends must be the same kind
    of quantity. Comparing our consensus estimate to the close would measure the
    consensus drifting, and would call a bet good when the whole board moved with
    it and we got the worst of it.
    """
    if entry.odds is None:
        return 0.0
    implied = american_to_prob(entry.odds)
    if entry.opposite_odds is None:
        return implied
    return devig([implied, american_to_prob(entry.opposite_odds)], method)[0]


# -- persistence ----------------------------------------------------------
def save_ledger(path: Path, entries: list[LedgerEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))


def load_ledger(path: Path) -> list[LedgerEntry]:
    if not path.exists():
        return []
    out: list[LedgerEntry] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out.append(
                LedgerEntry(
                    season=int(row.get("season") or 0),
                    week=int(row.get("week") or 0),
                    date=row.get("date", ""),
                    matchup=row.get("matchup", ""),
                    market=row.get("market", ""),
                    side=row.get("side", ""),
                    line=_float(row.get("line")),
                    book=row.get("book", ""),
                    odds=_float(row.get("odds")),
                    opposite_odds=_float(row.get("opposite_odds")),
                    tier=row.get("tier", Tier.PASS.value),
                    model_prob=_float(row.get("model_prob")) or 0.0,
                    fair_prob=_float(row.get("fair_prob")),
                    ev_model=_float(row.get("ev_model")),
                    ev_fair=_float(row.get("ev_fair")),
                    paired_books=int(row.get("paired_books") or 0),
                    screens=row.get("screens", ""),
                    result=row.get("result", ""),
                    pnl=_float(row.get("pnl")) or 0.0,
                    home_score=_int(row.get("home_score")),
                    away_score=_int(row.get("away_score")),
                    close_odds=_float(row.get("close_odds")),
                    close_prob=_float(row.get("close_prob")),
                    clv=_float(row.get("clv")),
                    clv_ev=_float(row.get("clv_ev")),
                    source=row.get("source", ENGINE),
                    mode=row.get("mode") or PAPER,
                    captured_at=row.get("captured_at", ""),
                )
            )
    return out


def update_ledger(path: Path, new_entries: list[LedgerEntry]) -> list[LedgerEntry]:
    """Replace every row for the weeks being written, then append.

    Keyed on season/week rather than appended blindly so that re-grading a week
    corrects it instead of counting it twice -- double-counted rows are the
    quietest way to make a record look longer than it is.
    """
    touched = {(entry.season, entry.week) for entry in new_entries}
    kept = [e for e in load_ledger(path) if (e.season, e.week) not in touched]
    merged = kept + new_entries
    save_ledger(path, merged)
    return merged


def position_key(entry: LedgerEntry) -> tuple[str, ...]:
    """What makes two rows the same position, for repeated captures.

    The book and the rung are part of the identity: -3 and -3.5 are different
    bets, and the same rung at two books is two prices. The price itself is not,
    which is the whole point -- a re-run at a moved number must not silently
    become a second position.
    """
    line = "" if entry.line is None else f"{entry.line:g}"
    return (
        str(entry.season),
        str(entry.week),
        entry.matchup,
        entry.market,
        entry.side,
        line,
        entry.book,
        entry.source,
    )


def merge_ledger(path: Path, new_entries: list[LedgerEntry]) -> list[LedgerEntry]:
    """Add positions that are new; leave every existing row exactly as it stands.

    This is what makes a repeated capture safe, and it encodes a betting decision
    rather than a storage one: **the price of record is the first one seen.** A dry
    run that re-priced Sunday's board every hour and kept the latest number would
    quietly grant itself the best price of the week in hindsight and destroy the
    CLV measurement, which is the only honest read available before hundreds of
    graded bets exist. Grading and closing rewrite rows through
    :func:`update_ledger`; pricing only ever appends.
    """
    existing = load_ledger(path)
    seen = {position_key(entry) for entry in existing}
    fresh = [entry for entry in new_entries if position_key(entry) not in seen]
    if fresh:
        save_ledger(path, existing + fresh)
    return fresh


def _float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int(value: str | None) -> int | None:
    out = _float(value)
    return None if out is None else int(out)


# -- metrics --------------------------------------------------------------
@dataclass
class Metrics:
    label: str
    n: int
    wins: int
    losses: int
    pushes: int
    win_pct: float
    required_win_pct: float
    ppv: float
    npv: float
    sensitivity: float
    specificity: float
    base_rate: float
    ppv_lift: float
    npv_lift: float
    roi: float
    units: float
    mean_clv: float
    clv_beat_pct: float


def _safe(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def metrics(
    entries: list[LedgerEntry], is_positive: Callable[[LedgerEntry], bool], label: str
) -> Metrics:
    """Confusion matrix plus ROI and CLV for one definition of "we picked it".

    ``base_rate`` is the win rate of *every* graded row in scope, so PPV lift
    answers the only question that matters about a screen: does selecting on it
    beat not selecting at all?
    """
    tp = fp = fn = tn = pushes = 0
    stake = units = breakeven = 0.0
    clvs: list[float] = []
    graded = [e for e in entries if e.result in (WIN, LOSS, PUSH)]
    live = [e for e in graded if e.result != PUSH]
    for entry in graded:
        positive = is_positive(entry)
        if entry.result == PUSH:
            if positive:
                pushes += 1
            continue
        won = entry.result == WIN
        if positive and won:
            tp += 1
        elif positive and not won:
            fp += 1
        elif not positive and won:
            fn += 1
        else:
            tn += 1
        if positive:
            stake += 1.0
            units += entry.pnl
            decimal = american_to_decimal(entry.odds) if entry.odds is not None else 1.91
            breakeven += 1.0 / decimal
            if entry.clv is not None:
                clvs.append(entry.clv)
    base = _safe(sum(1 for e in live if e.result == WIN), len(live))
    ppv = _safe(tp, tp + fp)
    npv = _safe(tn, tn + fn)
    return Metrics(
        label=label,
        n=tp + fp,
        wins=tp,
        losses=fp,
        pushes=pushes,
        win_pct=ppv,
        required_win_pct=_safe(breakeven, stake),
        ppv=ppv,
        npv=npv,
        sensitivity=_safe(tp, tp + fn),
        specificity=_safe(tn, tn + fp),
        base_rate=base,
        ppv_lift=round(ppv - base, 4),
        npv_lift=round(npv - (1.0 - base), 4),
        roi=_safe(units, stake),
        units=round(units, 3),
        mean_clv=round(sum(clvs) / len(clvs), 5) if clvs else 0.0,
        clv_beat_pct=_safe(sum(1 for c in clvs if c > 0), len(clvs)),
    )


def tier_metrics(entries: list[LedgerEntry]) -> list[Metrics]:
    engine = [e for e in entries if e.source == ENGINE]
    buys = {Tier.STRONG.value, Tier.MODERATE.value}
    return [
        metrics(engine, lambda e: e.tier == Tier.STRONG.value, Tier.STRONG.value),
        metrics(engine, lambda e: e.tier == Tier.MODERATE.value, Tier.MODERATE.value),
        metrics(engine, lambda e: e.tier in buys, "Buy (S+M)"),
        metrics(engine, lambda e: e.tier == Tier.PASS.value, Tier.PASS.value),
    ]


def screen_metrics(entries: list[LedgerEntry]) -> list[Metrics]:
    """One row per veto, over the bets it rejected.

    A screen earns its place by its rejections *losing*: NPV above the fade base
    rate means it removed losers. A screen whose rejections won is costing money
    and is visible here rather than in a year's hindsight.
    """
    engine = [e for e in entries if e.source == ENGINE]
    names = sorted({name for e in engine for name in e.screens.split(";") if name})

    def rejected_by(name: str) -> Callable[[LedgerEntry], bool]:
        return lambda entry: name in entry.screens.split(";")

    return [metrics(engine, rejected_by(name), f"screen:{name}") for name in names]


def market_metrics(entries: list[LedgerEntry]) -> list[Metrics]:
    engine = [e for e in entries if e.source == ENGINE]
    buys = {Tier.STRONG.value, Tier.MODERATE.value}
    out = []
    for market in (MONEYLINE, SPREAD, TOTAL):
        rows = [e for e in engine if e.market == market]
        if rows:
            out.append(metrics(rows, lambda e: e.tier in buys, f"market:{market}"))
    return out
