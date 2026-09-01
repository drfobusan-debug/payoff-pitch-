"""Where a slate's rows died: how many survive each stage, and what took the rest.

A card with no plays on it reads the same whether the board was quiet, the books
never quoted half of it, or one screen refused everything that survived the
others -- and those are opposite problems. The funnel separates them by counting
the same rows the ledger grades at each stage they have to clear:

    candidates -> priced by a book -> positive EV -> clears the price screen
    -> survives the market gates -> bought

and naming, per market, the gate that closed it. Nothing here changes a
decision; it reports the decisions already stamped on each row by
:attr:`Recommendation.pass_gate`, which is why a zero-buy slate can be read
without re-running the pipeline.

The stage a row is attributed to is the *first* one it failed, so the counts sum
to the candidate total and a gate's number is rows it alone refused rather than
rows it would have refused had they reached it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from mlb_engine.config import EVThresholds
from mlb_engine.features.lineup_lock import LineupLockGate
from mlb_engine.market.tiers import Tier
from mlb_engine.recommendations import Recommendation

# The screens in :func:`mlb_engine.market.tiers.price_screen`, which act on the
# price alone. Everything else stamped on ``pass_gate`` is a market gate applied
# after it, so the two stages can be told apart without the pipeline's help.
PRICE_SCREEN_GATES: frozenset[str] = frozenset(
    {
        "price_ceiling",
        "ev_floor",
        "thin_edge",
        "edge_ceiling",
        "prob_floor",
        "ev_ceiling",
    }
)

# ``ev_floor`` is reported as its own stage rather than inside the price screen:
# a row the book prices past our number is not a screen refusing a bet, it is
# the absence of one, and lumping the two hides whether a market is unbettable
# or merely unbought.
_EV_GATE = "ev_floor"

UNPRICED = "unpriced"

# The one gate a re-run can undo rather than an argument about thresholds: it
# refuses rows priced before the lineups, scratches and weather resolve them.
CLOCK_GATE = "lineup_clock"

# Only for the reader-facing table: "unpriced" is not a screen refusing a bet.
_GATE_LABELS = {UNPRICED: "no book price"}


@dataclass
class MarketFunnel:
    """One market's row counts by stage, and the gates that closed it."""

    market: str
    candidates: int = 0
    priced: int = 0
    positive_ev: int = 0
    cleared_price_screen: int = 0
    buys: int = 0
    gates: Counter[str] = field(default_factory=Counter)

    @property
    def closing_gate(self) -> str:
        """The gate that refused the most rows, or "" when nothing was refused."""
        if not self.gates:
            return ""
        return self.gates.most_common(1)[0][0]


@dataclass
class Funnel:
    """A slate's screening funnel, overall and per market."""

    overall: MarketFunnel
    markets: list[MarketFunnel]


def build(recs: list[Recommendation], thr: EVThresholds | None = None) -> Funnel:
    """Count a slate's rows through the stages, overall and by market.

    Rows carrying no price are counted as candidates and nothing else: an
    unquoted selection has not failed a screen, and counting it as one would
    charge the engine's own gates with the books' coverage.

    The EV stage is read off ``rec.ev`` against the market's own ``min_ev``
    rather than off the gate name. The two agree whenever the card is rendered
    under the thresholds it was priced under, and when they do not -- a floor
    moved between the run and the report, a later veto overwriting the gate --
    the number the reader wants is the one the row's own EV supports.
    """
    base = EVThresholds() if thr is None else thr
    per: dict[str, MarketFunnel] = {}
    overall = MarketFunnel("ALL")
    for rec in recs:
        mf = per.setdefault(rec.market, MarketFunnel(rec.market))
        for f in (mf, overall):
            f.candidates += 1
        if rec.market_american is None:
            for f in (mf, overall):
                f.gates[UNPRICED] += 1
            continue
        for f in (mf, overall):
            f.priced += 1
        gate = rec.pass_gate or ""
        if rec.ev is None or rec.ev <= base.for_market(rec.market).min_ev:
            for f in (mf, overall):
                f.gates[_EV_GATE] += 1
            continue
        for f in (mf, overall):
            f.positive_ev += 1
        if gate in PRICE_SCREEN_GATES:
            for f in (mf, overall):
                f.gates[gate] += 1
            continue
        for f in (mf, overall):
            f.cleared_price_screen += 1
        if rec.tier is Tier.PASS:
            for f in (mf, overall):
                f.gates[gate or "unnamed"] += 1
            continue
        for f in (mf, overall):
            f.buys += 1
    markets = sorted(per.values(), key=lambda f: (-f.priced, f.market))
    return Funnel(overall=overall, markets=markets)


def geometry_note(thr: EVThresholds) -> str:
    """Warn when the thresholds cannot all hold on a market priced near even.

    The conviction floor is an absolute probability and the edge ceiling is a
    distance from the devigged price, so together they require the market's own
    fair probability to reach ``min_prob - max_edge``. On a two-sided market
    quoted -110 both ways that number is .50, i.e. the engine can only back the
    side the book already favours, and any tightening of either knob narrows the
    window from both ends at once. Silent arithmetic, so it is printed.
    """
    required = thr.min_prob - thr.max_edge
    if required < 0.5 - 1e-9:
        return ""
    # Both screens are strict (``edge > max_edge``, ``model_prob < min_prob``),
    # so at exactly .50 the window is not empty -- it is one point wide, at the
    # edge ceiling itself. Worth the same warning and not the same sentence.
    tail = (
        "money the only row that clears both sits exactly on the ceiling"
        if required <= 0.5 + 1e-9
        else "money no row can clear both"
    )
    return (
        f"geometry: the conviction floor ({thr.min_prob:.2f}) and the edge "
        f"ceiling ({thr.max_edge:.2f}) together require the market's own fair "
        f"probability to reach {required:.2f} -- on a market quoted near even "
        f"{tail}"
    )


def gate_label(gate: str) -> str:
    """Render a gate name for a reader rather than for a grep."""
    return _GATE_LABELS.get(gate, gate)


def clock_note(f: Funnel) -> str:
    """Tell the operator to re-run when the clock, not the model, closed rows.

    ``lineup_clock`` is the only closer that is a scheduling artefact: the rows
    are not bad bets, they were priced too early to be bets at all, and the late
    pass re-prices them. Reported so a card thinned by the clock is not read as
    a card thinned by the screens.
    """
    refused = f.overall.gates.get(CLOCK_GATE, 0)
    if not refused:
        return ""
    # The gate reads its window from the environment, so the advice has to read
    # it from the same place or it will send the operator back outside it.
    hours = LineupLockGate.from_env().stale_hours
    return (
        f"clock: {refused} rows were refused for being priced more than "
        f"{hours:.0f}h before first pitch, not for their price -- "
        f"re-run inside the window (run --within-hours "
        f"{hours:.0f}) to price them off posted lineups"
    )


def _fmt(mf: MarketFunnel, *, top: int = 4) -> str:
    gates = ", ".join(f"{g} {n}" for g, n in mf.gates.most_common(top))
    return (
        f"{mf.market:<13} {mf.candidates:>5} cand {mf.priced:>5} priced "
        f"{mf.positive_ev:>4} +EV {mf.cleared_price_screen:>4} screened "
        f"{mf.buys:>3} buys | {gates}"
    )


def summary_lines(f: Funnel, thr: EVThresholds | None = None) -> list[str]:
    """The funnel as plain text, for the run log and the card."""
    o = f.overall
    lines = [
        f"Funnel: {o.candidates} candidates -> {o.priced} priced -> "
        f"{o.positive_ev} positive EV -> {o.cleared_price_screen} clear the "
        f"price screen -> {o.buys} buys",
    ]
    if o.gates:
        lines.append("  refused by: " + ", ".join(f"{g} {n}" for g, n in o.gates.most_common(8)))
    for mf in f.markets:
        lines.append("  " + _fmt(mf))
    clock = clock_note(f)
    if clock:
        lines.append("  " + clock)
    if thr is not None:
        note = geometry_note(thr)
        if note:
            lines.append("  " + note)
    return lines


def markdown(f: Funnel, thr: EVThresholds | None = None) -> list[str]:
    """The funnel as a Markdown table, best-covered market first."""
    o = f.overall
    lines = [
        "## How the slate was screened",
        "",
        f"*{o.candidates} candidate rows, {o.priced} of them quoted by a book, "
        f"{o.positive_ev} paying at the price offered, {o.cleared_price_screen} "
        f"clearing the price screen, **{o.buys} bought**.*",
        "",
        "| market | candidates | priced | +EV | screened | buys | closed mostly by |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for mf in f.markets:
        gate = gate_label(mf.closing_gate) or "—"
        lines.append(
            f"| {mf.market} | {mf.candidates} | {mf.priced} | {mf.positive_ev} "
            f"| {mf.cleared_price_screen} | {mf.buys} | {gate} |"
        )
    lines.append("")
    clock = clock_note(f)
    if clock:
        lines += [f"*{clock}.*", ""]
    if thr is not None:
        note = geometry_note(thr)
        if note:
            lines += [f"*{note}.*", ""]
    return lines
