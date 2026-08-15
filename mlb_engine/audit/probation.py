"""Probation: grade every market and every screen, and say when to act.

This exists because of a specific, repeated mistake. Screens and reopenings in
this engine have been decided on 30-60 graded buys -- a singles price floor
fitted on 34 minus-money bets, a doubles market reopened on 50 buys running
+25.7% -- and on the first slate after shipping, the floor refused thirteen bets
that went 10-3 while doubles lost 34.2% of stake. A 50-bet cell is not evidence,
and the way to stop trusting one is not to resolve to be more careful; it is to
delegate the decision to a rule that cannot be talked round.

Two symmetric questions, asked of the ledger rather than of anybody's judgement:

* **a market on probation** -- since it was last changed, are its buys losing?
* **a screen on probation** -- are the picks it refuses winning?

A screen that keeps deleting winners is as expensive as a market that keeps
losing, and neither shows up in a scorecard that only counts the bets we made.

The bar for acting is deliberately not a p-value. Three conditions, all
required:

1. **Volume.** ``min_n`` graded bets, default 100. Below that the verdict is
   ``WATCHING`` and nothing happens, however bad the number looks.
2. **Size.** The mean per-unit return must be worse than zero by more than one
   standard error. One, not the 1.96 of a 5% test: this is a stop-loss under
   asymmetric cost, where continuing to bet a losing market is not the safe
   option, so demanding proof beyond reasonable doubt has its own price.
3. **Consistency across time.** Both halves of the window must agree. This is
   the condition that does the real work, and it is here because it is what
   caught every false finding this engine has produced -- the +1.5 run line
   (+34.5% in July, -11.9% in August), the SIERA and velocity trend arrows, the
   doubles cell. A pooled average over a window whose halves disagree is an
   artefact of where the window was cut.

The same three tests also grade a screen that does not exist yet
(:func:`candidate_probation`): a proposed price band or probability floor is
judged on the buys it *would* have refused, before it is allowed to refuse one
for real. This is the third question, and it is asked for the same reason as the
other two -- a floor is normally proposed with a pooled ROI over the cell it was
found in, which is exactly the number the consistency test exists to distrust.

Nothing here changes a bet by itself. It emits findings, and a market it
condemns is shut by an explicit config change with the verdict quoted as the
reason -- so the decision stays reviewable and reversible.

One limit worth knowing when reading an early report: screen verdicts need the
``pass_gate`` attribution added in #117, and the audit grades the *pregame* card,
so refusals only start accumulating from the first slate whose pregame snapshot
carries the column. Until then the screen table is nearly empty -- which is a
gap in the data, not a clean bill of health for the screens.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean, stdev

from mlb_engine.audit.grade import PUSH
from mlb_engine.audit.ledger import LedgerEntry
from mlb_engine.calibration import FEATURE_BASIS, FEATURE_BASIS_SINCE
from mlb_engine.market.odds import american_to_decimal
from mlb_engine.market.tiers import Tier

# One slate contributes roughly 10-40 buys in a live market, so 100 is a handful
# of slates rather than a season: long enough that a single bad night cannot
# carry it, short enough to act inside the same month.
DEFAULT_MIN_N = 100
# Screen refusals accumulate faster than buys (a screen sees everything the EV
# floor let through), but they are counterfactual -- graded at a price we never
# actually took -- so they get the same bar rather than a lower one.
DEFAULT_MIN_REFUSED = 100

# Bets placed before the current feature basis were placed by a different
# engine: the basis retirement (#116) invalidated the calibration behind them,
# and the reopenings and screens of 08-12 changed which bets it makes at all.
# Pooling the two is how a market's old record gets to vouch for its new one, so
# probation starts at the basis and every verdict says so.
DEFAULT_SINCE = FEATURE_BASIS_SINCE.isoformat()
ALL_HISTORY = ""  # explicit opt-in to grading across a basis change

WATCHING = "WATCHING"  # not enough graded bets to judge
CLEAR = "CLEAR"  # judged, and not condemned
SHUT = "SHUT"  # market losing on all three tests
LIFT = "LIFT"  # screen refusing winners on all three tests
SHIP = "SHIP"  # proposed screen refusing losers on all three tests

_BUY = {Tier.STRONG.value, Tier.MODERATE.value}


def _min_n() -> int:
    return int(os.environ.get("MLBE_PROBATION_MIN_N", DEFAULT_MIN_N))


def _min_refused() -> int:
    return int(os.environ.get("MLBE_PROBATION_MIN_REFUSED", DEFAULT_MIN_REFUSED))


@dataclass
class Probation:
    """One market's or one screen's verdict."""

    name: str
    kind: str  # "market" | "screen" | "candidate"
    n: int
    roi: float  # mean per-unit return
    se: float  # standard error of that mean
    first_half: float  # ROI over the older half of the window
    second_half: float  # ROI over the newer half
    status: str
    finding: str

    @property
    def actionable(self) -> bool:
        return self.status in (SHUT, LIFT, SHIP)


def _decided(entries: list[LedgerEntry]) -> list[LedgerEntry]:
    """Graded rows carrying a real price, pushes excluded."""
    return [e for e in entries if e.result != PUSH and e.odds is not None]


def _roi_se(entries: list[LedgerEntry]) -> tuple[float, float]:
    """Mean per-unit return and its standard error.

    The unit of variance is the individual bet, not the day: a market's ROI is
    an average of one-unit stakes, so its uncertainty is the spread of those
    returns over the root of their count. Reporting ROI without this is how a
    50-bet cell comes to look like a finding.
    """
    pnl = [e.pnl for e in entries]
    if not pnl:
        return 0.0, 0.0
    mean = fmean(pnl)
    if len(pnl) < 2:
        return mean, 0.0
    return mean, stdev(pnl) / (len(pnl) ** 0.5)


def _halves(entries: list[LedgerEntry]) -> tuple[list[LedgerEntry], list[LedgerEntry]]:
    """Split chronologically into an older and a newer half.

    Split on the row order after sorting by date, not on the calendar midpoint:
    slates differ in size, and a calendar split can put 80% of the bets on one
    side and then call the other side unstable.
    """
    ordered = sorted(entries, key=lambda e: e.date)
    mid = len(ordered) // 2
    return ordered[:mid], ordered[mid:]


def _mean(entries: list[LedgerEntry]) -> float:
    return fmean([e.pnl for e in entries]) if entries else 0.0


def _judge(
    name: str,
    kind: str,
    entries: list[LedgerEntry],
    min_n: int,
    *,
    losing_is_bad: bool,
) -> Probation:
    """Apply the three tests. ``losing_is_bad`` flips them for screens.

    For a market we act when its buys lose. For a screen we act when the picks
    it refused *won* -- the same arithmetic with the sign reversed, because a
    screen's refusals are graded as the bets we would have made.
    """
    n = len(entries)
    roi, se = _roi_se(entries)
    older, newer = _halves(entries)
    h1, h2 = _mean(older), _mean(newer)
    sign = -1.0 if losing_is_bad else 1.0
    label = "buys" if kind == "market" else "refusals"
    if kind == "candidate":
        label = "buys it would refuse"

    if n < min_n:
        return Probation(
            name, kind, n, roi, se, h1, h2, WATCHING,
            f"{name}: {n} graded {label} ({roi * 100:+.1f}%) on the {FEATURE_BASIS} "
            f"basis -- under the {min_n} needed to judge; no action",
        )
    # Condition 2: worse (or better, for a screen) than zero by over one SE.
    beyond_se = sign * roi > se if se > 0 else sign * roi > 0
    # Condition 3: both halves agree.
    consistent = sign * h1 > 0 and sign * h2 > 0
    if beyond_se and consistent:
        status = {"market": SHUT, "screen": LIFT, "candidate": SHIP}[kind]
        verb = {
            "market": f"losing {abs(roi) * 100:.1f}% of stake",
            "screen": f"refusing winners at {roi * 100:+.1f}%",
            "candidate": f"losing {abs(roi) * 100:.1f}% of stake",
        }[kind]
        act = {
            "market": f"shut {name} until the refit",
            "screen": f"lift {name}; it is deleting money",
            "candidate": f"ship {name}",
        }[kind]
        return Probation(
            name, kind, n, roi, se, h1, h2, status,
            f"{name}: {n} {label} {verb} (se {se * 100:.1f}), "
            f"{h1 * 100:+.1f}% then {h2 * 100:+.1f}% across the halves -> {act}",
        )
    if not consistent:
        why = "the halves disagree"
    else:
        why = "inside one standard error of zero"
    return Probation(
        name, kind, n, roi, se, h1, h2, CLEAR,
        f"{name}: {n} {label} at {roi * 100:+.1f}% (se {se * 100:.1f}), "
        f"{h1 * 100:+.1f}% then {h2 * 100:+.1f}% -- {why}; no action",
    )


def market_probation(
    entries: list[LedgerEntry],
    since: str | None = None,
    min_n: int | None = None,
) -> list[Probation]:
    """Verdict per market, over its buys only.

    The window starts at ``since`` (an ISO date, as the ledger stores it), and
    defaults to the current feature basis rather than to all of history: a
    reopened market's record before it was shut is a different engine's record,
    and grading the two together hides exactly the change being measured. Pass
    :data:`ALL_HISTORY` to override that deliberately.
    """
    bar = _min_n() if min_n is None else min_n
    floor = DEFAULT_SINCE if since is None else since
    rows = [e for e in _decided(entries) if e.tier in _BUY and e.date >= floor]
    by_market: dict[str, list[LedgerEntry]] = {}
    for e in rows:
        by_market.setdefault(e.market, []).append(e)
    out = [
        _judge(m, "market", es, bar, losing_is_bad=True)
        for m, es in sorted(by_market.items())
    ]
    return sorted(out, key=lambda p: (p.status != SHUT, p.roi))


def screen_probation(
    entries: list[LedgerEntry],
    since: str | None = None,
    min_n: int | None = None,
) -> list[Probation]:
    """Verdict per screen, over the priced picks it refused.

    Only named screens count. ``tier_downgrade`` and ``unpriced`` are excluded:
    the first is a tier adjustment rather than a screen, and the second is the
    absence of a market, so neither is a decision anyone can lift.
    """
    bar = _min_refused() if min_n is None else min_n
    floor = DEFAULT_SINCE if since is None else since
    skip = {"", "tier_downgrade", "unpriced"}
    rows = [
        e
        for e in _decided(entries)
        if (e.pass_gate or e.veto_gate) not in skip and e.date >= floor
    ]
    by_gate: dict[str, list[LedgerEntry]] = {}
    for e in rows:
        by_gate.setdefault(e.pass_gate or e.veto_gate, []).append(e)
    out = [
        _judge(g, "screen", es, bar, losing_is_bad=False)
        for g, es in sorted(by_gate.items())
    ]
    return sorted(out, key=lambda p: (p.status != LIFT, -p.roi))


@dataclass(frozen=True)
class CandidateScreen:
    """A screen that does not exist yet, expressed as "would this row be refused?".

    ``refuses`` is evaluated against graded buys, so a candidate is graded on the
    money it would have saved rather than on the cell it was spotted in.
    """

    name: str
    refuses: Callable[[LedgerEntry], bool]
    rationale: str


def _is_ml(e: LedgerEntry) -> bool:
    return e.market in ("game_ml", "f5_ml")


def _is_home(e: LedgerEntry) -> bool:
    """Is the picked side the home team? Ledger matchups read ``AWAY @ HOME``."""
    parts = e.matchup.split(" @ ")
    if len(parts) != 2:
        return False
    return e.selection.split()[0] == parts[1]


def _home_ml_short_of(floor: float) -> Callable[[LedgerEntry], bool]:
    def refuses(e: LedgerEntry) -> bool:
        return _is_ml(e) and _is_home(e) and e.odds is not None and e.odds > floor

    return refuses


def _anchored_ev_negative(weight: float) -> Callable[[LedgerEntry], bool]:
    """Would a market-anchored probability have priced this ML buy at EV <= 0?

    Anchoring is affine, so on the *edge* screen a weight is only the edge floor
    restated (``edge -> edge x (1 - w)``); what it adds is a toll against the
    vig-inclusive break-even that grows with the price, and that is what this
    grades.
    """

    def refuses(e: LedgerEntry) -> bool:
        if not _is_ml(e) or e.fair_prob is None or e.odds is None:
            return False
        bet = (1.0 - weight) * e.model_prob + weight * e.fair_prob
        dec = american_to_decimal(e.odds)
        return bet * (dec - 1.0) - (1.0 - bet) <= 0.0

    return refuses


# The candidates asked for and re-graded so far. They stay here, graded every
# audit, rather than being settled once in a chat message: the verdicts below are
# what 27 slates said, and the point of keeping them is that another 27 may say
# something else.
CANDIDATE_SCREENS: tuple[CandidateScreen, ...] = (
    # Proposed as the home-side mirror of ``away_ml_refuse_odds`` after the
    # graded card put home ML at -7.9% over n=108, concentrated in the near
    # pick'em band. It failed the consistency test on the data that suggested it
    # -- the -120..-100 band ran +11.8% over the older half and -53.3% over the
    # newer -- and the rows it would have refused on the first five were +30.7%.
    CandidateScreen(
        "home_ml_refuse_longer_than_-120",
        _home_ml_short_of(-120.0),
        "mirror away_ml_refuse_odds on the home side",
    ),
    # The market blend, judged as the screen it actually is. Only the EV toll is
    # graded here, because the edge floor has to be rescaled by (1 - w) alongside
    # it or the blend is a floor hike wearing a different name.
    CandidateScreen(
        "game_ml_market_anchor_0.5",
        _anchored_ev_negative(0.5),
        "defer to the price where it out-forecasts the model",
    ),
)


def candidate_probation(
    entries: list[LedgerEntry],
    candidates: tuple[CandidateScreen, ...] = CANDIDATE_SCREENS,
    since: str | None = None,
    min_n: int | None = None,
) -> list[Probation]:
    """Verdict per proposed screen, over the graded buys it would have refused.

    Same three tests as a live market, because a candidate is the same claim: the
    rows in here lose. ``SHIP`` means they lose by more than a standard error in
    both halves of the window; anything else means the floor is a fit to where
    the window was cut, and the answer is to keep grading rather than to ship it
    and find out.
    """
    bar = _min_n() if min_n is None else min_n
    floor = DEFAULT_SINCE if since is None else since
    rows = [e for e in _decided(entries) if e.tier in _BUY and e.date >= floor]
    out = [
        _judge(c.name, "candidate", [e for e in rows if c.refuses(e)], bar, losing_is_bad=True)
        for c in candidates
    ]
    return sorted(out, key=lambda p: (p.status != SHIP, p.roi))


def probation_findings(
    entries: list[LedgerEntry], since: str | None = None
) -> list[str]:
    """Actionable verdicts only, markets first.

    Deliberately silent when nothing has crossed the bar. A monitor that prints
    a paragraph every morning is a monitor nobody reads, and the whole point of
    it is to be believed on the day it does speak.
    """
    rows = [
        *market_probation(entries, since),
        *screen_probation(entries, since),
        *candidate_probation(entries, since=since),
    ]
    return [p.finding for p in rows if p.actionable]
