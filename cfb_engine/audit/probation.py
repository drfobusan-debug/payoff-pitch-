"""Probation: grade every market and every screen, and say when to act.

This exists because of a mistake the MLB engine made repeatedly and this one has
not yet had the chance to. Screens and reopenings there were decided on 30-60
graded buys -- a price floor fitted on 34 minus-money bets, a market reopened on
50 buys running +25.7% -- and on the first slate after shipping, the floor
refused thirteen bets that went 10-3 while the reopened market lost 34.2% of
stake. A 50-bet cell is not evidence, and the way to stop trusting one is not to
resolve to be more careful; it is to delegate the decision to a rule that cannot
be talked round.

Two symmetric questions, asked of the ledger rather than of anybody's judgement:

* **a market on probation** -- since it was last changed, are its buys losing?
* **a screen on probation** -- are the picks it refuses winning?

A screen that keeps deleting winners is as expensive as a market that keeps
losing, and neither shows up in a scorecard that only counts the bets we made.
That matters more here than in baseball, because this engine ships with screens
switched off *on measured null results* (``CFBE_VSIN_HFA``, ``CFBE_STABILITY``,
``CFBE_INJURY_QB_PTS``) and with the drift gate measuring rather than vetoing --
those are decisions to keep re-testing, not settled facts.

The bar for acting is deliberately not a p-value. Three conditions, all required:

1. **Volume.** ``min_n`` graded bets, default 100. Below that the verdict is
   ``WATCHING`` and nothing happens, however bad the number looks. A Saturday
   contributes roughly 20-50 buys, so this is a few weeks rather than a season --
   which is the whole reason the bar can be this low: at ~800 games a year,
   waiting for ROI significance means waiting years.
2. **Size.** The mean per-unit return must be worse than zero by more than one
   standard error. One, not the 1.96 of a 5% test: this is a stop-loss under
   asymmetric cost, where continuing to bet a losing market is not the safe
   option, so demanding proof beyond reasonable doubt has its own price.
3. **Consistency across time.** Both halves of the window must agree. This is
   the condition that does the real work, and it is here because it is what
   caught every false finding the MLB engine produced -- a run line that went
   +34.5% in July and -11.9% in August, arrows fitted to a window's edge. A
   pooled average over a window whose halves disagree is an artefact of where the
   window was cut.

The same three tests also grade a screen that does not exist yet
(:func:`candidate_probation`): a proposed price band or drift floor is judged on
the buys it *would* have refused, before it is allowed to refuse one for real.
A floor is normally proposed with a pooled ROI over the cell it was found in,
which is exactly the number the consistency test exists to distrust.

Nothing here changes a bet by itself. It emits findings, and a market it
condemns is shut by an explicit config change with the verdict quoted as the
reason -- so the decision stays reviewable and reversible.

One limit worth knowing when reading an early report: screen verdicts need the
``pass_gate`` attribution, and the audit grades the *saved* card, so refusals
only accumulate from the first slate whose predictions carry the column. Until
then the screen table is nearly empty -- which is a gap in the data, not a clean
bill of health for the screens.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean, stdev

from cfb_engine.audit.grade import PUSH
from cfb_engine.audit.ledger import LedgerEntry
from cfb_engine.market.tiers import Tier

# One Saturday contributes roughly 20-50 buys, so 100 is a few weeks rather than
# a season: long enough that a single bad slate cannot carry it, short enough to
# act inside the same month.
DEFAULT_MIN_N = 100
# Screen refusals accumulate faster than buys (a screen sees everything the EV
# floor let through), but they are counterfactual -- graded at a price we never
# actually took -- so they get the same bar rather than a lower one.
DEFAULT_MIN_REFUSED = 100

# Unlike the MLB engine, this one has no retired feature basis to start the
# window at: there is no graded CFB history at all yet, so the default window is
# everything and the first verdicts will be WATCHING for weeks. When a screen is
# changed on the strength of a verdict, pass the date of that change as ``since``
# so the old regime's record cannot vouch for the new one.
ALL_HISTORY = ""

WATCHING = "WATCHING"  # not enough graded bets to judge
CLEAR = "CLEAR"  # judged, and not condemned
SHUT = "SHUT"  # market losing on all three tests
LIFT = "LIFT"  # screen refusing winners on all three tests
SHIP = "SHIP"  # proposed screen refusing losers on all three tests

_BUY = frozenset({Tier.STRONG.value, Tier.MODERATE.value})

# Attributions that are not screens anyone can lift: the absence of a market is
# not a decision, and a tier downgrade is an adjustment rather than a refusal.
_NOT_SCREENS = frozenset({"", "unpriced", "tier_downgrade"})


def _min_n() -> int:
    return int(os.environ.get("CFBE_PROBATION_MIN_N", DEFAULT_MIN_N))


def _min_refused() -> int:
    return int(os.environ.get("CFBE_PROBATION_MIN_REFUSED", DEFAULT_MIN_REFUSED))


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
    """Graded rows carrying a real price, pushes excluded.

    Pushes matter more in football than in baseball -- a spread or total lands on
    the number often enough that counting them as zero-return bets would dilute
    every ROI here toward nothing.
    """
    return [e for e in entries if e.result != PUSH and e.odds is not None]


def _roi_se(entries: list[LedgerEntry]) -> tuple[float, float]:
    """Mean per-unit return and its standard error.

    The unit of variance is the individual bet, not the day: a market's ROI is an
    average of one-unit stakes, so its uncertainty is the spread of those returns
    over the root of their count. Reporting ROI without this is how a 50-bet cell
    comes to look like a finding.
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

    Split on row order after sorting by date, not on the calendar midpoint:
    slates differ enormously in size in college football (a 60-game Saturday
    against a three-game Thursday), and a calendar split can put 80% of the bets
    on one side and then call the other side unstable.
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

    For a market we act when its buys lose. For a screen we act when the picks it
    refused *won* -- the same arithmetic with the sign reversed, because a
    screen's refusals are graded as the bets we would have made.
    """
    n = len(entries)
    roi, se = _roi_se(entries)
    older, newer = _halves(entries)
    h1, h2 = _mean(older), _mean(newer)
    sign = -1.0 if losing_is_bad else 1.0
    label = {"market": "buys", "screen": "refusals"}.get(kind, "buys it would refuse")

    if n < min_n:
        return Probation(
            name,
            kind,
            n,
            roi,
            se,
            h1,
            h2,
            WATCHING,
            f"{name}: {n} graded {label} ({roi * 100:+.1f}%) -- under the {min_n} "
            "needed to judge; no action",
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
            name,
            kind,
            n,
            roi,
            se,
            h1,
            h2,
            status,
            f"{name}: {n} {label} {verb} (se {se * 100:.1f}), "
            f"{h1 * 100:+.1f}% then {h2 * 100:+.1f}% across the halves -> {act}",
        )
    why = "the halves disagree" if not consistent else "inside one standard error of zero"
    return Probation(
        name,
        kind,
        n,
        roi,
        se,
        h1,
        h2,
        CLEAR,
        f"{name}: {n} {label} at {roi * 100:+.1f}% (se {se * 100:.1f}), "
        f"{h1 * 100:+.1f}% then {h2 * 100:+.1f}% -- {why}; no action",
    )


def market_probation(
    entries: list[LedgerEntry],
    since: str | None = None,
    min_n: int | None = None,
) -> list[Probation]:
    """Verdict per market, over its buys only."""
    bar = _min_n() if min_n is None else min_n
    floor = ALL_HISTORY if since is None else since
    rows = [e for e in _decided(entries) if e.tier in _BUY and e.date >= floor]
    by_market: dict[str, list[LedgerEntry]] = {}
    for e in rows:
        by_market.setdefault(e.market, []).append(e)
    out = [_judge(m, "market", es, bar, losing_is_bad=True) for m, es in sorted(by_market.items())]
    return sorted(out, key=lambda p: (p.status != SHUT, p.roi))


def screen_probation(
    entries: list[LedgerEntry],
    since: str | None = None,
    min_n: int | None = None,
) -> list[Probation]:
    """Verdict per screen, over the priced picks it refused."""
    bar = _min_refused() if min_n is None else min_n
    floor = ALL_HISTORY if since is None else since
    rows = [
        e for e in _decided(entries) if (e.pass_gate or "") not in _NOT_SCREENS and e.date >= floor
    ]
    by_gate: dict[str, list[LedgerEntry]] = {}
    for e in rows:
        by_gate.setdefault(e.pass_gate or "", []).append(e)
    out = [_judge(g, "screen", es, bar, losing_is_bad=False) for g, es in sorted(by_gate.items())]
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


def _drift_worse_than(floor: float) -> Callable[[LedgerEntry], bool]:
    """Refuse a buy the market moved *against* by more than ``floor``.

    ``drift`` is signed toward the side bet, so an adverse move is negative.
    """

    def refuses(e: LedgerEntry) -> bool:
        return e.drift is not None and e.drift <= -abs(floor)

    return refuses


def _run_up_over(ceiling: float) -> Callable[[LedgerEntry], bool]:
    """Refuse a buy the market had already run up toward before we took it.

    The mirror of the adverse test, and the less intuitive one: a number moving
    our way before we bet looks like confirmation, but it means the edge we are
    pricing has already been paid out to whoever moved it, and in the MLB ledger
    that group of bets underperformed.
    """

    def refuses(e: LedgerEntry) -> bool:
        return e.drift is not None and e.drift >= abs(ceiling)

    return refuses


def _shorter_than(ceiling: float) -> Callable[[LedgerEntry], bool]:
    """Refuse buys priced shorter than ``ceiling`` (a negative American number).

    The other half of the price band in :mod:`cfb_engine.market.priceband`. A
    short favourite has to be right about a near-certainty to earn anything, so
    most of the stake sits on the one outcome the price says will not happen.
    """

    def refuses(e: LedgerEntry) -> bool:
        return e.odds is not None and e.odds < ceiling

    return refuses


def _ml_longer_than(floor: float) -> Callable[[LedgerEntry], bool]:
    """Refuse moneyline dogs priced longer than ``floor``.

    A long dog's edge is the hardest thing this engine measures: the price
    demands little, so a small probability error is a large EV error, and the
    Markov simulator's tails are exactly where a fitted distribution is least
    trustworthy.
    """

    def refuses(e: LedgerEntry) -> bool:
        return e.market == "game_ml" and e.odds is not None and e.odds >= floor

    return refuses


# The candidates asked of every audit, rather than settled once in a chat
# message. Each is a screen someone has a good story for; the point of keeping
# them here is that the three tests, not the story, decide whether it ships.
CANDIDATE_SCREENS: tuple[CandidateScreen, ...] = (
    # The drift gate's veto half. It ships measuring only, because there was no
    # graded CFB row to set a threshold from when it was built -- this is how
    # that threshold gets set, out of the ledger instead of out of MLB's numbers.
    CandidateScreen(
        "drift_refuse_adverse_2pct",
        _drift_worse_than(0.02),
        "the number moved against us before we bet it",
    ),
    CandidateScreen(
        "drift_refuse_run_up_2pct",
        _run_up_over(0.02),
        "the edge was already paid out to whoever moved the line",
    ),
    CandidateScreen(
        "ml_refuse_dogs_longer_than_+200",
        _ml_longer_than(200.0),
        "a long dog's EV is dominated by the tail we model worst",
    ),
    # The price band's short end, graded before it is allowed to refuse anything.
    CandidateScreen(
        "price_refuse_shorter_than_-250",
        _shorter_than(-250.0),
        "a short favourite risks most of the stake on the outcome the price denies",
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
    floor = ALL_HISTORY if since is None else since
    rows = [e for e in _decided(entries) if e.tier in _BUY and e.date >= floor]
    out: list[Probation] = []
    for c in candidates:
        verdict = _judge(
            c.name, "candidate", [e for e in rows if c.refuses(e)], bar, losing_is_bad=True
        )
        if verdict.status == SHIP:
            # Carry the story into the finding, so a shipped screen arrives with
            # the reason it was proposed beside the number that justified it.
            verdict.finding = f"{verdict.finding} ({c.rationale})"
        out.append(verdict)
    return sorted(out, key=lambda p: (p.status != SHIP, p.roi))


def probation_rows(entries: list[LedgerEntry], since: str | None = None) -> list[Probation]:
    """Every verdict, markets first, then live screens, then candidates."""
    return [
        *market_probation(entries, since),
        *screen_probation(entries, since),
        *candidate_probation(entries, since=since),
    ]


def probation_findings(entries: list[LedgerEntry], since: str | None = None) -> list[str]:
    """Actionable verdicts only.

    Deliberately silent when nothing has crossed the bar. A monitor that prints a
    paragraph every morning is a monitor nobody reads, and the whole point of it
    is to be believed on the day it does speak.
    """
    return [p.finding for p in probation_rows(entries, since) if p.actionable]
