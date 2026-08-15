"""Per-prop false-positive / false-negative / true-positive analysis.

Runs over the persisted bet :class:`~mlb_engine.audit.ledger.LedgerEntry` history
(every graded prop across all audited slates) and turns the confusion breakdown
into plain-language, data-grounded recommendations:

* **False positives** (model favored the pick, ``model_prob >= 0.5``, but it
  lost) -> what is inflating PPV loss, and how to tighten it.
* **False negatives** (model faded the pick, ``model_prob < 0.5``, but it won)
  -> under-rated pockets to reclaim, lifting NPV.
* **True positives** (model favored the pick and it won) -> the pockets the
  model already nails, to concentrate/size up and reinforce PPV.

"Prop" = a ``batter_*`` / ``pitcher_*`` market (see
:func:`mlb_engine.audit.ledger.is_prop`). Every insight is derived only from
fields the ledger actually stores (model probability, line, EV, tier, book,
result), so nothing is fabricated.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from mlb_engine.audit.grade import PUSH, WIN
from mlb_engine.audit.ledger import ENGINE_PROB_THRESHOLD, LedgerEntry, is_prop
from mlb_engine.features.lineup_lock import POSTED, PROJECTED
from mlb_engine.market.odds import american_to_decimal
from mlb_engine.market.tiers import Tier

# A -110 line needs ~52.4% to break even; the model's own decision boundary is
# 0.5. We flag favored pockets under breakeven and faded pockets over it.
BREAKEVEN = 0.524
DEFAULT_MIN_N = 20
# Price-bucket samples are one bet per game, not one per batter, so they build a
# tenth as fast as the prop pockets and get a lower bar to be reported at all.
PRICE_MIN_N = 15
PRICE_LEAK_POINTS = 0.03  # win rate this far under break-even is worth naming
# A batter row's outcome is close to a coin flip, so the gap between two groups
# of them needs hundreds of rows before it clears its own standard error.
LINEUP_MIN_N = 300
_BUY = {Tier.STRONG.value, Tier.MODERATE.value}

FALSE_POSITIVE = "false_positive"
FALSE_NEGATIVE = "false_negative"
TRUE_POSITIVE = "true_positive"


@dataclass
class PropInsight:
    """One recommendation for a prop market."""

    market: str
    kind: str  # FALSE_POSITIVE | FALSE_NEGATIVE | TRUE_POSITIVE
    n: int  # sample size behind the headline rate
    rate: float  # headline rate (FP: win% of favored; FN: win% of faded; TP: win%)
    finding: str


def _decided(entries: list[LedgerEntry]) -> list[LedgerEntry]:
    return [e for e in entries if e.result != PUSH]


def _won(e: LedgerEntry) -> bool:
    return e.result == WIN


def _win_rate(entries: list[LedgerEntry]) -> float:
    d = _decided(entries)
    return sum(1 for e in d if _won(e)) / len(d) if d else 0.0


def _favored(entries: list[LedgerEntry]) -> list[LedgerEntry]:
    return [e for e in _decided(entries) if e.model_prob >= ENGINE_PROB_THRESHOLD]


def _faded(entries: list[LedgerEntry]) -> list[LedgerEntry]:
    return [e for e in _decided(entries) if e.model_prob < ENGINE_PROB_THRESHOLD]


def _fmt_line(line: float | None) -> str:
    return "?" if line is None else (f"{line:g}")


def _worst_line(entries: list[LedgerEntry], min_n: int) -> tuple[float | None, float, int] | None:
    """Line with the lowest win rate among ``entries`` (n>=min_n)."""
    by_line: dict[float | None, list[LedgerEntry]] = defaultdict(list)
    for e in entries:
        by_line[e.line].append(e)
    best: tuple[float | None, float, int] | None = None
    for line, es in by_line.items():
        if len(es) < min_n:
            continue
        wr = _win_rate(es)
        if best is None or wr < best[1]:
            best = (line, wr, len(es))
    return best


def _best_line(entries: list[LedgerEntry], min_n: int) -> tuple[float | None, float, int] | None:
    by_line: dict[float | None, list[LedgerEntry]] = defaultdict(list)
    for e in entries:
        by_line[e.line].append(e)
    best: tuple[float | None, float, int] | None = None
    for line, es in by_line.items():
        if len(es) < min_n:
            continue
        wr = _win_rate(es)
        if best is None or wr > best[1]:
            best = (line, wr, len(es))
    return best


def false_positive_insights(
    entries: list[LedgerEntry], min_n: int = DEFAULT_MIN_N
) -> list[PropInsight]:
    """Where the model's favored props under-perform -> tighten to raise PPV."""
    out: list[PropInsight] = []
    by_market: dict[str, list[LedgerEntry]] = defaultdict(list)
    for e in entries:
        if is_prop(e.market):
            by_market[e.market].append(e)

    for market in sorted(by_market):
        fav = _favored(by_market[market])
        if len(fav) < min_n:
            continue
        ppv = _win_rate(fav)
        avg_p = sum(e.model_prob for e in fav) / len(fav)
        gap = avg_p - ppv  # over-confidence among favored picks

        if ppv < BREAKEVEN:
            msg = (
                f"{market}: favored picks win only {ppv * 100:.1f}% (n={len(fav)}), "
                f"below the {BREAKEVEN * 100:.1f}% breakeven"
            )
            if gap > 0.03:
                msg += (
                    f"; model predicts {avg_p * 100:.1f}% -> over-confident by "
                    f"{gap * 100:.1f} pts. Shrink this projection / raise the buy "
                    f"threshold to cut false positives"
                )
            else:
                msg += ". Require a larger edge before recommending this prop"
            out.append(PropInsight(market, FALSE_POSITIVE, len(fav), ppv, msg))
        elif gap > 0.05:
            out.append(
                PropInsight(
                    market,
                    FALSE_POSITIVE,
                    len(fav),
                    ppv,
                    f"{market}: favored picks over-confident by {gap * 100:.1f} pts "
                    f"(predicts {avg_p * 100:.1f}%, hits {ppv * 100:.1f}%, n={len(fav)}) "
                    f"-> tighten calibration to trim phantom-edge false positives",
                )
            )

        # worst line inside the favored set
        wl = _worst_line(fav, min_n)
        if wl is not None and wl[1] < BREAKEVEN:
            line, wr, n = wl
            out.append(
                PropInsight(
                    market,
                    FALSE_POSITIVE,
                    n,
                    wr,
                    f"{market} line {_fmt_line(line)}: favored picks at this line win "
                    f"{wr * 100:.1f}% (n={n}) -> stop buying this line or demand a bigger edge",
                )
            )

        # EV-tiered buys that still lose (phantom edge that reached the sheet)
        buys = [e for e in fav if e.tier in _BUY]
        if len(buys) >= min_n:
            bwr = _win_rate(buys)
            if bwr < BREAKEVEN:
                out.append(
                    PropInsight(
                        market,
                        FALSE_POSITIVE,
                        len(buys),
                        bwr,
                        f"{market}: Strong/Moderate BUYS win {bwr * 100:.1f}% (n={len(buys)}) "
                        f"-> raise MLBE_MIN_EDGE_{market.upper()}; current buys are "
                        "false positives",
                    )
                )
    return out


def false_negative_insights(
    entries: list[LedgerEntry], min_n: int = DEFAULT_MIN_N
) -> list[PropInsight]:
    """Faded props that keep winning -> under-rated pockets to reclaim (NPV)."""
    out: list[PropInsight] = []
    by_market: dict[str, list[LedgerEntry]] = defaultdict(list)
    for e in entries:
        if is_prop(e.market):
            by_market[e.market].append(e)

    for market in sorted(by_market):
        faded = _faded(by_market[market])
        if len(faded) < min_n:
            continue
        fn_win = _win_rate(faded)  # faded picks that actually won
        if fn_win > ENGINE_PROB_THRESHOLD:
            avg_p = sum(e.model_prob for e in faded) / len(faded)
            out.append(
                PropInsight(
                    market,
                    FALSE_NEGATIVE,
                    len(faded),
                    fn_win,
                    f"{market}: faded picks still win {fn_win * 100:.1f}% (n={len(faded)}) "
                    f"while the model rates them {avg_p * 100:.1f}% -> systematically "
                    f"under-rated; raise the base projection to reclaim these false negatives",
                )
            )
        # most reclaimable line among fades (highest realized win rate over breakeven)
        bl = _best_line(faded, min_n)
        if bl is not None and bl[1] > BREAKEVEN:
            line, wr, n = bl
            out.append(
                PropInsight(
                    market,
                    FALSE_NEGATIVE,
                    n,
                    wr,
                    f"{market} line {_fmt_line(line)}: faded picks at this line win "
                    f"{wr * 100:.1f}% (n={n}) -> lift the projection here; it is a "
                    f"reclaimable false-negative pocket",
                )
            )
    return out


def true_positive_insights(
    entries: list[LedgerEntry], min_n: int = DEFAULT_MIN_N
) -> list[PropInsight]:
    """The prop pockets the model already nails -> concentrate/size up (PPV)."""
    out: list[PropInsight] = []
    by_market: dict[str, list[LedgerEntry]] = defaultdict(list)
    for e in entries:
        if is_prop(e.market):
            by_market[e.market].append(e)

    for market in sorted(by_market):
        fav = _favored(by_market[market])
        if len(fav) < min_n:
            continue
        ppv = _win_rate(fav)
        if ppv > BREAKEVEN:
            out.append(
                PropInsight(
                    market,
                    TRUE_POSITIVE,
                    len(fav),
                    ppv,
                    f"{market}: favored picks hit {ppv * 100:.1f}% (n={len(fav)}), above "
                    f"breakeven -> reliable; lean into it and size up",
                )
            )
        # sharpest line among the true positives
        bl = _best_line(fav, min_n)
        if bl is not None and bl[1] > max(ppv, BREAKEVEN):
            line, wr, n = bl
            out.append(
                PropInsight(
                    market,
                    TRUE_POSITIVE,
                    n,
                    wr,
                    f"{market} line {_fmt_line(line)}: favored picks at this line hit "
                    f"{wr * 100:.1f}% (n={n}) -> highest-conviction pocket; concentrate here",
                )
            )
        # tier that converts best
        buys = [e for e in fav if e.tier in _BUY]
        if len(buys) >= min_n:
            bwr = _win_rate(buys)
            if bwr > BREAKEVEN and bwr >= ppv:
                out.append(
                    PropInsight(
                        market,
                        TRUE_POSITIVE,
                        len(buys),
                        bwr,
                        f"{market}: Strong/Moderate BUYS convert at {bwr * 100:.1f}% "
                        f"(n={len(buys)}) -> the tiering is working here; keep/size up",
                    )
                )
    return out


def prop_insights(entries: list[LedgerEntry], min_n: int = DEFAULT_MIN_N) -> list[PropInsight]:
    """All FP + FN + TP prop insights, in that order."""
    return (
        false_positive_insights(entries, min_n)
        + false_negative_insights(entries, min_n)
        + true_positive_insights(entries, min_n)
    )


# --- run-line miss matrix --------------------------------------------------
# A backed -1.5 favorite that loses does so in one of two ways: the team won but
# only by 1 (a *one-run-win error* -> the model was right on the winner but the
# offense lacked the power to cover), or the team lost outright (wrong winner).
# A backed +1.5 dog that loses did so by 2+, split into a *blowout error*
# (lost by 5+ -> starter/bullpen variance blew the game open) and a moderate
# 2-4 run loss. Tracking which bucket dominates tells us which knob to turn.
BLOWOUT_MARGIN = 5  # dog lost by this many or more -> blowout error


@dataclass
class RunLineMissMatrix:
    """Where backed run-line picks miss (from ledger rows with a stored margin)."""

    fav_n: int  # decided, favored -1.5 picks with a margin
    fav_cover: int  # covered -1.5 (win)
    fav_one_run: int  # lost: backed team won by exactly 1
    fav_outright: int  # lost: backed team lost the game
    dog_n: int  # decided, favored +1.5 picks with a margin
    dog_cover: int  # covered +1.5 (win)
    dog_moderate: int  # lost by 2-4
    dog_blowout: int  # lost by 5+

    @property
    def fav_losses(self) -> int:
        return self.fav_one_run + self.fav_outright

    @property
    def dog_losses(self) -> int:
        return self.dog_moderate + self.dog_blowout

    @property
    def has_data(self) -> bool:
        return self.fav_n > 0 or self.dog_n > 0


def _is_fav_rl(e: LedgerEntry) -> bool:
    return e.line is not None and e.line < 0


def _is_dog_rl(e: LedgerEntry) -> bool:
    return e.line is not None and e.line > 0


def run_line_miss_matrix(entries: list[LedgerEntry]) -> RunLineMissMatrix:
    """Break backed run-line losses into one-run-win vs blowout errors.

    Only ``game_rl`` rows the model favored (``model_prob >= 0.5``), that were
    decided (not a push) and carry a stored margin, are counted.
    """
    rl = [
        e
        for e in _decided(entries)
        if e.market == "game_rl"
        and e.model_prob >= ENGINE_PROB_THRESHOLD
        and e.margin is not None
    ]
    m = RunLineMissMatrix(0, 0, 0, 0, 0, 0, 0, 0)
    for e in rl:
        margin = e.margin
        assert margin is not None  # narrowed by the filter above
        if _is_fav_rl(e):
            m.fav_n += 1
            if _won(e):
                m.fav_cover += 1
            elif margin == 1:
                m.fav_one_run += 1
            else:
                m.fav_outright += 1
        elif _is_dog_rl(e):
            m.dog_n += 1
            if _won(e):
                m.dog_cover += 1
            elif margin <= -BLOWOUT_MARGIN:
                m.dog_blowout += 1
            else:
                m.dog_moderate += 1
    return m


def run_line_miss_findings(entries: list[LedgerEntry]) -> list[str]:
    """Plain-language findings from the run-line miss matrix (empty if no data)."""
    m = run_line_miss_matrix(entries)
    out: list[str] = []
    if m.fav_losses > 0:
        share = m.fav_one_run / m.fav_losses
        if m.fav_one_run >= m.fav_outright:
            out.append(
                f"**-1.5 favorites die on the margin, not the pick.** {m.fav_one_run} of "
                f"{m.fav_losses} favorite run-line losses were one-run wins (the backed "
                f"team won but by 1). The model reads the winner but the offense lacks the "
                f"power to cover — lean -1.5 toward high-ISO/barrel lineups vs barrel-prone "
                f"starters, or take the ML instead. ({share * 100:.0f}% of the misses)"
            )
        else:
            out.append(
                f"**-1.5 favorites are the wrong team.** {m.fav_outright} of {m.fav_losses} "
                f"favorite run-line losses were outright losses (backed team didn't win) — "
                f"a side-selection problem, not a power problem. Tighten who we call the "
                f"favorite before stretching to -1.5."
            )
    if m.dog_losses > 0:
        share = m.dog_blowout / m.dog_losses
        if m.dog_blowout >= m.dog_moderate:
            out.append(
                f"**+1.5 dogs get blown out.** {m.dog_blowout} of {m.dog_losses} underdog "
                f"run-line losses were 5+ run blowouts — starter/bullpen variance is opening "
                f"games up. Gate +1.5 toward groundball underdog pitchers in low-total, "
                f"pitcher-friendly parks. ({share * 100:.0f}% of the misses)"
            )
        else:
            out.append(
                f"**+1.5 dogs lose close.** {m.dog_moderate} of {m.dog_losses} underdog "
                f"run-line losses were 2-4 run games, not blowouts — the +1.5 is landing "
                f"near the number; a small push toward tighter game scripts recovers them."
            )
    return out


# --- price buckets ---------------------------------------------------------
# A dog is *supposed* to win less than half the time, so a sub-50% win rate is
# not evidence of anything on its own: what matters is the win rate against the
# break-even the price demands. These buckets report that gap, so "we win 36% of
# our +150 dogs" can be read as the -4 point miss it is rather than a disaster,
# and a 52% hit rate on -140 favorites can be read as the loss it is.
PRICE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("Heavy favorite (-200 and shorter)", -1e9, -200.0),
    ("Favorite (-199 to -110)", -199.0, -110.0),
    ("Pick'em (-109 to +109)", -109.0, 109.0),
    ("Short dog (+110 to +199)", 110.0, 199.0),
    ("Mid dog (+200 to +399)", 200.0, 399.0),
    ("Longshot (+400 and up)", 400.0, 1e9),
)


@dataclass
class PriceBucket:
    """Realized performance of the buys in one price band."""

    label: str
    n: int
    win_rate: float
    breakeven: float  # mean break-even win rate the prices demand
    roi: float
    units: float

    @property
    def shortfall(self) -> float:
        """Points of win rate above (positive) or below the price's demand."""
        return self.win_rate - self.breakeven


def _buys(entries: list[LedgerEntry]) -> list[LedgerEntry]:
    """Decided, really-priced buys: the only rows whose ROI means anything."""
    return [
        e for e in _decided(entries)
        if e.tier in _BUY and e.odds is not None
    ]


def price_buckets(entries: list[LedgerEntry]) -> list[PriceBucket]:
    """Group real-priced buys by how long the price was."""
    buys = _buys(entries)
    out: list[PriceBucket] = []
    for label, lo, hi in PRICE_BUCKETS:
        rows = [e for e in buys if e.odds is not None and lo <= e.odds <= hi]
        if not rows:
            continue
        n = len(rows)
        wins = sum(1 for e in rows if _won(e))
        be = sum(1.0 / american_to_decimal(e.odds) for e in rows if e.odds is not None)
        units = sum(e.pnl for e in rows)
        out.append(
            PriceBucket(
                label=label,
                n=n,
                win_rate=wins / n,
                breakeven=be / n,
                roi=units / n,
                units=units,
            )
        )
    return out


def dog_vs_favorite(entries: list[LedgerEntry]) -> list[PriceBucket]:
    """The same measurement collapsed to two rows: plus-money vs minus-money."""
    buys = _buys(entries)
    out: list[PriceBucket] = []
    for label, keep in (
        ("Underdogs (plus money)", lambda o: o > 0),
        ("Favorites (minus money)", lambda o: o < 0),
    ):
        rows = [e for e in buys if e.odds is not None and keep(e.odds)]
        if not rows:
            continue
        n = len(rows)
        be = sum(1.0 / american_to_decimal(e.odds) for e in rows if e.odds is not None)
        units = sum(e.pnl for e in rows)
        out.append(
            PriceBucket(
                label=label,
                n=n,
                win_rate=sum(1 for e in rows if _won(e)) / n,
                breakeven=be / n,
                roi=units / n,
                units=units,
            )
        )
    return out


@dataclass
class LineupSplit:
    """Calibration and ROI for the rows priced against one lineup provenance."""

    status: str  # posted | projected
    n: int
    model: float  # mean model probability
    realized: float  # share that actually happened
    n_buys: int
    roi: float

    @property
    def bias(self) -> float:
        """Points of probability the model was long by on these rows."""
        return self.model - self.realized


def lineup_splits(entries: list[LedgerEntry]) -> list[LineupSplit]:
    """Split graded batter rows by whether the lineup was posted or projected.

    Calibration on *all* graded rows, buys and passes alike, because that is the
    measurement the gate needs: a projected lineup damages a probability whether
    or not the row cleared the EV screen, and restricting to buys would select on
    the very overstatement being measured. ROI is reported alongside, off buys
    only, since a pass has no price to have won or lost at.
    """
    rows = [
        e for e in _decided(entries)
        if e.market.startswith("batter_") and e.lineup_status
    ]
    out: list[LineupSplit] = []
    for status in (POSTED, PROJECTED):
        group = [e for e in rows if e.lineup_status == status]
        if not group:
            continue
        buys = [e for e in group if e.tier in _BUY and e.odds is not None]
        out.append(
            LineupSplit(
                status=status,
                n=len(group),
                model=sum(e.model_prob for e in group) / len(group),
                realized=sum(1 for e in group if _won(e)) / len(group),
                n_buys=len(buys),
                roi=(sum(e.pnl for e in buys) / len(buys)) if buys else 0.0,
            )
        )
    return out


def lineup_findings(entries: list[LedgerEntry], min_n: int = LINEUP_MIN_N) -> list[str]:
    """Does pricing a projected lineup cost what the lineup-lock gate assumes?

    Under-powered is stated as under-powered. The gate this feeds is off pending
    exactly this evidence, and "projected looks worse on 40 rows" is not evidence
    -- a batter probability carries a standard error near .5 per row, so the gap
    between two groups needs hundreds of rows before it separates from noise.
    """
    splits = {s.status: s for s in lineup_splits(entries)}
    posted, projected = splits.get(POSTED), splits.get(PROJECTED)
    if not splits:
        return [
            "Lineup provenance: no graded batter row carries it yet, so "
            "projected-vs-posted cannot be measured yet. Rows priced from here "
            "record it."
        ]
    if posted is None or projected is None:
        have = ", ".join(f"{s.status} n={s.n}" for s in splits.values())
        return [
            "Lineup provenance: only one side of the split has graded batter rows "
            f"({have}), so projected-vs-posted cannot be measured yet."
        ]
    gap = projected.bias - posted.bias
    # Two independent proportions, each row's variance bounded by .25.
    se = (0.25 / projected.n + 0.25 / posted.n) ** 0.5
    head = (
        f"**Lineup provenance: projected lineups run {projected.bias * 100:+.2f}pp "
        f"against posted {posted.bias * 100:+.2f}pp** on batter markets "
        f"(projected n={projected.n}, posted n={posted.n}), a "
        f"{gap * 100:+.2f}pp difference at {abs(gap) / se if se else 0:.1f} SE."
    )
    if projected.n < min_n or posted.n < min_n:
        return [
            head + f" Under the {min_n} rows a side needs to be read at all — "
            "reported to show the sample building, not to act on."
        ]
    if abs(gap) < se:
        return [
            head + " Inside one standard error: the two are indistinguishable so "
            "far, which is an argument against the lineup-lock demotion, not for it."
        ]
    worse = "projected" if gap > 0 else "posted"
    return [
        head + f" The {worse} rows are the worse-calibrated side. Buys: projected "
        f"{projected.roi * 100:+.1f}% on {projected.n_buys}, posted "
        f"{posted.roi * 100:+.1f}% on {posted.n_buys}. If it holds, the cheap "
        "version is to price props after lineups post rather than to demote them."
    ]


def price_bucket_findings(
    entries: list[LedgerEntry], min_n: int = PRICE_MIN_N
) -> list[str]:
    """Plain-language read on whether the long prices are paying for themselves."""
    out: list[str] = []
    sides = {b.label.split(" ")[0]: b for b in dog_vs_favorite(entries)}
    dogs = sides.get("Underdogs")
    favs = sides.get("Favorites")
    if dogs is not None and dogs.n >= min_n:
        verdict = (
            "clearing the bar the price sets"
            if dogs.shortfall >= 0
            else "not clearing the bar the price sets"
        )
        out.append(
            f"**Underdog buys win {dogs.win_rate * 100:.1f}% and need "
            f"{dogs.breakeven * 100:.1f}%** (n={dogs.n}) — {verdict}, "
            f"{dogs.shortfall * 100:+.1f} points, for {dogs.roi * 100:+.1f}% ROI. A "
            "sub-50% win rate on plus money is expected; this line is the honest "
            "test of whether the price compensates."
        )
    if favs is not None and favs.n >= min_n and dogs is not None and dogs.n >= min_n:
        worse = "underdogs" if dogs.shortfall < favs.shortfall else "favorites"
        gap = abs(dogs.shortfall - favs.shortfall) * 100
        out.append(
            f"Favorites miss their break-even by {favs.shortfall * 100:+.1f} points "
            f"({favs.win_rate * 100:.1f}% vs {favs.breakeven * 100:.1f}%, n={favs.n}), so the "
            f"leak is worse on **{worse}** by {gap:.1f} points. Price length, not "
            "side, is the axis to gate on."
        )
    for b in price_buckets(entries):
        if b.n >= min_n and b.shortfall <= -PRICE_LEAK_POINTS:
            out.append(
                f"**{b.label}: {b.win_rate * 100:.1f}% against a "
                f"{b.breakeven * 100:.1f}% break-even** (n={b.n}, {b.roi * 100:+.1f}% ROI) "
                f"— {abs(b.shortfall) * 100:.1f} points short. If it holds, cap buys in "
                "this band."
            )
    return out
