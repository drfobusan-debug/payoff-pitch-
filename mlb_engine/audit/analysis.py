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
from mlb_engine.market.tiers import Tier

# A -110 line needs ~52.4% to break even; the model's own decision boundary is
# 0.5. We flag favored pockets under breakeven and faded pockets over it.
BREAKEVEN = 0.524
DEFAULT_MIN_N = 20
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
                    f"{market} o{_fmt_line(line)}: favored picks at this line win "
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
                        f"-> raise the EV threshold; current buys are false positives",
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
                    f"{market} o{_fmt_line(line)}: faded picks at this line win "
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
                    f"{market} o{_fmt_line(line)}: favored picks at this line hit "
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
