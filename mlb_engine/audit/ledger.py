"""A persistent per-bet ledger plus cumulative engine performance metrics.

The ledger records every graded recommendation (one row per bet) across all
audited slates, and rolls the whole history up into overall sensitivity,
specificity, PPV, NPV, win rate, ROI and net units by tier — plus a single
**whole-engine** row (see :func:`engine_metrics`).

Confusion-matrix framing (same as the daily scorecard): for a tier T the
positive *prediction* is "pick is in tier T" and the positive *outcome* is "the
bet won" (pushes excluded):

    PPV         = TP / (TP + FP)   # win rate of that tier's picks
    NPV         = TN / (TN + FN)
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)

The whole-engine row uses the same math but a different positive *prediction*:
"the model favors this selection" (``model_prob >= 0.5``), aggregated across
every graded market and tier. It is keyed on the model's own probability
boundary (the same 0.5 boundary the backtest uses), so it measures the engine's
raw directional discrimination independent of EV/odds/tiering.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date as Date
from pathlib import Path

from mlb_engine.audit.grade import LOSS, PUSH, WIN, picked_margin
from mlb_engine.data.results import GameResult
from mlb_engine.market.odds import american_to_decimal
from mlb_engine.market.tiers import Tier
from mlb_engine.recommendations import Recommendation

_DEFAULT_DECIMAL = 1.91  # assume -110 when no price was captured


@dataclass
class LedgerEntry:
    date: str
    matchup: str
    category: str
    market: str
    selection: str
    line: float | None
    book: str
    odds: float | None  # american
    tier: str
    model_prob: float
    ev: float | None
    result: str  # win | loss | push
    pnl: float  # net units on a 1u stake (win: dec-1, loss: -1, push: 0)
    # Final margin from the picked side's perspective (team_runs - opp_runs), for
    # game/first-5 run-line rows only; None for every other market. Enables the
    # run-line miss matrix (one-run-win vs blowout errors) in the audit report.
    margin: float | None = None
    veto_gate: str = ""  # run-line NPV gate that removed this pick, "" if none
    # Which screen turned this selection into a Pass ("" when it was bought).
    # A gate is only gradeable once its own rows can be selected: see
    # :func:`gate_metrics`.
    pass_gate: str = ""
    # Pre-calibration probability. `mlb-engine calibrate` refits the isotonic
    # map from this column, not from `model_prob`: the map is applied to raw
    # simulation output, so fitting it on already-calibrated probabilities
    # would learn a correction that is then applied to the wrong input.
    raw_prob: float | None = None
    # Devigged market price at bet time and the probability the EV screen bet
    # (they differ only when MLBE_MARKET_ANCHOR is on).
    fair_prob: float | None = None
    bet_prob: float | None = None
    # Closing line value: the closing price, its no-vig probability, the points
    # the market moved our way, and the EV of our price under the closing
    # probability. Populated by `mlb-engine close` + audit; None when the closing
    # snapshot was not captured.
    close_odds: float | None = None
    close_prob: float | None = None
    clv: float | None = None
    clv_ev: float | None = None
    # American price of the other side (under for O/U props, opposing team for
    # ML/RL) at the same book. Persisted so the fade side can be graded/backtested
    # without re-fetching historical odds. None when the market was unpriced.
    under_odds: float | None = None
    # Was this row priced against the lineup that actually batted, and how long
    # before first pitch? ``features.lineup_lock`` stamps both on every
    # recommendation so the ledger can measure whether projected-lineup buys
    # underperform posted ones -- which is the evidence its own demotion gate is
    # waiting on, and which nothing could measure while the columns stopped at
    # the recommendation. "" on rows written before they were persisted.
    lineup_status: str = ""
    hours_to_first_pitch: float | None = None


LEDGER_FIELDS = [
    "date",
    "matchup",
    "category",
    "market",
    "selection",
    "line",
    "book",
    "odds",
    "under_odds",
    "tier",
    "model_prob",
    "ev",
    "result",
    "pnl",
    "margin",
    "veto_gate",
    "pass_gate",
    "raw_prob",
    "fair_prob",
    "bet_prob",
    "close_odds",
    "close_prob",
    "clv",
    "clv_ev",
    "lineup_status",
    "hours_to_first_pitch",
]
_OPTIONAL_FLOAT_FIELDS = (
    "line",
    "odds",
    "under_odds",
    "ev",
    "margin",
    "fair_prob",
    "bet_prob",
    "close_odds",
    "close_prob",
    "clv",
    "clv_ev",
    "hours_to_first_pitch",
)


def _pnl(result: str, odds: float | None) -> float:
    if result == WIN:
        dec = american_to_decimal(odds) if odds is not None else _DEFAULT_DECIMAL
        return round(dec - 1.0, 4)
    if result == LOSS:
        return -1.0
    return 0.0


def entries_from_graded(
    graded: list[tuple[Recommendation, str]],
    date: Date,
    results: dict[int, GameResult] | None = None,
) -> list[LedgerEntry]:
    """Build ledger rows from graded picks.

    ``results`` (game_pk -> :class:`GameResult`) is optional; when supplied, the
    picked-side run-line margin is recorded on run-line rows for the miss matrix.
    """
    entries: list[LedgerEntry] = []
    for rec, result in graded:
        margin: float | None = None
        if results is not None:
            res = results.get(rec.game_pk)
            if res is not None:
                margin = picked_margin(rec, res)
        entries.append(
            LedgerEntry(
                date=date.isoformat(),
                matchup=rec.matchup,
                category=rec.display_category,
                market=rec.market,
                selection=rec.selection,
                line=rec.line,
                book=rec.book or "",
                odds=rec.market_american,
                under_odds=rec.opposite_american,
                tier=rec.tier.value,
                model_prob=round(rec.model_prob, 4),
                ev=round(rec.ev, 4) if rec.ev is not None else None,
                result=result,
                pnl=_pnl(result, rec.market_american),
                margin=margin,
                veto_gate=rec.veto_gate or "",
                pass_gate=rec.pass_gate or "",
                raw_prob=round(rec.raw_prob, 4) if rec.raw_prob is not None else None,
                fair_prob=round(rec.fair_prob, 4) if rec.fair_prob is not None else None,
                bet_prob=round(rec.bet_prob, 4) if rec.bet_prob is not None else None,
                lineup_status=rec.lineup_status or "",
                hours_to_first_pitch=(
                    round(rec.hours_to_first_pitch, 2)
                    if rec.hours_to_first_pitch is not None
                    else None
                ),
            )
        )
    return entries


def _to_float(v: str) -> float | None:
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_ledger(path: Path) -> list[LedgerEntry]:
    if not path.exists():
        return []
    out: list[LedgerEntry] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out.append(
                LedgerEntry(
                    date=row["date"],
                    matchup=row["matchup"],
                    category=row["category"],
                    market=row["market"],
                    selection=row["selection"],
                    line=_to_float(row["line"]),
                    book=row["book"],
                    odds=_to_float(row["odds"]),
                    under_odds=_to_float(row.get("under_odds", "") or ""),
                    tier=row["tier"],
                    model_prob=_to_float(row["model_prob"]) or 0.0,
                    ev=_to_float(row["ev"]),
                    result=row["result"],
                    pnl=_to_float(row["pnl"]) or 0.0,
                    margin=_to_float(row.get("margin", "") or ""),
                    veto_gate=row.get("veto_gate", ""),
                    pass_gate=row.get("pass_gate", ""),
                    raw_prob=_to_float(row.get("raw_prob", "")),
                    fair_prob=_to_float(row.get("fair_prob", "")),
                    bet_prob=_to_float(row.get("bet_prob", "")),
                    close_odds=_to_float(row.get("close_odds", "")),
                    close_prob=_to_float(row.get("close_prob", "")),
                    clv=_to_float(row.get("clv", "")),
                    clv_ev=_to_float(row.get("clv_ev", "")),
                    lineup_status=row.get("lineup_status", ""),
                    hours_to_first_pitch=_to_float(row.get("hours_to_first_pitch", "")),
                )
            )
    return out


def update_ledger(path: Path, new_entries: list[LedgerEntry], date: Date) -> list[LedgerEntry]:
    """Replace any rows for ``date`` with ``new_entries`` (re-audit safe), persist, return all."""
    iso = date.isoformat()
    kept = [e for e in load_ledger(path) if e.date != iso]
    merged = kept + new_entries
    merged.sort(key=lambda e: (e.date, e.category, e.matchup))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        w.writeheader()
        for e in merged:
            row = asdict(e)
            for name in _OPTIONAL_FLOAT_FIELDS:
                if row[name] is None:
                    row[name] = ""
            w.writerow(row)
    return merged


def _safe(n: float, d: float) -> float:
    return round(n / d, 4) if d else 0.0


@dataclass
class OverallMetrics:
    tier: str
    n: int
    wins: int
    losses: int
    pushes: int
    win_pct: float
    ppv: float
    npv: float
    sensitivity: float
    specificity: float
    roi: float
    units: float
    # Win rate the prices actually charged for: mean 1/decimal over the bets this
    # row counts as positive. Nine retro-priced slates won 59.6% of favoured bets
    # into an average break-even of 60.5%, so win% alone reads as a success and
    # loses money. Every reported win% now travels with the bar it had to clear.
    required_win_pct: float = 0.0


def _metrics(
    entries: list[LedgerEntry],
    is_positive: Callable[[LedgerEntry], bool],
    label: str,
) -> OverallMetrics:
    tp = fp = fn = tn = pushes = 0
    stake = 0.0
    units = 0.0
    breakeven = 0.0
    for e in entries:
        pred_pos = is_positive(e)
        if e.result == PUSH:
            if pred_pos:
                pushes += 1
            continue
        actual_pos = e.result == WIN
        if pred_pos and actual_pos:
            tp += 1
        elif pred_pos and not actual_pos:
            fp += 1
        elif not pred_pos and actual_pos:
            fn += 1
        else:
            tn += 1
        if pred_pos:
            stake += 1.0
            units += e.pnl
            dec = american_to_decimal(e.odds) if e.odds is not None else _DEFAULT_DECIMAL
            breakeven += 1.0 / dec
    return OverallMetrics(
        tier=label,
        n=tp + fp,
        wins=tp,
        losses=fp,
        pushes=pushes,
        win_pct=_safe(tp, tp + fp),
        ppv=_safe(tp, tp + fp),
        npv=_safe(tn, tn + fn),
        sensitivity=_safe(tp, tp + fn),
        specificity=_safe(tn, tn + fp),
        roi=_safe(units, stake),
        units=round(units, 3),
        required_win_pct=_safe(breakeven, stake),
    )


def _metrics_for(entries: list[LedgerEntry], positive: set[str], label: str) -> OverallMetrics:
    return _metrics(entries, lambda e: e.tier in positive, label)


def overall_metrics(entries: list[LedgerEntry]) -> list[OverallMetrics]:
    return [
        _metrics_for(entries, {Tier.STRONG.value}, Tier.STRONG.value),
        _metrics_for(entries, {Tier.MODERATE.value}, Tier.MODERATE.value),
        _metrics_for(entries, {Tier.PASS.value}, Tier.PASS.value),
        _metrics_for(entries, {Tier.STRONG.value, Tier.MODERATE.value}, "Buy (S+M)"),
    ]


# Probability boundary at which the model is said to "favor" a selection. The
# backtest confusion matrix uses the same 0.5 boundary on the model's own prob.
ENGINE_PROB_THRESHOLD = 0.5
ENGINE_LABEL = "ENGINE (p>=.5)"


def engine_metrics(entries: list[LedgerEntry]) -> OverallMetrics:
    """Whole-engine PPV/NPV across every graded market and tier.

    Positive prediction = the model favors the selection
    (``model_prob >= ENGINE_PROB_THRESHOLD``); positive outcome = it won.
    Measures the engine's raw directional discrimination — how often the side
    the model prefers actually wins (PPV) and how often the side it fades
    actually loses (NPV) — independent of EV, odds or tiering.
    """
    return _metrics(entries, lambda e: e.model_prob >= ENGINE_PROB_THRESHOLD, ENGINE_LABEL)


def daily_rollup(entries: list[LedgerEntry]) -> list[OverallMetrics]:
    """One Buy (S+M) metrics row per audited date, oldest first."""
    buy = {Tier.STRONG.value, Tier.MODERATE.value}
    by_date: dict[str, list[LedgerEntry]] = {}
    for e in entries:
        by_date.setdefault(e.date, []).append(e)
    return [_metrics_for(by_date[d], buy, d) for d in sorted(by_date)]


def _by(entries: list[LedgerEntry], key: Callable[[LedgerEntry], str]) -> dict[str, list[LedgerEntry]]:
    out: dict[str, list[LedgerEntry]] = {}
    for e in entries:
        out.setdefault(key(e), []).append(e)
    return out


def _favors(e: LedgerEntry) -> bool:
    return e.model_prob >= ENGINE_PROB_THRESHOLD


def daily_engine_metrics(entries: list[LedgerEntry]) -> list[OverallMetrics]:
    """Whole-engine PPV/NPV for each audited date (oldest first).

    Same directional framing as :func:`engine_metrics`, but computed per slate so
    day-to-day discrimination can be tracked.
    """
    by_date = _by(entries, lambda e: e.date)
    return [_metrics(by_date[d], _favors, d) for d in sorted(by_date)]


# --- props: batter/pitcher prop markets only -------------------------------
PROP_PREFIXES = ("batter_", "pitcher_")


def is_prop(market: str) -> bool:
    return market.startswith(PROP_PREFIXES)


_BUY_TIERS = frozenset({Tier.STRONG.value, Tier.MODERATE.value})


def _side_marker(token: str) -> str:
    """``o``/``u`` when ``token`` is a prop side+line marker like ``o1.5``."""
    if len(token) > 1 and token[0] in ("o", "u") and token[1].isdigit():
        return token[0]
    return ""


def prop_subject(selection: str) -> str:
    """A prop selection with its side/line marker dropped: who and what, not which way.

    ``"Bobby Witt Jr. H o0.5"`` and ``"Bobby Witt Jr. H u0.5"`` are the same
    wager seen from both ends, so both return ``"Bobby Witt Jr. H"``.
    """
    head, _, last = selection.rpartition(" ")
    if head and _side_marker(last):
        return head
    return selection


def prop_side_of(selection: str) -> str:
    """``o``, ``u``, or "" for a selection that carries no prop side marker."""
    return _side_marker(selection.rpartition(" ")[2])


def _is_over(selection: str) -> bool:
    return prop_side_of(selection) == "o"


def one_side_per_prop(entries: list[LedgerEntry]) -> list[LedgerEntry]:
    """Collapse the two rows of a prop to the one the engine actually stood behind.

    Since both sides of every prop are priced, each prop lands in the ledger
    twice: the side that was bought, and its complement as a Pass so the fade is
    still graded. Keeping both in a *measurement* double-counts one wager, and
    not neutrally -- the two rows are complements, so one of them is whichever
    side is nearly certain. Grading "Ohtani HR u0.5" at .96 as a correct positive
    prediction is free credit for the base rate, and it flatters every rate keyed
    on ``model_prob >= 0.5``: on the current ledger, adding the complements moves
    whole-engine PPV .531 -> .706, ``batter_hr`` .20 -> .90, and lifts five
    markets off Fade while promoting ``pitcher_k`` to Play. NPV becomes .707
    against a PPV of .706, which is the tell -- with both sides present it is no
    longer independent information, it is the same measurement restated.

    So a prop contributes one row: the side that was bought if either was, else
    the **over**. The bought side wins the tie-break because a long price can be
    taken on a side the model puts under .5 (a +560 double), and dropping that
    row would erase a real bet and its P&L. Where nothing was bought the over is
    kept rather than the favored side, for two reasons: it is the side the ledger
    has always recorded, so the calibration series stays continuous across the
    change; and "keep whichever side the model favors" would reintroduce exactly
    the problem, because the favored side of a prop is generally the near-certain
    one, and a rate built from always choosing it measures the base rate rather
    than the model.

    A row is only ever dropped where the group is exactly one over and one under,
    so the function can only remove a row it has proved is the other half of one
    already counted. Everything else passes through: non-prop markets, selections
    with no side marker, and a prop with only one side in the ledger -- which is
    every row graded before both sides were priced, so no historical number
    moves. Verified against the real ledger: synthesising the complement of all
    86,762 rows and measuring the deduped result reproduces today's numbers
    exactly, market verdicts included.
    """
    Key = tuple[str, str, str, float | None, str]
    groups: dict[Key, list[LedgerEntry]] = {}
    order: list[Key] = []
    out: list[LedgerEntry] = []
    for e in entries:
        if not is_prop(e.market) or not prop_side_of(e.selection):
            out.append(e)
            continue
        key = (e.date, e.matchup, e.market, e.line, prop_subject(e.selection))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(e)
    for key in order:
        rows = groups[key]
        overs = [r for r in rows if _is_over(r.selection)]
        unders = [r for r in rows if not _is_over(r.selection)]
        if not (len(overs) == 1 and len(unders) == 1):
            # Not a complementary pair. Anything else -- one side alone, or a key
            # that somehow repeats -- is left exactly as it is, so this can only
            # ever remove a row it has proved is the other half of one already
            # counted.
            out.extend(rows)
            continue
        bought = [r for r in rows if r.tier in _BUY_TIERS]
        out.append(bought[0] if bought else overs[0])
    return out


def prop_metrics(entries: list[LedgerEntry]) -> list[OverallMetrics]:
    """Whole-engine-style PPV/NPV for every prop market, plus an ALL PROPS row.

    One row per distinct ``batter_*`` / ``pitcher_*`` market (e.g. ``batter_hr``,
    ``pitcher_k``), keyed on the model-favored boundary, oldest-market-name first,
    followed by an aggregate ``ALL PROPS`` row.
    """
    props = [e for e in entries if is_prop(e.market)]
    by_market = _by(props, lambda e: e.market)
    rows = [_metrics(by_market[m], _favors, m) for m in sorted(by_market)]
    if props:
        rows.append(_metrics(props, _favors, "ALL PROPS"))
    return rows


# --- run lines: per-line metrics + NPV gate attribution --------------------
RUNLINE_MARKETS = ("game_rl", "f5_rl")


def runline_metrics(entries: list[LedgerEntry]) -> list[OverallMetrics]:
    """PPV/NPV for run lines, split by side, plus one row per NPV gate.

    The gate rows are the point of the exercise: each reports how the selections
    a gate *removed* actually finished. A gate is earning its keep when its row
    shows a low win rate (it deleted losers); a gate whose vetoed picks won near
    50% is destroying bet volume for nothing. ``KEPT`` is the complement — every
    run line that cleared the gates.
    """
    rls = [e for e in entries if e.market in RUNLINE_MARKETS]
    if not rls:
        return []

    rows = [_metrics(rls, _favors, "ALL RUN LINES")]
    for label, line in (("FAVORITE (-1.5)", -1.5), ("UNDERDOG (+1.5)", 1.5)):
        side = [e for e in rls if e.line == line]
        if side:
            rows.append(_metrics(side, _favors, label))

    vetoed = [e for e in rls if e.veto_gate]
    for gate in sorted({e.veto_gate for e in vetoed}):
        rows.append(_metrics([e for e in vetoed if e.veto_gate == gate], _favors, f"VETO {gate}"))
    if vetoed:
        rows.append(_metrics([e for e in rls if not e.veto_gate], _favors, "KEPT (no veto)"))
    return rows


# --- gates: what each screen rejected, and how those picks finished ---------
def _all(_e: LedgerEntry) -> bool:
    return True


def gate_metrics(entries: list[LedgerEntry]) -> list[OverallMetrics]:
    """How the selections each screen *rejected* would actually have finished.

    Every rejected pick counts as a bet here rather than only the model-favored
    side, because that is the counterfactual being tested: had the screen not
    fired, the engine would have bet these selections at these prices. So read
    each row against its ``required_win_pct`` and not against 50% -- a screen is
    earning its keep when the picks it deleted finished *below* the bar their
    price set, and is manufacturing false negatives when they cleared it.

    Unpriced rows are excluded: there was no bet to forgo. ``BOUGHT`` is the
    complement, every priced selection that survived every screen.
    """
    priced = [e for e in entries if e.odds is not None]
    by_gate = _by([e for e in priced if e.pass_gate], lambda e: e.pass_gate)
    rows = [
        _metrics(by_gate[g], _all, f"GATE {g}")
        for g in sorted(by_gate, key=lambda g: -len(by_gate[g]))
    ]
    bought = [e for e in priced if not e.pass_gate]
    if bought:
        rows.append(_metrics(bought, _all, "BOUGHT (no gate)"))
    return rows


def market_metrics(entries: list[LedgerEntry]) -> list[OverallMetrics]:
    """Whole-engine-style PPV/NPV for *every* market, sorted by ROI (high to low).

    One :class:`OverallMetrics` row per distinct market (game and F5 lines as well
    as props), each keyed on the model-favored boundary (``model_prob >= 0.5``).
    Markets the model never favored (``n == 0``) still appear so the report can
    show that the engine correctly abstained; they sort last.
    """
    by_market = _by(entries, lambda e: e.market)
    rows = [_metrics(by_market[m], _favors, m) for m in by_market]
    rows.sort(key=lambda m: (m.n == 0, -m.roi))
    return rows
