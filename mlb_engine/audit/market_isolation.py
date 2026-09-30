"""Isolate a prop market's model accuracy from price, tier and selection.

The power-screen audit found a handful of markets in the black -- batter hits and
walks, a starter's walks and outs -- and units alone cannot say why. A market
can pay because the model's probability sits closer to the outcome than the
book's (skill), or because the rows it happened to hold were priced long, or
because the buy gates picked the right handful (selection), or because forty
rows went its way (luck). This module scores each of those on its own:

* **Accuracy.** Brier and log-loss of the model probability, the printed
  (anchored) probability and the two-sided no-vig market probability against the
  same outcomes, and a paired bootstrap interval on model-minus-market Brier so
  "the model beats the market" carries its uncertainty.
* **Calibration.** Predicted against realised, by probability decile, for the
  model and for the market.
* **Splits.** Over/under, price band, bought versus passed, edge tercile and
  line value, each with its record and its accuracy.
* **Selection.** Betting every row of the market blind at the recorded price,
  against betting only the rows whose model-minus-market edge cleared a
  threshold, so the value of the pick is read apart from the value of the number.

One-sided quotes are reported on their own: with no opposite price there is no
fair two-sided mark, and scoring the model against a vigged single price would
flatter it.

Rows come from either ledger the engine keeps -- the power screen's own receipt,
graded off the box score, or the main audit ledger, which arrives graded -- and
are reduced to one shape (:class:`Row`) before anything is scored. Nothing here
writes a price, a tier or a gate.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date as Date
from pathlib import Path

import numpy as np

from mlb_engine.audit.grade import LOSS, PUSH, WIN
from mlb_engine.audit.ledger import pnl_units
from mlb_engine.audit.power_ledger import GradedPosition, Position
from mlb_engine.output.power_board import BUY_TIERS

#: The markets the 9/18 audit found in the black, in the order they are reported.
FOCUS_MARKETS: tuple[str, ...] = ("H", "BB", "SP BB", "SP outs")

#: How the main ledger's market codes map onto the power screen's labels.
ENGINE_MARKET_LABEL: dict[str, str] = {
    "batter_h": "H",
    "batter_bb": "BB",
    "batter_hrr": "H+R+RBI",
    "batter_tb": "TB",
    "batter_rbi": "RBI",
    "batter_1b": "1B",
    "batter_2b": "2B",
    "batter_r": "R",
    "batter_hr": "HR",
    "batter_k": "K (bat)",
    "pitcher_bb": "SP BB",
    "pitcher_outs": "SP outs",
    "pitcher_h": "SP H",
    "pitcher_er": "SP ER",
    "pitcher_k": "SP K",
}

PRICE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("<= -150", -math.inf, -150),
    ("-149..+149", -149, 149),
    ("+150..+299", 150, 299),
    (">= +300", 300, math.inf),
)

_BAND_ORDER = {label: i for i, (label, _, _) in enumerate(PRICE_BANDS)}

#: Edge thresholds, in probability points, for the selection ladder.
EDGE_STEPS_PTS: tuple[int, ...] = (0, 1, 2, 3)

_EPS = 1e-6


@dataclass(frozen=True)
class Row:
    """One graded prop, reduced to what accuracy and price need to know."""

    market: str
    date: str
    side: str
    line: float | None
    odds: float | None
    model_prob: float
    bet_prob: float
    fair_prob: float | None
    tier: str
    gate: str
    result: str
    units: float
    one_way: bool = False

    @property
    def decided(self) -> bool:
        return self.result in (WIN, LOSS)

    @property
    def outcome(self) -> int:
        return 1 if self.result == WIN else 0

    @property
    def two_sided(self) -> bool:
        """Scoreable against a fair mark: decided, priced both ways, mark known."""
        return self.decided and not self.one_way and self.fair_prob is not None

    @property
    def edge(self) -> float | None:
        """Model less the no-vig market, in probability; None with no mark."""
        return None if self.fair_prob is None else self.model_prob - self.fair_prob

    @property
    def is_buy(self) -> bool:
        return self.tier in BUY_TIERS

    @property
    def line_label(self) -> str:
        if self.line is None:
            return "(no line)"
        return f"{'o' if self.side == 'over' else 'u'}{self.line:g}"


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


def rows_from_power(graded: Iterable[GradedPosition]) -> list[Row]:
    """The power screen's graded positions, as rows."""
    out: list[Row] = []
    for g in graded:
        p: Position = g.position
        out.append(
            Row(
                market=p.market,
                date=p.date,
                side=p.side,
                line=p.line,
                odds=p.odds,
                model_prob=p.model_prob,
                bet_prob=p.shown_prob,
                fair_prob=p.fair_prob,
                tier=p.tier,
                gate=p.gate,
                result=g.result,
                units=g.units,
                one_way=not p.devigged or p.gate == "one_way_quote",
            )
        )
    return out


def _float(v: str | None) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _in_range(day: str, start: Date | None, end: Date | None) -> bool:
    if not day:
        return False
    try:
        d = Date.fromisoformat(day)
    except ValueError:
        return False
    return (start is None or d >= start) and (end is None or d <= end)


def _side_of(selection: str) -> str:
    """``Cam Smith 1B o0.5`` is an over, ``... u0.5`` an under."""
    tail = selection.strip().rsplit(" ", 1)[-1].lower()
    return "under" if tail.startswith("u") else "over"


def rows_from_engine_ledger(
    lines: Iterable[str], start: Date | None = None, end: Date | None = None
) -> list[Row]:
    """The main audit ledger's graded, priced prop rows, as rows.

    Only rows that carry a recorded price and a result are kept -- an unpriced
    row has no price to be graded at and no market to be scored against -- and
    game markets are left out, since this is a prop-market question.
    """
    out: list[Row] = []
    for r in csv.DictReader(lines):
        market = ENGINE_MARKET_LABEL.get(r.get("market", ""))
        if market is None or not _in_range(r.get("date", ""), start, end):
            continue
        result = (r.get("result") or "").strip().lower()
        odds = _float(r.get("odds"))
        model = _float(r.get("model_prob"))
        if result not in (WIN, LOSS, PUSH) or odds is None or model is None:
            continue
        side = _side_of(r.get("selection", ""))
        pnl = _float(r.get("pnl"))
        gate = (r.get("pass_gate") or r.get("veto_gate") or "").strip()
        bet = _float(r.get("bet_prob"))
        out.append(
            Row(
                market=market,
                date=r.get("date", ""),
                side=side,
                line=_float(r.get("line")),
                odds=odds,
                model_prob=model,
                bet_prob=model if bet is None else bet,
                fair_prob=_float(r.get("fair_prob")),
                tier=(r.get("tier") or "").strip(),
                gate=gate,
                result=result,
                units=pnl if pnl is not None else pnl_units(result, odds),
                one_way=gate == "one_way_quote" or _float(r.get("under_odds")) is None,
            )
        )
    return out


def last_run_per_day(positions: Sequence[Position]) -> list[Position]:
    """One capture per day: the last run recorded for it."""
    keep: dict[str, str] = {}
    for p in positions:
        keep[p.date] = max(keep.get(p.date, ""), p.run_id)
    return [p for p in positions if p.run_id == keep[p.date]]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def brier(probs: Sequence[float], outcomes: Sequence[int]) -> float | None:
    if not probs:
        return None
    return float(np.mean((np.asarray(probs) - np.asarray(outcomes)) ** 2))


def log_loss(probs: Sequence[float], outcomes: Sequence[int]) -> float | None:
    if not probs:
        return None
    p = np.clip(np.asarray(probs, dtype=float), _EPS, 1 - _EPS)
    o = np.asarray(outcomes, dtype=float)
    return float(-np.mean(o * np.log(p) + (1 - o) * np.log(1 - p)))


@dataclass(frozen=True)
class BootstrapCI:
    """Paired bootstrap on the mean per-row Brier difference, model less market."""

    n: int
    mean: float
    lo: float
    hi: float
    resamples: int
    #: Share of resamples in which the model scored strictly better.
    p_model_better: float

    @property
    def verdict(self) -> str:
        if self.hi < 0:
            return "model beats market"
        if self.lo > 0:
            return "market beats model"
        return "indistinguishable"


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    outcomes: Sequence[int],
    resamples: int = 2000,
    seed: int = 7,
    level: float = 0.95,
) -> BootstrapCI | None:
    """CI on mean(sq_err(a) - sq_err(b)); negative favours ``a``."""
    n = len(outcomes)
    if n < 2:
        return None
    o = np.asarray(outcomes, dtype=float)
    diff = (np.asarray(a) - o) ** 2 - (np.asarray(b) - o) ** 2
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    means = diff[idx].mean(axis=1)
    tail = (1 - level) / 2
    return BootstrapCI(
        n=n,
        mean=float(diff.mean()),
        lo=float(np.quantile(means, tail)),
        hi=float(np.quantile(means, 1 - tail)),
        resamples=resamples,
        p_model_better=float(np.mean(means < 0)),
    )


@dataclass(frozen=True)
class Record:
    """Wins, losses, pushes and units over some rows, with accuracy where scoreable."""

    label: str
    n: int
    wins: int
    losses: int
    pushes: int
    units: float
    brier_model: float | None
    brier_bet: float | None
    brier_market: float | None
    scored: int

    @property
    def decided(self) -> int:
        return self.wins + self.losses

    @property
    def hit_rate(self) -> float | None:
        return self.wins / self.decided if self.decided else None

    @property
    def roi(self) -> float | None:
        return self.units / self.n if self.n else None

    @property
    def brier_gap(self) -> float | None:
        """Model less market Brier; negative means the model was closer."""
        if self.brier_model is None or self.brier_market is None:
            return None
        return self.brier_model - self.brier_market


def record(label: str, rows: Sequence[Row]) -> Record:
    scored = [r for r in rows if r.two_sided]
    outcomes = [r.outcome for r in scored]
    return Record(
        label=label,
        n=len(rows),
        wins=sum(1 for r in rows if r.result == WIN),
        losses=sum(1 for r in rows if r.result == LOSS),
        pushes=sum(1 for r in rows if r.result == PUSH),
        units=round(sum(r.units for r in rows), 4),
        brier_model=brier([r.model_prob for r in scored], outcomes),
        brier_bet=brier([r.bet_prob for r in scored], outcomes),
        brier_market=brier([r.fair_prob or 0.0 for r in scored], outcomes),
        scored=len(scored),
    )


@dataclass(frozen=True)
class CalibrationBin:
    label: str
    n: int
    predicted: float
    realised: float


def calibration(
    rows: Sequence[Row], prob: Callable[[Row], float | None], bins: int = 10
) -> list[CalibrationBin]:
    """Predicted against realised, in equal-width probability bins."""
    buckets: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for r in rows:
        if not r.decided:
            continue
        p = prob(r)
        if p is None:
            continue
        k = min(int(p * bins), bins - 1)
        buckets[k].append((p, r.outcome))
    out: list[CalibrationBin] = []
    for k in sorted(buckets):
        pairs = buckets[k]
        out.append(
            CalibrationBin(
                label=f"{k / bins:.1f}-{(k + 1) / bins:.1f}",
                n=len(pairs),
                predicted=sum(p for p, _ in pairs) / len(pairs),
                realised=sum(o for _, o in pairs) / len(pairs),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


def price_band(odds: float | None) -> str:
    if odds is None:
        return "(no price)"
    for label, lo, hi in PRICE_BANDS:
        if lo <= odds <= hi:
            return label
    return "(no price)"


def edge_terciles(rows: Sequence[Row]) -> list[tuple[str, list[Row]]]:
    """Rows with a mark, cut into thirds of model-minus-market edge, lowest first."""
    marked = sorted((r for r in rows if r.edge is not None), key=lambda r: r.edge or 0.0)
    if not marked:
        return []
    n = len(marked)
    cuts = [0, n // 3, (2 * n) // 3, n]
    out: list[tuple[str, list[Row]]] = []
    for i, name in enumerate(("low edge", "mid edge", "high edge")):
        chunk = marked[cuts[i] : cuts[i + 1]]
        if not chunk:
            continue
        lo = 100 * (chunk[0].edge or 0.0)
        hi = 100 * (chunk[-1].edge or 0.0)
        out.append((f"{name} ({lo:+.1f}..{hi:+.1f}pts)", chunk))
    return out


def _grouped(rows: Sequence[Row], key: Callable[[Row], str]) -> list[tuple[str, list[Row]]]:
    groups: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    return sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))


def _line_sort(label: str) -> tuple[float, str]:
    try:
        return float(label[1:]), label[0]
    except ValueError:
        return math.inf, label


def selection_ladder(rows: Sequence[Row], steps: Sequence[int] = EDGE_STEPS_PTS) -> list[Record]:
    """Blind at the recorded price, then only rows clearing each edge threshold.

    Only rows with a two-sided mark can be ranked on edge, so the blind line is
    restricted to them too: otherwise the ladder's first rung would hold rows
    the later rungs could never hold.
    """
    marked = [r for r in rows if r.fair_prob is not None and not r.one_way]
    out = [record("every two-sided row, blind", marked)]
    for x in steps:
        picked = [r for r in marked if (r.edge or 0.0) * 100 >= x - 1e-9]
        out.append(record(f"edge >= {x} pt", picked))
    return out


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


@dataclass
class MarketReport:
    market: str
    all_rows: Record
    two_sided: Record
    one_way: Record
    logloss_model: float | None
    logloss_bet: float | None
    logloss_market: float | None
    bootstrap: BootstrapCI | None
    calibration_model: list[CalibrationBin]
    calibration_market: list[CalibrationBin]
    splits: dict[str, list[Record]] = field(default_factory=dict)
    ladder: list[Record] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.bootstrap is None:
            return "no two-sided rows"
        return self.bootstrap.verdict

    @property
    def thin(self) -> bool:
        return self.two_sided.scored < 100


def build_report(
    market: str, rows: Sequence[Row], resamples: int = 2000, seed: int = 7
) -> MarketReport:
    mine = [r for r in rows if r.market == market]
    scored = [r for r in mine if r.two_sided]
    one_way = [r for r in mine if r.one_way]
    outcomes = [r.outcome for r in scored]
    model = [r.model_prob for r in scored]
    fair = [r.fair_prob or 0.0 for r in scored]
    splits: dict[str, list[Record]] = {
        "side": [record(k, g) for k, g in _grouped(mine, lambda r: r.side)],
        "price band": [
            record(k, g)
            for k, g in sorted(
                _grouped(mine, lambda r: price_band(r.odds)),
                key=lambda kv: _BAND_ORDER.get(kv[0], len(PRICE_BANDS)),
            )
        ],
        "tier": [
            record(k, g) for k, g in _grouped(mine, lambda r: "bought" if r.is_buy else "Pass")
        ],
        "edge tercile": [record(k, g) for k, g in edge_terciles(scored)],
        "line": [
            record(k, g)
            for k, g in sorted(
                _grouped(mine, lambda r: r.line_label), key=lambda kv: _line_sort(kv[0])
            )
        ],
    }
    return MarketReport(
        market=market,
        all_rows=record("all rows", mine),
        two_sided=record("two-sided", scored),
        one_way=record("one-sided quotes", one_way),
        logloss_model=log_loss(model, outcomes),
        logloss_bet=log_loss([r.bet_prob for r in scored], outcomes),
        logloss_market=log_loss(fair, outcomes),
        bootstrap=paired_bootstrap(model, fair, outcomes, resamples=resamples, seed=seed),
        calibration_model=calibration(scored, lambda r: r.model_prob),
        calibration_market=calibration(scored, lambda r: r.fair_prob),
        splits=splits,
        ladder=selection_ladder(mine),
    )


def markets_present(rows: Sequence[Row]) -> list[str]:
    """Focus markets first, then the rest by row count."""
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r.market] += 1
    rest = sorted((m for m in counts if m not in FOCUS_MARKETS), key=lambda m: -counts[m])
    return [m for m in FOCUS_MARKETS if m in counts] + rest


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _f(v: float | None, fmt: str, blank: str = "-") -> str:
    return blank if v is None else format(v, fmt)


def _pct(v: float | None) -> str:
    return _f(v, ".1%")


def _signed_pct(v: float | None) -> str:
    return _f(v, "+.1%")


def record_line(rec: Record, width: int = 30) -> str:
    rec_str = f"{rec.wins}-{rec.losses}" + (f"-{rec.pushes}" if rec.pushes else "")
    return (
        f"  {rec.label:<{width}} n={rec.n:<4} {rec_str:>9}  hit {_pct(rec.hit_rate):>6}"
        f"  {rec.units:+7.2f}u  ROI {_signed_pct(rec.roi):>7}"
        f"  Brier m/b/mkt {_f(rec.brier_model, '.4f')}/{_f(rec.brier_bet, '.4f')}"
        f"/{_f(rec.brier_market, '.4f')} (gap {_f(rec.brier_gap, '+.4f')}, {rec.scored} scored)"
    )


def verdict_text(rep: MarketReport) -> str:
    b = rep.bootstrap
    if b is None:
        return "no two-sided rows to score"
    caveat = f"; n={b.n} < 100, thin" if rep.thin else f"; n={b.n}"
    return (
        f"{b.verdict}: model-minus-market Brier {b.mean:+.4f} "
        f"[{b.lo:+.4f}, {b.hi:+.4f}] 95% CI, P(model better)={b.p_model_better:.2f}{caveat}"
    )


def render_console(rep: MarketReport) -> str:
    out: list[str] = [f"=== {rep.market} ===", record_line(rep.all_rows)]
    out.append(record_line(rep.two_sided))
    out.append(record_line(rep.one_way))
    out.append(
        f"  log-loss model/printed/market: {_f(rep.logloss_model, '.4f')} / "
        f"{_f(rep.logloss_bet, '.4f')} / {_f(rep.logloss_market, '.4f')}"
    )
    out.append(f"  bootstrap: {verdict_text(rep)}")
    for name, recs in rep.splits.items():
        out.append(f"  -- by {name} --")
        out.extend(record_line(r) for r in recs)
    out.append("  -- selection: blind vs edge threshold --")
    out.extend(record_line(r) for r in rep.ladder)
    out.append("  -- calibration (model | market) --")
    mk = {b.label: b for b in rep.calibration_market}
    for b in rep.calibration_model:
        m = mk.get(b.label)
        out.append(
            f"  {b.label:<9} model n={b.n:<4} pred {b.predicted:.3f} real {b.realised:.3f}"
            + (f"   | market n={m.n:<4} pred {m.predicted:.3f} real {m.realised:.3f}" if m else "")
        )
    for b in rep.calibration_market:
        if b.label not in {c.label for c in rep.calibration_model}:
            out.append(
                f"  {b.label:<9} {'':<36}   | market n={b.n:<4} pred {b.predicted:.3f}"
                f" real {b.realised:.3f}"
            )
    return "\n".join(out)


def summary_rows(reports: Sequence[MarketReport]) -> list[dict[str, str]]:
    """One line per market: the comparison table, as strings ready to print or write."""
    rows: list[dict[str, str]] = []
    for rep in reports:
        b = rep.bootstrap
        rows.append(
            {
                "market": rep.market,
                "n": str(rep.all_rows.n),
                "hit": _pct(rep.all_rows.hit_rate),
                "units": f"{rep.all_rows.units:+.2f}",
                "roi": _signed_pct(rep.all_rows.roi),
                "scored": str(rep.two_sided.scored),
                "brier_model": _f(rep.two_sided.brier_model, ".4f"),
                "brier_printed": _f(rep.two_sided.brier_bet, ".4f"),
                "brier_market": _f(rep.two_sided.brier_market, ".4f"),
                "logloss_model": _f(rep.logloss_model, ".4f"),
                "logloss_market": _f(rep.logloss_market, ".4f"),
                "gap": _f(rep.two_sided.brier_gap, "+.4f"),
                "ci_lo": "-" if b is None else f"{b.lo:+.4f}",
                "ci_hi": "-" if b is None else f"{b.hi:+.4f}",
                "p_model_better": "-" if b is None else f"{b.p_model_better:.2f}",
                "verdict": rep.verdict + (" (thin)" if rep.thin else ""),
                "one_way_n": str(rep.one_way.n),
                "one_way_units": f"{rep.one_way.units:+.2f}",
            }
        )
    return rows


def _md_table(header: Sequence[str], body: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(str(c) for c in row) + " |" for row in body)
    return "\n".join(lines)


def _md_records(recs: Sequence[Record]) -> str:
    return _md_table(
        ("split", "n", "W-L-P", "hit", "units", "ROI", "Brier model", "Brier mkt", "gap", "scored"),
        (
            (
                r.label,
                str(r.n),
                f"{r.wins}-{r.losses}-{r.pushes}",
                _pct(r.hit_rate),
                f"{r.units:+.2f}",
                _signed_pct(r.roi),
                _f(r.brier_model, ".4f"),
                _f(r.brier_market, ".4f"),
                _f(r.brier_gap, "+.4f"),
                str(r.scored),
            )
            for r in recs
        ),
    )


def render_markdown(
    reports: Sequence[MarketReport], title: str, focus: Sequence[str] = FOCUS_MARKETS
) -> str:
    out: list[str] = [f"# {title}", ""]
    summ = summary_rows(reports)
    cols = (
        "market",
        "n",
        "hit",
        "units",
        "roi",
        "scored",
        "brier_model",
        "brier_printed",
        "brier_market",
        "gap",
        "ci_lo",
        "ci_hi",
        "p_model_better",
        "verdict",
    )
    out.append("## Four-market summary")
    out.append("")
    out.append(_md_table(cols, ([s[c] for c in cols] for s in summ if s["market"] in focus)))
    out.append("")
    out.append("## Every market")
    out.append("")
    out.append(_md_table(cols, ([s[c] for c in cols] for s in summ)))
    out.append("")
    out.append(
        "Brier gap is model minus no-vig market on two-sided decided rows; negative favours "
        "the model. CI is a paired bootstrap on that gap. Units at the recorded price, one "
        "unit a row, pushes at zero. `scored` < 100 is flagged thin."
    )
    for rep in reports:
        out.append("")
        out.append(f"## {rep.market}")
        out.append("")
        out.append(f"Verdict: {verdict_text(rep)}")
        out.append("")
        out.append(
            f"Log-loss model / printed / market: {_f(rep.logloss_model, '.4f')} / "
            f"{_f(rep.logloss_bet, '.4f')} / {_f(rep.logloss_market, '.4f')}"
        )
        out.append("")
        out.append(_md_records([rep.all_rows, rep.two_sided, rep.one_way]))
        for name, recs in rep.splits.items():
            out.append("")
            out.append(f"### by {name}")
            out.append("")
            out.append(_md_records(recs))
        out.append("")
        out.append("### selection: blind vs edge threshold")
        out.append("")
        out.append(_md_records(rep.ladder))
        out.append("")
        out.append("### calibration")
        out.append("")
        mk = {b.label: b for b in rep.calibration_market}
        labels = sorted({b.label for b in rep.calibration_model} | set(mk))
        mm = {b.label: b for b in rep.calibration_model}
        out.append(
            _md_table(
                ("bin", "model n", "model pred", "model real", "mkt n", "mkt pred", "mkt real"),
                (
                    (
                        lab,
                        str(mm[lab].n) if lab in mm else "-",
                        f"{mm[lab].predicted:.3f}" if lab in mm else "-",
                        f"{mm[lab].realised:.3f}" if lab in mm else "-",
                        str(mk[lab].n) if lab in mk else "-",
                        f"{mk[lab].predicted:.3f}" if lab in mk else "-",
                        f"{mk[lab].realised:.3f}" if lab in mk else "-",
                    )
                    for lab in labels
                ),
            )
        )
    return "\n".join(out) + "\n"


def write_csv(reports: Sequence[MarketReport], path: Path) -> None:
    """Every record of every market, one row each, for a spreadsheet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "market",
        "section",
        "label",
        "n",
        "wins",
        "losses",
        "pushes",
        "hit_rate",
        "units",
        "roi",
        "scored",
        "brier_model",
        "brier_printed",
        "brier_market",
        "brier_gap",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for rep in reports:
            sections: list[tuple[str, Sequence[Record]]] = [
                ("total", [rep.all_rows, rep.two_sided, rep.one_way])
            ]
            sections.extend(rep.splits.items())
            sections.append(("selection", rep.ladder))
            for section, recs in sections:
                for r in recs:
                    w.writerow(
                        {
                            "market": rep.market,
                            "section": section,
                            "label": r.label,
                            "n": r.n,
                            "wins": r.wins,
                            "losses": r.losses,
                            "pushes": r.pushes,
                            "hit_rate": "" if r.hit_rate is None else round(r.hit_rate, 4),
                            "units": r.units,
                            "roi": "" if r.roi is None else round(r.roi, 4),
                            "scored": r.scored,
                            "brier_model": "" if r.brier_model is None else round(r.brier_model, 5),
                            "brier_printed": "" if r.brier_bet is None else round(r.brier_bet, 5),
                            "brier_market": (
                                "" if r.brier_market is None else round(r.brier_market, 5)
                            ),
                            "brier_gap": "" if r.brier_gap is None else round(r.brier_gap, 5),
                        }
                    )
