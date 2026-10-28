"""Ledger audit report: every graded night rolled up the way the ledger is read
by hand -- buys with ROI and a bootstrap CI, the CLV sign test, one side per
game against the market by market, the totals lean, goalie-status accuracy and
the probation table (master plan §7).

Rows priced at or after puck drop are not predictions and are left out of
every read (listed on their own). CLV counts only closes taken within
``STALE_CLOSE_MIN`` of puck drop; older or unverified closes are counted apart.

Nothing here is a model input. The report is rebuilt from ``graded_*.json``
on every ``nhl-engine audit`` run and attached to the card email, so the
verdict is always read off the whole ledger, never argued from one slate.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date as Date
from html import escape
from pathlib import Path

from nhl_engine.audit import bands as bnd
from nhl_engine.audit import screens as scr
from nhl_engine.audit.ledger import STALE_CLOSE_MIN, LedgerRow
from nhl_engine.audit.scorecard import scorecard
from nhl_engine.market.board import EXEC_BOOKS

BOOTSTRAP_N = 4000
SIGN_TEST_P = 0.01
SIGN_TEST_MIN = 20
CLV_FLAT = 1e-6  # consensus unchanged: neither side of the sign test
SIDE_MARKETS: dict[str, tuple[str, Callable[[LedgerRow], bool]]] = {
    "home ML": ("game_ml", lambda r: r.side == r.home),
    "Over": ("game_total", lambda r: r.side == "over"),
    "home PL": ("game_pl", lambda r: r.side == r.home),
}
Starters = Mapping[str, tuple[str, str]]  # "slate|matchup" -> (away starter, home starter)


@dataclass
class Tally:
    n: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    pnl: float = 0.0
    clv_sum: float = 0.0
    clv_n: int = 0
    clv_neg: int = 0
    clv_pos: int = 0
    clv_stale: int = 0  # a close older than STALE_CLOSE_MIN, or no puck drop to check it

    def add(self, r: LedgerRow) -> None:
        if r.outcome is None:
            return
        self.n += 1
        self.wins += r.outcome == "win"
        self.losses += r.outcome == "loss"
        self.pushes += r.outcome == "push"
        self.pnl += r.pnl or 0.0
        if r.clv is not None and not r.close_fresh:
            self.clv_stale += 1
        elif r.clv is not None:
            self.clv_sum += r.clv
            self.clv_n += 1
            self.clv_neg += r.clv < -CLV_FLAT
            self.clv_pos += r.clv > CLV_FLAT

    @property
    def clv_flat(self) -> int:
        return self.clv_n - self.clv_neg - self.clv_pos

    @property
    def clv_moves(self) -> str:
        return f"against {self.clv_neg}, toward {self.clv_pos}, flat {self.clv_flat}"

    @property
    def roi(self) -> float:
        return self.pnl / self.n if self.n else 0.0

    @property
    def clv(self) -> float:
        return self.clv_sum / self.clv_n if self.clv_n else 0.0

    @property
    def record(self) -> str:
        return f"{self.wins}-{self.losses}-{self.pushes}"


@dataclass
class SideRead:
    """One representative side per game for a market (the row nearest a coin flip)."""

    label: str
    n: int
    brier_model: float
    brier_market: float
    lean: float  # mean model - market
    below: int  # games where the model sat under the market
    won_above: tuple[int, int]  # (wins, n) when the model liked the side more than the market
    won_below: tuple[int, int]
    outcomes: Tally

    @property
    def lean_p(self) -> float:
        return sign_test_p(self.below, self.n)


@dataclass
class NightTiming:
    """Per night: how long before puck drop each game's close was taken."""

    night: str
    games: int
    lead_median: float | None  # minutes, over games with a known close and puck drop
    lead_max: float | None  # the stalest close
    stale: list[str]  # games whose close is older than STALE_CLOSE_MIN
    unknown: int  # games with no close or no puck drop
    late_rows: int  # rows priced at or after puck drop
    unpriced: list[str] = field(default_factory=list)  # scheduled games no card priced


@dataclass
class GoalieRead:
    status: str
    n: int = 0
    wrong: int = 0


@dataclass
class LedgerAudit:
    nights: list[str]
    rows: int
    buys: Tally
    buys_by_market: dict[str, Tally]
    buys_by_tier: dict[str, Tally]
    buys_by_night: dict[str, Tally]
    roi_ci: tuple[float, float]
    # What the unanchored sim alone would have bought (graded-only shadow).
    sim_buys: Tally
    sim_buys_by_market: dict[str, Tally]
    sim_buys_by_tier: dict[str, Tally]
    sim_roi_ci: tuple[float, float]
    sides: list[SideRead]
    goalies: list[GoalieRead]
    scorecard_text: str
    findings: list[str] = field(default_factory=list)
    screens: list[scr.ScreenRead] = field(default_factory=list)
    bands: list[bnd.BandRead] = field(default_factory=list)
    late: Tally = field(default_factory=Tally)  # every graded row priced after puck drop
    late_buys: Tally = field(default_factory=Tally)
    late_games: list[str] = field(default_factory=list)  # "night matchup"
    timing: list[NightTiming] = field(default_factory=list)
    exec_buys: Tally = field(default_factory=Tally)  # buys placed at an EXEC_BOOKS book

    @property
    def clv_p(self) -> float:
        return sign_test_p(self.buys.clv_neg, self.buys.clv_neg + self.buys.clv_pos)


SIM_NOTE = (
    "Rows the sim alone (no market anchor) would have bought under the same gates. "
    "Nothing here was staked; it exists so probation can grade the sim's own picks "
    "while the anchor holds. Call at 100+ per market."
)


def sign_test_p(k: int, n: int) -> float:
    """Two-sided exact binomial p-value for ``k`` of ``n`` under p = 0.5."""
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_roi(pnls: list[float], *, n: int = BOOTSTRAP_N, seed: int = 1) -> tuple[float, float]:
    if not pnls:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(pnls, k=len(pnls))) / len(pnls) for _ in range(n))
    return (means[int(0.025 * n)], means[min(n - 1, int(0.975 * n))])


def _tally(rows: list[LedgerRow]) -> Tally:
    t = Tally()
    for r in rows:
        t.add(r)
    return t


def _group(rows: list[LedgerRow], key: Callable[[LedgerRow], str]) -> dict[str, Tally]:
    out: dict[str, Tally] = defaultdict(Tally)
    for r in rows:
        out[key(r)].add(r)
    return dict(sorted(out.items()))


def one_side(rows: list[LedgerRow], label: str) -> SideRead | None:
    market, pick = SIDE_MARKETS[label]
    by_game: dict[tuple[str, str], list[LedgerRow]] = defaultdict(list)
    for r in rows:
        if r.market == market and pick(r) and r.outcome in ("win", "loss", "push"):
            by_game[(r.slate_date, r.matchup)].append(r)
    reps = [min(c, key=lambda r: abs(r.consensus - 0.5)) for c in by_game.values()]
    decided = [r for r in reps if r.outcome != "push"]
    if not decided:
        return None
    won = {id(r): float(r.outcome == "win") for r in decided}
    above = [r for r in decided if r.model_prob > r.consensus]
    below = [r for r in decided if r.model_prob <= r.consensus]
    return SideRead(
        label=label,
        n=len(decided),
        brier_model=sum((r.model_prob - won[id(r)]) ** 2 for r in decided) / len(decided),
        brier_market=sum((r.consensus - won[id(r)]) ** 2 for r in decided) / len(decided),
        lean=sum(r.model_prob - r.consensus for r in decided) / len(decided),
        below=sum(1 for r in decided if r.model_prob < r.consensus),
        won_above=(sum(r.outcome == "win" for r in above), len(above)),
        won_below=(sum(r.outcome == "win" for r in below), len(below)),
        outcomes=_tally(reps),
    )


def night_timing(
    rows: list[LedgerRow], scheduled: Mapping[str, Sequence[str]] | None = None
) -> list[NightTiming]:
    by_game: dict[tuple[str, str], list[LedgerRow]] = defaultdict(list)
    for r in rows:
        by_game[(r.slate_date, r.matchup)].append(r)
    nights: dict[str, list[tuple[str, float | None]]] = defaultdict(list)
    late: dict[str, int] = defaultdict(int)
    for (night, matchup), game in by_game.items():
        closes = [r for r in game if r.close_at and r.puck_drop]
        last = max(closes, key=lambda r: r.close_at) if closes else None
        nights[night].append((matchup, last.close_lead_min if last else None))
        late[night] += sum(r.priced_late for r in game)
    out: list[NightTiming] = []
    for night in sorted(nights):
        leads = [x for _, x in nights[night] if x is not None]
        out.append(
            NightTiming(
                night=night,
                games=len(nights[night]),
                lead_median=statistics.median(leads) if leads else None,
                lead_max=max(leads) if leads else None,
                stale=sorted(m for m, x in nights[night] if x is not None and x > STALE_CLOSE_MIN),
                unknown=len(nights[night]) - len(leads),
                late_rows=late[night],
                unpriced=sorted(
                    set((scheduled or {}).get(night, ())) - {m for m, _ in nights[night]}
                ),
            )
        )
    return out


def _surname(name: str) -> str:
    return name.replace(".", " ").split()[-1].lower() if name.strip() else ""


def goalie_reads(rows: list[LedgerRow], starters: Starters) -> list[GoalieRead]:
    seen: set[tuple[str, str]] = set()
    out: dict[str, GoalieRead] = {}
    for r in rows:
        k = (r.slate_date, r.matchup)
        actual = starters.get(f"{r.slate_date}|{r.matchup}")
        if k in seen or actual is None:
            continue
        seen.add(k)
        status = (r.goalie_status.split("/") + ["", ""])[:2]
        for named, real, st in zip((r.away_goalie, r.home_goalie), actual, status, strict=True):
            if not real or not st:
                continue
            g = out.setdefault(st, GoalieRead(st))
            g.n += 1
            g.wrong += _surname(real) not in named.lower()
    return [out[k] for k in sorted(out)]


def _findings(a: LedgerAudit) -> list[str]:
    out: list[str] = []
    b = a.buys
    if a.late.n:
        lb = a.late_buys
        out.append(
            f"{a.late.n} graded rows on {len(a.late_games)} game{'s' if len(a.late_games) != 1 else ''} were priced at or after puck drop "
            f"({', '.join(a.late_games)}) and are left out of every read"
            + (
                f", including {lb.n} buy{'s' if lb.n != 1 else ''} ({lb.record}, {lb.pnl:+.2f}u)."
                if lb.n
                else "."
            )
        )
    if b.n:
        out.append(
            f"Buys {b.record}, {b.pnl:+.2f}u, ROI {b.roi:+.1%} "
            f"(95% CI {a.roi_ci[0]:+.0%}..{a.roi_ci[1]:+.0%}, n={b.n}) -- "
            + ("not callable at this n." if b.n < 100 else "callable n.")
        )
    if b.n and a.exec_buys.n < b.n:
        e = a.exec_buys
        out.append(
            f"{e.n} of the {b.n} buys were at DraftKings/BetMGM/Novig ({e.record}, {e.pnl:+.2f}u); "
            "the rest took the best price at another book, which buys no longer do."
        )
    if b.clv_n:
        verdict = (
            "the market moves against our buys after we price."
            if a.clv_p < SIGN_TEST_P and b.clv_neg > b.clv_pos
            else "no CLV verdict yet."
            if a.clv_p >= SIGN_TEST_P
            else "the market moves our way after we price."
        )
        out.append(
            f"Close vs our pricing on {b.clv_n} buys: {b.clv_moves} (avg {b.clv * 100:+.1f} pts, "
            f"sign test on the moved p={a.clv_p:.3g}): {verdict}"
        )
    if b.clv_stale:
        out.append(
            f"CLV leaves out {b.clv_stale} buys whose close was more than {STALE_CLOSE_MIN:.0f} min "
            "before puck drop or could not be timed."
        )
    unpriced = [f"{t.night} ({', '.join(t.unpriced)})" for t in a.timing if t.unpriced]
    if unpriced:
        out.append(
            "Scheduled games that no card priced (early or 17:00 pass missed them): "
            + "; ".join(unpriced)
            + "."
        )
    stale = [f"{t.night} ({len(t.stale)})" for t in a.timing if t.stale]
    if stale:
        out.append(
            f"No archived quote in the last {STALE_CLOSE_MIN:.0f} min before puck drop for some games: "
            + ", ".join(stale)
            + " -- the board capture did not run near puck drop."
        )
    for s in a.sides:
        if s.n >= SIGN_TEST_MIN and s.lean_p < SIGN_TEST_P:
            direction = "under" if s.below > s.n / 2 else "over"
            out.append(
                f"{s.label}: model {direction} the market in {s.below if direction == 'under' else s.n - s.below}/{s.n} games "
                f"(avg {s.lean * 100:+.1f} pts, p={s.lean_p:.3g}); Brier model {s.brier_model:.3f} vs market "
                f"{s.brier_market:.3f} -- a bias, not variance."
            )
    for g in a.goalies:
        if g.status != "confirmed" and g.n >= 5 and g.wrong / g.n >= 0.25:
            out.append(
                f"Goalie status '{g.status}' was wrong {g.wrong}/{g.n} times; 'confirmed' "
                + next((f"{c.wrong}/{c.n}" for c in a.goalies if c.status == "confirmed"), "n/a")
                + "."
            )
    return out


def build(
    rows: list[LedgerRow],
    starters: Starters | None = None,
    context: scr.Context | None = None,
    scheduled: Mapping[str, Sequence[str]] | None = None,
) -> LedgerAudit:
    total = len(rows)
    late = [r for r in rows if r.priced_late and r.outcome is not None]
    timing = night_timing(rows, scheduled)
    rows = [r for r in rows if not r.priced_late]
    graded = [r for r in rows if r.outcome is not None]
    buys = [r for r in graded if r.is_buy]
    sim_buys = [r for r in graded if r.sim_buy]
    audit = LedgerAudit(
        nights=sorted({r.slate_date for r in rows}),
        rows=total,
        buys=_tally(buys),
        buys_by_market=_group(buys, lambda r: r.market),
        buys_by_tier=_group(buys, lambda r: r.tier),
        buys_by_night=_group(buys, lambda r: r.slate_date),
        roi_ci=bootstrap_roi([r.pnl or 0.0 for r in buys]),
        sim_buys=_tally(sim_buys),
        sim_buys_by_market=_group(sim_buys, lambda r: r.market),
        sim_buys_by_tier=_group(sim_buys, lambda r: r.sim_tier),
        sim_roi_ci=bootstrap_roi([r.pnl or 0.0 for r in sim_buys]),
        sides=[s for s in (one_side(graded, k) for k in SIDE_MARKETS) if s is not None],
        goalies=goalie_reads(rows, starters or {}),
        scorecard_text=scorecard(rows).render(),
        screens=scr.read_screens(rows, context or scr.Context()),
        bands=bnd.read_bands(rows, context or scr.Context()),
        late=_tally(late),
        late_buys=_tally([r for r in late if r.is_buy]),
        late_games=sorted({f"{r.slate_date} {r.matchup}" for r in late}),
        timing=timing,
        exec_buys=_tally([r for r in buys if r.book in EXEC_BOOKS]),
    )
    audit.findings = _findings(audit)
    return audit


# -- rendering ---------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}"


def _tally_rows(d: Mapping[str, Tally]) -> list[tuple[str, ...]]:
    return [
        (k, t.record, f"{t.roi:+.1%}", f"{t.pnl:+.2f}u", _pct(t.clv) if t.clv_n else "")
        for k, t in d.items()
    ]


def _screen_rows(reads: list[scr.ScreenRead]) -> list[tuple[str, ...]]:
    return [
        (
            s.name,
            s.rule,
            str(s.n),
            s.record,
            f"{s.saved:+.2f}",
            f"{s.se:.2f}" if len(s.games) >= 2 else "",
            f"{s.halves[0]:+.2f} / {s.halves[1]:+.2f}",
            s.verdict,
        )
        for s in reads
    ]


TIMING_HEAD = [
    "night",
    "games",
    "close lead median (min)",
    "stalest close (min)",
    f"closes > {STALE_CLOSE_MIN:.0f} min",
    "no close/drop",
    "rows after drop",
    "never priced",
]


def _min(x: float | None) -> str:
    return f"{x:.0f}" if x is not None else "n/a"


def _timing_rows(ts: list[NightTiming]) -> list[tuple[str, ...]]:
    return [
        (
            t.night,
            str(t.games),
            _min(t.lead_median),
            _min(t.lead_max),
            ", ".join(t.stale) or "0",
            str(t.unknown),
            str(t.late_rows),
            ", ".join(t.unpriced) or "0",
        )
        for t in ts
    ]


def _clv_line(a: LedgerAudit) -> str:
    b = a.buys
    left = f"; {b.clv_stale} stale or untimed closes left out" if b.clv_stale else ""
    return (
        f"CLV avg {_pct(b.clv)} pts on {b.clv_n} buys closed within {STALE_CLOSE_MIN:.0f} min of "
        f"puck drop{left}; close {b.clv_moves} (sign test p={a.clv_p:.3g})."
    )


HEADER_NOTE = (
    "Graded on NHL API finals; close = last archived quote before puck drop. "
    "Rows priced at or after puck drop are left out."
)


BAND_HEAD = ["band", "bets", "W-L", "units", "vs market", "ROI", "SE (u, by game)", "halves", "verdict"]


def _band_title(a: LedgerAudit) -> str:
    return (
        f"Graded-only bands (flat 1u on every bet at the card's price; call at {bnd.MIN_N}+ games)"
    )


def _band_note(a: LedgerAudit) -> str:
    k = len(a.bands)
    return (
        f"{k} bands are read at once, so about {k / 20:.0f} will sit past 2 SE by chance. "
        "Mirror bands (home favourite / away underdog) are the same games."
    )


def _band_rows(reads: list[bnd.BandRead]) -> list[tuple[str, ...]]:
    return [
        (
            b.name,
            str(b.n),
            b.record,
            f"{b.units:+.2f}",
            f"{b.excess:+.2f}" if b.expected else "",
            f"{b.roi:+.1%}" if b.n else "",
            f"{b.se:.2f}" if b.n >= 2 else "",
            f"{b.halves[0]:+.2f} / {b.halves[1]:+.2f}",
            b.verdict,
        )
        for b in reads
    ]


def render_md(a: LedgerAudit, *, as_of: str) -> str:
    b = a.buys
    nights = ", ".join(a.nights) if a.nights else "none"
    lines = [
        f"# NHL ledger audit (as of {as_of})",
        "",
        f"{len(a.nights)} graded night{'s' if len(a.nights) != 1 else ''} ({nights}); {a.rows} rows. "
        + HEADER_NOTE,
        "",
        "## Findings",
        "",
        *(f"- {f}" for f in a.findings or ["nothing graded yet"]),
        "",
        "## Buys",
        "",
        f"**{b.record}, {b.pnl:+.2f}u, ROI {b.roi:+.1%}**, 95% CI {a.roi_ci[0]:+.0%}..{a.roi_ci[1]:+.0%} (n={b.n}). "
        + _clv_line(a),
        "",
        "| | W-L-P | ROI | P&L | CLV pts |",
        "|---|---|---|---|---|",
    ]
    for section in (a.buys_by_market, a.buys_by_tier, a.buys_by_night):
        lines += [f"| {' | '.join(r)} |" for r in _tally_rows(section)]
    sb = a.sim_buys
    lines += [
        "",
        "## Sim would-be buys (unanchored sim at the card's price; graded-only)",
        "",
        SIM_NOTE,
        "",
        f"**{sb.record}, {sb.pnl:+.2f}u, ROI {sb.roi:+.1%}**, 95% CI {a.sim_roi_ci[0]:+.0%}..{a.sim_roi_ci[1]:+.0%} (n={sb.n}).",
        "",
        "| | W-L-P | ROI | P&L | CLV pts |",
        "|---|---|---|---|---|",
    ]
    for section in (a.sim_buys_by_market, a.sim_buys_by_tier):
        lines += [f"| {' | '.join(r)} |" for r in _tally_rows(section)]
    lines += [
        "",
        "## Model vs market (one side per game; Brier, lower is better)",
        "",
        "| side | n | model | market | lean | model<mkt | won when model>mkt | won when model<=mkt |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in a.sides:
        lines.append(
            f"| {s.label} | {s.n} | {s.brier_model:.3f} | {s.brier_market:.3f} | {_pct(s.lean)} pts | "
            f"{s.below}/{s.n} (p={s.lean_p:.2g}) | {s.won_above[0]}/{s.won_above[1]} | {s.won_below[0]}/{s.won_below[1]} |"
        )
    over = next((s for s in a.sides if s.label == "Over"), None)
    if over:
        lines.append(f"\nFull-game totals went Over {over.outcomes.record} (one line per game).")
    lines += ["", "## Pricing time vs puck drop", ""]
    lines += [f"| {' | '.join(TIMING_HEAD)} |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {' | '.join(r)} |" for r in _timing_rows(a.timing)]
    lines += ["", "## Goalie accuracy (card's named starter vs actual)", ""]
    if a.goalies:
        lines += ["| status | games | wrong |", "|---|---|---|"]
        lines += [f"| {g.status} | {g.n} | {g.wrong} |" for g in a.goalies]
    else:
        lines.append("no starter results available")
    lines += ["", f"## Graded-only screens (refuse nothing; promote at {scr.MIN_N}+ buys)", ""]
    lines += ["| screen | rule | buys hit | W-L | saved u | SE (by game) | halves | verdict |"]
    lines += ["|---|---|---|---|---|---|---|---|"]
    lines += [f"| {' | '.join(r)} |" for r in _screen_rows(a.screens)]
    lines += ["", f"## {_band_title(a)}", "", _band_note(a), ""]
    lines += [f"| {' | '.join(BAND_HEAD)} |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {' | '.join(r)} |" for r in _band_rows(a.bands)]
    lines += ["", "## Scorecard (all rows)", "", "```", a.scorecard_text.rstrip(), "```", ""]
    return "\n".join(lines)


CSS = """
body{font-family:Helvetica,Arial,sans-serif;font-size:10.5px;color:#111;margin:18px}
h1{font-size:18px;margin:0 0 4px} h2{font-size:12.5px;margin:14px 0 4px;border-bottom:1px solid #999}
table{border-collapse:collapse;margin:4px 0} td,th{border:1px solid #ccc;padding:2px 6px;text-align:right}
th{background:#eee} td:first-child,th:first-child{text-align:left}
pre{font-size:8.5px;line-height:1.2} .find{background:#fff8e1;padding:6px 10px;border-left:3px solid #e0a800}
.sub{color:#555}
"""


def _table(head: list[str], body: list[tuple[str, ...]]) -> str:
    h = "".join(f"<th>{escape(x)}</th>" for x in head)
    rows = "".join("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in r) + "</tr>" for r in body)
    return f"<table><tr>{h}</tr>{rows}</table>"


def render_html(a: LedgerAudit, *, as_of: str) -> str:
    b = a.buys
    nights = ", ".join(a.nights) if a.nights else "none"
    parts = [
        f"<h1>NHL ledger audit <span class='sub'>as of {escape(as_of)}</span></h1>",
        f"<p class='sub'>{len(a.nights)} graded nights ({escape(nights)}); {a.rows} rows. "
        f"{escape(HEADER_NOTE)}</p>",
        "<div class='find'><b>Findings</b><ul>"
        + "".join(f"<li>{escape(f)}</li>" for f in a.findings or ["nothing graded yet"])
        + "</ul></div>",
        "<h2>Buys</h2>",
        f"<p><b>{b.record}, {b.pnl:+.2f}u, ROI {b.roi:+.1%}</b>, 95% CI {a.roi_ci[0]:+.0%}..{a.roi_ci[1]:+.0%} "
        f"(n={b.n}). {escape(_clv_line(a))}</p>",
    ]
    for title, d in (
        ("by market", a.buys_by_market),
        ("by tier", a.buys_by_tier),
        ("by night", a.buys_by_night),
    ):
        parts.append(_table([title, "W-L-P", "ROI", "P&L", "CLV pts"], _tally_rows(d)))
    sb = a.sim_buys
    parts.append("<h2>Sim would-be buys (unanchored sim at the card's price; graded-only)</h2>")
    parts.append(f"<p class='sub'>{escape(SIM_NOTE)}</p>")
    parts.append(
        f"<p><b>{sb.record}, {sb.pnl:+.2f}u, ROI {sb.roi:+.1%}</b>, 95% CI {a.sim_roi_ci[0]:+.0%}..{a.sim_roi_ci[1]:+.0%} "
        f"(n={sb.n}).</p>"
    )
    for title, d in (("by market", a.sim_buys_by_market), ("by tier", a.sim_buys_by_tier)):
        parts.append(_table([title, "W-L-P", "ROI", "P&L", "CLV pts"], _tally_rows(d)))
    parts.append("<h2>Model vs market (one side per game; Brier, lower is better)</h2>")
    parts.append(
        _table(
            [
                "side",
                "n",
                "model",
                "market",
                "lean pts",
                "model<mkt",
                "won model>mkt",
                "won model<=mkt",
            ],
            [
                (
                    s.label,
                    str(s.n),
                    f"{s.brier_model:.3f}",
                    f"{s.brier_market:.3f}",
                    _pct(s.lean),
                    f"{s.below}/{s.n} (p={s.lean_p:.2g})",
                    f"{s.won_above[0]}/{s.won_above[1]}",
                    f"{s.won_below[0]}/{s.won_below[1]}",
                )
                for s in a.sides
            ],
        )
    )
    parts.append("<h2>Pricing time vs puck drop</h2>")
    parts.append(_table(TIMING_HEAD, _timing_rows(a.timing)))
    parts.append("<h2>Goalie accuracy (card's named starter vs actual)</h2>")
    parts.append(
        _table(
            ["status", "games", "wrong"], [(g.status, str(g.n), str(g.wrong)) for g in a.goalies]
        )
        if a.goalies
        else "<p>no starter results available</p>"
    )
    parts.append(f"<h2>Graded-only screens (refuse nothing; promote at {scr.MIN_N}+ buys)</h2>")
    parts.append(
        _table(
            ["screen", "rule", "buys hit", "W-L", "saved u", "SE (by game)", "halves", "verdict"],
            _screen_rows(a.screens),
        )
    )
    parts.append(f"<h2>{escape(_band_title(a))}</h2><p class='sub'>{escape(_band_note(a))}</p>")
    parts.append(_table(BAND_HEAD, _band_rows(a.bands)))
    parts.append(f"<h2>Scorecard (all rows)</h2><pre>{escape(a.scorecard_text)}</pre>")
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{''.join(parts)}</body></html>"


def audit_dir(out_dir: Path) -> Path:
    return out_dir / "audit"


def write(
    a: LedgerAudit, out_dir: Path, day: Date, *, as_of: str, pdf: bool = True
) -> dict[str, Path]:
    """Write ``ledger_audit_<day>.md`` (+ ``.pdf``) and return the paths by kind."""
    d = audit_dir(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    stem = d / f"ledger_audit_{day.isoformat()}"
    paths = {"audit_md": stem.with_suffix(".md")}
    paths["audit_md"].write_text(render_md(a, as_of=as_of), encoding="utf-8")
    if pdf:
        from weasyprint import HTML

        paths["audit_pdf"] = stem.with_suffix(".pdf")
        paths["audit_pdf"].write_bytes(bytes(HTML(string=render_html(a, as_of=as_of)).write_pdf()))
    return paths


def latest(out_dir: Path) -> dict[str, Path]:
    """Newest audit report on disk (md + pdf), for attaching to the card email."""
    d = audit_dir(out_dir)
    out: dict[str, Path] = {}
    for kind, suffix in (("audit_md", ".md"), ("audit_pdf", ".pdf")):
        files = sorted(d.glob(f"ledger_audit_*{suffix}")) if d.exists() else []
        if files:
            out[kind] = files[-1]
    return out


__all__ = [
    "GoalieRead",
    "LedgerAudit",
    "NightTiming",
    "SideRead",
    "Tally",
    "bootstrap_roi",
    "build",
    "goalie_reads",
    "latest",
    "night_timing",
    "one_side",
    "render_html",
    "render_md",
    "sign_test_p",
    "write",
]
