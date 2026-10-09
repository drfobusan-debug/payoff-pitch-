"""Ledger audit report: every graded night rolled up the way the ledger is read
by hand -- buys with ROI and a bootstrap CI, the CLV sign test, one side per
game against the market by market, the totals lean, goalie-status accuracy and
the probation table (master plan §7).

Nothing here is a model input. The report is rebuilt from ``graded_*.json``
on every ``nhl-engine audit`` run and attached to the card email, so the
verdict is always read off the whole ledger, never argued from one slate.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date as Date
from html import escape
from pathlib import Path

from nhl_engine.audit.ledger import LedgerRow
from nhl_engine.audit.scorecard import scorecard

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

    def add(self, r: LedgerRow) -> None:
        if r.outcome is None:
            return
        self.n += 1
        self.wins += r.outcome == "win"
        self.losses += r.outcome == "loss"
        self.pushes += r.outcome == "push"
        self.pnl += r.pnl or 0.0
        if r.clv is not None:
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
    sides: list[SideRead]
    goalies: list[GoalieRead]
    scorecard_text: str
    findings: list[str] = field(default_factory=list)

    @property
    def clv_p(self) -> float:
        return sign_test_p(self.buys.clv_neg, self.buys.clv_neg + self.buys.clv_pos)


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
    if b.n:
        out.append(
            f"Buys {b.record}, {b.pnl:+.2f}u, ROI {b.roi:+.1%} "
            f"(95% CI {a.roi_ci[0]:+.0%}..{a.roi_ci[1]:+.0%}, n={b.n}) -- "
            + ("not callable at this n." if b.n < 100 else "callable n.")
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


def build(rows: list[LedgerRow], starters: Starters | None = None) -> LedgerAudit:
    graded = [r for r in rows if r.outcome is not None]
    buys = [r for r in graded if r.is_buy]
    audit = LedgerAudit(
        nights=sorted({r.slate_date for r in rows}),
        rows=len(rows),
        buys=_tally(buys),
        buys_by_market=_group(buys, lambda r: r.market),
        buys_by_tier=_group(buys, lambda r: r.tier),
        buys_by_night=_group(buys, lambda r: r.slate_date),
        roi_ci=bootstrap_roi([r.pnl or 0.0 for r in buys]),
        sides=[s for s in (one_side(graded, k) for k in SIDE_MARKETS) if s is not None],
        goalies=goalie_reads(rows, starters or {}),
        scorecard_text=scorecard(rows).render(),
    )
    audit.findings = _findings(audit)
    return audit


# -- rendering ---------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}"


def _tally_rows(d: Mapping[str, Tally]) -> list[tuple[str, ...]]:
    return [(k, t.record, f"{t.roi:+.1%}", f"{t.pnl:+.2f}u", _pct(t.clv)) for k, t in d.items()]


def render_md(a: LedgerAudit, *, as_of: str) -> str:
    b = a.buys
    nights = ", ".join(a.nights) if a.nights else "none"
    lines = [
        f"# NHL ledger audit (as of {as_of})",
        "",
        f"{len(a.nights)} graded night{'s' if len(a.nights) != 1 else ''} ({nights}); {a.rows} rows. "
        "Graded on NHL API finals; close = last archived quote before puck drop.",
        "",
        "## Findings",
        "",
        *(f"- {f}" for f in a.findings or ["- nothing graded yet"]),
        "",
        "## Buys",
        "",
        f"**{b.record}, {b.pnl:+.2f}u, ROI {b.roi:+.1%}**, 95% CI {a.roi_ci[0]:+.0%}..{a.roi_ci[1]:+.0%} (n={b.n}). "
        f"CLV avg {_pct(b.clv)} pts; close {b.clv_moves} (sign test p={a.clv_p:.3g}).",
        "",
        "| | W-L-P | ROI | P&L | CLV pts |",
        "|---|---|---|---|---|",
    ]
    for section in (a.buys_by_market, a.buys_by_tier, a.buys_by_night):
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
    lines += ["", "## Goalie accuracy (card's named starter vs actual)", ""]
    if a.goalies:
        lines += ["| status | games | wrong |", "|---|---|---|"]
        lines += [f"| {g.status} | {g.n} | {g.wrong} |" for g in a.goalies]
    else:
        lines.append("no starter results available")
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
        "Graded on NHL API finals; close = last archived quote before puck drop.</p>",
        "<div class='find'><b>Findings</b><ul>"
        + "".join(f"<li>{escape(f)}</li>" for f in a.findings or ["nothing graded yet"])
        + "</ul></div>",
        "<h2>Buys</h2>",
        f"<p><b>{b.record}, {b.pnl:+.2f}u, ROI {b.roi:+.1%}</b>, 95% CI {a.roi_ci[0]:+.0%}..{a.roi_ci[1]:+.0%} "
        f"(n={b.n}). CLV avg {_pct(b.clv)} pts; close {b.clv_moves} (p={a.clv_p:.3g}).</p>",
    ]
    for title, d in (
        ("by market", a.buys_by_market),
        ("by tier", a.buys_by_tier),
        ("by night", a.buys_by_night),
    ):
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
    parts.append("<h2>Goalie accuracy (card's named starter vs actual)</h2>")
    parts.append(
        _table(
            ["status", "games", "wrong"], [(g.status, str(g.n), str(g.wrong)) for g in a.goalies]
        )
        if a.goalies
        else "<p>no starter results available</p>"
    )
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
    "SideRead",
    "Tally",
    "bootstrap_roi",
    "build",
    "goalie_reads",
    "latest",
    "one_side",
    "render_html",
    "render_md",
    "sign_test_p",
    "write",
]
