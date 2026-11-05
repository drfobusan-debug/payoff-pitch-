"""The CBB ledger audit as a report (text body + HTML/PDF): last 7 slates, then the season.

Graded only: it reads bet-of-record rows and changes no price, gate or buy. Each
scope opens on the close (CLV) and model-vs-market Brier, with the record beneath;
the season adds the promotion bar, the bands, the gate counterfactual and the
push shape. One row per position per slate (``ledger.record``), so a selection on
two card passes counts once.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta
from html import escape

from cbb_engine import teamgroups
from cbb_engine.bands import main_rows, one_slate_per_game
from engine_common.bands import MIN_N as BAND_MIN_N
from engine_common.bands import BandRead
from engine_common.promotion import MIN_BUYS
from engine_runtime import altlines as alt_rt
from engine_runtime import faststart, ledger, views

Z = 1.96
EDGE_BANDS = (("edge 2-3%", 0.02, 0.03), ("edge 3-5%", 0.03, 0.05), ("edge 5%+", 0.05, 1.0))
NOTE = "Graded only, paper mode: this report changes no price, gate or buy."


def _decimal(american: float) -> float:
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / -american)


@dataclass(frozen=True)
class Seg:
    label: str
    n: int  # decided (pushes excluded)
    wins: int
    losses: int
    pushes: int
    units: float
    roi: float
    lo: float
    hi: float
    breakeven: float
    clv_n: int
    clv: float  # mean consensus CLV, probability points
    beat: float  # share of rows with a close that beat it

    @property
    def record(self) -> str:
        return f"{self.wins}-{self.losses}" + (f"-{self.pushes}" if self.pushes else "")

    @property
    def verdict(self) -> str:
        if self.n < 2:
            return "no sample"
        if self.n >= MIN_BUYS and self.lo > 0:
            return "winning beyond noise"
        if self.n >= MIN_BUYS and self.hi < 0:
            return "losing beyond noise"
        return f"underpowered ({self.n}/{MIN_BUYS})" if self.n < MIN_BUYS else "within noise"


def segment(label: str, rows: Iterable[ledger.Row]) -> Seg:
    rs = [r for r in rows if r.outcome in ("win", "loss", "push") and r.pnl is not None]
    decided = [r for r in rs if r.outcome != "push" and r.american is not None]
    n = len(decided)
    units = sum(r.pnl or 0.0 for r in decided)
    roi = units / n if n else 0.0
    be = sum(1.0 / _decimal(float(r.american or 0.0)) for r in decided) / n if n else 0.0
    half = 0.0
    if n >= 2:
        # Clustered by game, and never narrower than a coin at the prices' break-even.
        games: dict[tuple[str, str], float] = defaultdict(float)
        for r in decided:
            games[(r.slate_date, r.event_id)] += (r.pnl or 0.0) - roi
        se = math.sqrt(sum(v * v for v in games.values()) * n / (n - 1)) / n
        floor = math.sqrt((1.0 - be) / be / n) if 0 < be < 1 else 0.0
        half = Z * max(se, floor)
    clv = [r.clv_consensus for r in rs if r.clv_consensus is not None]
    return Seg(
        label=label,
        n=n,
        wins=sum(r.outcome == "win" for r in decided),
        losses=sum(r.outcome == "loss" for r in decided),
        pushes=sum(r.outcome == "push" for r in rs),
        units=round(units, 3),
        roi=roi,
        lo=roi - half,
        hi=roi + half,
        breakeven=be,
        clv_n=len(clv),
        clv=sum(clv) / len(clv) if clv else 0.0,
        beat=sum(c > 0 for c in clv) / len(clv) if clv else 0.0,
    )


def _where(label: str, rows: Sequence[ledger.Row], keep: Callable[[ledger.Row], bool]) -> Seg:
    return segment(label, [r for r in rows if keep(r)])


def _is(attr: str, value: str) -> Callable[[ledger.Row], bool]:
    return lambda r: getattr(r, attr) == value


def _edge(lo: float, hi: float) -> Callable[[ledger.Row], bool]:
    return lambda r: lo <= (r.edge or 0.0) < hi


@dataclass(frozen=True)
class Table:
    title: str
    head: tuple[str, ...]
    rows: list[tuple[str, ...]]
    note: str = ""


@dataclass
class Section:
    title: str
    lines: list[str] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)


@dataclass
class Report:
    day: Date
    findings: list[str]
    sections: list[Section]
    subject: str


SEG_HEAD = (
    "segment",
    "bets",
    "W-L-P",
    "break-even",
    "units",
    "ROI",
    "95% range",
    "CLV pts",
    "beat close",
    "verdict",
)


def _seg_rows(segs: Iterable[Seg]) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for s in segs:
        if not (s.n or s.pushes):
            continue
        out.append(
            (
                s.label,
                str(s.n + s.pushes),
                s.record,
                f"{s.breakeven:.1%}",
                f"{s.units:+.2f}",
                f"{s.roi:+.1%}",
                f"{s.lo:+.0%} to {s.hi:+.0%}" if s.n >= 2 else "-",
                f"{s.clv:+.2f} (n={s.clv_n})" if s.clv_n else "-",
                f"{s.beat:.0%}" if s.clv_n else "-",
                s.verdict,
            )
        )
    return out


def _seg_table(title: str, segs: Iterable[Seg], note: str = "") -> Table:
    return Table(title, SEG_HEAD, _seg_rows(segs), note)


def _graded(rows: Iterable[ledger.Row]) -> list[ledger.Row]:
    """Graded rows of record, each game from one slate (a game on several daily slates counts once)."""
    return one_slate_per_game(r for r in rows if r.bet_of_record and r.outcome in ("win", "loss", "push"))


def live_buys(rows: Iterable[ledger.Row]) -> list[ledger.Row]:
    return one_slate_per_game(
        r for r in rows if r.bet_of_record and r.outcome in ("win", "loss", "push") and r.status == "live" and r.is_buy
    )


def paper_buys(rows: Iterable[ledger.Row]) -> list[ledger.Row]:
    """Research/shadow rows that cleared every gate: what the engine would have bought."""
    return one_slate_per_game(
        r for r in rows if r.bet_of_record and r.outcome in ("win", "loss", "push") and r.status != "live" and r.is_buy
    )


def briers(rows: Iterable[ledger.Row]) -> list[tuple[str, ...]]:
    """One side per game (home / over), main line: model and bet-time no-vig, plus the close."""
    acc: dict[str, list[tuple[float, float, float, float | None]]] = defaultdict(list)
    for r in main_rows(rows):
        if (
            r.side not in ("home", "over")
            or r.outcome == "push"
            or r.p_model is None
            or r.p_fair is None
        ):
            continue
        won = 1.0 if r.outcome == "win" else 0.0
        acc[r.market].append((won, r.p_model, r.p_fair, r.close_consensus))
    out: list[tuple[str, ...]] = []
    for market, xs in sorted(acc.items()):
        closed = [x for x in xs if x[3] is not None]
        out.append(
            (
                market,
                str(len(xs)),
                f"{sum((m - w) ** 2 for w, m, _, _ in xs) / len(xs):.4f}",
                f"{sum((f - w) ** 2 for w, _, f, _ in xs) / len(xs):.4f}",
                f"{sum((float(c or 0) - w) ** 2 for w, _, _, c in closed) / len(closed):.4f} (n={len(closed)})"
                if closed
                else "-",
            )
        )
    return out


def coverage(rows: Iterable[ledger.Row]) -> list[tuple[str, ...]]:
    """Per slate: rows of record, graded, with a pre-tip close, and with no close (pulled)."""
    by_day: dict[str, list[ledger.Row]] = defaultdict(list)
    for r in rows:
        if r.bet_of_record:
            by_day[r.slate_date].append(r)
    out: list[tuple[str, ...]] = []
    for d, rs in sorted(by_day.items()):
        buys = [r for r in rs if r.is_buy]
        out.append(
            (
                d,
                str(len(rs)),
                str(sum(bool(r.outcome) for r in rs)),
                str(sum(r.close_status == "pre_tip" for r in rs)),
                str(sum(r.close_status == "pulled" for r in rs)),
                f"{sum(bool(r.outcome) for r in buys)}/{len(buys)}",
            )
        )
    return out


def _by_week(rows: Sequence[ledger.Row], end: Date) -> list[Seg]:
    if not rows:
        return []
    first = min(Date.fromisoformat(r.slate_date) for r in rows)
    out: list[Seg] = []
    hi = end
    while hi >= first:
        lo = hi - timedelta(days=6)
        span = [r for r in rows if lo <= Date.fromisoformat(r.slate_date) <= hi]
        if span:
            out.append(segment(f"{lo:%b} {lo.day}-{hi:%b} {hi.day}", span))
        hi = lo - timedelta(days=1)
    return out[::-1]


def _scope(title: str, rows: list[ledger.Row], end: Date, *, weekly: bool) -> tuple[Section, Seg]:
    buys = live_buys(rows)
    paper = paper_buys(rows)
    slates = sorted({r.slate_date for r in rows})
    all_buys = segment("all live buys", buys)
    sec = Section(
        title, [f"{len(slates)} slate(s) with rows of record; {len(_graded(rows))} graded row(s)."]
    )
    sec.tables += [
        _seg_table(
            "Live buys (full-game totals at DraftKings / BetMGM)",
            [all_buys]
            + [_where(t, buys, _is("tier", t)) for t in ("Strong buy", "Moderate buy")]
            + [_where(b, buys, _is("book", b)) for b in sorted({r.book for r in buys})]
            + [_where(s, buys, _is("side", s)) for s in ("over", "under")]
            + [_where(lab, buys, _edge(lo, hi)) for lab, lo, hi in EDGE_BANDS],
        ),
        _seg_table(
            "Would-be buys in research markets (cleared every gate; not bet)",
            [_where(m, paper, _is("market", m)) for m in sorted({r.market for r in paper})],
        ),
        Table(
            "Model vs market (Brier, one side per game, lower is better)",
            ("market", "games", "model", "market at card", "close"),
            briers(rows),
            "The model is anchored to the devigged consensus, so model and market should be close; "
            "the close column is the bar both are judged against.",
        ),
        _seg_table(
            "By slate" if weekly else "By week",
            [segment(d, [r for r in buys if r.slate_date == d]) for d in slates]
            if weekly
            else _by_week(buys, end),
        ),
        Table(
            "Grading coverage",
            ("slate", "rows of record", "graded", "close pre-tip", "no close", "buys graded"),
            coverage(rows),
        ),
    ]
    return sec, all_buys


def _findings(
    week: Seg,
    season: Seg,
    rows: list[ledger.Row],
    promo: list[dict[str, object]],
    reads: Sequence[BandRead],
) -> list[str]:
    out: list[str] = []
    if season.n + season.pushes == 0:
        out.append("No graded live buys yet this season.")
    else:
        out.append(
            f"Season live buys {season.record}, {season.units:+.2f}u, ROI {season.roi:+.1%} "
            f"(95% range {season.lo:+.0%} to {season.hi:+.0%}, n={season.n}): {season.verdict}."
        )
        if week.n + week.pushes:
            out.append(f"Last 7 slates {week.record}, {week.units:+.2f}u, ROI {week.roi:+.1%}.")
    if season.clv_n:
        side = "toward" if season.clv > 0 else "against"
        out.append(
            f"The close moved {side} the buys: {season.clv:+.2f} probability points on average, "
            f"{season.beat:.0%} beat it (n={season.clv_n})."
        )
    ungraded = [r for r in coverage(rows) if r[2] == "0" and r[1] != "0"]
    if ungraded:
        out.append(
            f"{len(ungraded)} slate(s) have rows of record but nothing graded ({', '.join(r[0] for r in ungraded)}): "
            "no ESPN final was matched; check the 03:00 audit log."
        )
    for p in promo:
        out.append(
            f"Promotion bar, {p['market']}: {p['status']} at n={p['n']} ({p['why_not'] or 'cleared'})."
        )
    called = [b for b in reads if b.n >= BAND_MIN_N and b.verdict != "no edge"]
    out += [
        f"Band {b.name}: {b.verdict} ({b.record}, excess {b.excess:+.1f}u ± {b.se:.1f})."
        for b in called
    ]
    return out


def build(
    rows_by_day: Mapping[str, Iterable[ledger.Row]],
    day: Date,
    *,
    bands: Sequence[BandRead] | None = None,
    groups: Sequence[teamgroups.GroupRead] | None = None,
    fast: Sequence[faststart.Read] | None = None,
    checklist_lines: Sequence[str] = (),
    recent: int = 7,
    alts: Sequence[alt_rt.Alt] = (),
) -> Report:
    season = [r for d, rs in sorted(rows_by_day.items()) if d <= day.isoformat() for r in rs]
    start = (day - timedelta(days=recent - 1)).isoformat()
    week = [r for r in season if r.slate_date >= start]
    w_sec, w_buys = _scope(f"Last {recent} days ({start} .. {day})", week, day, weekly=True)
    s_sec, s_buys = _scope("Season to date", season, day, weekly=False)
    promo = views.v_promotion(season)
    s_sec.tables.append(
        Table(
            f"Promotion bar (paper -> Lean at {MIN_BUYS}+ graded buys, CLV and Brier)",
            ("market", "buys", "ROI", "CLV low", "status", "why not"),
            [
                (
                    str(p["market"]),
                    str(p["n"]),
                    f"{float(str(p['roi'] or 0)):+.1%}",
                    str(p["clv_lo"]),
                    str(p["status"]),
                    str(p["why_not"]),
                )
                for p in promo
            ],
        )
    )
    reads = list(bands or [])
    s_sec.tables.append(
        Table(
            f"Bands (graded only, flat 1u per main-line row; called at {BAND_MIN_N} games and 2 SE)",
            ("band", "n", "W-L", "units", "ROI", "excess vs no-vig", "halves", "verdict"),
            [
                (
                    b.name,
                    str(b.n),
                    b.record,
                    f"{b.units:+.1f}",
                    f"{b.roi:+.1%}",
                    f"{b.excess:+.1f} ± {b.se:.1f}" if b.n > 1 else "-",
                    f"{b.halves[0]:+.1f} / {b.halves[1]:+.1f}",
                    b.verdict,
                )
                for b in reads
                if b.n
            ],
            "" if reads else "bands unavailable (see log)",
        )
    )
    s_sec.tables.append(
        Table(
            f"Team groups: how well the close called each kind of game (one row per game; called at {BAND_MIN_N} games and 2 SE)",
            teamgroups.HEAD,
            teamgroups.table(groups or []),
            ("" if groups is not None else "team groups unavailable (see log). ")
            + "Close miss = mean |result - closing line|; vs rest = this group's miss less every other game's; "
            "bias > 0 = favourites / Overs beat the close. " + teamgroups.CONTINUITY_NOTE,
        )
    )
    s_sec.tables.append(
        Table(
            "Fast starts: does the first half stay in the final? (one row per game with a timely close)",
            faststart.HEAD,
            faststart.table(fast or []),
            ("" if fast is not None else "fast-start study unavailable (see log). ")
            + ("" if fast else f"needs {faststart.MIN_PRIOR}+ earlier games first. ")
            + faststart.NOTE,
        )
    )
    gates = [g for g in views.v_gate(season) if int(g["n"]) > 0]  # type: ignore[call-overload]
    s_sec.tables.append(
        Table(
            "Refused rows by gate (CLV the gate gave up; positive = the gate refused a winner on price)",
            ("gate", "market", "status", "rows", "CLV pts"),
            [
                (
                    str(g["gate"]),
                    str(g["market"]),
                    str(g["status"]),
                    str(g["n"]),
                    "-" if g["clv_consensus"] is None else f"{g['clv_consensus']:+.2f}",
                )
                for g in gates
            ],
        )
    )
    s_sec.tables.append(
        Table(
            "Push mass on live totals (predicted vs realised, by number)",
            ("line", "rows", "predicted push", "realised push"),
            [
                (
                    str(v["line"]),
                    str(v["n"]),
                    "-" if v["pred_push"] is None else f"{v['pred_push']:.1%}",
                    f"{v['real_push']:.1%}",
                )
                for v in views.v_shape(season)
            ],
        )
    )
    s_sec.tables.append(
        Table(
            "Alternate lines beside bought games: points bought vs sold ('beside buys' = live buys)",
            tuple(alt_rt.HEAD),
            alt_rt.table(alts),
            alt_rt.NOTE if alts else "no graded alternate lines yet",
        )
    )
    findings = _findings(w_buys, s_buys, season, promo, reads)
    findings += [
        f"Team group {g.group} ({g.market}): {g.verdict} (n={g.games} games)."
        for g in groups or []
        if g.dimension != "all" and g.games >= BAND_MIN_N and g.verdict != "no difference"
    ]
    sections = [w_sec, s_sec]
    if checklist_lines:
        sections.append(Section("Go/no-go checklist", list(checklist_lines)))
    subject = (
        f"CBB audit {day} -- season buys {s_buys.record or '0-0'}, {s_buys.units:+.2f}u [paper]"
    )
    return Report(day, findings, sections, subject)


def _text_table(t: Table) -> list[str]:
    lines = [f"-- {t.title} --"]
    if t.note:
        lines.append(t.note)
    if not t.rows:
        return [*lines, "  (none)"]
    widths = [max(len(c) for c in col) for col in zip(t.head, *t.rows, strict=False)]
    for row in (t.head, *t.rows):
        lines.append(
            "  " + "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=False)).rstrip()
        )
    return lines


def render_text(rep: Report) -> str:
    out = [NOTE, "", "Findings:"] + [f"  - {f}" for f in rep.findings]
    for s in rep.sections:
        out += ["", f"== {s.title} ==", *s.lines]
        for t in s.tables:
            out += ["", *_text_table(t)]
    return "\n".join(out) + "\n"


CSS = """
@page { size: letter landscape; margin: 0.4in; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 8.5pt; color: #1a1a1a; }
h1 { font-size: 15pt; color: #16324f; margin: 0 0 4px; }
h2 { font-size: 12pt; color: #16324f; border-bottom: 1px solid #16324f; margin: 14px 0 4px; }
h3 { font-size: 10pt; color: #16324f; margin: 10px 0 3px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 4px; }
th, td { border: 1px solid #ccd; padding: 2px 4px; text-align: left; }
th { background: #e8edf3; }
.sub { color: #555; }
.find { background: #f4f7fa; border: 1px solid #ccd; padding: 4px 10px; }
pre { font-size: 8pt; }
"""


def _html_table(t: Table) -> str:
    note = f"<p class='sub'>{escape(t.note)}</p>" if t.note else ""
    if not t.rows:
        return f"<h3>{escape(t.title)}</h3>{note}<p class='sub'>none</p>"
    head = "".join(f"<th>{escape(h)}</th>" for h in t.head)
    body = "".join("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in r) + "</tr>" for r in t.rows)
    return f"<h3>{escape(t.title)}</h3>{note}<table><tr>{head}</tr>{body}</table>"


def render_html(rep: Report) -> str:
    parts = [
        f"<h1>CBB ledger audit <span class='sub'>through {rep.day}</span></h1>",
        f"<p class='sub'>{escape(NOTE)}</p>",
        "<div class='find'><b>Findings</b><ul>"
        + "".join(f"<li>{escape(f)}</li>" for f in rep.findings)
        + "</ul></div>",
    ]
    for s in rep.sections:
        parts.append(f"<h2>{escape(s.title)}</h2>")
        if s.lines and not s.tables:
            parts.append(f"<pre>{escape(chr(10).join(s.lines))}</pre>")
        else:
            parts += [f"<p>{escape(line)}</p>" for line in s.lines]
        parts += [_html_table(t) for t in s.tables]
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{''.join(parts)}</body></html>"


__all__ = [
    "Report",
    "Seg",
    "Table",
    "briers",
    "build",
    "coverage",
    "live_buys",
    "paper_buys",
    "render_html",
    "render_text",
    "segment",
]
