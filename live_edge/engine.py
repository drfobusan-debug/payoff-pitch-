"""One live tick: score + board -> rolled probabilities -> flags, with a tape and a ledger.

Storage (``LIVE_EDGE_DIR``, default ``~/.live_edge``):

* ``priors.json``   pregame consensus per event, captured from the board before kickoff
                    (or ESPN's pickcenter closing line for games already under way)
* ``streaks.json``  consecutive qualifying ticks per (event, market, side, book)
* ``tape/{sport}_{day}.jsonl``  one line per game per tick: score, clock, model, best prices
* ``flags.csv``     every flag ever raised, graded in place once the game is final
"""

from __future__ import annotations

import csv
import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from live_edge.espn import LiveGame, fetch_pregame_line, norm_team
from live_edge.oddsapi import EventQuotes, Quote, age_seconds
from live_edge.roller import LiveDist, Prior, Roller
from nfl_engine.market.odds import american_to_decimal, american_to_prob

log = logging.getLogger(__name__)

FLAG_FIELDS = [
    "ts",
    "sport",
    "event_id",
    "matchup",
    "market",
    "side",
    "line",
    "book",
    "price",
    "p_model",
    "p_market",
    "edge",
    "ev",
    "period",
    "clock",
    "home_score",
    "away_score",
    "prior_margin",
    "prior_total",
    "espn_home_wp",
    "result",
    "pnl",
    "final_home",
    "final_away",
]


@dataclass(frozen=True)
class Thresholds:
    ml_edge: float = 0.06
    spread_edge: float = 0.05
    total_edge: float = 0.05
    min_ticks: int = 2
    min_books: int = 2
    min_fraction_left: float = 0.15
    max_quote_age: float = 90.0
    min_price_prob: float = 0.12
    max_price_prob: float = 0.88

    def edge_for(self, market: str) -> float:
        return {"ml": self.ml_edge, "spread": self.spread_edge, "total": self.total_edge}[market]


@dataclass
class Flag:
    ts: str
    sport: str
    event_id: str
    matchup: str
    market: str
    side: str
    line: float | None
    book: str
    price: float
    p_model: float
    p_market: float
    edge: float
    ev: float
    period: int
    clock: int
    home_score: int
    away_score: int
    prior_margin: float
    prior_total: float
    espn_home_wp: float | None
    result: str = ""
    pnl: float | None = None
    final_home: int | None = None
    final_away: int | None = None


@dataclass
class Candidate:
    market: str
    side: str
    line: float | None
    book: str
    price: float
    p_model: float
    p_market: float

    @property
    def edge(self) -> float:
        return self.p_model - self.p_market

    @property
    def ev(self) -> float:
        dec = american_to_decimal(self.price)
        return self.p_model * (dec - 1.0) - (1.0 - self.p_model)


@dataclass
class TickReport:
    live: int = 0
    priced: int = 0
    flags: list[Flag] = field(default_factory=list)
    graded: int = 0
    unmatched: list[str] = field(default_factory=list)
    no_prior: list[str] = field(default_factory=list)


def data_dir() -> Path:
    raw = os.getenv("LIVE_EDGE_DIR")
    return Path(raw).expanduser() if raw else Path.home() / ".live_edge"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except ValueError:
        return {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
    tmp.replace(path)


def match_event(game: LiveGame, events: list[EventQuotes]) -> EventQuotes | None:
    for ev in events:
        if norm_team(ev.home) in game.home_aliases and norm_team(ev.away) in game.away_aliases:
            return ev
    for ev in events:
        h, a = norm_team(ev.home), norm_team(ev.away)
        if any(x and x in h for x in game.home_aliases[:3]) and any(
            x and x in a for x in game.away_aliases[:3]
        ):
            return ev
    return None


def consensus_prior(ev: EventQuotes, sport: str) -> Prior | None:
    """Median pregame home spread and total across books, as a Prior."""
    spreads = [
        q.line
        for q in ev.quotes
        if q.market == "spread" and q.side == "home" and q.line is not None
    ]
    totals = [
        q.line for q in ev.quotes if q.market == "total" and q.side == "over" and q.line is not None
    ]
    if not spreads or not totals:
        return None
    sd_m, sd_t = (16.0, 13.0) if sport == "cfb" else (13.2, 13.4)
    return Prior(margin=-median(spreads), total=median(totals), margin_sd=sd_m, total_sd=sd_t)


def devig_pairs(
    ev: EventQuotes, now: datetime, max_age: float
) -> list[tuple[Quote, float, Quote, float]]:
    """(quote_a, fair_a, quote_b, fair_b) for every fresh two-way pair."""
    out = []
    for a, b in ev.pairs():
        age_a, age_b = age_seconds(a.updated, now), age_seconds(b.updated, now)
        if age_a is None or age_b is None or max(age_a, age_b) > max_age:
            continue
        pa, pb = american_to_prob(a.price), american_to_prob(b.price)
        if pa + pb <= 0:
            continue
        out.append((a, pa / (pa + pb), b, pb / (pa + pb)))
    return out


def model_prob(dist: LiveDist, q: Quote) -> float | None:
    if q.market == "ml":
        p = dist.p_home_win()
        return p if q.side == "home" else 1.0 - p
    if q.line is None:
        return None
    if q.market == "spread":
        home_line = q.line if q.side == "home" else -q.line
        p = dist.p_home_cover(home_line)
        return p if q.side == "home" else 1.0 - p
    p = dist.p_over(q.line)
    return p if q.side == "over" else 1.0 - p


def candidates(dist: LiveDist, ev: EventQuotes, th: Thresholds, now: datetime) -> list[Candidate]:
    """Best price per (market, side, line) whose devigged prob the model beats by the threshold.

    The edge is measured against the *median* devigged price across the books
    quoting that line (at least ``min_books`` of them), so one stale or slow
    book cannot raise a flag on its own; the best price among them is kept.
    """
    by_key: dict[tuple[str, str, float | None], list[tuple[Quote, float]]] = {}
    for a, fa, b, fb in devig_pairs(ev, now, th.max_quote_age):
        for q, fair in ((a, fa), (b, fb)):
            by_key.setdefault((q.market, q.side, q.line), []).append((q, fair))
    out: list[Candidate] = []
    for (market, side, line), rows in by_key.items():
        if len({q.book for q, _ in rows}) < th.min_books:
            continue
        pm = model_prob(dist, rows[0][0])
        if pm is None:
            continue
        consensus = median(fair for _, fair in rows)
        if pm - consensus < th.edge_for(market):
            continue
        best: Candidate | None = None
        for q, _ in rows:
            price_p = american_to_prob(q.price)
            if not th.min_price_prob <= price_p <= th.max_price_prob:
                continue
            c = Candidate(market, side, line, q.book, q.price, pm, consensus)
            if c.ev > 0 and (best is None or c.ev > best.ev):
                best = c
        if best is not None:
            out.append(best)
    return out


class LiveEngine:
    def __init__(
        self,
        sport: str,
        root: Path | None = None,
        thresholds: Thresholds | None = None,
        prior_fallback: Callable[[str, str], tuple[float, float] | None]
        | None = fetch_pregame_line,
    ) -> None:
        self.sport = sport
        self.prior_fallback = prior_fallback
        self.root = root or data_dir()
        self.th = thresholds or Thresholds()
        self.roller = Roller(sport)
        self.priors_path = self.root / "priors.json"
        self.streaks_path = self.root / "streaks.json"
        self.flags_path = self.root / "flags.csv"

    # -- persistence -------------------------------------------------------
    def load_flags(self) -> list[Flag]:
        if not self.flags_path.exists():
            return []
        rows = []
        with self.flags_path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(_flag_from_row(r))
        return rows

    def save_flags(self, flags: list[Flag]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.flags_path.with_suffix(".tmp")
        with tmp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FLAG_FIELDS)
            w.writeheader()
            for f in flags:
                w.writerow({k: ("" if v is None else v) for k, v in asdict(f).items()})
        tmp.replace(self.flags_path)

    def _tape(self, record: dict) -> None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self.root / "tape" / f"{self.sport}_{day}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    # -- the tick ----------------------------------------------------------
    def tick(
        self, games: list[LiveGame], events: list[EventQuotes], now: datetime | None = None
    ) -> TickReport:
        now = now or datetime.now(timezone.utc)
        ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        priors = _read_json(self.priors_path)
        streaks = _read_json(self.streaks_path)
        flags = self.load_flags()
        open_keys = {(f.event_id, f.market, f.side, f.line, f.book) for f in flags}
        rep = TickReport()

        for g in games:
            ev = match_event(g, events)
            if g.status == "pre":
                if ev is not None and g.event_id not in priors:
                    pr = consensus_prior(ev, self.sport)
                    if pr is not None:
                        priors[g.event_id] = {
                            "margin": pr.margin,
                            "total": pr.total,
                            "matchup": g.matchup,
                        }
                continue
            if g.status == "post":
                rep.graded += self._grade(flags, g)
                continue
            rep.live += 1
            if ev is None:
                rep.unmatched.append(g.matchup)
                continue
            raw = priors.get(g.event_id)
            if raw is None and self.prior_fallback is not None:
                line = self.prior_fallback(self.sport, g.event_id)
                if line is not None:
                    raw = {
                        "margin": -line[0],
                        "total": line[1],
                        "matchup": g.matchup,
                        "source": "espn",
                    }
                    priors[g.event_id] = raw
            if raw is None:
                rep.no_prior.append(g.matchup)
                continue
            sd_m, sd_t = (16.0, 13.0) if self.sport == "cfb" else (13.2, 13.4)
            prior = Prior(float(raw["margin"]), float(raw["total"]), sd_m, sd_t)
            dist = self.roller.roll(prior, g.state)
            rep.priced += 1
            cands = candidates(dist, ev, self.th, now)
            live_keys = {f"{c.market}|{c.side}|{c.line}|{c.book}" for c in cands}
            self._tape(
                {
                    "ts": ts,
                    "event_id": g.event_id,
                    "matchup": g.matchup,
                    "home": g.state.home_score,
                    "away": g.state.away_score,
                    "period": g.state.period,
                    "clock": g.state.clock_secs,
                    "f_left": round(g.state.fraction_left, 4),
                    "prior": raw,
                    "espn_home_wp": g.espn_home_wp,
                    "model": {
                        "margin_mu": round(dist.margin_mu, 2),
                        "margin_sd": round(dist.margin_sd, 2),
                        "total_mu": round(dist.total_mu, 2),
                        "total_sd": round(dist.total_sd, 2),
                        "p_home": round(dist.p_home_win(), 4),
                    },
                    "candidates": [
                        asdict(c) | {"edge": round(c.edge, 4), "ev": round(c.ev, 4)} for c in cands
                    ],
                    "n_quotes": len(ev.quotes),
                }
            )
            # Persistence: a candidate must qualify on consecutive ticks.
            ev_streaks = streaks.setdefault(g.event_id, {})
            for k in list(ev_streaks):
                if k not in live_keys:
                    del ev_streaks[k]
            if g.state.fraction_left < self.th.min_fraction_left:
                continue
            for c in cands:
                k = f"{c.market}|{c.side}|{c.line}|{c.book}"
                ev_streaks[k] = int(ev_streaks.get(k, 0)) + 1
                if ev_streaks[k] < self.th.min_ticks:
                    continue
                if (g.event_id, c.market, c.side, c.line, c.book) in open_keys:
                    continue
                fl = Flag(
                    ts=ts,
                    sport=self.sport,
                    event_id=g.event_id,
                    matchup=g.matchup,
                    market=c.market,
                    side=c.side,
                    line=c.line,
                    book=c.book,
                    price=c.price,
                    p_model=round(c.p_model, 4),
                    p_market=round(c.p_market, 4),
                    edge=round(c.edge, 4),
                    ev=round(c.ev, 4),
                    period=g.state.period,
                    clock=g.state.clock_secs,
                    home_score=g.state.home_score,
                    away_score=g.state.away_score,
                    prior_margin=prior.margin,
                    prior_total=prior.total,
                    espn_home_wp=g.espn_home_wp,
                )
                flags.append(fl)
                rep.flags.append(fl)
                open_keys.add((g.event_id, c.market, c.side, c.line, c.book))

        _write_json(self.priors_path, priors)
        _write_json(self.streaks_path, streaks)
        self.save_flags(flags)
        return rep

    def _grade(self, flags: list[Flag], g: LiveGame) -> int:
        n = 0
        for f in flags:
            if f.event_id != g.event_id or f.result:
                continue
            f.final_home, f.final_away = g.state.home_score, g.state.away_score
            f.result, f.pnl = grade_flag(f, g.state.home_score, g.state.away_score)
            n += 1
        return n


def grade_flag(f: Flag, home: int, away: int) -> tuple[str, float]:
    margin, total = home - away, home + away
    if f.market == "ml":
        if margin == 0:
            return "push", 0.0
        won = (margin > 0) == (f.side == "home")
    elif f.market == "spread":
        line = f.line if f.line is not None else 0.0
        m = margin + line if f.side == "home" else -margin + line
        if m == 0:
            return "push", 0.0
        won = m > 0
    else:
        line = f.line if f.line is not None else 0.0
        if total == line:
            return "push", 0.0
        won = (total > line) == (f.side == "over")
    pnl = american_to_decimal(f.price) - 1.0 if won else -1.0
    return ("win" if won else "loss"), round(pnl, 4)


def _flag_from_row(r: dict[str, str]) -> Flag:
    def fl(k: str) -> float | None:
        v = r.get(k, "")
        return float(v) if v not in ("", None) else None

    def it(k: str) -> int | None:
        v = r.get(k, "")
        return int(float(v)) if v not in ("", None) else None

    return Flag(
        ts=r["ts"],
        sport=r["sport"],
        event_id=r["event_id"],
        matchup=r["matchup"],
        market=r["market"],
        side=r["side"],
        line=fl("line"),
        book=r["book"],
        price=float(r["price"]),
        p_model=float(r["p_model"]),
        p_market=float(r["p_market"]),
        edge=float(r["edge"]),
        ev=float(r["ev"]),
        period=int(r["period"]),
        clock=int(r["clock"]),
        home_score=int(r["home_score"]),
        away_score=int(r["away_score"]),
        prior_margin=float(r["prior_margin"]),
        prior_total=float(r["prior_total"]),
        espn_home_wp=fl("espn_home_wp"),
        result=r.get("result", ""),
        pnl=fl("pnl"),
        final_home=it("final_home"),
        final_away=it("final_away"),
    )


def summarize(flags: list[Flag]) -> str:
    graded = [f for f in flags if f.result in ("win", "loss", "push")]
    lines = [f"{len(flags)} flags, {len(graded)} graded, {len(flags) - len(graded)} open"]
    for market in ("ml", "spread", "total"):
        fs = [f for f in graded if f.market == market]
        if not fs:
            continue
        w = sum(f.result == "win" for f in fs)
        lo = sum(f.result == "loss" for f in fs)
        p = sum(f.result == "push" for f in fs)
        pnl = sum(f.pnl or 0.0 for f in fs)
        lines.append(f"  {market:6s} {w}-{lo}-{p}  pnl {pnl:+.2f}u  roi {pnl / len(fs):+.1%}")
    by_q: dict[int, list[float]] = {}
    for f in graded:
        by_q.setdefault(f.period, []).append(f.pnl or 0.0)
    if by_q:
        lines.append(
            "  by quarter: "
            + ", ".join(f"Q{q} n={len(v)} {sum(v):+.2f}u" for q, v in sorted(by_q.items()))
        )
    return "\n".join(lines)
