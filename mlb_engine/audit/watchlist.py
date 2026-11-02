"""What the MLB engine is watching: pre-registered graded-only bands for the card.

Nothing here prices or stakes a bet. Each band was written down before the
games it is judged on (``REGISTERED``): rows dated earlier never count, so the
2023-2026 study that suggested a band cannot also confirm it. Bands are read
with the shared band reader (``engine_common.bands``): flat 1u at the card's
price, judged against what the no-vig price expects, SE over per-game sums,
and a call only at 150 games with both halves agreeing.

The 2023-2026 schedule study (8,658 games with a true close, 50 bands, none
past Holm) left these as the only candidates:

* Under, getaway day game: +5.1u over the market on 2,162 (se 2.1)
* Under, a team played last night and today is a day game: +4.4u on 2,098 (se 2.1)
* the two together: +7.6u on 1,432 (se 2.5), found after looking at both
* home team on its 10th+ straight day with a game: +7.5u on 1,053 (se 2.9)
* the same at 13+: +12.4u on 350 (se 4.9)
* home team in a series opener after the visitor flew 1,000+ miles: -5.4u
  on 928 (se 3.0), a fade candidate

The engine screens from the 2026 power-screen audit sit beside them: pitcher
walks and outs buys, and the batter-over-vs-elite-arm refusal. The worksheet
contributes a bullpen-only side rule, and the totals sheet's SUM call is shown
against simply betting every Over.
"""

from __future__ import annotations

import csv
import html
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta
from pathlib import Path
from typing import Any

from engine_common.bands import MIN_N, BandRead
from mlb_engine.audit.grade import LOSS, WIN
from mlb_engine.audit.ledger import ENGINE, LedgerEntry, load_ledger, one_side_per_prop
from mlb_engine.data.parks import get_park
from mlb_engine.market.odds import american_to_decimal, american_to_prob
from mlb_engine.market.tiers import Tier

log = logging.getLogger(__name__)

REGISTERED = "2027-01-01"
LONG_TRIP_MILES = 1000.0
BULLPEN_GAP = 5.0
WINS, LOSES = "rows win", "rows lose"
_BUY = frozenset({Tier.STRONG.value, Tier.MODERATE.value})
_SCHEDULE_TTL_S = 12 * 3600


# ---- schedule context -------------------------------------------------------
@dataclass(frozen=True)
class GameContext:
    game_pk: int
    date: str
    matchup: str
    home: str
    away: str
    getaway: bool  # last game of a series, played in the day
    day_after_night: bool  # a day game with either team off a night game yesterday
    home_streak: int  # consecutive calendar days the home team has played, today included
    opener_visitor_miles: float | None  # series opener: how far the visitor came


def _miles(a: int | None, b: int | None) -> float | None:
    pa, pb = (get_park(a) if a else None), (get_park(b) if b else None)
    if pa is None or pb is None:
        return None
    la1, lo1, la2, lo2 = map(math.radians, (pa.lat, pa.lon, pb.lat, pb.lon))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 3958.8 * math.asin(math.sqrt(h))


def context(games: list[dict[str, Any]]) -> dict[int, GameContext]:
    """Schedule context per game from a season's StatsAPI schedule (see ``fetch_schedule``)."""
    games = sorted(games, key=lambda g: (g["start"], g["pk"]))
    by_team: dict[int, list[dict[str, Any]]] = {}
    for g in games:
        for tid in (g["home_id"], g["away_id"]):
            by_team.setdefault(tid, []).append(g)

    def previous(tid: int, g: dict[str, Any]) -> dict[str, Any] | None:
        seq = by_team[tid]
        i = next(k for k, x in enumerate(seq) if x["pk"] == g["pk"])
        return seq[i - 1] if i else None

    def streak(tid: int, g: dict[str, Any]) -> int:
        days = {Date.fromisoformat(x["date"]) for x in by_team[tid] if x["start"] <= g["start"]}
        d, n = Date.fromisoformat(g["date"]), 0
        while d in days:
            n, d = n + 1, d - timedelta(days=1)
        return n

    out: dict[int, GameContext] = {}
    for g in games:
        today = Date.fromisoformat(g["date"])
        off_night = False
        for tid in (g["home_id"], g["away_id"]):
            p = previous(tid, g)
            if p and Date.fromisoformat(p["date"]) == today - timedelta(days=1) and p["dn"] == "night":
                off_night = True
        miles = None
        if g["sgn"] == 1:
            p = previous(g["away_id"], g)
            miles = _miles(p["venue"], g["venue"]) if p else None
        out[g["pk"]] = GameContext(
            game_pk=g["pk"],
            date=g["date"],
            matchup=f"{g['away']} @ {g['home']}",
            home=g["home"],
            away=g["away"],
            getaway=g["sgn"] == g["gis"] and g["gis"] >= 2 and g["dn"] == "day",
            day_after_night=off_night and g["dn"] == "day",
            home_streak=streak(g["home_id"], g),
            opener_visitor_miles=miles,
        )
    return out


def parse_schedule(data: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for block in data.get("dates", []):
        for g in block.get("games", []):
            state = (g.get("status") or {}).get("detailedState", "")
            if state in ("Postponed", "Cancelled"):
                continue
            teams = g.get("teams", {})
            home, away = teams.get("home", {}).get("team", {}), teams.get("away", {}).get("team", {})
            out.append({
                "pk": g.get("gamePk"),
                "date": g.get("officialDate") or block.get("date"),
                "start": g.get("gameDate", ""),
                "dn": g.get("dayNight", ""),
                "sgn": g.get("seriesGameNumber") or 0,
                "gis": g.get("gamesInSeries") or 0,
                "home": home.get("abbreviation", ""),
                "away": away.get("abbreviation", ""),
                "home_id": home.get("id", 0),
                "away_id": away.get("id", 0),
                "venue": (g.get("venue") or {}).get("id"),
            })
    return out


def fetch_schedule(get: Callable[..., dict[str, Any]], season: int, cache_dir: Path) -> list[dict[str, Any]]:
    """The regular season's schedule, cached for 12 hours (one free StatsAPI call)."""
    path = cache_dir / f"schedule_{season}.json"
    if path.exists() and time.time() - path.stat().st_mtime < _SCHEDULE_TTL_S:
        return list(json.loads(path.read_text()))
    games = parse_schedule(get("schedule", sportId=1, season=season, gameType="R", hydrate="team,venue"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(games))
    return games


# ---- bands ------------------------------------------------------------------
def _team(selection: str) -> str:
    return selection.rsplit(" ", 1)[0]


def would_be_buy(e: LedgerEntry) -> bool:
    return e.tier in _BUY or e.pass_gate.startswith("graded_only_")


@dataclass(frozen=True)
class Band:
    name: str
    what: str
    wants: str
    pick: Callable[[LedgerEntry, GameContext | None], bool]
    needs_schedule: bool = True


def _under(e: LedgerEntry) -> bool:
    return e.market == "game_total" and e.selection.startswith("Under")


def _home_ml(e: LedgerEntry, c: GameContext | None) -> bool:
    return c is not None and e.market == "game_ml" and _team(e.selection) == c.home


BANDS: tuple[Band, ...] = (
    Band("under_getaway_day", "Under, getaway day game", WINS,
         lambda e, c: c is not None and _under(e) and c.getaway),
    Band("under_day_after_night", "Under, a team played last night, day game today", WINS,
         lambda e, c: c is not None and _under(e) and c.day_after_night),
    Band("under_getaway_and_day_after_night", "Under, both of the above", WINS,
         lambda e, c: c is not None and _under(e) and c.getaway and c.day_after_night),
    Band("home_ml_10_straight_days", "Home ML, home team's 10th+ straight day with a game", WINS,
         lambda e, c: _home_ml(e, c) and c is not None and c.home_streak >= 10),
    Band("home_ml_13_straight_days", "Home ML, 13th+ straight day", WINS,
         lambda e, c: _home_ml(e, c) and c is not None and c.home_streak >= 13),
    Band("home_ml_opener_visitor_flew_1000", "Home ML, series opener, visitor flew 1,000+ mi", LOSES,
         lambda e, c: _home_ml(e, c) and c is not None
         and (c.opener_visitor_miles or 0.0) >= LONG_TRIP_MILES),
    Band("pitcher_bb_buys", "Pitcher walks, would-be buys", WINS,
         lambda e, c: e.market == "pitcher_bb" and would_be_buy(e), needs_schedule=False),
    Band("pitcher_outs_buys", "Pitcher outs, would-be buys", WINS,
         lambda e, c: e.market == "pitcher_outs" and would_be_buy(e), needs_schedule=False),
    Band("batter_over_elite_arm", "Batter overs refused vs an elite arm", LOSES,
         lambda e, c: e.pass_gate == "batter_over_elite_arm", needs_schedule=False),
)


@dataclass(frozen=True)
class WatchRow:
    name: str
    what: str
    wants: str
    read: BandRead

    @property
    def games(self) -> int:
        return len(self.read.games)


def _expected(odds: float, fair: float) -> float:
    return fair * (american_to_decimal(odds) - 1.0) - (1.0 - fair)


def _read(name: str, rows: list[tuple[str, str, float, float]]) -> BandRead:
    """``rows``: (date, game key, pnl, expected), any order."""
    rows = sorted(rows)
    r = BandRead(name)
    per_game: dict[tuple[str, str], list[float]] = {}
    for d, g, pnl, exp in rows:
        r.pnls.append(pnl)
        r.expected.append(exp)
        acc = per_game.setdefault((d, g), [0.0, 0.0])
        acc[0] += pnl
        acc[1] += exp
    for (_, _), (pnl, exp) in sorted(per_game.items()):
        r.games.append(pnl)
        r.expected_games.append(exp)
    return r


def band_rows(entries: list[LedgerEntry], ctx: dict[int, GameContext]) -> list[WatchRow]:
    by_game: dict[tuple[str, str], list[GameContext]] = {}
    for gc in ctx.values():
        by_game.setdefault((gc.date, gc.matchup), []).append(gc)
    rows = [
        e for e in one_side_per_prop(entries)
        if e.source == ENGINE and e.date >= REGISTERED and e.result in (WIN, LOSS)
        and e.odds is not None and e.fair_prob is not None
    ]
    out = []
    for b in BANDS:
        picked = []
        for e in rows:
            cs = by_game.get((e.date, e.matchup), [])
            c = cs[0] if len(cs) == 1 else None  # a doubleheader is not guessed at
            if b.needs_schedule and c is None:
                continue
            if b.pick(e, c):
                picked.append((e.date, e.matchup, e.pnl, _expected(e.odds or 0.0, e.fair_prob or 0.0)))
        out.append(WatchRow(b.name, b.what, b.wants, _read(b.name, picked)))
    return out


def _f(v: str | None) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def bullpen_row(worksheet: list[dict[str, str]], weights: str) -> WatchRow:
    """Bet the side whose bullpen scores ``BULLPEN_GAP``+ points better, at its moneyline."""
    picked = []
    for r in worksheet:
        if r.get("date", "") < REGISTERED or r.get("weights") != weights or r.get("result") not in ("fav", "dog"):
            continue
        a_bp, h_bp = _f(r.get("away_bp")), _f(r.get("home_bp"))
        a_ml, h_ml = _f(r.get("away_ml")), _f(r.get("home_ml"))
        if None in (a_bp, h_bp, a_ml, h_ml) or abs((a_bp or 0.0) - (h_bp or 0.0)) < BULLPEN_GAP:
            continue
        side = r["away"] if (a_bp or 0.0) > (h_bp or 0.0) else r["home"]
        price, other = (a_ml, h_ml) if side == r["away"] else (h_ml, a_ml)
        winner = r["fav"] if r["result"] == "fav" else (r["home"] if r["fav"] == r["away"] else r["away"])
        p, q = american_to_prob(price or 0.0), american_to_prob(other or 0.0)
        pnl = american_to_decimal(price or 0.0) - 1.0 if winner == side else -1.0
        picked.append((r["date"], r.get("game", ""), pnl, _expected(price or 0.0, p / (p + q))))
    return WatchRow("worksheet_bullpen_side", f"Worksheet: the side with a {BULLPEN_GAP:g}+ pt bullpen edge",
                    WINS, _read("worksheet_bullpen_side", picked))


@dataclass(frozen=True)
class TotalsBenchmark:
    n: int
    sum_wins: int
    sum_calls: int
    over_wins: int

    @property
    def sum_rate(self) -> float:
        return self.sum_wins / self.sum_calls if self.sum_calls else 0.0

    @property
    def over_rate(self) -> float:
        return self.over_wins / self.n if self.n else 0.0


def totals_benchmark(rows: list[dict[str, str]]) -> TotalsBenchmark:
    """The totals sheet's SUM-sign call against betting every Over (pushes and voids out)."""
    graded = [r for r in rows if r.get("date", "") >= REGISTERED and r.get("result") in ("over", "under")]
    calls = [(r, s) for r in graded if (s := _f(r.get("sum_pts"))) not in (None, 0.0)]
    sum_wins = sum(1 for r, s in calls if (r["result"] == "over") == ((s or 0.0) > 0))
    return TotalsBenchmark(len(graded), sum_wins, len(calls), sum(1 for r in graded if r["result"] == "over"))


@dataclass(frozen=True)
class Watchlist:
    rows: list[WatchRow]
    totals: TotalsBenchmark
    weights: str
    schedule_ok: bool


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def build(audit_dir: Path, schedule: list[dict[str, Any]], weights: str) -> Watchlist:
    ledger = audit_dir / "ledger.csv"
    entries = [e for e in load_ledger(ledger) if e.date >= REGISTERED] if ledger.exists() else []
    rows = band_rows(entries, context(schedule) if schedule else {})
    rows.append(bullpen_row(_csv(audit_dir / "worksheet_ledger.csv"), weights))
    return Watchlist(rows, totals_benchmark(_csv(audit_dir / "totals_ledger.csv")), weights, bool(schedule))


# ---- rendering --------------------------------------------------------------
def _status(row: WatchRow) -> str:
    v = row.read.verdict
    if v.startswith("probation"):
        return f"{MIN_N - row.games} more games"
    return v


def _cells(row: WatchRow) -> list[str]:
    r = row.read
    se = "n/a" if math.isnan(r.se) else f"{r.se:.1f}"
    h1, h2 = r.halves
    return [row.what, "win" if row.wants == WINS else "lose", f"{row.games}", r.record,
            f"{r.excess:+.1f}u (se {se})", f"{h1:+.1f} / {h2:+.1f}", _status(row)]


_HEAD = ["Band", "Wants rows to", "Games", "W-L", "vs market", "Halves", "Status"]


def _intro(w: Watchlist) -> str:
    note = "" if w.schedule_ok else " Schedule unavailable: the schedule bands read nothing this card."
    return (f"Graded only, never bet. Judged on slates from {REGISTERED} on, at the card's price "
            f"against the no-vig market; a call needs {MIN_N} games, 2 SE and both halves agreeing.{note}")


def _totals_line(w: Watchlist) -> str:
    t = w.totals
    return (f"Totals sheet: SUM call {t.sum_wins}-{t.sum_calls - t.sum_wins} ({t.sum_rate:.1%}) "
            f"vs always-Over {t.over_wins}-{t.n - t.over_wins} ({t.over_rate:.1%}). "
            f"Worksheet weights frozen at {w.weights}.")


def markdown(w: Watchlist) -> list[str]:
    lines = ["## What we're watching", "", f"*{_intro(w)}*", "",
             "| " + " | ".join(_HEAD) + " |", "|" + "---|" * len(_HEAD)]
    lines += ["| " + " | ".join(_cells(r)) + " |" for r in w.rows]
    return [*lines, "", _totals_line(w), ""]


def html_block(w: Watchlist) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in _HEAD)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in _cells(r)) + "</tr>" for r in w.rows)
    return (f"<h2>What we're watching</h2><p><em>{html.escape(_intro(w))}</em></p>"
            f"<table><tr>{head}</tr>{body}</table><p>{html.escape(_totals_line(w))}</p>")
