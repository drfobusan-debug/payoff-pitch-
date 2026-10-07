"""Top-down MLB game-line backtest: soft-book prices against Pinnacle's no-vig line.

The question: when DraftKings, BetMGM or another US book offers a moneyline, run
line or total at a better price than Pinnacle's devigged price, is that bet +EV
at Pinnacle's close, and does it win money? No model is involved. Pinnacle is
the reference price and the soft books are where the bet would be struck.

Data comes from The Odds API historical endpoint (paid plans only; 5-minute
snapshots since September 2022) with ``markets=h2h,spreads,totals`` and an
explicit ``bookmakers`` list of at most 10, which bills as one region: 30
credits per snapshot for the whole slate (measured 2026-10-06). A day is
snapshotted every ``--interval`` minutes from ``--lead`` hours before its first
pitch to its last first pitch. Each raw response is cached gzipped before it is
parsed, so ``analyze`` re-runs cost nothing and an interrupted ``pull`` resumes
where it stopped. Final scores come from the free MLB Stats API.

    .venv/bin/python scripts/topdown_backtest.py plan    --seasons 2024 2025
    .venv/bin/python scripts/topdown_backtest.py pull    --seasons 2024 2025 --cap 800000
    .venv/bin/python scripts/topdown_backtest.py analyze --seasons 2024 2025 --fit 2024 --test 2025

Method (``analyze``):

* Fair price: Pinnacle no-vig for the same market at the same point.
* A soft-book quote triggers at the first snapshot where its EV against that
  fair price clears the threshold. There is one bet per game x market x side x
  point: the best-EV book at the first snapshot that triggers.
* Close: Pinnacle no-vig at the last snapshot before first pitch, at the same
  point. CLV-EV = close fair x decimal price - 1. It is left blank when Pinnacle
  no longer quotes that point at the close.
* ROI comes from the final score; pushes return 0.
* CIs are bootstrapped over slates (dates), because a day's games share news,
  weather and the same Pinnacle feed.
* Pre-registered rule: the threshold is picked on ``--fit`` (the lowest one
  whose CLV-EV lower bound is above zero with at least ``MIN_BETS`` bets), then
  judged once on ``--test`` by the same bound.

Pinnacle reaches The Odds API through its public website, which "may incur a
delay", so a gap can be Pinnacle lagging the soft book rather than the other
way round. Scoring against Pinnacle's own close is what catches that: a stale
reference shows up as negative CLV.

This is analysis only. Nothing here touches live scoring.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.oos_odds_history import (  # noqa: E402
    Budget,
    american_profit,
    american_to_prob,
    no_vig,
    team_abbr,
)

log = logging.getLogger("topdown_backtest")

HIST_URL = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds"
SCHED_URL = "https://statsapi.mlb.com/api/v1/schedule"
MARKETS = ("h2h", "spreads", "totals")
SHARP = "pinnacle"
DEFAULT_BOOKS = (
    SHARP,
    "draftkings",
    "betmgm",
    "fanduel",
    "williamhill_us",
    "betrivers",
    "espnbet",
    "fanatics",
    "bovada",
    "betonlineag",
)
#: The vendor bills each group of up to 10 named bookmakers as one region.
BOOKS_PER_REGION = 10
CREDITS_PER_SNAPSHOT = 10 * len(MARKETS)
DEFAULT_CACHE = Path.home() / ".mlb_engine" / "cache" / "topdown_hist"
DEFAULT_OUT = Path.home() / ".mlb_engine" / "audit" / "topdown"
THRESHOLDS = (0.005, 0.01, 0.02, 0.03, 0.05)
MIN_BETS = 50
#: Stats API abbreviations that differ from the odds-name mapping.
ABBR_ALIAS = {"OAK": "ATH"}
SKIP_STATES = ("postponed", "cancelled", "canceled", "suspended")
#: An odds event and a scheduled game further apart than this are different games.
MATCH_WINDOW = timedelta(hours=4)

GameKey = tuple[str, str, str]  # home, away, commence (ISO, UTC)
LineKey = tuple[str, "float | None"]  # market, line (home point for spreads, total for totals)


# --- schedule and finals --------------------------------------------------------------


@dataclass(frozen=True)
class Game:
    pk: int
    day: Date
    home: str
    away: str
    commence: datetime
    state: str
    detailed: str
    home_runs: int | None
    away_runs: int | None

    @property
    def played(self) -> bool:
        return not self.detailed.lower().startswith(SKIP_STATES)

    @property
    def final(self) -> bool:
        return (
            self.played
            and self.state == "Final"
            and self.home_runs is not None
            and self.away_runs is not None
        )


def _ts(raw: object) -> datetime | None:
    if not raw:
        return None
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(timezone.utc)


def _abbr(team: dict) -> str:
    ab = str(team.get("abbreviation", ""))
    return ABBR_ALIAS.get(ab, ab)


def parse_schedule(dates: list[dict]) -> list[Game]:
    out: list[Game] = []
    for block in dates:
        day = Date.fromisoformat(block["date"])
        for g in block.get("games", []):
            commence = _ts(g.get("gameDate"))
            if commence is None:
                continue
            home, away = g["teams"]["home"], g["teams"]["away"]
            status = g.get("status", {}) or {}
            hr, ar = home.get("score"), away.get("score")
            out.append(
                Game(
                    int(g["gamePk"]),
                    day,
                    _abbr(home.get("team", {})),
                    _abbr(away.get("team", {})),
                    commence,
                    str(status.get("abstractGameState", "")),
                    str(status.get("detailedState", "")),
                    int(hr) if hr is not None else None,
                    int(ar) if ar is not None else None,
                )
            )
    return out


def season_games(
    season: int, cache_dir: Path, session: requests.Session | None = None
) -> list[Game]:
    """Regular-season games with scores, one Stats API call per month (free).

    Cached once the season is over, so ``plan`` and ``analyze`` agree on the slate.
    """
    path = cache_dir / f"schedule_{season}.json"
    if path.exists():
        return parse_schedule(json.loads(path.read_text()))
    sess = session or requests.Session()
    dates: list[dict] = []
    for month in range(3, 11):
        start = Date(season, month, 1)
        end = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        params: dict[str, str | int] = {
            "sportId": 1,
            "gameType": "R",
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "hydrate": "team,linescore",
        }
        r = sess.get(
            SCHED_URL,
            params=params,
            timeout=60,
        )
        r.raise_for_status()
        dates.extend(r.json().get("dates", []))
    if Date(season, 11, 1) < Date.today():
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dates))
    return parse_schedule(dates)


# --- plan ------------------------------------------------------------------------------


def day_grid(commences: list[datetime], lead_hours: float, interval_min: int) -> list[datetime]:
    """Snapshot times on an ``interval_min`` grid from ``lead_hours`` before the
    first pitch to the last first pitch."""
    start = min(commences) - timedelta(hours=lead_hours)
    end = max(commences)
    step = interval_min * 60
    t = datetime.fromtimestamp(math.ceil(start.timestamp() / step) * step, tz=timezone.utc)
    out: list[datetime] = []
    while t <= end:
        out.append(t)
        t += timedelta(minutes=interval_min)
    return out


def plan(games: list[Game], lead_hours: float, interval_min: int) -> dict[Date, list[datetime]]:
    by_day: dict[Date, list[datetime]] = defaultdict(list)
    for g in games:
        if g.played:
            by_day[g.day].append(g.commence)
    return {d: day_grid(ts, lead_hours, interval_min) for d, ts in sorted(by_day.items())}


def snap_path(cache_dir: Path, day: Date, at: datetime) -> Path:
    return cache_dir / day.isoformat() / f"{at.strftime('%Y%m%dT%H%MZ')}.json.gz"


# --- pull ------------------------------------------------------------------------------


def fetch_snapshot(
    at: datetime,
    path: Path,
    api_key: str,
    budget: Budget,
    session: requests.Session,
    books: tuple[str, ...],
) -> bool:
    """Cache one historical snapshot. ``True`` when it is on disk afterwards."""
    if path.exists():
        return True
    if not budget.can_spend(CREDITS_PER_SNAPSHOT):
        return False
    params = {
        "apiKey": api_key,
        "bookmakers": ",".join(books),
        "markets": ",".join(MARKETS),
        "oddsFormat": "american",
        "date": at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for attempt in range(5):
        try:
            resp = session.get(HIST_URL, params=params, timeout=60)
        except requests.RequestException as exc:  # message would carry the key
            log.warning("%s: %s", params["date"], type(exc).__name__)
            time.sleep(2 * (attempt + 1))
            continue
        if resp.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        if resp.status_code != 200:
            log.error("%s: HTTP %s", params["date"], resp.status_code)
            if resp.status_code in (401, 402, 403):
                budget.remaining = 0
            return False
        try:
            cost = int(float(resp.headers.get("x-requests-last", CREDITS_PER_SNAPSHOT)))
        except ValueError:
            cost = CREDITS_PER_SNAPSHOT
        budget.record(resp, cost)
        raw = resp.json()
        raw["_meta"] = {
            "requested": params["date"],
            "credits": cost,
            "x_requests_remaining": budget.remaining,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as fh:
            json.dump(raw, fh)
        tmp.replace(path)
        return True
    return False


def pull(
    snaps: dict[Date, list[datetime]],
    cache_dir: Path,
    api_key: str,
    budget: Budget,
    session: requests.Session,
    books: tuple[str, ...],
) -> int:
    total = sum(len(v) for v in snaps.values())
    done = 0
    for day, times in snaps.items():
        for at in times:
            path = snap_path(cache_dir, day, at)
            if not fetch_snapshot(at, path, api_key, budget, session, books):
                if budget.remaining == 0 or not budget.can_spend(CREDITS_PER_SNAPSHOT):
                    print(
                        f"STOP: spent={budget.spent} remaining={budget.remaining} "
                        f"at {day} ({done}/{total} snapshots on disk)"
                    )
                    return 1
                log.warning("%s %s: not fetched, continuing", day, at)
            done += 1
            if done % 200 == 0:
                log.info("%s/%s spent=%s remaining=%s", done, total, budget.spent, budget.remaining)
    print(f"done: {done}/{total} snapshots, spent={budget.spent} remaining={budget.remaining}")
    return 0


# --- parse -----------------------------------------------------------------------------


def _side_line(market: str, oc: dict, home: str, away: str) -> tuple[str, float | None] | None:
    if market == "totals":
        name, pt = str(oc.get("name", "")).lower(), oc.get("point")
        if name not in ("over", "under") or pt is None:
            return None
        return name, float(pt)
    ab = team_abbr(str(oc.get("name", "")))
    side = "home" if ab == home else "away" if ab == away else None
    if side is None:
        return None
    if market == "h2h":
        return side, None
    pt = oc.get("point")
    if pt is None:
        return None
    return side, float(pt) if side == "home" else -float(pt)


def parse_snapshot(
    raw: dict,
) -> tuple[datetime | None, dict[GameKey, dict[str, dict[LineKey, dict[str, float]]]]]:
    """``(snapshot time, {game: {book: {(market, line): {side: american}}}})``.

    Games already under way at the snapshot are dropped: those are live prices.
    """
    ts = _ts(raw.get("timestamp"))
    out: dict[GameKey, dict[str, dict[LineKey, dict[str, float]]]] = {}
    if ts is None:
        return None, out
    for ev in raw.get("data", []) or []:
        home, away = (
            team_abbr(str(ev.get("home_team", ""))),
            team_abbr(str(ev.get("away_team", ""))),
        )
        commence = _ts(ev.get("commence_time"))
        if not home or not away or commence is None or commence <= ts:
            continue
        books: dict[str, dict[LineKey, dict[str, float]]] = {}
        for bk in ev.get("bookmakers", []) or []:
            lines: dict[LineKey, dict[str, float]] = {}
            for mkt in bk.get("markets", []) or []:
                key = str(mkt.get("key", ""))
                if key not in MARKETS:
                    continue
                for oc in mkt.get("outcomes", []) or []:
                    if oc.get("price") is None:
                        continue
                    sl = _side_line(key, oc, home, away)
                    if sl is not None:
                        lines.setdefault((key, sl[1]), {})[sl[0]] = float(oc["price"])
            if lines:
                books[str(bk.get("key", ""))] = lines
        if books:
            out[(home, away, commence.isoformat())] = books
    return ts, out


def _dec(american: float) -> float:
    return 1.0 + american_profit(american)


@dataclass(frozen=True)
class Obs:
    ts: datetime
    ev: float
    price: float
    fair: float


@dataclass
class Scan:
    obs: dict[tuple[GameKey, LineKey, str, str], list[Obs]]
    close: dict[tuple[GameKey, LineKey], tuple[datetime, dict[str, float]]]
    #: Pinnacle's last pre-game snapshot per game x market. A line whose own last
    #: quote is older than this had moved off that point by the close.
    last_seen: dict[tuple[GameKey, str], datetime]
    compared: Counter[str]

    def closing_fair(self, gk: GameKey, lk: LineKey) -> dict[str, float] | None:
        hit = self.close.get((gk, lk))
        if hit is None or hit[0] != self.last_seen.get((gk, lk[0])):
            return None
        return hit[1]


def _index(games: list[Game]) -> dict[tuple[str, str], list[Game]]:
    index: dict[tuple[str, str], list[Game]] = defaultdict(list)
    for g in games:
        index[(g.home, g.away)].append(g)
    return index


def scan(paths: list[Path], min_threshold: float, games: list[Game] | None = None) -> Scan:
    """Every soft-book side that cleared ``min_threshold`` EV against Pinnacle,
    plus Pinnacle's last pre-game fair price for every line.

    With ``games``, an event is only read from its own slate day's snapshots: an
    evening snapshot also carries tomorrow's board, and that day's grid (not
    today's) is what reaches its first pitch, so only it gives a real close."""
    index = _index(games) if games is not None else None
    obs: dict[tuple[GameKey, LineKey, str, str], list[Obs]] = defaultdict(list)
    close: dict[tuple[GameKey, LineKey], tuple[datetime, dict[str, float]]] = {}
    last_seen: dict[tuple[GameKey, str], datetime] = {}
    compared: Counter[str] = Counter()
    seen: set[datetime] = set()
    for p in sorted(paths):
        with gzip.open(p, "rt") as fh:
            ts, events = parse_snapshot(json.load(fh))
        if ts is None or ts in seen:  # two requests can resolve to one snapshot
            continue
        seen.add(ts)
        day = Date.fromisoformat(p.parent.name)
        for gk, books in events.items():
            if index is not None:
                g = match_game(gk, index)
                if g is None or g.day != day:
                    continue
            for lk, sides in books.get(SHARP, {}).items():
                if len(sides) != 2:
                    continue
                a, b = sorted(sides)
                fa, fb = no_vig(american_to_prob(sides[a]), american_to_prob(sides[b]))
                fair = {a: fa, b: fb}
                prev = close.get((gk, lk))
                if prev is None or ts > prev[0]:
                    close[(gk, lk)] = (ts, fair)
                if ts > last_seen.get((gk, lk[0]), ts - timedelta(seconds=1)):
                    last_seen[(gk, lk[0])] = ts
                for book, lines in books.items():
                    if book == SHARP or lk not in lines:
                        continue
                    for side, price in lines[lk].items():
                        if side not in fair:
                            continue
                        compared[lk[0]] += 1
                        ev = fair[side] * _dec(price) - 1.0
                        if ev >= min_threshold:
                            obs[(gk, lk, side, book)].append(Obs(ts, ev, price, fair[side]))
    return Scan(dict(obs), close, last_seen, compared)


# --- grade -----------------------------------------------------------------------------


def grade(market: str, side: str, line: float | None, home_runs: int, away_runs: int) -> str:
    """``win`` / ``loss`` / ``push`` for one side. ``line`` is the home point for
    spreads and the total for totals."""
    if market == "h2h":
        margin = float(home_runs - away_runs)
    elif market == "spreads":
        margin = home_runs - away_runs + float(line or 0.0)
    else:
        margin = home_runs + away_runs - float(line or 0.0)
        side = "home" if side == "over" else "away"
    if margin == 0:
        return "push"
    return "win" if (margin > 0) == (side == "home") else "loss"


def match_game(gk: GameKey, index: dict[tuple[str, str], list[Game]]) -> Game | None:
    commence = _ts(gk[2])
    assert commence is not None
    best = min(
        index.get((gk[0], gk[1]), []), key=lambda g: abs(g.commence - commence), default=None
    )
    if best is None or abs(best.commence - commence) > MATCH_WINDOW:
        return None
    return best


def _persist_minutes(hits: list[Obs], interval_min: int) -> float:
    """How long the first gap stayed open over consecutive snapshots."""
    limit = timedelta(minutes=1.5 * interval_min)
    last = hits[0].ts
    for o in hits[1:]:
        if o.ts - last > limit:
            break
        last = o.ts
    return (last - hits[0].ts).total_seconds() / 60.0


def bets(sc: Scan, games: list[Game], threshold: float, interval_min: int) -> pd.DataFrame:
    index = _index(games)
    picks: dict[tuple[GameKey, LineKey, str], list[tuple[str, list[Obs]]]] = defaultdict(list)
    for (gk, lk, side, book), series in sc.obs.items():
        hits = sorted((o for o in series if o.ev >= threshold), key=lambda o: o.ts)
        if hits:
            picks[(gk, lk, side)].append((book, hits))
    rows: list[dict] = []
    for (gk, lk, side), cands in picks.items():
        first = min(h[0].ts for _, h in cands)
        book, hits = max(((b, h) for b, h in cands if h[0].ts == first), key=lambda c: c[1][0].ev)
        o = hits[0]
        game = match_game(gk, index)
        commence = _ts(gk[2])
        assert commence is not None
        market, line = lk
        row: dict = {
            "day": game.day if game else commence.date(),
            "season": (game.day if game else commence.date()).year,
            "home": gk[0],
            "away": gk[1],
            "commence": gk[2],
            "market": market,
            "line": line,
            "side": side,
            "book": book,
            "price": o.price,
            "fair_bet": o.fair,
            "ev_bet": o.ev,
            "bet_ts": o.ts.isoformat(),
            "min_to_pitch": (commence - o.ts).total_seconds() / 60.0,
            "persist_min": _persist_minutes(hits, interval_min),
            "close_fair": math.nan,
            "clv_pts": math.nan,
            "clv_ev": math.nan,
            "result": None,
            "pnl": math.nan,
        }
        closing = sc.closing_fair(gk, lk)
        if closing is not None and side in closing:
            cf = closing[side]
            row.update(close_fair=cf, clv_pts=cf - o.fair, clv_ev=cf * _dec(o.price) - 1.0)
        if game is not None and game.final:
            assert game.home_runs is not None and game.away_runs is not None
            res = grade(market, side, line, game.home_runs, game.away_runs)
            row["result"] = res
            row["pnl"] = (
                american_profit(o.price) if res == "win" else 0.0 if res == "push" else -1.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


# --- summarize ------------------------------------------------------------------------


def boot_ci(df: pd.DataFrame, col: str, b: int = 1000, seed: int = 7) -> tuple[float, float]:
    """95% CI of the row mean of ``col``, resampling whole slates."""
    d = df[df[col].notna()]
    if d.empty:
        return math.nan, math.nan
    agg = d.groupby("day")[col].agg(["sum", "count"])
    s, n = agg["sum"].to_numpy(), agg["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(b, len(s)))
    means = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for market in ("all", *MARKETS):
        d = df if market == "all" else df[df["market"] == market]
        if d.empty:
            continue
        clv = d[d["clv_ev"].notna()]
        graded = d[d["pnl"].notna()]
        clo, chi = boot_ci(d, "clv_ev")
        rlo, rhi = boot_ci(d, "pnl")
        rows.append(
            {
                "market": market,
                "n": len(d),
                "slates": d["day"].nunique(),
                "ev_bet%": round(100 * d["ev_bet"].mean(), 2),
                "clv_ev%": round(100 * clv["clv_ev"].mean(), 2) if len(clv) else math.nan,
                "clv_ci": f"[{100 * clo:+.2f}, {100 * chi:+.2f}]",
                "close_cov%": round(100 * len(clv) / len(d), 1),
                "beat_close%": round(100 * (clv["clv_pts"] > 0).mean(), 1)
                if len(clv)
                else math.nan,
                "lost_close%": round(100 * (clv["clv_pts"] < 0).mean(), 1)
                if len(clv)
                else math.nan,
                "roi%": round(100 * graded["pnl"].mean(), 2) if len(graded) else math.nan,
                "roi_ci": f"[{100 * rlo:+.1f}, {100 * rhi:+.1f}]",
                "persist_med_min": round(float(d["persist_min"].median()), 1),
                "min_to_pitch_med": round(float(d["min_to_pitch"].median()), 0),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(by_thr: dict[float, pd.DataFrame], season: int) -> float | None:
    """Lowest threshold whose CLV-EV lower bound is above zero on ``season``."""
    for thr in sorted(by_thr):
        d = by_thr[thr]
        d = d[(d["season"] == season) & d["clv_ev"].notna()] if not d.empty else d
        if len(d) >= MIN_BETS and boot_ci(d, "clv_ev")[0] > 0:
            return thr
    return None


# --- cli -------------------------------------------------------------------------------


def _books(raw: str) -> tuple[str, ...]:
    return tuple(b.strip() for b in raw.split(",") if b.strip())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["plan", "pull", "analyze"])
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    ap.add_argument("--start", type=Date.fromisoformat, help="only days on/after this date")
    ap.add_argument("--end", type=Date.fromisoformat, help="only days on/before this date")
    ap.add_argument("--interval", type=int, default=10, help="minutes between snapshots")
    ap.add_argument("--lead", type=float, default=2.0, help="hours before a day's first pitch")
    ap.add_argument("--books", type=_books, default=DEFAULT_BOOKS)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--cap", type=int, default=50_000, help="hard credit cap for this run")
    ap.add_argument(
        "--floor",
        type=int,
        default=1_000_000,
        help="never let x-requests-remaining drop below this (the other engines' share)",
    )
    ap.add_argument("--thresholds", type=float, nargs="+", default=list(THRESHOLDS))
    ap.add_argument("--fit", type=int, help="season the threshold is chosen on")
    ap.add_argument("--test", type=int, help="season the chosen threshold is judged on")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    books: tuple[str, ...] = args.books
    if SHARP not in books:
        print(f"--books must include {SHARP}", file=sys.stderr)
        return 2
    if len(books) > BOOKS_PER_REGION:
        print(
            f"{len(books)} books bill as {math.ceil(len(books) / BOOKS_PER_REGION)} regions; "
            f"keep it to {BOOKS_PER_REGION}",
            file=sys.stderr,
        )
        return 2

    games: list[Game] = []
    for s in args.seasons:
        games.extend(season_games(s, args.cache_dir))
    games = [
        g
        for g in games
        if (args.start is None or g.day >= args.start) and (args.end is None or g.day <= args.end)
    ]
    snaps = plan(games, args.lead, args.interval)

    if args.cmd in ("plan", "pull"):
        for s in args.seasons:
            mine = {d: v for d, v in snaps.items() if d.year == s}
            n = sum(len(v) for v in mine.values())
            cached = sum(
                snap_path(args.cache_dir, d, t).exists() for d, v in mine.items() for t in v
            )
            print(
                f"{s}: game days={len(mine)} snapshots={n} cached={cached} "
                f"credits_needed={(n - cached) * CREDITS_PER_SNAPSHOT:,}"
            )
        if args.cmd == "plan":
            return 0
        key = os.environ.get("THE_ODDS_API_KEY") or os.environ.get("ODDS_API_KEY")
        if not key:
            print("THE_ODDS_API_KEY not set", file=sys.stderr)
            return 2
        return pull(
            snaps, args.cache_dir, key, Budget(args.cap, args.floor), requests.Session(), books
        )

    paths = [snap_path(args.cache_dir, d, t) for d, v in snaps.items() for t in v]
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("no cached snapshots; run pull first", file=sys.stderr)
        return 1
    sc = scan(paths, min(args.thresholds), games)
    print(f"snapshots={len(paths)} soft-book quotes compared: {dict(sc.compared)}")
    args.out.mkdir(parents=True, exist_ok=True)
    by_thr: dict[float, pd.DataFrame] = {}
    report: list[str] = []
    for thr in sorted(args.thresholds):
        df = bets(sc, games, thr, args.interval)
        by_thr[thr] = df
        if df.empty:
            report.append(f"\n== EV >= {thr:.1%}: no bets")
            continue
        df.to_csv(args.out / f"bets_ev{thr * 1000:.0f}.csv", index=False)
        for s in sorted(df["season"].unique()):
            report.append(f"\n== EV >= {thr:.1%}, season {s}")
            report.append(summarize(df[df["season"] == s]).to_string(index=False))
    if args.fit is not None and args.test is not None:
        chosen = choose_threshold(by_thr, args.fit)
        report.append(
            f"\n== Pre-registered test: threshold chosen on {args.fit}: "
            f"{'none passed' if chosen is None else f'{chosen:.1%}'}"
        )
        if chosen is not None:
            d = by_thr[chosen]
            d = d[d["season"] == args.test]
            lo, hi = boot_ci(d, "clv_ev")
            verdict = "PASS" if len(d[d["clv_ev"].notna()]) >= MIN_BETS and lo > 0 else "FAIL"
            report.append(f"{args.test}: CLV-EV CI [{100 * lo:+.2f}, {100 * hi:+.2f}] -> {verdict}")
            report.append(summarize(d).to_string(index=False))
            by_book = d.groupby("book").agg(
                n=("pnl", "size"), clv_ev=("clv_ev", "mean"), roi=("pnl", "mean")
            )
            report.append("\nby book:\n" + (by_book * [1, 100, 100]).round(2).to_string())
    text = "\n".join(report)
    (args.out / "summary.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
