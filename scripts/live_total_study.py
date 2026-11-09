"""Four-sport live-total study: is "pregame total + the start" the fair live total, and is
there anything left to bet against it? Research only; prices nothing, gates nothing.

    python -m scripts.live_total_study pull --sport nba     # ESPN, free, cached per day
    python -m scripts.live_total_study report               # every sport pulled

Data, all keyless ESPN: the scoreboard gives each final's per-period line score, and the
core ``/odds`` endpoint gives the pregame totals ESPN kept on the game (live-odds providers
and model sites dropped). The pregame number is the median across books after dropping any
book more than ``OUTLIER`` of the median away. ESPN BET's own open and close lines, where
ESPN has them (2024-25 on), are kept too, for the noisy-pregame check.

The finals are graded as books grade totals: overtime in, and the NHL shootout as one goal.

Sections, per checkpoint:
  1. the slope ``final - T = a + b * (so far - share * T)`` (``engine_runtime.faststart``,
     share fit on earlier days only), and per season;
  2. the same games read off ESPN BET's open and its close: if the pregame number is noisy,
     the start also says how wrong it was, so ``b`` rises above 1 as the number gets staler.
     Continuation of that kind is real for a live price that starts from the same number;
  3. the miss by scoreboard lead at the checkpoint (fouling, pulled goalies, garbage time):
     a + b * gap is fit on all games, then the mean miss of that rule per lead band;
  4. what a gap is worth: the miss SD, the gap at which a -110 ticket breaks even, and a
     walk-forward ceiling. The ceiling assumes the live book posts exactly pregame + start
     (rounded to the half point) and asks how often the fitted rule (seasons before only)
     beat it. That line is hypothetical: no live price is used or invented, so this is an
     upper bound on what continuation could be worth, not an edge.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import statistics
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import requests

from engine_runtime import faststart

SITE = "https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard"
CORE = "https://sports.core.api.espn.com/v2/sports/{core}/events/{eid}/competitions/{eid}/odds"
OUTLIER = 0.06  # a book this far (as a share of the median) from the median is dropped
SKIP_BOOKS = ("live", "accuscore", "teamrankings", "consensus", "numberfire")
BREAK_EVEN = 110 / 210  # a -110 ticket


@dataclass(frozen=True)
class Sport:
    path: str
    core: str
    cuts: dict[str, int]
    periods: int  # regulation periods
    leads: tuple[int, ...]  # lead-band edges at the checkpoint
    seasons: dict[str, tuple[str, str]]
    groups: str = ""


SPORTS: dict[str, Sport] = {
    "nfl": Sport(
        "football/nfl",
        "football/leagues/nfl",
        {"Q1": 1, "half": 2, "Q3": 3},
        4,
        (4, 8, 15),
        {
            "2022": ("2022-09-08", "2023-01-09"),
            "2023": ("2023-09-07", "2024-01-08"),
            "2024": ("2024-09-05", "2025-01-06"),
            "2025": ("2025-09-04", "2026-01-05"),
            "2026": ("2026-09-09", "2027-01-04"),
        },
    ),
    "nba": Sport(
        "basketball/nba",
        "basketball/leagues/nba",
        {"Q1": 1, "half": 2, "Q3": 3},
        4,
        (6, 11, 16, 21),
        {
            "2022-23": ("2022-10-18", "2023-04-09"),
            "2023-24": ("2023-10-24", "2024-04-14"),
            "2024-25": ("2024-10-22", "2025-04-13"),
            "2025-26": ("2025-10-21", "2026-04-12"),
            "2026-27": ("2026-10-20", "2027-04-11"),
        },
    ),
    "cbb": Sport(
        "basketball/mens-college-basketball",
        "basketball/leagues/mens-college-basketball",
        {"half": 1},
        2,
        (6, 11, 16),
        {
            "2023-24": ("2023-11-06", "2024-03-17"),
            "2024-25": ("2024-11-04", "2025-03-16"),
            "2025-26": ("2025-11-03", "2026-03-15"),
            "2026-27": ("2026-11-02", "2027-03-14"),
        },
        groups="50",
    ),
    "nhl": Sport(
        "hockey/nhl",
        "hockey/leagues/nhl",
        {"P1": 1, "P2": 2},
        3,
        (1, 2, 3),
        {
            "2022-23": ("2022-10-07", "2023-04-14"),
            "2023-24": ("2023-10-10", "2024-04-18"),
            "2024-25": ("2024-10-04", "2025-04-17"),
            "2025-26": ("2025-10-07", "2026-04-16"),
            "2026-27": ("2026-10-06", "2027-04-15"),
        },
    ),
}


def data_root() -> Path:
    return Path(os.environ.get("LIVE_STUDY_DIR", Path.home() / ".live_edge" / "total_study"))


# ---------------------------------------------------------------- pull


def _get(s: requests.Session, url: str, params: dict | None = None) -> dict | None:
    for attempt in range(4):
        try:
            r = s.get(url, params=params, timeout=20)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            time.sleep(1.5 * (attempt + 1))
    return None


def _line(raw: object) -> float | None:
    """A total from ESPN's odds blocks; signed values there are prices, not lines."""
    text = str(raw or "").strip()
    if not text or text[0] in "+-":
        return None
    try:
        v = float(text)
    except ValueError:
        return None
    return v if v > 0 else None


def parse_odds(doc: dict) -> dict:
    books: dict[str, float] = {}
    out: dict = {"books": books, "open": None, "close": None}
    for it in doc.get("items", []):
        name = str((it.get("provider") or {}).get("name", ""))
        if any(k in name.lower() for k in SKIP_BOOKS):
            continue
        ou = it.get("overUnder")
        if isinstance(ou, (int, float)) and ou > 0:
            books[name] = float(ou)
        if name == "ESPN BET":
            out["open"] = _line(((it.get("open") or {}).get("total") or {}).get("american"))
            out["close"] = _line(((it.get("close") or {}).get("total") or {}).get("american"))
    return out


def parse_game(ev: dict, day: str, season: str) -> dict | None:
    comp = (ev.get("competitions") or [{}])[0]
    st = (comp.get("status") or {}).get("type") or {}
    if not st.get("completed") or (ev.get("season") or {}).get("type") != 2:
        return None
    teams = comp.get("competitors") or []
    if len(teams) != 2:
        return None
    lines, scores = [], []
    for t in teams:
        lines.append([int(float(p.get("value", 0))) for p in t.get("linescores") or []])
        scores.append(int(float(t.get("score", 0))))
    home = 0 if teams[0].get("homeAway") == "home" else 1
    if len(lines[0]) != len(lines[1]) or sum(lines[0]) != scores[0] or sum(lines[1]) != scores[1]:
        return None
    return {
        "id": str(ev["id"]),
        "day": day,
        "season": season,
        "home": lines[home],
        "away": lines[1 - home],
    }


def pull_day(s: requests.Session, sp: Sport, day: date, season: str) -> list[dict]:
    params = {"dates": day.strftime("%Y%m%d"), "limit": "500"}
    if sp.groups:
        params["groups"] = sp.groups
    sb = _get(s, SITE.format(path=sp.path), params) or {}
    games = [g for ev in sb.get("events", []) if (g := parse_game(ev, day.isoformat(), season))]
    for g in games:
        doc = _get(s, CORE.format(core=sp.core, eid=g["id"])) or {}
        g.update(parse_odds(doc))
    return games


def pull(sport: str, seasons: Sequence[str] | None, workers: int = 12) -> int:
    sp = SPORTS[sport]
    root = data_root() / sport
    root.mkdir(parents=True, exist_ok=True)
    today = date.today()
    todo: list[tuple[date, str]] = []
    for season, (a, b) in sp.seasons.items():
        if seasons and season not in seasons:
            continue
        d, end = date.fromisoformat(a), min(date.fromisoformat(b), today - timedelta(days=1))
        while d <= end:
            if not (root / f"{d.isoformat()}.json.gz").exists():
                todo.append((d, season))
            d += timedelta(days=1)

    def one(item: tuple[date, str]) -> int:
        d, season = item
        with requests.Session() as s:
            games = pull_day(s, sp, d, season)
        tmp = root / f"{d.isoformat()}.tmp"
        with gzip.open(tmp, "wt") as fh:
            json.dump(games, fh)
        tmp.rename(root / f"{d.isoformat()}.json.gz")
        return len(games)

    with ThreadPoolExecutor(workers) as ex:
        n = sum(ex.map(one, todo))
    print(f"{sport}: {len(todo)} days pulled, {n} finals")
    return n


def load(sport: str) -> list[dict]:
    out: list[dict] = []
    for p in sorted((data_root() / sport).glob("*.json.gz")):
        with gzip.open(p, "rt") as fh:
            out += json.load(fh)
    return out


# ---------------------------------------------------------------- study


def consensus(books: dict[str, float]) -> float | None:
    vals = list(books.values())
    if not vals:
        return None
    med = statistics.median(vals)
    kept = [v for v in vals if abs(v - med) <= OUTLIER * med]
    return statistics.median(kept) if kept else None


@dataclass(frozen=True)
class Game:
    id: str
    day: str
    season: str
    pregame: float
    home: list[int]
    away: list[int]
    open: float | None
    close: float | None

    @property
    def final(self) -> float:
        return float(sum(self.home) + sum(self.away))

    def so_far(self, k: int) -> float:
        return float(sum(self.home[:k]) + sum(self.away[:k]))

    def lead(self, k: int) -> int:
        return abs(sum(self.home[:k]) - sum(self.away[:k]))

    def obs(self, k: int, pregame: float | None = None) -> faststart.Obs:
        return faststart.Obs(self.id, self.day, pregame or self.pregame, self.so_far(k), self.final)


def games(sport: str) -> list[Game]:
    sp = SPORTS[sport]
    out = []
    for g in load(sport):
        t = consensus(g.get("books") or {})
        if t is None or len(g["home"]) < sp.periods:
            continue
        out.append(
            Game(
                g["id"],
                g["day"],
                g["season"],
                t,
                g["home"],
                g["away"],
                g.get("open"),
                g.get("close"),
            )
        )
    return out


def walk(rows: Sequence[Game], k: int, pregame: str = "pregame") -> list[tuple[Game, float, float]]:
    """(game, share, gap) with share from earlier days only, as faststart.gaps."""
    obs = [g.obs(k, getattr(g, pregame)) for g in rows if getattr(g, pregame)]
    by_id = {g.id: g for g in rows}
    return [(by_id[o.game_id], s, gap) for o, s, gap in faststart.gaps(obs)]


def _fit(g: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """a, b, se(b), residual SD."""
    beta, cov, e = faststart._ols(np.c_[np.ones(len(g)), g], y)
    return float(beta[0]), float(beta[1]), float(math.sqrt(cov[1, 1])), float(e.std(ddof=2))


def _phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _z(p: float) -> float:
    lo, hi = -6.0, 6.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _phi(mid) < p else (lo, mid)
    return (lo + hi) / 2


def _bands(edges: Sequence[int]) -> list[tuple[int, int | None, str]]:
    out: list[tuple[int, int | None, str]] = []
    lo = 0
    for e in edges:
        out.append((lo, e - 1, f"{lo}-{e - 1}" if e - 1 > lo else f"{lo}"))
        lo = e
    out.append((lo, None, f"{lo}+"))
    return out


def md(head: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def study(sport: str) -> list[str]:
    sp = SPORTS[sport]
    gs = games(sport)
    if not gs:
        return [f"## {sport.upper()}", "", "no games pulled"]
    unit = "goals" if sport == "nhl" else "points"
    mean_miss = statistics.fmean(g.final - g.pregame for g in gs)
    out = [
        f"## {sport.upper()}",
        "",
        f"{len(gs):,} regular-season finals with a pregame total, {gs[0].day} to {gs[-1].day}; "
        f"final − pregame averages {mean_miss:+.2f} {unit}.",
        "",
        "**1. Slope, pooled (share fit on earlier days only)**",
        "",
    ]
    reads = [r for c, k in sp.cuts.items() if (r := faststart.read(c, [g.obs(k) for g in gs]))]
    out += md(faststart.HEAD, faststart.table(reads))

    out += ["", "**By season** (b ± SE; n)", ""]
    seasons = sorted({g.season for g in gs})
    rows = []
    for c, k in sp.cuts.items():
        w = walk(gs, k)
        cells = [c]
        for s in seasons:
            sub = [(g, gap) for g, _, gap in w if g.season == s]
            if len(sub) < 30:
                cells.append("–")
                continue
            a, b, se, _ = _fit(
                np.array([x[1] for x in sub]), np.array([x[0].final - x[0].pregame for x in sub])
            )
            cells.append(f"{b:.2f} ± {se:.2f} ({len(sub)})")
        rows.append(cells)
    out += md(["checkpoint", *seasons], rows)

    out += [
        "",
        "**2. Same games off ESPN BET's open vs its close vs the book median** "
        "(b ± SE; residual SD). A noisy pregame number pushes b above 1.",
        "",
    ]
    both = [g for g in gs if g.open and g.close]
    rows = []
    for c, k in sp.cuts.items():
        if len(both) < 100:
            rows.append([c, str(len(both)), "too few", "", ""])
            continue
        cells = [c, str(len(both))]
        for attr in ("open", "close", "pregame"):
            w = walk(both, k, attr)
            g_ = np.array([x[2] for x in w])
            y = np.array([x[0].final - getattr(x[0], attr) for x in w])
            _, b, se, sd = _fit(g_, y)
            cells.append(f"{b:.2f} ± {se:.2f}; {sd:.2f}")
        rows.append(cells)
    out += md(["checkpoint", "games", "ESPN BET open", "ESPN BET close", "book median"], rows)

    out += [
        "",
        "**3. Miss of the fitted rule by lead at the checkpoint** "
        f"(final − (T + a + b·gap), mean ± SE {unit}; games)",
        "",
    ]
    bands = _bands(sp.leads)
    rows = []
    for c, k in sp.cuts.items():
        w = walk(gs, k)
        g_ = np.array([x[2] for x in w])
        y = np.array([x[0].final - x[0].pregame for x in w])
        a, b, _, _ = _fit(g_, y)
        res = y - (a + b * g_)
        lead = np.array([x[0].lead(k) for x in w])
        cells = [c]
        for lo, hi, _ in bands:
            m = (lead >= lo) & (lead <= (hi if hi is not None else 10**6))
            if m.sum() < 30:
                cells.append(f"– ({int(m.sum())})")
                continue
            band = res[m]
            cells.append(
                f"{band.mean():+.2f} ± {band.std(ddof=1) / math.sqrt(m.sum()):.2f} ({int(m.sum())})"
            )
        rows.append(cells)
    out += md(["checkpoint", *(f"lead {lab}" for _, _, lab in bands)], rows)

    out += [
        "",
        "**4. What a gap is worth.** Break-even = the gap between the fair number and the live "
        "line a −110 ticket needs (Normal miss). Typical continuation = |b − 1| × SD of the "
        "surprise. Ceiling: walk-forward (each season's a, b and lead bands fit on the seasons "
        "before it) against a hypothetical live line of exactly T + surprise, rounded to the "
        "half point, betting only where the rule is at least the break-even gap off it; hit rate with pushes out. No live price is used. NHL lines move by juice, not half goals, so its ceiling is rough.",
        "",
    ]
    rows = []
    for c, k in sp.cuts.items():
        w = walk(gs, k)
        g_ = np.array([x[2] for x in w])
        y = np.array([x[0].final - x[0].pregame for x in w])
        a, b, _, sd = _fit(g_, y)
        be = sd * _z(BREAK_EVEN)
        typ = abs(b - 1) * float(g_.std())
        cells = [c, f"{sd:.2f}", f"{be:.2f}", f"{typ:.2f}", f"{_phi(typ / sd):.1%}"]
        cells += ceiling(w, sp, k, be)
        rows.append(cells)
    out += md(
        [
            "checkpoint",
            "miss SD",
            "break-even gap",
            "typical continuation",
            "its win rate",
            "ceiling: slope (edge ≥ break-even)",
            "ceiling: slope + lead band (edge ≥ break-even)",
        ],
        rows,
    )
    return out


def ceiling(
    w: Sequence[tuple[Game, float, float]], sp: Sport, k: int, min_edge: float = 0.0
) -> list[str]:
    """Walk-forward hit rate of the fitted rule against a hypothetical line of T + surprise,
    betting only where the rule sits at least ``min_edge`` off that line."""
    seasons = sorted({x[0].season for x in w})
    bands = _bands(sp.leads)
    tallies = {"slope": [0, 0], "lead": [0, 0]}
    for i, s in enumerate(seasons[1:], 1):
        train = [x for x in w if x[0].season in seasons[:i]]
        test = [x for x in w if x[0].season == s]
        g_ = np.array([x[2] for x in train])
        y = np.array([x[0].final - x[0].pregame for x in train])
        a, b, _, _ = _fit(g_, y)
        res = y - (a + b * g_)
        lead = np.array([x[0].lead(k) for x in train])
        adj = {}
        for lo, hi, lab in bands:
            m = (lead >= lo) & (lead <= (hi if hi is not None else 10**6))
            adj[lab] = float(res[m].mean()) if m.sum() >= 30 else 0.0
        for gm, _, gap in test:
            line = round((gm.pregame + gap) * 2) / 2
            if gm.final == line:
                continue
            band = next(
                lab
                for lo, hi, lab in bands
                if gm.lead(k) >= lo and (hi is None or gm.lead(k) <= hi)
            )
            for key, fair in (
                ("slope", gm.pregame + a + b * gap),
                ("lead", gm.pregame + a + b * gap + adj[band]),
            ):
                if abs(fair - line) < max(min_edge, 1e-9):
                    continue
                won = (fair > line) == (gm.final > line)
                tallies[key][0] += int(won)
                tallies[key][1] += 1
    out = []
    for key in ("slope", "lead"):
        wins, n = tallies[key]
        if n == 0:
            out.append("–")
            continue
        p = wins / n
        out.append(f"{p:.1%} ± {math.sqrt(p * (1 - p) / n):.1%} (n={n})")
    return out


def report(sports: Sequence[str]) -> str:
    lines = [
        "# Live-total study: pregame total + the start, four sports",
        "",
        "Research only. ESPN finals and ESPN-kept pregame totals; no live price is used. "
        "Totals graded as books grade them (overtime in; NHL shootout = 1 goal). "
        f"Break-even uses −110 ({BREAK_EVEN:.1%}).",
        "",
    ]
    for s in sports:
        lines += [*study(s), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="live_total_study")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull")
    p.add_argument("--sport", choices=tuple(SPORTS), action="append")
    p.add_argument("--season", action="append")
    p.add_argument("--workers", type=int, default=12)
    r = sub.add_parser("report")
    r.add_argument("--sport", choices=tuple(SPORTS), action="append")
    r.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    sports = args.sport or list(SPORTS)
    if args.cmd == "pull":
        for s in sports:
            pull(s, args.season, args.workers)
        return 0
    text = report([s for s in sports if (data_root() / s).exists()])
    if args.out:
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
