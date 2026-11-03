"""§4.e / decision 10.1: buy 2024-25 game-line closes from the Odds API historical endpoint.

One bulk ``/historical/.../odds`` snapshot per distinct tip cluster at T-5 (floored to the
10-minute snapshot grid) covers every game tipping in that window; each snapshot is written
as its own immutable ``close`` shard under the slate date, so `fit_width.py` and the audit
read it exactly like a live close pass. ``--dense`` adds the 10-minute featured grid from
T-3h on named slates (stale-line frequency / persistence / per-bet CLV SD for the MDEs).

Nothing here spends until the plan has been printed and ``--yes`` given; a rerun skips
snapshots already in the day's manifest, so an interrupted pull resumes.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Protocol

from cbb_engine.config import SPORT
from cbb_engine.data import espn
from cbb_engine.data.capture import rows_from_events, slate_date_of
from cbb_engine.market import consensus
from engine_runtime import quotes
from engine_runtime.odds_client import HIST_MULT

HIST_PREFIX = "hist"
CLOSE_MARKETS: tuple[str, ...] = ("spreads", "totals")
GRID_MIN = 10
CLOSE_BEFORE_MIN = 5
DENSE_WINDOW_H = 3.0
CSV_FIELDS = (
    "game_date", "event_id", "home", "away", "tip_utc", "close_total", "final_total",
    "paired_books", "spread_abs", "close_shard", "home_spread", "home_pts", "away_pts", "h1_home", "h1_away",
)


def _parse(stamp: str) -> datetime | None:
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _floor(moment: datetime, step: int = GRID_MIN) -> datetime:
    return moment.replace(second=0, microsecond=0, minute=(moment.minute // step) * step)


def close_times(tips: Iterable[str]) -> list[str]:
    """One snapshot per distinct T-5 grid cell; the snapshot at or before that time is the close."""
    out: set[datetime] = set()
    for t in tips:
        dt = _parse(t)
        if dt is not None:
            out.add(_floor(dt - timedelta(minutes=CLOSE_BEFORE_MIN)))
    return [_iso(m) for m in sorted(out)]


def dense_times(tips: Iterable[str], *, window_h: float = DENSE_WINDOW_H) -> list[str]:
    """The 10-minute featured grid from T-window to T-5, unioned across the slate's tips."""
    out: set[datetime] = set()
    for t in tips:
        dt = _parse(t)
        if dt is None:
            continue
        m = _floor(dt - timedelta(hours=window_h))
        last = _floor(dt - timedelta(minutes=CLOSE_BEFORE_MIN))
        while m <= last:
            out.add(m)
            m += timedelta(minutes=GRID_MIN)
    return [_iso(m) for m in sorted(out)]


def plan(
    tips_by_day: dict[str, list[str]],
    *,
    dense_days: Sequence[str] = (),
    markets: tuple[str, ...] = CLOSE_MARKETS,
    regions: int = 1,
) -> dict[str, object]:
    """Snapshots and credits the pull will cost, per day and in total, before any spend."""
    per_call = HIST_MULT * len(markets) * regions
    days: dict[str, dict[str, int]] = {}
    for day, tips in sorted(tips_by_day.items()):
        n_close = len(close_times(tips))
        n_dense = len(dense_times(tips)) if day in dense_days else 0
        days[day] = {
            "games": len(tips), "close_snapshots": n_close, "dense_snapshots": n_dense,
            "credits": per_call * (n_close + n_dense),
        }
    return {
        "markets": list(markets), "credits_per_snapshot": per_call, "days": days,
        "snapshots": sum(d["close_snapshots"] + d["dense_snapshots"] for d in days.values()),
        "credits": sum(d["credits"] for d in days.values()),
    }


def _pass_id(kind: str, at: str) -> str:
    return f"{HIST_PREFIX}_{kind}_{at.replace('-', '').replace(':', '')}"


def _in_window(ev: dict[str, object], at: datetime, horizon: timedelta) -> bool:
    tip = _parse(str(ev.get("commence_time", "")))
    return tip is not None and at < tip <= at + horizon


class HistoricalSource(Protocol):
    def historical_board(
        self, markets: tuple[str, ...], *, at: str, pass_id: str = "", **params: str
    ) -> tuple[str, list[dict[str, object]]]: ...


def pull_day(
    client: HistoricalSource,
    root: Path,
    day: str,
    tips: Iterable[str],
    *,
    dense: bool = False,
    markets: tuple[str, ...] = CLOSE_MARKETS,
) -> dict[str, int]:
    """Buy the day's close (and optionally dense) snapshots; skip any already in the manifest."""
    tips = list(tips)
    done = {str(m.get("pass_id", "")) for m in quotes.read_manifest(root, SPORT, day)}
    counts = {"snapshots": 0, "skipped": 0, "rows": 0, "events": 0}
    jobs = [("close", at, timedelta(minutes=GRID_MIN + CLOSE_BEFORE_MIN)) for at in close_times(tips)]
    if dense:
        jobs += [("featured", at, timedelta(hours=DENSE_WINDOW_H)) for at in dense_times(tips)]
    for kind, at, horizon in jobs:
        pid = _pass_id(kind, at)
        if pid in done:
            counts["skipped"] += 1
            continue
        at_dt = _parse(at)
        assert at_dt is not None
        stamp, evs = client.historical_board(markets, at=at, pass_id=pid)
        if not stamp:
            continue
        evs = [e for e in evs if _in_window(e, at_dt, horizon) and slate_date_of(str(e.get("commence_time", ""))) == day]
        rows = rows_from_events(evs, captured_at=stamp, pass_id=pid, pass_kind=kind)
        teams = {str(e.get("id", "")): [str(e.get("home_team", "")), str(e.get("away_team", ""))] for e in evs}
        written = quotes.write_shard(
            root, SPORT, day, rows, pass_id=pid, pass_kind=kind,
            extra={"hist": True, "requested_at": at, "teams": teams},
        )
        counts["snapshots"] += 1
        if written is not None:
            counts["rows"] += len(rows)
            counts["events"] += len(evs)
    return counts


_FOLD = {
    "state": "st", "saint": "st", "st.": "st", "north": "n", "south": "s", "east": "e", "west": "w",
    "northern": "n", "southern": "s", "eastern": "e", "western": "w", "central": "c", "university": "",
    "&": "", "and": "", "the": "",
}


# Odds API names ESPN spells differently, as whole tokens.
_ALIAS = {"umkc": "kansas city", "kangaroos": "roos"}


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _tokens(name: str) -> frozenset[str]:
    """Team-name tokens with the Odds API / ESPN abbreviation differences folded (State↔St, Saint↔St, North↔N)."""
    out = set()
    for word in name.lower().replace("-", " ").replace("(", " ").replace(")", " ").split():
        for raw in _ALIAS.get(word, word).split():
            tok = _FOLD.get(raw, raw)
            tok = "".join(ch for ch in tok if ch.isalnum())
            if tok:
                out.add(tok)
    return frozenset(out)


_WEAK = {"st", "n", "s", "e", "w", "c"}


def _sim(a: str, b: str) -> float:
    """Jaccard over folded tokens; 0 unless the names share a token beyond the direction/State fillers."""
    ta, tb = _tokens(a), _tokens(b)
    if not (ta & tb) - _WEAK:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def match_final(home: str, away: str, games: Sequence[espn.Game]) -> tuple[int, int] | None:
    """The day's ESPN final for (home, away), as (away, home) points; see ``scores``."""
    hit = scores(home, away, games)
    return hit[0] if hit else None


def scores(
    home: str, away: str, games: Sequence[espn.Game]
) -> tuple[tuple[int, int], tuple[int, int] | None] | None:
    """(final, first half) of the day's finished game for (home, away), each as (away, home) points
    in the caller's orientation, even when ESPN lists the teams the other way round."""
    hit = match_oriented(home, away, [g for g in games if g.final])
    if hit is None:
        return None
    game, swapped = hit
    final = game.final
    assert final is not None
    h1 = (game.away_halves[0], game.home_halves[0]) if len(game.home_halves) >= 2 else None
    if swapped:
        final, h1 = (final[1], final[0]), ((h1[1], h1[0]) if h1 else None)
    return final, h1


def match_oriented(home: str, away: str, games: Sequence[espn.Game]) -> tuple[espn.Game, bool] | None:
    """The day's ESPN game for (home, away), and whether ESPN lists the teams the other way round
    (at neutral sites the Odds API's home team can be ESPN's away team). The reversed match is the
    fallback and needs each team to share at least half its tokens, so a weak overlap such as
    Kansas / Kansas City never pairs a game the other way round."""
    game = match_game(home, away, games)
    if game is not None:
        return game, False
    game = match_game(away, home, games, min_side=0.5)
    return (game, True) if game is not None else None


def match_game(
    home: str, away: str, games: Sequence[espn.Game], *, min_side: float = 0.0
) -> espn.Game | None:
    """The day's ESPN game for (home, away): exact normalized match, else the unique best token match
    where both sides share a token (score ≥ 1.0 of 2 and ahead of the runner-up), each side's
    similarity above zero and at least ``min_side``."""
    exact = {(_norm(g.home), _norm(g.away)): g for g in games}
    hit = exact.get((_norm(home), _norm(away)))
    if hit:
        return hit
    scored = sorted(
        (
            (_sim(home, g.home) + _sim(away, g.away), g)
            for g in games
            if min(_sim(home, g.home), _sim(away, g.away)) > 0
            and min(_sim(home, g.home), _sim(away, g.away)) >= min_side
        ),
        key=lambda t: -t[0],
    )
    if not scored or scored[0][0] < 1.0:
        return None
    if len(scored) > 1 and scored[1][0] >= scored[0][0] - 1e-9:
        return None
    return scored[0][1]


def latest_scoreboard(data_dir: Path, day: str) -> list[espn.Game]:
    files = sorted((data_dir / "espn" / day).glob("scoreboard_*.json"))
    if not files:
        return []
    payload = json.loads(files[-1].read_text())
    return espn.parse_scoreboard(payload) if isinstance(payload, dict) else []


def export_day(root: Path, data_dir: Path, day: str) -> list[dict[str, object]]:
    """Join the day's hist close shards to ESPN finals: one row per game for `fit_width.py` and
    `fit_margin.py`.

    The close is the last hist close shard captured before the game's tip; ``close_total``,
    ``spread_abs`` and ``home_spread`` (home side's line) are consensus lines (execution books
    excluded), ``paired_books`` the total's; ``h1_*`` are first-half points.
    """
    man = quotes.read_manifest(root, SPORT, day)
    teams: dict[str, list[str]] = {}
    for m in man:
        tm = m.get("teams")
        if m.get("hist") and m.get("pass_kind") == "close" and isinstance(tm, dict):
            teams.update({str(k): [str(x) for x in v] for k, v in tm.items()})
    shards = [q for q in quotes.read_day(root, SPORT, day) if q.pass_kind == "close" and q.pass_id.startswith(HIST_PREFIX)]
    by_event: dict[str, list[quotes.Quote]] = {}
    for q in shards:
        by_event.setdefault(q.event_id, []).append(q)
    games = latest_scoreboard(data_dir, day)
    out: list[dict[str, object]] = []
    for eid, qs in sorted(by_event.items()):
        tip = _parse(qs[0].tip_utc)
        pre = [q for q in qs if (c := _parse(q.captured_at)) is not None and tip is not None and c < tip]
        if not pre:
            continue
        last = max(q.pass_id for q in pre)
        anchors = consensus.build([q for q in pre if q.pass_id == last], shard=last)
        total = [a for a in anchors if a.market == "game_total" and a.paired_books]
        if not total:
            continue
        close_total = consensus.consensus_line(anchors, eid, "game_total")
        ats = [abs(a.line) for a in anchors if a.market == "game_ats" and a.line is not None and a.paired_books]
        spread = median(ats) if ats else None
        home_lines = [
            a.line if a.sides[0] == "home" else -a.line
            for a in anchors
            if a.market == "game_ats" and a.line is not None and a.paired_books and a.sides[0] in ("home", "away")
        ]
        home, away = teams.get(eid, ["", ""])
        final, h1 = scores(home, away, games) or (None, None)
        out.append({
            "game_date": day, "event_id": eid, "home": home, "away": away, "tip_utc": qs[0].tip_utc,
            "close_total": close_total,
            "final_total": sum(final) if final else "",
            "paired_books": max(a.paired_books for a in total),
            "spread_abs": abs(spread) if spread is not None else "",
            "close_shard": last,
            "home_spread": median(home_lines) if home_lines else "",
            "home_pts": final[1] if final else "", "away_pts": final[0] if final else "",
            "h1_home": h1[1] if h1 else "", "h1_away": h1[0] if h1 else "",
        })
    return out


def write_csv(rows: Iterable[dict[str, object]], path: Path) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def days_between(start: Date, end: Date) -> list[str]:
    n = (end - start).days
    return [(start + timedelta(days=i)).isoformat() for i in range(max(0, n) + 1)]


def tips_from_games(games: Iterable[espn.Game], day: str) -> list[str]:
    return [g.start_utc for g in games if slate_date_of(g.start_utc) == day]


__all__ = [
    "CLOSE_MARKETS", "CSV_FIELDS", "HIST_PREFIX", "close_times", "days_between", "dense_times", "match_game",
    "export_day", "latest_scoreboard", "match_final", "match_oriented", "plan", "pull_day", "scores", "tips_from_games", "write_csv",
]
