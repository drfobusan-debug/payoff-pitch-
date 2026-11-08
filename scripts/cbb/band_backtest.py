"""Grade the CBB bands on the bought 2024-25 closes, at their recorded prices (no credits).

Each game's last hist close pass before tip gives one row per spread/total side: the
median line across books, the median price at that line, the no-vig consensus there.
Finals and AP ranks come from ESPN scoreboards; Torvik rank, tempo third and last result from Torvik's archived table and game
logs dated before the slate; conference membership from Torvik's team results.
The hist pull has no moneylines and no openers, so the ML and line-move bands stay empty.

    python scripts/cbb/band_backtest.py --root ~/.cbb_engine/runtime --espn ~/.cbb_engine/data \
        --logs getgamestats_2025.csv --results 2025_team_results.csv [--games out.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

from cbb_engine import audit, bands, history
from cbb_engine.config import SPORT
from cbb_engine.data import espn, torvik
from cbb_engine.market import consensus
from cbb_engine.output import teamstats
from engine_runtime import ledger, quotes


def board(espn_dir: Path, day: Date) -> list[espn.Game]:
    """The day's ESPN board: the engine's archive layout, else a flat ``<day>.json``."""
    games = history.latest_scoreboard(espn_dir, day.isoformat())
    flat = espn_dir / f"{day.isoformat()}.json"
    if not games and flat.exists():
        games = espn.parse_scoreboard(json.loads(flat.read_text()))
    return games


def hist_teams(root: Path, day: str) -> dict[str, tuple[str, str]]:
    teams: dict[str, tuple[str, str]] = {}
    for m in quotes.read_manifest(root, SPORT, day):
        tm = m.get("teams")
        if m.get("hist") and m.get("pass_kind") == "close" and isinstance(tm, dict):
            teams.update({str(k): (str(v[0]), str(v[1])) for k, v in tm.items()})
    return teams


def close_rows(root: Path, day: str, teams: dict[str, tuple[str, str]], games: list[espn.Game]) -> list[ledger.Row]:
    by_event: dict[str, list[quotes.Quote]] = defaultdict(list)
    for q in quotes.read_day(root, SPORT, day):
        if q.pass_kind == "close" and q.pass_id.startswith(history.HIST_PREFIX):
            by_event[q.event_id].append(q)
    rows: list[ledger.Row] = []
    for eid, qs in sorted(by_event.items()):
        tip = history._parse(qs[0].tip_utc)
        pre = [q for q in qs if (c := history._parse(q.captured_at)) is not None and tip is not None and c < tip]
        if not pre or eid not in teams:
            continue
        final = history.match_final(*teams[eid], games)
        if final is None:
            continue
        last = max(q.pass_id for q in pre)
        snap = [q for q in pre if q.pass_id == last]
        anchors = consensus.build(snap, shard=last)
        for (market, side), line in bands.main_lines(snap).items():
            a = next((a for a in anchors if a.market == market and a.line == line and side in a.p_fair), None)
            prices = sorted(q.american for q in snap if q.market == market and q.side == side and q.line == line)
            if a is None or not prices:
                continue
            r = ledger.Row(
                slate_date=day, event_id=eid, tip_utc=qs[0].tip_utc, market=market, side=side, line=line,
                book="median", pass_id=last, pass_kind="close", bet_of_record=True, status="backtest",
                american=prices[len(prices) // 2], p_fair=a.p_fair[side], paired_books=a.paired_books,
            )
            r.outcome = audit.outcome(r, final)
            r.pnl = audit.pnl(r, r.outcome)
            rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True, help="runtime root holding quotes/cbb/<day>/")
    ap.add_argument("--espn", type=Path, required=True, help="ESPN scoreboards: <dir>/espn/<day>/scoreboard_*.json or <dir>/<day>.json")
    ap.add_argument("--logs", type=Path, required=True, help="Torvik getgamestats.php?year=YYYY&csv=1")
    ap.add_argument("--results", type=Path, required=True, help="Torvik YYYY_team_results.csv (conference membership only)")
    ap.add_argument("--start", default="2024-11-04")
    ap.add_argument("--end", default="2025-04-08")
    ap.add_argument("--games", type=Path, help="write one row per graded game (close lines, finals, context) here")
    args = ap.parse_args()
    logs = bands.parse_logs(teamstats.read_logs(args.logs))
    confs = {r.get("team", ""): r.get("conf", "") for r in torvik.read_csv(args.results)}
    start, end = Date.fromisoformat(args.start), Date.fromisoformat(args.end)
    boards: dict[Date, list[espn.Game]] = {}
    day = start - timedelta(days=bands.REST_LOOKBACK)
    while day <= end:
        boards[day] = board(args.espn, day)
        day += timedelta(days=1)
    rows: list[ledger.Row] = []
    ctx: dict[str, bands.GameCtx] = {}
    out: list[dict[str, object]] = []
    day = start
    while day <= end:
        teams = hist_teams(args.root, day.isoformat())
        if teams:
            got = close_rows(args.root, day.isoformat(), teams, boards[day])
            c = bands.context(teams, day, boards, logs, confs)
            rows += got
            ctx.update(c)
            for eid in sorted({r.event_id for r in got}):
                lines = {(r.market, r.side): r.line for r in got if r.event_id == eid}
                g = history.match_game(*teams[eid], boards[day])
                out.append({
                    "game_date": day.isoformat(), "event_id": eid, "home": teams[eid][0], "away": teams[eid][1],
                    "home_line": lines.get(("game_ats", "home")), "total_line": lines.get(("game_total", "over")),
                    "home_pts": g.final[1] if g and g.final else "", "away_pts": g.final[0] if g and g.final else "",
                    "neutral": c[eid].neutral, "conference": c[eid].conference,
                    "home_net_rank": c[eid].home.net_rank, "away_net_rank": c[eid].away.net_rank,
                })
        day += timedelta(days=1)
    print(f"{len(out)} games, {len(rows)} graded rows, {args.start} to {args.end}")
    print(bands.render(bands.read_bands(rows, ctx)))
    if args.games and out:
        with args.games.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
