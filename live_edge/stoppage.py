"""Live-total stoppage test: is the live total already the fair final? Paper only.

    surprise = points so far - share x pregame total
    fair     = pregame total + b x surprise
    gap      = live total - fair

Share, b, the screen and the margin are frozen per sport and never refit once live
prices are seen. Only the primary stoppage (the half; the end of period 2 in the
NHL) makes tickets; the other stoppages are written to the tape only. A gap counts
when the median book is past the screen and the next poll still has the same score
and a gap past the screen on the same side. A game whose lead at the stoppage is
at or past the sport's margin is a pass. One ticket a game, priced at the better of
DraftKings/BetMGM on the confirming snapshot; a missing price stays missing. Both the
pregame number and the live total are medians of at least three fresh two-sided books.

Where the final-score study found the lead at the stoppage moves the final (CBB), a
lead-adjusted fair number is written next to the frozen one and graded on each ticket.
It never makes or blocks a ticket.

    python -m live_edge.stoppage run --sport nba --hours 7
    python -m live_edge.stoppage report --sport nba
    python -m live_edge.stoppage run --sport cbb --hours 14 --email   # email the report at the end
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from live_edge import mail
from live_edge.engine import data_dir, match_event, now_iso
from live_edge.espn import LiveGame, fetch_scoreboard
from live_edge.oddsapi import EventQuotes, LiveOddsClient, age_seconds

log = logging.getLogger(__name__)

EXEC_BOOKS = ("draftkings", "betmgm")
CLOSE_MIN = 45.0  # the pregame number is the newest median within this many minutes of start
PRE_EVERY_MIN = 10.0
STALE_SECS = 300.0  # a book's total older than this is left out of the median
MIN_BOOKS = 3  # fewer fresh two-sided books than this and there is no median


@dataclass(frozen=True)
class Rule:
    share: float
    b: float
    screen: float
    margin: int  # lead at the stoppage at or past this is a pass
    target: int  # settled gaps one slate must show before buying history
    labels: dict[int, str]
    primary: int  # period whose end is the primary stoppage


_FOOTBALL = {1: "end of Q1", 2: "half", 3: "end of Q3"}
HEARTBEAT_POLLS = 20  # one "waiting" line per ~30 min at the default 90 s interval

RULES: dict[str, Rule] = {
    # CFB share is the half share fit in the CFB fast-start study (#636, 49 games).
    "cfb": Rule(0.518, 1.0, 4.0, 17, 8, _FOOTBALL, 2),
    "nfl": Rule(0.51, 1.0, 3.0, 17, 6, _FOOTBALL, 2),
    "nba": Rule(0.50, 1.0, 5.0, 20, 8, _FOOTBALL, 2),
    # CBB screen 3 from 2026-11-09 (4 through 11-07: 1 of 40 halftime gaps reached it).
    "cbb": Rule(0.47, 1.12, 3.0, 15, 8, {1: "half"}, 1),
    "nhl": Rule(0.64, 1.0, 1.0, 3, 8, {1: "end of P1", 2: "end of P2"}, 2),
}

# Graded only. (lead from, points) added to the frozen fair number, centered on the games
# under the pass margin. CBB half, 14,018 regular-season games 2023-24 to 2026-27
# (scripts/live_total_study.py): lead 0-5 +0.39 +/- 0.15, 6-10 -0.18 +/- 0.18,
# 11-14 -0.73 +/- 0.24. NFL, NBA and NHL bands are within 1 SE of zero, so none.
LEAD_ADJ: dict[str, tuple[tuple[int, float], ...]] = {
    "cbb": ((0, 0.4), (6, -0.2), (11, -0.7)),
}


def lead_adj(sport: str, lead: int) -> float | None:
    bands = LEAD_ADJ.get(sport)
    if not bands:
        return None
    return next(adj for start, adj in reversed(bands) if lead >= start)


def stoppage(rule: Rule, g: LiveGame) -> tuple[str, bool] | None:
    """(label, primary) when the game sits at a recorded break, else None."""
    if g.status != "in":
        return None
    at_break = g.detail in ("STATUS_HALFTIME", "STATUS_END_PERIOD") or g.state.clock_secs == 0
    label = rule.labels.get(g.state.period)
    if not at_break or label is None:
        return None
    return label, g.state.period == rule.primary


def _ts(raw: str) -> datetime | None:
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _decimal(american: float) -> float:
    return 1 + (american / 100 if american > 0 else 100 / -american)


def books(ev: EventQuotes, now: datetime) -> dict[str, dict[str, object]]:
    """Each book's main total: line, Over and Under price, quote age in seconds."""
    out: dict[str, dict[str, object]] = {}
    for q in ev.quotes:
        if q.market != "total" or q.line is None:
            continue
        row = out.setdefault(q.book, {"line": q.line, "over": None, "under": None, "age": None})
        if row["line"] != q.line:
            continue
        row[q.side] = q.price
        age = age_seconds(q.updated, now)
        row["age"] = None if age is None else round(age)
    return out


def median_line(rows: dict[str, dict[str, object]]) -> tuple[float | None, int]:
    lines = [
        float(r["line"])  # type: ignore[arg-type]
        for r in rows.values()
        if r["over"] is not None
        and r["under"] is not None
        and (r["age"] is None or float(r["age"]) <= STALE_SECS)  # type: ignore[arg-type]
    ]
    return (statistics.median(lines) if len(lines) >= MIN_BOOKS else None), len(lines)


def fair(rule: Rule, pregame: float, so_far: int) -> tuple[float, float]:
    surprise = so_far - rule.share * pregame
    return surprise, pregame + rule.b * surprise


def best_book(rows: dict[str, dict[str, object]], side: str) -> tuple[str, float, float] | None:
    """Better of DraftKings/BetMGM for ``side``: the kinder line first, then the price."""
    picks = []
    for bk in EXEC_BOOKS:
        r = rows.get(bk)
        if not r or r.get(side) is None:
            continue
        line, price = float(r["line"]), float(r[side])  # type: ignore[arg-type]
        picks.append(((line if side == "under" else -line), _decimal(price), bk, line, price))
    if not picks:
        return None
    _, _, bk, line, price = max(picks)
    return bk, line, price


def settle(side: str, line: float, final: int, price: float | None) -> tuple[str, float | None]:
    if final == line:
        return "push", 0.0
    won = final < line if side == "under" else final > line
    if price is None:
        return ("win" if won else "loss"), None
    return ("win", round(_decimal(price) - 1, 4)) if won else ("loss", -1.0)


class Study:
    def __init__(self, sport: str, root: Path | None = None) -> None:
        self.sport = sport
        self.rule = RULES[sport]
        self.dir = (root or data_dir()) / "stoppage" / sport
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.state: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        try:
            raw = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            raw = {}
        return {k: raw.get(k, {}) for k in ("pregame", "last", "pre_polled")}

    def _save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True))
        tmp.replace(self.state_path)

    def _append(self, name: str, row: dict) -> None:
        with (self.dir / name).open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    def read(self, name: str) -> list[dict]:
        path = self.dir / name
        if not path.exists():
            return []
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

    def needs_board(self, games: list[LiveGame], now: datetime) -> bool:
        if any(stoppage(self.rule, g) for g in games):
            return True
        last = _ts(self.state["pre_polled"].get("at", "")) if self.state["pre_polled"] else None
        if last and (now - last).total_seconds() < PRE_EVERY_MIN * 60:
            return False
        return any(self._closing(g, now) for g in games)

    @staticmethod
    def _closing(g: LiveGame, now: datetime) -> bool:
        ko = _ts(g.kickoff) if g.status == "pre" and g.kickoff else None
        return bool(ko and 0 < (ko - now).total_seconds() <= CLOSE_MIN * 60)

    def tick(self, games: list[LiveGame], events: list[EventQuotes] | None, now: datetime) -> dict:
        stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        out = {"snapshots": 0, "tickets": 0, "graded": 0}
        tickets = {t["espn_id"] for t in self.read("tickets.jsonl")}
        graded = {g["espn_id"] for g in self.read("grades.jsonl")}
        if events is not None and any(self._closing(g, now) for g in games):
            self.state["pre_polled"] = {"at": stamp}
        for g in games:
            ev = match_event(g, events) if events else None
            if ev is not None and self._closing(g, now):
                rows = books(ev, now)
                line, n = median_line(rows)
                if line is not None:
                    self.state["pregame"][g.event_id] = {"line": line, "at": stamp, "books": n}
                    self._append(
                        "tape.jsonl",
                        {
                            "kind": "pregame",
                            "at": stamp,
                            "espn_id": g.event_id,
                            "odds_id": ev.event_id,
                            "matchup": g.matchup,
                            "start": g.kickoff,
                            "median": line,
                            "n_books": n,
                            "books": rows,
                        },
                    )
            stop = stoppage(self.rule, g)
            if stop and ev is not None:
                snap = self._snapshot(g, ev, stop, stamp, now)
                self._append("tape.jsonl", snap)
                out["snapshots"] += 1
                if stop[1]:
                    prev = self.state["last"].get(g.event_id)
                    if g.event_id not in tickets and self._confirms(prev, snap):
                        self._append("tickets.jsonl", self._ticket(snap))
                        tickets.add(g.event_id)
                        out["tickets"] += 1
                    self.state["last"][g.event_id] = snap
            if g.status == "post" and g.event_id in tickets and g.event_id not in graded:
                final = g.state.home_score + g.state.away_score
                for t in self.read("tickets.jsonl"):
                    if t["espn_id"] == g.event_id:
                        self._append("grades.jsonl", self._grade(t, final, stamp))
                graded.add(g.event_id)
                out["graded"] += 1
        self._save()
        return out

    def _snapshot(
        self, g: LiveGame, ev: EventQuotes, stop: tuple[str, bool], stamp: str, now: datetime
    ) -> dict:
        rows = books(ev, now)
        live, n = median_line(rows)
        home, away = g.state.home_score, g.state.away_score
        pre = self.state["pregame"].get(g.event_id)
        snap: dict = {
            "kind": "stoppage",
            "at": stamp,
            "espn_id": g.event_id,
            "odds_id": ev.event_id,
            "matchup": g.matchup,
            "stoppage": stop[0],
            "primary": stop[1],
            "home": home,
            "away": away,
            "margin": abs(home - away),
            "pass_margin": abs(home - away) >= self.rule.margin,
            "pregame": pre["line"] if pre else None,
            "pregame_at": pre["at"] if pre else None,
            "live": live,
            "n_books": n,
            "surprise": None,
            "fair": None,
            "gap": None,
            "fair_lead": None,
            "gap_lead": None,
            "books": rows,
        }
        if stop[1] and pre and live is not None:
            surprise, fair_total = fair(self.rule, float(pre["line"]), home + away)
            snap.update(
                surprise=round(surprise, 3),
                fair=round(fair_total, 3),
                gap=round(live - fair_total, 3),
            )
            adj = lead_adj(self.sport, abs(home - away))
            if adj is not None:
                snap.update(
                    fair_lead=round(fair_total + adj, 3),
                    gap_lead=round(live - fair_total - adj, 3),
                )
        return snap

    def _confirms(self, prev: dict | None, cur: dict) -> bool:
        if not prev or cur["pass_margin"] or cur["gap"] is None or prev.get("gap") is None:
            return False
        if (prev["home"], prev["away"]) != (cur["home"], cur["away"]):
            return False
        s = self.rule.screen
        return (prev["gap"] >= s and cur["gap"] >= s) or (prev["gap"] <= -s and cur["gap"] <= -s)

    def _ticket(self, snap: dict) -> dict:
        side = "under" if snap["gap"] > 0 else "over"
        naive = (
            "under"
            if snap["live"] > snap["pregame"]
            else "over"
            if snap["live"] < snap["pregame"]
            else None
        )
        gl, s = snap.get("gap_lead"), self.rule.screen
        lead = None if gl is None else "under" if gl >= s else "over" if gl <= -s else None
        row = {
            "espn_id": snap["espn_id"],
            "odds_id": snap["odds_id"],
            "matchup": snap["matchup"],
            "at": snap["at"],
            "stoppage": snap["stoppage"],
            "home": snap["home"],
            "away": snap["away"],
            "pregame": snap["pregame"],
            "live": snap["live"],
            "fair": snap["fair"],
            "gap": snap["gap"],
            "share": self.rule.share,
            "b": self.rule.b,
            "screen": self.rule.screen,
            "side": side,
            "naive_side": naive,
            "fair_lead": snap.get("fair_lead"),
            "gap_lead": gl,
            "lead_side": lead,
            "paper_only": True,
        }
        for key, which in (("", side), ("naive_", naive), ("lead_", lead)):
            pick = best_book(snap["books"], which) if which else None
            # No DK/BetMGM quote: graded at the median line, its price left missing.
            row[f"{key}book"], row[f"{key}line"], row[f"{key}price"] = pick or (
                None,
                snap["live"] if which else None,
                None,
            )
        return row

    @staticmethod
    def _grade(t: dict, final: int, stamp: str) -> dict:
        row = {"espn_id": t["espn_id"], "final": final, "graded_at": stamp}
        for key in ("", "naive_", "lead_"):
            side, line = t.get(f"{key}side"), t.get(f"{key}line")
            if side is None or line is None:
                row[f"{key}result"], row[f"{key}units"] = None, None
                continue
            row[f"{key}result"], row[f"{key}units"] = settle(side, line, final, t[f"{key}price"])
        return row

    def report(self) -> str:
        r = self.rule
        tape = [s for s in self.read("tape.jsonl") if s["kind"] == "stoppage" and s["primary"]]
        seen = {s["espn_id"] for s in tape}
        priced = {s["espn_id"] for s in tape if s["live"] is not None}
        passed = {s["espn_id"] for s in tape if s["pass_margin"]}
        no_pre = {s["espn_id"] for s in tape if s["pregame"] is None}
        tickets = self.read("tickets.jsonl")
        grades = {g["espn_id"]: g for g in self.read("grades.jsonl")}
        lines = [
            f"{self.sport.upper()} live-total test (paper only). share {r.share}, b {r.b}, "
            f"screen {r.screen:g}, pass at a lead of {r.margin}+",
            f"games at the primary stoppage: {len(seen)}; with a live total: {len(priced)}; "
            f"no timely pregame number: {len(no_pre)}; passed on the margin: {len(passed)}",
            f"confirmed gaps (tickets): {len(tickets)}; settled: "
            f"{sum(t['espn_id'] in grades for t in tickets)}",
        ]
        named = [("", "screened"), ("naive_", "naive (live > pregame: under)")]
        if self.sport in LEAD_ADJ:
            named.append(("lead_", "lead-adjusted (graded only; blank when inside the screen)"))
        for key, name in named:
            res = [grades[t["espn_id"]] for t in tickets if t["espn_id"] in grades]
            w = sum(g.get(f"{key}result") == "win" for g in res)
            lo = sum(g.get(f"{key}result") == "loss" for g in res)
            p = sum(g.get(f"{key}result") == "push" for g in res)
            units = [g[f"{key}units"] for g in res if g.get(f"{key}units") is not None]
            missing = sum(
                g.get(f"{key}result") is not None and g.get(f"{key}units") is None for g in res
            )
            roi = f"{sum(units) / len(units):+.1%}" if units else "n/a"
            lines.append(
                f"{name}: {w}-{lo}-{p}, ROI {roi} on {len(units)} priced; "
                f"{missing} with no DK/BetMGM price"
            )
        settled = sum(t["espn_id"] in grades for t in tickets)
        verdict = (
            f"clears the count ({settled} of {r.target} settled gaps): buy the history"
            if settled >= r.target
            else f"under its count ({settled} of {r.target} settled gaps): stop"
        )
        lines.append(f"slate verdict: {verdict}. One slate is a go/no-go count, not a result.")
        return "\n".join(lines)


def slate(study: Study, since: str) -> tuple[str, str]:
    """(subject, body): this run's tickets, then the running report."""
    grades = {g["espn_id"]: g for g in study.read("grades.jsonl")}
    mine = [t for t in study.read("tickets.jsonl") if t["at"] >= since]
    seen = {
        s["espn_id"]
        for s in study.read("tape.jsonl")
        if s["kind"] == "stoppage" and s["primary"] and s["at"] >= since
    }
    rows = [f"This run (since {since}): {len(seen)} games at the stoppage, {len(mine)} tickets"]
    for t in mine:
        g = grades.get(t["espn_id"], {})
        price = "no DK/BetMGM price" if t["price"] is None else f"{t['book']} {t['price']:+g}"
        lead = f"; lead-adjusted {t['lead_side'] or '-'} {g.get('lead_result') or ''}".rstrip()
        rows.append(
            f"  {t['matchup']}: {t['side']} {t['line']:g} ({price}), live {t['live']:g} "
            f"fair {t['fair']:g} gap {t['gap']:+g}; final {g.get('final', '-')} "
            f"{g.get('result') or 'unsettled'}{lead if study.sport in LEAD_ADJ else ''}"
        )
    settled = sum(t["espn_id"] in grades for t in mine)
    subject = (
        f"{study.sport.upper()} live-total paper test {since[:10]}: "
        f"{len(mine)} tickets, {settled} settled"
    )
    return subject, "\n".join([*rows, "", "Running totals", study.report()])


def email_slate(study: Study, since: str) -> int:
    subject, body = slate(study, since)
    try:
        to = mail.send(subject, body)
    except mail.EmailNotConfigured as exc:
        print(f"not emailed: {exc}", file=sys.stderr)
        return 2
    print(f"emailed {subject!r} -> {to}")
    return 0


def run(
    study: Study,
    client: LiveOddsClient,
    *,
    interval: int,
    hours: float,
    max_credits: int,
    idle_exit: int,
) -> None:
    deadline = time.monotonic() + hours * 3600
    spent = idle = polls = 0
    seen_live = False
    while time.monotonic() < deadline:
        now = datetime.now(timezone.utc)
        try:
            games = fetch_scoreboard(study.sport)
            events = None
            if study.needs_board(games, now) and spent < max_credits:
                events = client.fetch(study.sport, markets="totals")
                spent += 1
            out = study.tick(games, events, now)
            if out["snapshots"] or out["tickets"] or out["graded"]:
                print(f"[{now_iso()}] {study.sport}: {out}, credits used {spent}", flush=True)
            live = any(g.status == "in" for g in games)
            seen_live = seen_live or live
            board = {s: sum(g.status == s for g in games) for s in ("pre", "in", "post")}
        except Exception as exc:  # noqa: BLE001 - the loop must survive a bad tick
            log.warning("tick failed: %s", exc)
            live = False
            board = {}
        # Before the first tip the loop waits, however long: a launchd job can start hours
        # early and an empty or failed board must not look like a finished slate. After it,
        # only polls with nothing in play and nothing still to tip count toward the exit, so
        # an afternoon-to-evening gap does not end the run either. --hours bounds both.
        if not seen_live:
            polls += 1
            if polls % HEARTBEAT_POLLS == 1:
                print(f"[{now_iso()}] waiting for the first tip; board {board}", flush=True)
            time.sleep(interval)
            continue
        idle = 0 if live or board.get("pre") else idle + 1
        if idle >= idle_exit:
            print("slate over: no live games; stopping")
            break
        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="live_edge.stoppage",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("cmd", choices=("run", "tick", "report"))
    ap.add_argument("--sport", choices=tuple(RULES), required=True)
    ap.add_argument("--interval", type=int, default=90, help="seconds between polls")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--max-credits", type=int, default=600, help="Odds API calls (1 credit each)")
    ap.add_argument(
        "--idle-exit",
        type=int,
        default=10,
        help="polls with nothing live, after the first tip, before stopping",
    )
    ap.add_argument("--email", action="store_true", help="email the report (run: this run's slate)")
    ap.add_argument("--since", help="report --email: tickets from this UTC stamp (default today)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    study = Study(args.sport)
    if args.cmd == "report":
        print(study.report())
        if args.email:
            return email_slate(study, args.since or now_iso()[:10])
        return 0
    client = LiveOddsClient(os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY"))
    if not client.available():
        print("THE_ODDS_API_KEY is not set; cannot read live totals", file=sys.stderr)
        return 2
    if args.cmd == "tick":
        now = datetime.now(timezone.utc)
        games = fetch_scoreboard(args.sport)
        events = (
            client.fetch(args.sport, markets="totals") if study.needs_board(games, now) else None
        )
        print(study.tick(games, events, now))
        return 0
    since = now_iso()
    run(
        study,
        client,
        interval=args.interval,
        hours=args.hours,
        max_credits=args.max_credits,
        idle_exit=args.idle_exit,
    )
    print(study.report())
    return email_slate(study, since) if args.email else 0


if __name__ == "__main__":
    sys.exit(main())
