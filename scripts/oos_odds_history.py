"""Pre-game MLB prices from The Odds API historical endpoint, cached to disk.

Pulls two snapshots per game day -- one just before the earliest first pitch of
the day (the afternoon slate) and one just before the earliest evening first
pitch -- with ``markets=h2h,spreads,totals`` and ``regions=us`` (30 credits a
call). Each raw response is written to
``~/.mlb_engine/cache/oddsapi_hist/<date>_<slot>.json`` *before* it is parsed,
so a re-run of the parse step costs nothing.

The credit accounting is strict: a hard session cap, a floor the account's
``x-requests-remaining`` may not drop below, and a refusal to spend when the
next call would breach either.

Parsing keys a game by (home, away, commence_time) so doubleheaders do not
collide, takes for each game the snapshot nearest to but before first pitch,
and records the hours to first pitch. The prices are *pre-game*, not closing.

    .venv/bin/python scripts/oos_odds_history.py plan   --seasons 2024 2025 2026
    .venv/bin/python scripts/oos_odds_history.py fetch  --seasons 2024 2025 2026
    .venv/bin/python scripts/oos_odds_history.py parse  --out ~/.mlb_engine/audit/oos_prices.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

log = logging.getLogger("oos_odds_history")

HIST_URL = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds"
SCHED_URL = "https://statsapi.mlb.com/api/v1/schedule"
MARKETS = "h2h,spreads,totals"
REGIONS = "us"
CREDITS_PER_CALL = 10 * 3 * 1  # 10 x markets x regions
DEFAULT_CACHE = Path.home() / ".mlb_engine" / "cache" / "oddsapi_hist"
SEASON_END = {2024: Date(2024, 9, 29), 2025: Date(2025, 9, 28), 2026: Date(2026, 7, 18)}
SEASON_START = {2024: Date(2024, 3, 20), 2025: Date(2025, 3, 18), 2026: Date(2026, 3, 26)}
#: Snapshots are taken this long before the first pitch they target.
LEAD = timedelta(minutes=10)
#: A second slot is only worth a call when it is this far after the first.
SLOT_GAP = timedelta(hours=3)
#: Games earlier than this UTC hour (international openers) are not chased.
EARLIEST_UTC_HOUR = 15
PREFERRED_BOOKS = ("pinnacle", "fanduel", "draftkings", "betmgm", "williamhill_us", "bovada")

TEAM_ABBR = {
    "arizona diamondbacks": "AZ", "atlanta braves": "ATL", "baltimore orioles": "BAL",
    "boston red sox": "BOS", "chicago cubs": "CHC", "chicago white sox": "CWS",
    "cincinnati reds": "CIN", "cleveland guardians": "CLE", "colorado rockies": "COL",
    "detroit tigers": "DET", "houston astros": "HOU", "kansas city royals": "KC",
    "los angeles angels": "LAA", "los angeles dodgers": "LAD", "miami marlins": "MIA",
    "milwaukee brewers": "MIL", "minnesota twins": "MIN", "new york mets": "NYM",
    "new york yankees": "NYY", "oakland athletics": "ATH", "athletics": "ATH",
    "las vegas athletics": "ATH", "sacramento athletics": "ATH",
    "philadelphia phillies": "PHI", "pittsburgh pirates": "PIT", "san diego padres": "SD",
    "san francisco giants": "SF", "seattle mariners": "SEA", "st louis cardinals": "STL",
    "st. louis cardinals": "STL", "tampa bay rays": "TB", "texas rangers": "TEX",
    "toronto blue jays": "TOR", "washington nationals": "WSH",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def team_abbr(name: str) -> str | None:
    return TEAM_ABBR.get(_norm(name)) or TEAM_ABBR.get(_norm(name).replace(".", ""))


def american_to_prob(price: float) -> float:
    p = float(price)
    return 100.0 / (p + 100.0) if p > 0 else -p / (-p + 100.0)


def no_vig(p_a: float, p_b: float) -> tuple[float, float]:
    s = p_a + p_b
    return p_a / s, p_b / s


def american_profit(price: float) -> float:
    """Units won on a 1u stake when the side wins."""
    p = float(price)
    return p / 100.0 if p > 0 else 100.0 / -p


# --- schedule -------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedGame:
    game_pk: int
    date: Date
    home: str
    away: str
    commence: datetime
    dh: str
    status: str


def schedule(start: Date, end: Date, session: requests.Session | None = None) -> list[SchedGame]:
    """Regular-season games with UTC first pitch, one statsapi call per month."""
    sess = session or requests.Session()
    out: list[SchedGame] = []
    cur = start
    while cur <= end:
        stop = min((cur.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1), end)
        params: dict[str, str | int] = {
            "sportId": 1, "gameType": "R", "startDate": cur.isoformat(), "endDate": stop.isoformat(),
            "hydrate": "team",
        }
        r = sess.get(SCHED_URL, params=params, timeout=60)
        r.raise_for_status()
        for day in r.json().get("dates", []):
            d = Date.fromisoformat(day["date"])
            for g in day.get("games", []):
                iso = g.get("gameDate")
                if not iso:
                    continue
                out.append(SchedGame(
                    int(g["gamePk"]), d,
                    str(g["teams"]["home"]["team"].get("abbreviation", "")),
                    str(g["teams"]["away"]["team"].get("abbreviation", "")),
                    datetime.fromisoformat(iso.replace("Z", "+00:00")),
                    str(g.get("doubleHeader", "N")), str(g.get("status", {}).get("detailedState", "")),
                ))
        cur = stop + timedelta(days=1)
    return out


def slots_for_day(commences: list[datetime]) -> list[datetime]:
    """Snapshot timestamps for one game day: ``LEAD`` before the first pitch of
    each of up to two waves (afternoon, evening) at least ``SLOT_GAP`` apart."""
    times = sorted(t for t in commences if t.hour >= EARLIEST_UTC_HOUR or t.date() != min(
        c.date() for c in commences))
    if not times:
        times = sorted(commences)
    slots = [times[0] - LEAD]
    later = [t for t in times if t >= slots[0] + SLOT_GAP]
    if later:
        slots.append(later[0] - LEAD)
    return slots


def plan(games: list[SchedGame]) -> dict[Date, list[datetime]]:
    by_day: dict[Date, list[datetime]] = {}
    for g in games:
        if g.status.lower().startswith(("postponed", "cancelled", "canceled", "suspended")):
            continue
        by_day.setdefault(g.date, []).append(g.commence)
    return {d: slots_for_day(ts) for d, ts in sorted(by_day.items())}


# --- fetch ----------------------------------------------------------------------------


class Budget:
    def __init__(self, cap: int, floor: int) -> None:
        self.cap, self.floor = cap, floor
        self.spent = 0
        self.remaining: int | None = None

    def can_spend(self, cost: int) -> bool:
        if self.spent + cost > self.cap:
            return False
        return self.remaining is None or self.remaining - cost >= self.floor

    def record(self, resp: requests.Response, cost: int) -> None:
        self.spent += cost
        rem = resp.headers.get("x-requests-remaining")
        if rem is not None:
            try:
                self.remaining = int(float(rem))
            except ValueError:
                pass


def slot_path(cache_dir: Path, day: Date, slot: datetime) -> Path:
    return cache_dir / f"{day.isoformat()}_{slot.strftime('%H%M')}Z.json"


def fetch_snapshot(day: Date, slot: datetime, cache_dir: Path, api_key: str, budget: Budget,
                   session: requests.Session) -> dict | None:
    path = slot_path(cache_dir, day, slot)
    if path.exists():
        return json.loads(path.read_text())
    if not budget.can_spend(CREDITS_PER_CALL):
        log.warning("budget stop: spent=%s remaining=%s", budget.spent, budget.remaining)
        return None
    params = {
        "apiKey": api_key, "regions": REGIONS, "markets": MARKETS, "oddsFormat": "american",
        "date": slot.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for attempt in range(4):
        try:
            resp = session.get(HIST_URL, params=params, timeout=60)
        except requests.RequestException as exc:
            log.warning("%s %s: %s", day, slot, type(exc).__name__)
            time.sleep(2 * (attempt + 1))
            continue
        if resp.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        if resp.status_code != 200:
            log.error("%s %s: HTTP %s %s", day, slot, resp.status_code, resp.text[:120])
            if resp.status_code in (401, 402, 403):
                budget.remaining = 0
            return None
        budget.record(resp, CREDITS_PER_CALL)
        raw = resp.json()
        raw["_meta"] = {
            "requested": params["date"], "credits": CREDITS_PER_CALL,
            "x_requests_remaining": budget.remaining, "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(raw))
        tmp.replace(path)
        return raw
    return None


# --- parse ----------------------------------------------------------------------------


def _side_prices(outcomes: list[dict], home: str, away: str) -> tuple[float, float] | None:
    ph = pa = None
    for oc in outcomes:
        ab = team_abbr(str(oc.get("name", "")))
        if oc.get("price") is None:
            continue
        if ab == home:
            ph = float(oc["price"])
        elif ab == away:
            pa = float(oc["price"])
    return (ph, pa) if ph is not None and pa is not None else None


def _rl_prices(outcomes: list[dict], home: str, away: str) -> tuple[float, float, float] | None:
    """(home point, home price, away price) for the standard -1.5/+1.5 line."""
    ph = pa = pt = None
    for oc in outcomes:
        ab = team_abbr(str(oc.get("name", "")))
        pt_i = oc.get("point")
        if oc.get("price") is None or pt_i is None or abs(abs(float(pt_i)) - 1.5) > 1e-9:
            continue
        if ab == home:
            ph, pt = float(oc["price"]), float(pt_i)
        elif ab == away:
            pa = float(oc["price"])
    return (pt, ph, pa) if ph is not None and pa is not None and pt is not None else None


def _total_prices(outcomes: list[dict]) -> tuple[float, float, float] | None:
    over = under = pt = None
    for oc in outcomes:
        if oc.get("price") is None or oc.get("point") is None:
            continue
        nm = str(oc.get("name", "")).lower()
        if nm == "over":
            over, pt = float(oc["price"]), float(oc["point"])
        elif nm == "under":
            under = float(oc["price"])
    return (pt, over, under) if over is not None and under is not None and pt is not None else None


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else math.nan


def parse_event(ev: dict, snap_ts: datetime) -> dict | None:
    home, away = team_abbr(str(ev.get("home_team", ""))), team_abbr(str(ev.get("away_team", "")))
    if not home or not away:
        return None
    commence = datetime.fromisoformat(str(ev["commence_time"]).replace("Z", "+00:00"))
    if commence <= snap_ts:
        return None
    row: dict = {
        "home": home, "away": away, "commence": commence.isoformat(), "snapshot": snap_ts.isoformat(),
        "hours_to_pitch": (commence - snap_ts).total_seconds() / 3600.0,
    }
    ml: dict[str, tuple[float, float]] = {}
    rl: dict[str, tuple[float, float, float]] = {}
    tot: dict[str, tuple[float, float, float]] = {}
    for bk in ev.get("bookmakers", []):
        key = str(bk.get("key", ""))
        for mkt in bk.get("markets", []):
            ocs = mkt.get("outcomes", [])
            if mkt.get("key") == "h2h":
                v = _side_prices(ocs, home, away)
                if v:
                    ml[key] = v
            elif mkt.get("key") == "spreads":
                v3 = _rl_prices(ocs, home, away)
                if v3:
                    rl[key] = v3
            elif mkt.get("key") == "totals":
                v3 = _total_prices(ocs)
                if v3:
                    tot[key] = v3
    if not ml:
        return None
    row["n_books_ml"] = len(ml)
    book = next((b for b in PREFERRED_BOOKS if b in ml), sorted(ml)[0])
    row["book"] = book
    row["ml_home"], row["ml_away"] = ml[book]
    ph, pa = no_vig(american_to_prob(ml[book][0]), american_to_prob(ml[book][1]))
    row["p_home_book"] = ph
    row["ml_home_med"] = _median([v[0] for v in ml.values()])
    row["ml_away_med"] = _median([v[1] for v in ml.values()])
    probs = [no_vig(american_to_prob(v[0]), american_to_prob(v[1]))[0] for v in ml.values()]
    row["p_home_cons"] = _median(probs)
    if rl:
        rb = book if book in rl else next((b for b in PREFERRED_BOOKS if b in rl), sorted(rl)[0])
        row["rl_book"] = rb
        row["rl_home_pt"], row["rl_home"], row["rl_away"] = rl[rb]
        row["rl_home_med"] = _median([v[1] for v in rl.values()])
        row["rl_away_med"] = _median([v[2] for v in rl.values()])
    if tot:
        tb = book if book in tot else next((b for b in PREFERRED_BOOKS if b in tot), sorted(tot)[0])
        row["tot_book"] = tb
        row["total_pt"], row["over"], row["under"] = tot[tb]
        row["total_med"] = _median([v[0] for v in tot.values()])
    return row


def parse_cache(cache_dir: Path) -> list[dict]:
    """One row per (home, away, commence): the snapshot nearest before first pitch."""
    best: dict[tuple[str, str, str], dict] = {}
    for path in sorted(cache_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        ts_raw = raw.get("timestamp") or raw.get("_meta", {}).get("requested")
        if not ts_raw:
            continue
        snap_ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        for ev in raw.get("data", []):
            row = parse_event(ev, snap_ts)
            if row is None:
                continue
            key = (row["home"], row["away"], row["commence"])
            if key not in best or row["hours_to_pitch"] < best[key]["hours_to_pitch"]:
                best[key] = row
    return _collapse_jitter(sorted(best.values(), key=lambda r: (r["commence"], r["home"])))


def _collapse_jitter(rows: list[dict], gap_hours: float = 2.0) -> list[dict]:
    """Snapshots list the same game with first pitch drifting by a minute or two
    (02:05 vs 02:06); keep one row per matchup within ``gap_hours`` (the nearest
    snapshot), so that only genuine doubleheaders keep two rows."""
    out: list[dict] = []
    last: dict[tuple[str, str], int] = {}
    for r in rows:
        key = (r["home"], r["away"])
        prev_i = last.get(key)
        if prev_i is not None:
            prev = out[prev_i]
            dt = datetime.fromisoformat(r["commence"]) - datetime.fromisoformat(prev["commence"])
            if abs(dt.total_seconds()) <= gap_hours * 3600:
                if r["hours_to_pitch"] < prev["hours_to_pitch"]:
                    out[prev_i] = r
                continue
        last[key] = len(out)
        out.append(r)
    return out


def write_csv(rows: list[dict], out: Path) -> None:
    import csv
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


# --- cli ------------------------------------------------------------------------------


def _seasons_plan(seasons: list[int], start: Date | None = None, end: Date | None = None) -> dict[Date, list[datetime]]:
    sess = requests.Session()
    out: dict[Date, list[datetime]] = {}
    for s in seasons:
        out.update(plan(schedule(start or SEASON_START[s], end or SEASON_END[s], sess)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "fetch", "parse"])
    ap.add_argument("--seasons", type=int, nargs="*", default=[2024, 2025, 2026])
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--cap", type=int, default=35_000, help="hard credit cap for this run")
    ap.add_argument("--floor", type=int, default=25_000, help="never let x-requests-remaining drop below")
    ap.add_argument("--out", type=Path, default=Path.home() / ".mlb_engine" / "audit" / "oos_prices.csv")
    ap.add_argument("--plan-out", type=Path, default=None)
    ap.add_argument("--start", type=Date.fromisoformat, default=None, help="override the season start date")
    ap.add_argument("--end", type=Date.fromisoformat, default=None, help="override the season end date")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd in ("plan", "fetch"):
        p = _seasons_plan(args.seasons, args.start, args.end)
        n_calls = sum(len(v) for v in p.values())
        cached = sum(1 for d, ss in p.items() for s in ss if slot_path(args.cache_dir, d, s).exists())
        print(f"game days={len(p)} calls={n_calls} cached={cached} "
              f"credits_needed={(n_calls - cached) * CREDITS_PER_CALL}")
        if args.plan_out:
            args.plan_out.parent.mkdir(parents=True, exist_ok=True)
            with args.plan_out.open("w") as fh:
                fh.write("date,slot_utc,cached\n")
                for d, ss in p.items():
                    for s in ss:
                        fh.write(f"{d},{s.isoformat()},{int(slot_path(args.cache_dir, d, s).exists())}\n")
        if args.cmd == "plan":
            return 0
        key = os.environ.get("THE_ODDS_API_KEY")
        if not key:
            print("THE_ODDS_API_KEY not set", file=sys.stderr)
            return 2
        budget = Budget(args.cap, args.floor)
        sess = requests.Session()
        done = 0
        for d, ss in p.items():
            for s in ss:
                raw = fetch_snapshot(d, s, args.cache_dir, key, budget, sess)
                if raw is None and not slot_path(args.cache_dir, d, s).exists():
                    if not budget.can_spend(CREDITS_PER_CALL):
                        print(f"STOP budget: spent={budget.spent} remaining={budget.remaining}")
                        return 1
                done += 1
                if done % 25 == 0:
                    log.info("%s/%s spent=%s remaining=%s", done, n_calls, budget.spent, budget.remaining)
        print(f"done: spent={budget.spent} remaining={budget.remaining}")
        return 0

    rows = parse_cache(args.cache_dir)
    write_csv(rows, args.out)
    hrs = sorted(r["hours_to_pitch"] for r in rows)
    print(f"games priced={len(rows)} median hours to first pitch="
          f"{hrs[len(hrs) // 2]:.2f} -> {args.out}" if rows else "no rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
