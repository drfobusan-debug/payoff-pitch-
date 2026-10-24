"""Alternate spreads and totals beside every spread buy and graded-only total.

Tracked, never bought. At card time the books' alternate lines are fetched for
each game the card took a side in (``alternate_spreads`` and
``alternate_totals``, about two credits a game) and appended to
``altlines_<date>.jsonl``. The audit then grades every alternate number on the
same side, both ways: points bought (California +20.5 at a worse price) and
points sold (California +16.5 at a better one), against the real final, in
``alt_ledger.csv``. ``cfb-engine alt-report`` prints the record by direction and
by how many points moved. Nothing here is read by pricing.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

from cfb_engine.market import gradedonly
from cfb_engine.market.odds import american_to_decimal

ALT_MARKETS = ("alternate_spreads", "alternate_totals")
_MARKET = {"alternate_spreads": "game_ats", "alternate_totals": "game_total"}


def tracked(market: str, tier: str, pass_gate: str | None) -> bool:
    """A spread buy, or a total the card would have bought (bought or graded-only)."""
    if market == "game_ats":
        return tier != "Pass"
    if market == "game_total":
        return tier != "Pass" or gradedonly.is_gate(pass_gate)
    return False


def parse_event(
    raw: dict, home: str, away: str, home_name: str, away_name: str, captured_at: str
) -> list[dict]:
    """One row per book, side and alternate number in an event-odds reply."""
    matchup = f"{away} @ {home}"
    out: list[dict] = []
    for bm in raw.get("bookmakers", []) or []:
        for mkt in bm.get("markets", []) or []:
            market = _MARKET.get(str(mkt.get("key", "")))
            if market is None:
                continue
            for oc in mkt.get("outcomes", []) or []:
                price, point = oc.get("price"), oc.get("point")
                if price is None or point is None:
                    continue
                name = str(oc.get("name", ""))
                if market == "game_total":
                    side = "over" if name.lower().startswith("over") else "under"
                else:
                    low = re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()
                    side = "home" if low == home_name else "away" if low == away_name else ""
                    if not side:
                        continue
                out.append(
                    {
                        "matchup": matchup,
                        "captured_at": captured_at,
                        "market": market,
                        "side": side,
                        "point": float(point),
                        "book": str(bm.get("key", "")),
                        "american": float(price),
                    }
                )
    return out


def append(rows: Iterable[dict], path: Path) -> int:
    rows = list(rows)
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
    return len(rows)


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@dataclass
class AltRow:
    date: str
    matchup: str
    market: str
    main_selection: str
    main_line: float
    tier: str
    side: str
    alt_line: float
    #: Points moved in our favour: positive = bought, negative = sold.
    favour: float
    book: str
    american: float
    result: str
    pnl: float


def favour(market: str, side: str, main: float, alt: float) -> float:
    """How many points the alternate number gives us over the main one."""
    if market == "game_total" and side == "over":
        return round(main - alt, 1)
    return round(alt - main, 1)


def outcome(market: str, side: str, line: float, home_pts: int, away_pts: int) -> str:
    if market == "game_total":
        diff = (home_pts + away_pts) - line
        diff = diff if side == "over" else -diff
    else:
        mine, theirs = (home_pts, away_pts) if side == "home" else (away_pts, home_pts)
        diff = mine + line - theirs
    return "push" if diff == 0 else "win" if diff > 0 else "loss"


def _pnl(result: str, american: float) -> float:
    if result == "win":
        return round(american_to_decimal(american) - 1.0, 4)
    return -1.0 if result == "loss" else 0.0


def grade(
    *,
    date: str,
    matchup: str,
    market: str,
    selection: str,
    main_line: float,
    tier: str,
    side: str,
    home_pts: int,
    away_pts: int,
    rows: list[dict],
    kickoff_utc: str = "",
) -> list[AltRow]:
    """Every alternate number on our side from the first capture, at its best price."""
    mine = [
        r
        for r in rows
        if r["matchup"] == matchup
        and r["market"] == market
        and r["side"] == side
        and (not kickoff_utc or r["captured_at"] < kickoff_utc)
    ]
    if not mine:
        return []
    first = min(r["captured_at"] for r in mine)
    best: dict[float, dict] = {}
    for r in mine:
        if r["captured_at"] != first:
            continue
        cur = best.get(r["point"])
        if cur is None or american_to_decimal(r["american"]) > american_to_decimal(cur["american"]):
            best[r["point"]] = r
    out = []
    for point, r in sorted(best.items()):
        f = favour(market, side, main_line, point)
        if f == 0:
            continue
        res = outcome(market, side, point, home_pts, away_pts)
        out.append(
            AltRow(
                date,
                matchup,
                market,
                selection,
                main_line,
                tier,
                side,
                point,
                f,
                r["book"],
                r["american"],
                res,
                _pnl(res, r["american"]),
            )
        )
    return out


FIELDS = [f.name for f in fields(AltRow)]


def save_day(path: Path, day: str, rows: list[AltRow]) -> int:
    """Replace one slate's rows in the alt ledger; other slates are kept as written."""
    kept: list[dict] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            kept = [r for r in csv.DictReader(fh) if r["date"] != day]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(kept)
        w.writerows(asdict(r) for r in rows)
    return len(rows)


def report(path: Path) -> list[tuple[str, int, int, int, float, float]]:
    """(label, n, wins, losses, units, ROI) by market, direction and points moved."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["result"] in ("win", "loss", "push")]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        f = float(r["favour"])
        way = "bought" if f > 0 else "sold"
        size = min(abs(f), 7.0)
        groups[f"{r['market']} {way} {size:g}{'+' if size == 7.0 else ''} pts"].append(r)
        groups[f"{r['market']} {way} (all)"].append(r)
    out = []
    for label in sorted(groups):
        g = groups[label]
        units = sum(float(r["pnl"]) for r in g)
        out.append(
            (
                label,
                len(g),
                sum(r["result"] == "win" for r in g),
                sum(r["result"] == "loss" for r in g),
                round(units, 2),
                round(units / len(g), 4),
            )
        )
    return out


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
