"""Alternate spreads and totals beside every spread buy and graded-only total.

Tracked, never bought. At card time the books' alternate lines are fetched for
each game the card took a side in (``alternate_spreads`` and
``alternate_totals``, about two credits a game) and appended to
``altlines_<date>.jsonl``. The audit then grades every alternate number on the
same side, both ways: points bought (California +20.5 at a worse price) and
points sold (California +16.5 at a better one), against the real final, in
``alt_ledger.csv``. ``cfb-engine alt-report`` prints the record by direction and
by how many points moved. Nothing here is read by pricing.

Each rung is also priced by the engine: the main-line probability the card bet
at, moved along the sim's normal margin (or total) to the alternate number, so
the market owns the main line and the sim owns the shape around it. A rung
whose edge over that book's two-way no-vig price passes the same EV and edge
bars as a main-line buy, and whose edge beats the main line's own by
``EDGE_MARGIN``, is a candidate; the best candidate per game and side is ``flagged``. That is the
graded-only answer to which number to take instead of the main one. Only rungs
within ``WINDOW`` points of the main number are graded. Each row carries the
main line's own result, so a flagged rung is read against the main line on the
same games.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

from cfb_engine.config import EVThresholds
from cfb_engine.market import gradedonly
from cfb_engine.market.odds import american_to_decimal, no_vig_two_way
from engine_runtime import altlines as rt

ALT_MARKETS = ("alternate_spreads", "alternate_totals")
_MARKET = {"alternate_spreads": "game_ats", "alternate_totals": "game_total"}
#: Points either side of the main number graded: a spread window that crosses the 3.
WINDOW = 3.5
#: A rung must beat the main line's edge by this much: the best of ~14 rungs edges past it by chance.
EDGE_MARGIN = 0.01


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
    #: Engine probability at the alternate number; blank on rows graded before it was priced.
    p_engine: float | None = None
    #: The book's two-way no-vig probability at that number, when it posted the other side.
    fair_prob: float | None = None
    edge: float | None = None
    #: The game's best rung that passes the main-line buy bars and beats the main line's edge.
    flagged: bool = False
    #: The main line's own result and 1u P&L on the card's price, for the same-games comparison.
    main_result: str = ""
    main_pnl: float | None = None


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


def side_prob(market: str, side: str, line: float, mean: float, sd: float) -> float:
    """P(our side covers ``line``) on a normal margin (home minus away) or total."""
    if market == "game_total":
        z = (mean - line) / sd if side == "over" else (line - mean) / sd
    else:
        z = (mean + line) / sd if side == "home" else (line - mean) / sd
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def engine_prob(
    market: str, side: str, main: float, alt: float, p_main: float, mean: float, sd: float
) -> float:
    """The card's main-line probability, moved by the sim's shape to the alternate number."""
    shift = side_prob(market, side, alt, mean, sd) - side_prob(market, side, main, mean, sd)
    return min(max(p_main + shift, 0.001), 0.999)


def _opposite(market: str, side: str, point: float) -> tuple[str, float]:
    if market == "game_total":
        return ("under" if side == "over" else "over"), point
    return ("away" if side == "home" else "home"), -point


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
    p_main: float | None = None,
    mean: float | None = None,
    sd: float | None = None,
    thr: EVThresholds | None = None,
    main_edge: float | None = None,
    main_result: str = "",
    main_american: float | None = None,
) -> list[AltRow]:
    """Every alternate number on our side within ``WINDOW``, from the first capture, at its best price.

    With the card's ``p_main`` and the sim's ``mean``/``sd`` each rung is priced;
    under ``thr`` the rung with the biggest edge among those that clear the
    main-line buy bars and beat ``main_edge`` by ``EDGE_MARGIN`` is flagged.
    """
    pregame = [
        r
        for r in rows
        if r["matchup"] == matchup
        and r["market"] == market
        and (not kickoff_utc or r["captured_at"] < kickoff_utc)
    ]
    mine = [r for r in pregame if r["side"] == side]
    if not mine:
        return []
    first = min(r["captured_at"] for r in mine)
    other = {
        (r["book"], r["side"], r["point"]): r["american"] for r in pregame if r["captured_at"] == first
    }
    best: dict[float, dict] = {}
    for r in mine:
        if r["captured_at"] != first:
            continue
        cur = best.get(r["point"])
        if cur is None or american_to_decimal(r["american"]) > american_to_decimal(cur["american"]):
            best[r["point"]] = r
    main_pnl = _pnl(main_result, main_american) if main_result and main_american is not None else None
    out = []
    for point, r in sorted(best.items()):
        f = favour(market, side, main_line, point)
        if f == 0 or abs(f) > WINDOW:
            continue
        res = outcome(market, side, point, home_pts, away_pts)
        p_eng = fair = edge = None
        candidate = False
        if p_main is not None and mean is not None and sd:
            p_eng = engine_prob(market, side, main_line, point, p_main, mean, sd)
            opp = other.get((r["book"], *_opposite(market, side, point)))
            if opp is not None:
                fair = no_vig_two_way(r["american"], opp)[0]
                edge = p_eng - fair
                ev = p_eng * american_to_decimal(r["american"]) - 1.0
                if thr is not None and main_edge is not None:
                    candidate = (
                        ev > thr.min_ev
                        and thr.min_edge <= edge <= thr.max_edge
                        and edge >= main_edge + EDGE_MARGIN
                    )
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
                None if p_eng is None else round(p_eng, 4),
                None if fair is None else round(fair, 4),
                None if edge is None else round(edge, 4),
                candidate,
                main_result,
                main_pnl,
            )
        )
    picks = [a for a in out if a.flagged]
    if picks:
        top = max(picks, key=lambda a: a.edge or 0.0)
        for a in picks:
            a.flagged = a is top
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
    """(label, n, wins, losses, units, ROI) by market, direction and points moved.

    Each engine-flagged group is followed by the main line on the same games.
    """
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [
            r
            for r in csv.DictReader(fh)
            if r["result"] in ("win", "loss", "push") and abs(float(r["favour"])) <= WINDOW
        ]
    groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in rows:
        f = float(r["favour"])
        way = "bought" if f > 0 else "sold"
        mine = (r["result"], float(r["pnl"]))
        groups[f"{r['market']} {way} {abs(f):g} pts"].append(mine)
        groups[f"{r['market']} {way} (all)"].append(mine)
        if r.get("flagged") == "True":
            groups[f"{r['market']} {way} engine-flagged"].append(mine)
            if r.get("main_pnl"):
                groups[f"{r['market']} {way} engine-flagged: main line"].append(
                    (r["main_result"], float(r["main_pnl"]))
                )
    out = []
    for label in sorted(groups):
        g = groups[label]
        units = sum(p for _, p in g)
        out.append(
            (
                label,
                len(g),
                sum(res == "win" for res, _ in g),
                sum(res == "loss" for res, _ in g),
                round(units, 2),
                round(units / len(g), 4),
            )
        )
    return out


def alts(path: Path, *, flagged_only: bool = False, main_line: bool = False) -> list[rt.Alt]:
    """The alt ledger as records for the shared alt-lines table (every row sits beside a tracked side).

    ``main_line`` reads the flagged rungs' games at the main number instead (same games, same table).
    """
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [
            r
            for r in csv.DictReader(fh)
            if r["result"] in ("win", "loss", "push")
            and abs(float(r["favour"])) <= WINDOW
            and (not (flagged_only or main_line) or r.get("flagged") == "True")
            and (not main_line or bool(r.get("main_pnl")))
        ]
    return [
        rt.Alt(
            game=f"{r['date']} {r['matchup']}",
            kind="spread" if r["market"] == "game_ats" else "total",
            favour=float(r["favour"]),
            beside_buy=r["tier"] != "Pass",
            outcome=r["main_result"] if main_line else r["result"],
            pnl=float(r["main_pnl"] if main_line else r["pnl"]),
        )
        for r in rows
    ]

def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
