"""The news alarm: re-price a game the moment its market or its players change.

A scheduled card prices against whatever it last saw. In the NBA a star ruled
out twenty minutes before tip moves the line at once, and a card still holding
the old number reads the move as value on the short-handed side. So every
watch tick (minutes apart, not hours) compares the latest two looks at:

* the featured board -- consensus no-vig ML probability, spread and total;
* ESPN's injury feed -- every status change, as teams announce it;
* the official league report -- each new hourly file;

and raises an ``Alert`` per affected game. Moves are measured from the board
seen at the game's last alert (``reference``), so a slow creep adds up. An
alerted game is re-captured at once (first half and props too) and stays
*pending* until (1) its line has settled -- two consecutive fast polls each
moving less than the settle tolerance (``settle``), recorded as a ``settled``
alert -- and (2) a pricing run newer than that has seen it. The card may not
recommend a pending game (``pending``): a line still moving after news is a
falling knife. Alerts are archived, immutably, under
``<data>/alerts/<YYYY-MM-DD>/<stamp>_<fp>.csv``, so the audit can grade what the
alarm caught and what it missed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import date as Date
from pathlib import Path

from nba_engine.config import AlarmParams
from nba_engine.data import teamnames
from nba_engine.data.capture import QuoteRow
from nba_engine.data.espn_injuries import FeedRow
from nba_engine.data.injuries import InjuryRow
from nba_engine.schemas import Game

ACTIVE = frozenset({"Available", "Active", ""})


@dataclass(frozen=True)
class Alert:
    detected_at: str
    game_date: str
    event_id: str
    matchup: str
    kind: str  # ml_move | spread_move | total_move | injury | injury_official | settled
    detail: str
    before: str
    after: str


FIELDS = tuple(f.name for f in fields(Alert))


def _implied(american: float) -> float:
    return 100.0 / (american + 100.0) if american > 0 else -american / (-american + 100.0)


def consensus(rows: Iterable[QuoteRow]) -> dict[str, dict[str, float]]:
    """Per event: median no-vig home ML probability, home spread and total line.

    Only paired quotes count (a side without its partner has no fair price);
    a market no book pairs is simply absent.
    """
    ml: dict[str, list[float]] = {}
    spread: dict[str, list[float]] = {}
    total: dict[str, list[float]] = {}
    homes: dict[str, str] = {}
    for r in rows:
        if r.opposite_american is None or not r.event_id:
            continue
        home = r.matchup.split(" @ ")[-1]
        homes[r.event_id] = home
        if r.market == "game_ml" and r.side == home:
            a, b = _implied(r.american), _implied(r.opposite_american)
            ml.setdefault(r.event_id, []).append(a / (a + b))
        elif r.market == "game_ats" and r.side == home and r.line is not None:
            spread.setdefault(r.event_id, []).append(r.line)
        elif r.market == "game_total" and r.side == "over" and r.line is not None:
            total.setdefault(r.event_id, []).append(r.line)
    out: dict[str, dict[str, float]] = {}
    for key, table in (("ml", ml), ("spread", spread), ("total", total)):
        for eid, values in table.items():
            out.setdefault(eid, {})[key] = statistics.median(values)
    return out


def board_alerts(
    before: Iterable[QuoteRow],
    after: Iterable[QuoteRow],
    games: Mapping[str, Game],
    params: AlarmParams,
    detected_at: str,
) -> list[Alert]:
    """Consensus moves at or past the alarm thresholds between two boards."""
    return move_alerts(consensus(before), consensus(after), games, params, detected_at)


def reference(
    boards: Sequence[Sequence[QuoteRow]], alerts: Iterable[Alert]
) -> dict[str, dict[str, float]]:
    """Per event, the consensus the next move is measured from.

    That is the board seen at the game's latest alert, or the day's first board
    if it has none -- not merely the previous look, so a line that creeps a
    little each tick still trips the alarm once the creep adds up. ``boards``
    are the day's snapshots, oldest first.
    """
    anchor: dict[str, str] = {}
    for a in alerts:
        if a.event_id and a.detected_at > anchor.get(a.event_id, ""):
            anchor[a.event_id] = a.detected_at
    out: dict[str, dict[str, float]] = {}
    for rows in boards:
        if not rows:
            continue
        taken = rows[0].captured_at
        for eid, cons in consensus(rows).items():
            if eid not in out or taken <= anchor.get(eid, ""):
                out[eid] = cons
    return out


def move_alerts(
    prev: Mapping[str, Mapping[str, float]],
    cur: Mapping[str, Mapping[str, float]],
    games: Mapping[str, Game],
    params: AlarmParams,
    detected_at: str,
) -> list[Alert]:
    """Consensus moves at or past the alarm thresholds, ``prev`` -> ``cur``."""
    limits = {"ml": params.ml_prob, "spread": params.spread_pts, "total": params.total_pts}
    out: list[Alert] = []
    for eid, now in sorted(cur.items()):
        game = games.get(eid)
        if game is None or eid not in prev:
            continue
        for key, limit in limits.items():
            a, b = prev[eid].get(key), now.get(key)
            if a is None or b is None or abs(b - a) < limit - 1e-9:
                continue
            fmt = (lambda v: f"{v:.3f}") if key == "ml" else (lambda v: f"{v:g}")
            label = f"home {key}" if key != "total" else "total"
            out.append(
                Alert(
                    detected_at=detected_at,
                    game_date=game.game_date.isoformat(),
                    event_id=eid,
                    matchup=game.matchup,
                    kind=f"{key}_move",
                    detail=f"{label} {fmt(a)} -> {fmt(b)}",
                    before=fmt(a),
                    after=fmt(b),
                )
            )
    return out


def _games_by_team(games: Iterable[Game]) -> dict[str, Game]:
    out: dict[str, Game] = {}
    for g in games:
        out.setdefault(g.home, g)
        out.setdefault(g.away, g)
    return out


def _status_alerts(
    before: Mapping[tuple[str, str], tuple[str, str]],
    after: Mapping[tuple[str, str], tuple[str, str]],
    games: Iterable[Game],
    kind: str,
    detected_at: str,
) -> list[Alert]:
    """``before``/``after``: (team, player key) -> (display name, status)."""
    by_team = _games_by_team(games)
    out: list[Alert] = []
    for key in sorted(set(before) | set(after)):
        team = key[0]
        game = by_team.get(team)
        if game is None:
            continue
        name_a, was = before.get(key, ("", ""))
        name_b, now = after.get(key, ("", ""))
        if was == now or (was in ACTIVE and now in ACTIVE):
            continue
        out.append(
            Alert(
                detected_at=detected_at,
                game_date=game.game_date.isoformat(),
                event_id=game.event_id,
                matchup=game.matchup,
                kind=kind,
                detail=f"{team} {name_b or name_a}: {was or 'not listed'} -> {now or 'cleared'}",
                before=was,
                after=now,
            )
        )
    return out


def feed_alerts(
    before: Iterable[FeedRow], after: Iterable[FeedRow], games: Iterable[Game], detected_at: str
) -> list[Alert]:
    def table(rows: Iterable[FeedRow]) -> dict[tuple[str, str], tuple[str, str]]:
        return {r.key: (r.player, r.status) for r in rows if r.team}

    return _status_alerts(table(before), table(after), games, "injury", detected_at)


def official_alerts(
    before: Iterable[InjuryRow], after: Iterable[InjuryRow], games: Iterable[Game], detected_at: str
) -> list[Alert]:
    def table(rows: Iterable[InjuryRow]) -> dict[tuple[str, str], tuple[str, str]]:
        out: dict[tuple[str, str], tuple[str, str]] = {}
        for r in rows:
            code = teamnames.code_for(r.team)
            if code and r.player:
                out[(code, r.player)] = (r.player, r.status)
        return out

    return _status_alerts(table(before), table(after), games, "injury_official", detected_at)


def alerts_dir(data_dir: Path, day: Date) -> Path:
    return data_dir / "alerts" / day.isoformat()


def write_alerts(alerts: list[Alert], data_dir: Path, day: Date, stamp: str) -> Path | None:
    if not alerts:
        return None
    body = "\n".join("|".join(str(v) for v in asdict(a).values()) for a in alerts)
    fp = hashlib.sha256(body.encode()).hexdigest()[:10]
    directory = alerts_dir(data_dir, day)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stamp.replace(':', '').replace('-', '')}_{fp}.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for a in alerts:
            writer.writerow(asdict(a))
    return path


def read_alerts(data_dir: Path, day: Date) -> list[Alert]:
    directory = alerts_dir(data_dir, day)
    out: list[Alert] = []
    for path in sorted(directory.glob("*.csv")) if directory.is_dir() else []:
        with path.open(newline="") as fh:
            out.extend(Alert(**{k: r.get(k, "") for k in FIELDS}) for r in csv.DictReader(fh))
    return out


SETTLED = "settled"


def _latest(alerts: Iterable[Alert]) -> tuple[dict[str, str], dict[str, str]]:
    news: dict[str, str] = {}
    calm: dict[str, str] = {}
    for a in alerts:
        if not a.event_id:
            continue
        table = calm if a.kind == SETTLED else news
        if a.detected_at > table.get(a.event_id, ""):
            table[a.event_id] = a.detected_at
    return news, calm


def unsettled(alerts: Iterable[Alert]) -> set[str]:
    """Events whose latest news has not yet been followed by a settled line."""
    news, calm = _latest(alerts)
    return {eid for eid, at in news.items() if calm.get(eid, "") <= at}


def pending(alerts: Iterable[Alert], priced_at: Mapping[str, str]) -> set[str]:
    """Events that may not be recommended.

    An alerted game is pending until its line has settled after the latest news
    *and* a pricing run strictly newer than the settle has seen it.
    ``priced_at`` maps event id -> ISO UTC time of the last pricing run that saw
    the game.
    """
    news, calm = _latest(alerts)
    out: set[str] = set()
    for eid, at in news.items():
        released = calm.get(eid, "")
        if released <= at or priced_at.get(eid, "") <= released:
            out.add(eid)
    return out


def _quiet(a: Mapping[str, float], b: Mapping[str, float], params: AlarmParams) -> bool:
    if not a or set(a) != set(b):
        return False
    for key, value in a.items():
        tol = params.settle_ml if key == "ml" else params.settle_pts
        if abs(b[key] - value) >= tol:
            return False
    return True


def settle(
    events: Iterable[str],
    first: Mapping[str, Mapping[str, float]],
    poll: Callable[[set[str]], Mapping[str, Mapping[str, float]]],
    params: AlarmParams,
    sleep: Callable[[float], None],
) -> dict[str, list[Mapping[str, float]]]:
    """Fast-poll alerted games until each line holds still; return the settled ones.

    ``first`` is each game's consensus at the alert; every round sleeps
    ``settle_interval_s`` then polls the games not yet settled. A game settles
    when its last two moves are both inside the tolerance (three looks) on every
    market it had at the alert; a game whose market was pulled or keeps moving
    stays unsettled for the next tick.
    Returns event id -> the consensus path that settled it.
    """
    seen: dict[str, list[Mapping[str, float]]] = {
        eid: [first[eid]] if eid in first else [] for eid in events
    }
    needed = {eid: set(first.get(eid, {})) for eid in seen}
    done: dict[str, list[Mapping[str, float]]] = {}
    for _ in range(params.settle_polls):
        open_ = set(seen) - set(done)
        if not open_:
            break
        sleep(params.settle_interval_s)
        now = poll(open_)
        for eid in open_:
            if eid in now:
                seen[eid].append(now[eid])
            path = seen[eid]
            if (
                len(path) >= 3
                and needed[eid] <= set(path[-1])
                and _quiet(path[-3], path[-2], params)
                and _quiet(path[-2], path[-1], params)
            ):
                done[eid] = path
    return done


def settled_alerts(
    done: Mapping[str, Sequence[Mapping[str, float]]],
    games: Mapping[str, Game],
    detected_at: str,
) -> list[Alert]:
    out: list[Alert] = []
    for eid, path in sorted(done.items()):
        game = games.get(eid)
        if game is None or not path:
            continue
        out.append(
            Alert(
                detected_at=detected_at,
                game_date=game.game_date.isoformat(),
                event_id=eid,
                matchup=game.matchup,
                kind=SETTLED,
                detail=f"line steady over the last {len(path) - 1} poll(s)",
                before=json.dumps(dict(path[0]), sort_keys=True),
                after=json.dumps(dict(path[-1]), sort_keys=True),
            )
        )
    return out


__all__ = [
    "FIELDS",
    "SETTLED",
    "Alert",
    "alerts_dir",
    "board_alerts",
    "consensus",
    "feed_alerts",
    "move_alerts",
    "official_alerts",
    "pending",
    "read_alerts",
    "reference",
    "settle",
    "settled_alerts",
    "unsettled",
    "write_alerts",
]
