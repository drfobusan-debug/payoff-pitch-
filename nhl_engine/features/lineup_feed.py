"""Turn a RotoWire slate into starter overrides and availability records.

Goalies: a ``Confirmed`` goalie becomes a ``confirmed`` override, ``Expected``
a ``probable`` one -- both clear the ``goalie_unconfirmed`` gate. An existing
override is never downgraded (a hand-entered ``confirmed`` beats a RotoWire
``Expected``); it is replaced when RotoWire names a different goalie at an
equal or stronger status, because the newer read is the better one.

Injuries: RotoWire designations map onto the availability ladder::

    IR, IR-NR, IR-LT, O, OUT, SUSP  -> out
    D, DTD-D                         -> doubtful
    DTD, Q, GTD                      -> questionable

Names resolve to NHL ids through MoneyPuck's season summaries (this season,
then last) by normalised full name within the team; ``M. Blackwood`` style
short names fall back to first-initial + surname within the team. Nothing
here changes a price directly: the override feeds ``starter_for`` and the
availability log feeds the lineup rebuild, both of which already carry their
own evidence rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date as Date
from pathlib import Path

import pandas as pd

from nhl_engine.data import availability as avail
from nhl_engine.data.capture import QuoteRow
from nhl_engine.data.moneypuck import MoneyPuckClient
from nhl_engine.data.rotowire import SOURCE, RotoPlayer, RotoSlate, norm_name
from nhl_engine.data.teamnames import canonical
from nhl_engine.features import starters

log = logging.getLogger("nhl_engine")

RANK = {"projected": 0, "probable": 1, "confirmed": 2}

INJURY_MAP = {
    "IR": "out",
    "IR-NR": "out",
    "IR-LT": "out",
    "LTIR": "out",
    "O": "out",
    "OUT": "out",
    "SUSP": "out",
    "D": "doubtful",
    "DTD-D": "doubtful",
    "DTD": "questionable",
    "Q": "questionable",
    "GTD": "questionable",
}


@dataclass
class FeedSummary:
    goalies_set: list[str] = field(default_factory=list)
    goalies_kept: list[str] = field(default_factory=list)
    goalies_unresolved: list[str] = field(default_factory=list)
    injuries_logged: int = 0
    injuries_unresolved: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [
            f"goalies: {len(self.goalies_set)} set, {len(self.goalies_kept)} kept, "
            f"{len(self.goalies_unresolved)} unresolved",
            f"injuries: {self.injuries_logged} new record(s), "
            f"{len(self.injuries_unresolved)} name(s) without an NHL id",
        ]
        out += [f"  {g}" for g in self.goalies_set]
        out += [f"  kept {g}" for g in self.goalies_kept]
        out += [f"  unresolved goalie {g}" for g in self.goalies_unresolved]
        return out


class NameResolver:
    """Full-name -> NHL id within a team, from MoneyPuck season summaries."""

    def __init__(self, mp: MoneyPuckClient, season: int) -> None:
        self._mp = mp
        self._season = season
        self._tables: dict[str, pd.DataFrame] = {}

    def _table(self, kind: str) -> pd.DataFrame:
        if kind not in self._tables:
            frames = []
            for s in (self._season, self._season - 1):
                t = self._mp.season_goalies(s) if kind == "G" else self._mp.season_skaters(s)
                if not t.empty and {"playerId", "name", "team", "situation"} <= set(t.columns):
                    frames.append(t[t["situation"] == "all"][["playerId", "name", "team"]])
            if frames:
                df = pd.concat(frames, ignore_index=True).drop_duplicates("playerId", keep="first")
                df = df.assign(key=df["name"].map(norm_name), code=df["team"].map(canonical))
            else:
                df = pd.DataFrame(columns=["playerId", "name", "team", "key", "code"])
            self._tables[kind] = df
        return self._tables[kind]

    def resolve(self, name: str, team: str, kind: str) -> tuple[int, str] | None:
        df = self._table(kind)
        if df.empty:
            return None
        key = norm_name(name)
        team = canonical(team)
        hit = df[df["key"] == key]
        if len(hit) > 1:
            hit = hit[hit["code"] == team] if (hit["code"] == team).any() else hit.iloc[:1]
        if hit.empty:
            parts = key.split()
            if len(parts) >= 2 and len(parts[0]) == 1:
                initial, last = parts[0], " ".join(parts[1:])
                cand = df[(df["code"] == team) & df["key"].str.endswith(" " + last)]
                cand = cand[cand["key"].str.startswith(initial)]
                if len(cand) == 1:
                    hit = cand
        if hit.empty or len(hit) > 1:
            return None
        row = hit.iloc[0]
        return int(row["playerId"]), str(row["name"])


def _role(p: RotoPlayer) -> str:
    if p.pos == "G":
        return "G2"
    if p.pos in ("LD", "RD", "D"):
        return "D"
    return "F"


def apply_slate(
    roto: RotoSlate,
    *,
    mp: MoneyPuckClient,
    data_dir: Path,
    season: int,
    quotes: list[QuoteRow] | None = None,
    seen_at: str | None = None,
) -> FeedSummary:
    slate: Date = roto.slate
    summary = FeedSummary()
    resolver = NameResolver(mp, season)
    ov_path = starters.overrides_path(data_dir, slate)
    current = starters.load_overrides(ov_path)
    records: list[avail.Availability] = []
    now = seen_at or roto.fetched_at or avail.now_utc()

    for game in roto.games:
        market = avail.market_snapshot(quotes or [], game.matchup)
        for side in (game.away, game.home):
            team = side.team
            if side.goalie and side.goalie_status:
                hit = resolver.resolve(side.goalie, team, "G")
                if hit is None:
                    summary.goalies_unresolved.append(f"{team} {side.goalie}")
                else:
                    pid, name = hit
                    have = current.get(team) or {}
                    have_rank = RANK.get(str(have.get("status", "")), -1) if have else -1
                    new_rank = RANK[side.goalie_status]
                    same = int(str(have.get("player_id", 0))) == pid
                    if have and (have_rank > new_rank or (same and have_rank == new_rank)):
                        summary.goalies_kept.append(
                            f"{team} {have.get('name', have.get('player_id'))} "
                            f"{have.get('status')} ({have.get('source')})"
                        )
                    else:
                        starters.save_override(
                            ov_path, team, pid, side.goalie_status, SOURCE, name=name
                        )
                        current = starters.load_overrides(ov_path)
                        summary.goalies_set.append(f"{team} {name} {side.goalie_status}")
            for p in side.injuries:
                status = INJURY_MAP.get(p.status.upper())
                if status is None:
                    continue
                kind = "G" if p.pos == "G" else "S"
                hit = resolver.resolve(p.name, team, kind)
                pid, name = hit if hit else (0, p.name)
                if hit is None:
                    summary.injuries_unresolved.append(f"{team} {p.name}")
                records.append(
                    avail.Availability(
                        slate=slate.isoformat(),
                        team=team,
                        player_id=pid,
                        name=name,
                        status=status,
                        source=SOURCE,
                        posted_at="",
                        seen_at=now,
                        role=_role(p),
                        market_home_ml=market[0],
                        market_book=market[1],
                        note=p.status,
                    )
                )
    written = avail.append(data_dir, records)
    summary.injuries_logged = len(written)
    return summary


__all__ = ["INJURY_MAP", "FeedSummary", "NameResolver", "apply_slate"]
