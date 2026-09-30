"""Posted lineups, stamped with the moment the engine first saw them.

The box score says who batted; it does not say when that was known. Any study
of lineup intent -- a clinched team resting regulars, an eliminated team
playing its call-ups -- needs the nine names *as posted before first pitch*,
and the only way to have that later is to write it down now. Every pass that
fetches the slate records each confirmed lineup the first time it appears,
with the capture time and the lead to first pitch. A lineup that changes after
that (a scratch) keeps its first view and appends the revision, so the file
holds both what was posted and what was finally sent out.

No scoring reads this file. It is a ledger for next season's calibration.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from mlb_engine.schemas import Slate, TeamGameInfo


@dataclass(frozen=True)
class LineupPlayer:
    order: int
    mlbam_id: int
    name: str
    bats: str | None = None
    position: str | None = None


@dataclass
class LineupRevision:
    captured_at: str
    players: list[LineupPlayer]


@dataclass
class LineupCapture:
    game_pk: int
    side: str  # "away" | "home"
    team: str
    opponent: str
    first_pitch_utc: str | None
    captured_at: str
    lead_hours: float | None
    starter: str | None
    starter_id: int | None
    starter_throws: str | None
    players: list[LineupPlayer]
    revisions: list[LineupRevision] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.game_pk}:{self.side}"

    @property
    def player_ids(self) -> tuple[int, ...]:
        return tuple(p.mlbam_id for p in self.players)

    @property
    def final_players(self) -> list[LineupPlayer]:
        return self.revisions[-1].players if self.revisions else self.players


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _lead_hours(first_pitch: str | None, now: datetime) -> float | None:
    if not first_pitch:
        return None
    try:
        fp = datetime.fromisoformat(first_pitch.replace("Z", "+00:00"))
    except ValueError:
        return None
    return round((fp - now.astimezone(timezone.utc)).total_seconds() / 3600, 2)


def _players(team: TeamGameInfo) -> list[LineupPlayer]:
    return [
        LineupPlayer(
            order=slot.order,
            mlbam_id=slot.player.mlbam_id,
            name=slot.player.name,
            bats=slot.player.bats.value if slot.player.bats else None,
            position=slot.player.position,
        )
        for slot in team.lineup
    ]


def captures_from_slate(slate: Slate, now: datetime | None = None) -> list[LineupCapture]:
    """One capture per confirmed lineup on the slate, stamped ``now``."""
    now = now or datetime.now(timezone.utc)
    stamp = _iso(now)
    out: list[LineupCapture] = []
    for game in slate.games:
        for side, team, opp in (("away", game.away, game.home), ("home", game.home, game.away)):
            if not team.lineup_confirmed():
                continue
            sp = opp.probable_pitcher
            out.append(
                LineupCapture(
                    game_pk=game.game_pk,
                    side=side,
                    team=team.abbrev,
                    opponent=opp.abbrev,
                    first_pitch_utc=game.game_datetime_utc,
                    captured_at=stamp,
                    lead_hours=_lead_hours(game.game_datetime_utc, now),
                    starter=sp.name if sp else None,
                    starter_id=sp.mlbam_id if sp else None,
                    starter_throws=sp.throws.value if sp and sp.throws else None,
                    players=_players(team),
                )
            )
    return out


def merge_lineups(
    old: dict[str, LineupCapture], new: list[LineupCapture]
) -> dict[str, LineupCapture]:
    """Union two captures; the earliest view of a lineup is the one that stays.

    The point of the file is *when* the nine were known, so a later capture
    never overwrites an earlier one. If the names differ it is appended as a
    revision instead, in capture order, so a scratch is visible without
    losing the lineup as first posted. Captures travel between machines and
    arrive in either order, hence the sort rather than "first call wins".
    """
    merged = {k: v for k, v in old.items()}
    for cap in new:
        seen = merged.get(cap.key)
        if seen is None:
            merged[cap.key] = cap
            continue
        views = [seen, *(_as_capture(seen, r) for r in seen.revisions), cap]
        views.sort(key=lambda c: c.captured_at)
        first = views[0]
        revisions: list[LineupRevision] = []
        last_ids = first.player_ids
        for v in views[1:]:
            if v.player_ids != last_ids:
                revisions.append(LineupRevision(captured_at=v.captured_at, players=list(v.players)))
                last_ids = v.player_ids
        merged[cap.key] = LineupCapture(
            **{**asdict(first), "players": first.players, "revisions": revisions}
        )
    return merged


def _as_capture(base: LineupCapture, rev: LineupRevision) -> LineupCapture:
    return LineupCapture(
        **{
            **asdict(base),
            "captured_at": rev.captured_at,
            "players": rev.players,
            "revisions": [],
        }
    )


def save_lineups(path: Path, captures: dict[str, LineupCapture]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(c) for _, c in sorted(captures.items())]
    path.write_text(json.dumps(rows, indent=1), encoding="utf-8")


def load_lineups(path: Path) -> dict[str, LineupCapture]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    out: dict[str, LineupCapture] = {}
    for r in raw:
        if not isinstance(r, dict):
            continue
        cap = LineupCapture(
            **{
                **r,
                "players": [LineupPlayer(**p) for p in r.get("players", [])],
                "revisions": [
                    LineupRevision(
                        captured_at=v["captured_at"],
                        players=[LineupPlayer(**p) for p in v.get("players", [])],
                    )
                    for v in r.get("revisions", [])
                ],
            }
        )
        out[cap.key] = cap
    return out


__all__ = [
    "LineupCapture",
    "LineupPlayer",
    "LineupRevision",
    "captures_from_slate",
    "load_lineups",
    "merge_lineups",
    "save_lineups",
]
