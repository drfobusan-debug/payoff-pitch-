"""Did we hear it before the number moved?

The historical study says a team missing an established starting quarterback is
worth -2.2 points against the *closing* spread, and that the entire effect lives
in the first game of the absence -- 59.6% fading in the 2024-25 holdout against
47.1% once the backup is common knowledge. Box scores cannot say when the news
broke, so they cannot say whether a live engine gets there in time. That is the
one thing this reader measures.

For every absence in the availability log:

* **lead** -- our first sighting minus the feed's own posting time. Large is bad
  in a different way from small: it means we polled late.
* **move after** -- how far the spread travelled between our first sighting and
  our last one, signed so positive means the market moved *against* the team
  missing him. Points still on the table after we knew.

If ``move after`` is near zero the market had already absorbed the news and the
coefficient should stay at 0.0; if the line keeps moving, the points are real and
``CFBE_INJURY_QB_PTS`` has something to buy.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sighting:
    game: str
    team: str
    player: str
    position: str
    designation: str
    first_seen: datetime
    last_seen: datetime
    news_posted: datetime | None
    first_spread: float | None
    last_spread: float | None
    observations: int
    team_is_home: bool

    @property
    def lead_s(self) -> float | None:
        """Seconds between the news being posted and us reading it."""
        if self.news_posted is None:
            return None
        return (self.first_seen - self.news_posted).total_seconds()

    @property
    def move_after(self) -> float | None:
        """Spread movement against the short-handed team after our first sighting."""
        if self.first_spread is None or self.last_spread is None:
            return None
        delta = self.last_spread - self.first_spread
        # The log stores the home spread; a home absence moves it up (less
        # favoured), so flip the sign to read "against the team missing him".
        return delta if self.team_is_home else -delta


def _stamp(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def read_log(path: Path) -> list[Sighting]:
    """Collapse the append-only log into one row per player per game."""
    if not path.exists():
        return []
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                log.warning("skipping unparseable availability row")
                continue
            if not isinstance(row, dict):
                continue
            game = f"{row.get('away', '')} @ {row.get('home', '')}"
            key = (game, str(row.get("player_id") or row.get("player", "")))
            groups.setdefault(key, []).append(row)

    out: list[Sighting] = []
    for (game, _pid), rows in groups.items():
        ordered = sorted(rows, key=lambda r: str(r.get("observed_at", "")))
        first, last = ordered[0], ordered[-1]
        seen_first = _stamp(first.get("observed_at"))
        seen_last = _stamp(last.get("observed_at"))
        if seen_first is None or seen_last is None:
            continue
        out.append(
            Sighting(
                game=game,
                team=str(first.get("team", "")),
                player=str(first.get("player", "")),
                position=str(first.get("position", "")),
                designation=str(last.get("designation", "")),
                first_seen=seen_first,
                last_seen=seen_last,
                news_posted=_stamp(first.get("news_posted_at")),
                first_spread=_float(first.get("spread_home")),
                last_spread=_float(last.get("spread_home")),
                observations=len(ordered),
                team_is_home=str(first.get("home_key", "")) == str(first.get("team", "")),
            )
        )
    return sorted(out, key=lambda s: s.first_seen)


def _float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def summarize(sightings: list[Sighting]) -> str:
    """The verdict paragraph: lead time, and points left after we knew."""
    if not sightings:
        return "No absences logged yet."
    leads = [s.lead_s / 3600.0 for s in sightings if s.lead_s is not None]
    moves = [s.move_after for s in sightings if s.move_after is not None]
    lines = [
        f"{len(sightings)} absences logged, {len(leads)} with a posting time.",
    ]
    if leads:
        lines.append(
            f"  lead time (news -> first sighting): median {median(leads):+.1f}h, "
            f"best {min(leads):+.1f}h, worst {max(leads):+.1f}h"
        )
    if moves:
        against = sum(1 for m in moves if m > 0.25)
        lines.append(
            f"  spread movement after we knew: median {median(moves):+.2f} pts, "
            f"{against}/{len(moves)} still moved 0.25+ against the team"
        )
        lines.append(
            "  points left on the table after detection is what CFBE_INJURY_QB_PTS "
            "would be buying."
        )
    else:
        lines.append("  no absence has been seen twice yet, so no movement to measure.")
    return "\n".join(lines)
