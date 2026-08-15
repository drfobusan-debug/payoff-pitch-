"""Transfer-portal churn per team, from CFBD's free ``/player/portal`` feed.

Why it is reported and not priced
---------------------------------
Star-weighted net portal talent (incoming minus outgoing, counting only moves
dated before the season opens, so no roster is read backwards) was measured on
3,464 games from 2021-2025 against the **closing spread's residual**:

    net portal gap vs residual        r=+0.0100 (t=+0.59)
    net QB portal gap vs residual     r=+0.0343 (t=+2.02)
    returning production, same test   r=+0.0200 (t=+1.16)
    R2 over the closing spread        +0.00004

Deciles are not monotone (rank vs residual r=+0.0055, p=0.75): the entire effect
sits in the top decile, and inside that decile it is entirely recent -- 2021-2023
gave +0.12 pts and a 52.2% cover, 2024-2025 gave +3.14 pts and 58.9% on n=190.
That may be real (portal volume roughly tripled over the same span) or it may be
one bucket found by looking, which is what returning production turned out to be.

Churn does not widen the distribution either, so it is not usable as an
uncertainty term: mean |residual| is 12.18 / 12.00 / 12.20 / 12.05 / 12.29 points
across churn quintiles, and churn vs |residual| is r=+0.0032 (p=0.85).

So the book is computed, printed on the card, and left out of the price. A season
graded on this basis settles whether the 2024-2025 tail survives contact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from cfb_engine.data.teamnames import school_key

log = logging.getLogger(__name__)

# Recruiting stars are ordinal; these are the conventional talent-point values,
# and they are only ever used to rank churn, never to add points to a rating.
STAR_POINTS: dict[int, float] = {2: 0.3, 3: 1.0, 4: 3.0, 5: 7.0}
# A quarterback moves a line far more than a rotational safety does.
POSITION_WEIGHT: dict[str, float] = {
    "QB": 2.0,
    "RB": 1.0,
    "WR": 1.0,
    "TE": 0.8,
    "OT": 1.2,
    "OL": 1.0,
    "IOL": 1.0,
    "G": 1.0,
    "C": 1.0,
    "EDGE": 1.2,
    "DL": 1.1,
    "DT": 1.0,
    "LB": 0.9,
    "CB": 1.1,
    "S": 0.9,
    "ATH": 0.8,
}
_DEFAULT_WEIGHT = 0.8


@dataclass(frozen=True)
class TeamPortal:
    """One team's pre-season portal ledger, in star-weighted talent points."""

    team: str
    talent_in: float = 0.0
    talent_out: float = 0.0
    players_in: int = 0
    players_out: int = 0
    qb_in: float = 0.0
    qb_out: float = 0.0

    @property
    def net(self) -> float:
        return self.talent_in - self.talent_out

    @property
    def churn(self) -> float:
        return self.talent_in + self.talent_out

    @property
    def qb_net(self) -> float:
        return self.qb_in - self.qb_out

    def summary(self) -> str:
        return (
            f"portal {self.net:+.1f} net talent "
            f"({self.players_in} in / {self.players_out} out)"
        )


PortalBook = dict[str, TeamPortal]


def _player_value(stars: int, position: str) -> float:
    return STAR_POINTS.get(stars, 0.0) * POSITION_WEIGHT.get(position, _DEFAULT_WEIGHT)


def build_portal_book(entries: list[dict[str, object]], season: int) -> PortalBook:
    """Aggregate raw ``/player/portal`` rows into a per-team book.

    Moves dated on or after August 1 of the season are dropped: an in-season
    transfer is not information the pre-season roster had, and including it would
    let a backtest see churn the market could not.
    """
    totals: dict[str, dict[str, float]] = {}

    def bucket(team: str) -> dict[str, float]:
        return totals.setdefault(
            school_key(team),
            {"in": 0.0, "out": 0.0, "n_in": 0.0, "n_out": 0.0, "qb_in": 0.0, "qb_out": 0.0},
        )

    cutoff = f"{season}-08-01"
    for row in entries:
        moved = row.get("transferDate")
        if isinstance(moved, str) and moved[:10] >= cutoff:
            continue
        stars_raw = row.get("stars")
        stars = int(stars_raw) if isinstance(stars_raw, (int, float)) else 0
        position = str(row.get("position") or "").upper()
        value = _player_value(stars, position)
        star_points = STAR_POINTS.get(stars, 0.0)

        destination = row.get("destination")
        if isinstance(destination, str) and destination:
            b = bucket(destination)
            b["in"] += value
            b["n_in"] += 1
            if position == "QB":
                b["qb_in"] += star_points

        origin = row.get("origin")
        if isinstance(origin, str) and origin:
            b = bucket(origin)
            b["out"] += value
            b["n_out"] += 1
            if position == "QB":
                b["qb_out"] += star_points

    return {
        team: TeamPortal(
            team=team,
            talent_in=v["in"],
            talent_out=v["out"],
            players_in=int(v["n_in"]),
            players_out=int(v["n_out"]),
            qb_in=v["qb_in"],
            qb_out=v["qb_out"],
        )
        for team, v in totals.items()
    }


def portal_for(book: PortalBook, team: str) -> TeamPortal | None:
    return book.get(school_key(team))


def portal_note(book: PortalBook, home: str, away: str) -> str | None:
    """A single reported line for the card, or ``None`` when nothing stands out.

    Only the pre-season weeks say anything a reader cannot get from the results,
    and only when the two rosters churned unevenly, so the note is deliberately
    rare rather than printed on every game.
    """
    h, a = portal_for(book, home), portal_for(book, away)
    if h is None or a is None:
        return None
    gap = h.net - a.net
    if abs(gap) < 5.0:
        return None
    ahead, behind = (home, away) if gap > 0 else (away, home)
    return (
        f"Portal: {ahead} out-recruited {behind} by {abs(gap):.0f} talent pts "
        f"[reported, not scored]"
    )
