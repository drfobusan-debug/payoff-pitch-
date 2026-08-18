"""Roster continuity for the ratings-only margin: production kept *and* bought.

Why this exists
---------------
CFBD's returning production (``percentPPA``, see :mod:`cfb_engine.data.returning`)
is the share of last season's own output a team brings back. It was a real
predictor and has decayed into nothing -- measured on games with a closing spread,
the home-minus-away gap carries **+2.82 points per unit (t +4.02) over 2014-2019
and +0.65 (t +1.14) over 2021-2025**, and its mean fell from 0.59 to 0.375.

The decay is substantially a *measurement* failure rather than a fact about
football: the talent did not evaporate, it transferred, and ``percentPPA`` cannot
see an arrival. Crediting each pre-season arrival with the production he posted at
his previous school (``/ppa/players/season``, garbage time excluded) restores it,
on 3,634 games from 2021-2025:

    margin ~ spread + retained                  retained +0.56  t +0.91
    margin ~ spread + retained + bought         retained +1.04  t +1.99
                                                bought   +1.82  t +2.14
    margin ~ spread + (retained + bought)       roster   +1.41  t +3.54

Summing the two beats fitting them apart -- the equal-weight restriction is not
rejected -- so the book carries one share per team. A quarterback multiplier was
tested and goes the *wrong* way: bought QB production carries -1.51 (t -1.61) on
top of bought production generally, i.e. a transfer quarterback's numbers at his
old school predict less than anyone else's, so no positional weight is applied.

Why it is not priced against the market
---------------------------------------
+1.41 points per unit sounds usable and is not: the gap's SD is 0.423, so a
typical game moves 0.60 points, and backing that side goes **51.11% ATS
(1846-1766-22, -2.4% ROI)** against a 52.38% break-even -- *worse* than the
broken retained-only version it corrects (51.50%). Fitted on 2021-2023 and held
out on 2024-2025 it goes 725-750 (49.15%). It is a correctly measured,
unprofitable fact, so nothing here ever contests a closing spread.

What it *is* worth
------------------
The ratings-only margin -- the fallback the engine prices from when a game has no
consensus spread -- has no market number to be redundant with, and there the term
is large and stable. Walk-forward, fitting on all earlier seasons and scoring the
next one, added to the engine's own SP+ log5 margin:

    engine rating margin                RMSE 18.2314   MAE 14.5514
    + returning gap only                RMSE 18.1448   MAE 14.4691
    + roster gap (this module)           RMSE 17.9010   MAE 14.2989

better in all four held-out seasons, with the coefficient landing at +5.98,
+6.40, +6.48 and +7.28 across folds. The dose curve is flat past ~7 and the cap
sweep costs nothing above 8 points (17.9028 vs 17.9010 uncapped, clipping 0.7% of
games), which is where ``FITTED_MAX_PTS`` comes from -- the old 3-point cap would
have given away a third of the gain.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field

from cfb_engine.data.teamnames import school_key

log = logging.getLogger(__name__)

# Points of margin per unit of roster gap, fit on the ratings-only margin and
# held out season by season (folds: +5.98, +6.40, +6.48, +7.28).
FITTED_PTS_PER_UNIT = 6.5
# Cap chosen by sweep, not by eye: RMSE is flat above 8 points and 8 clips 0.7%.
FITTED_MAX_PTS = 8.0
# One arrival cannot be worth more than a whole prior season of a team's output.
# Shares above 1 are a denominator artefact (an elite offence's producer joining a
# weak roster), and the raw maximum observed is 3.08.
SHARE_CAP = 1.0


def normalize_name(name: str) -> str:
    """Fold a player name to the form both CFBD feeds agree on."""
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    for ch in (".", "'", "-"):
        folded = folded.replace(ch, " " if ch == "-" else "")
    return " ".join(folded.lower().split())


@dataclass(frozen=True)
class ProductionBook:
    """A season's player PPA, and each team's total, for the transfer join."""

    players: dict[tuple[str, str], float]  # (normalised name, school_key) -> total PPA
    teams: dict[str, float]  # school_key -> summed total PPA

    def total_for(self, team: str) -> float:
        return self.teams.get(school_key(team), 0.0)


def parse_production(rows: list[dict[str, object]]) -> ProductionBook:
    """Parse ``/ppa/players/season`` rows into a joinable book.

    The endpoint lists only players with real usage (~30 a team), so an arrival
    absent from it contributed ~nothing and is correctly credited zero rather
    than treated as missing data.
    """
    players: dict[tuple[str, str], float] = {}
    teams: dict[str, float] = {}
    for row in rows:
        name, team = row.get("name"), row.get("team")
        totals = row.get("totalPPA")
        total = totals.get("all") if isinstance(totals, dict) else None
        if not isinstance(name, str) or not isinstance(team, str):
            continue
        if not isinstance(total, (int, float)):
            continue
        key = school_key(team)
        players[(normalize_name(name), key)] = float(total)
        teams[key] = teams.get(key, 0.0) + float(total)
    return ProductionBook(players=players, teams=teams)


def build_incoming_shares(
    entries: list[dict[str, object]], prior: ProductionBook, season: int
) -> dict[str, float]:
    """Production each team *bought*, as a share of its own prior-season output.

    Moves dated on or after August 1 are dropped for the same reason
    :func:`cfb_engine.data.portal.build_portal_book` drops them: an in-season
    transfer is not information the pre-season roster had.
    """
    bought: dict[str, float] = {}
    cutoff = f"{season}-08-01"
    for row in entries:
        destination = row.get("destination")
        if not isinstance(destination, str) or not destination:
            continue
        moved = row.get("transferDate")
        if isinstance(moved, str) and moved[:10] >= cutoff:
            continue
        first, last = row.get("firstName"), row.get("lastName")
        origin = row.get("origin")
        if not isinstance(origin, str) or not origin:
            continue
        name = normalize_name(f"{first if isinstance(first, str) else ''} "
                              f"{last if isinstance(last, str) else ''}")
        ppa = prior.players.get((name, school_key(origin)))
        if ppa is None or ppa <= 0:
            continue
        key = school_key(destination)
        bought[key] = bought.get(key, 0.0) + float(ppa)
    return {
        team: min(SHARE_CAP, total / prior.teams[team])
        for team, total in bought.items()
        if prior.teams.get(team, 0.0) > 0
    }


@dataclass
class RosterBook:
    """Share of last season's production a team kept, plus what it bought."""

    retained: dict[str, float]  # school_key -> percentPPA
    bought: dict[str, float] = field(default_factory=dict)

    def share(self, team_name: str) -> float | None:
        """Retained plus bought production, or ``None`` when unmeasured.

        A team missing from the returning-production table is usually an FCS
        opponent, and an unknown share is not a league-average one, so it stays
        ``None`` rather than being imputed. A team merely absent from the portal
        feed bought nothing, which *is* zero.
        """
        kept = self.retained.get(school_key(team_name))
        if kept is None:
            return None
        return kept + self.bought.get(school_key(team_name), 0.0)

    def gap(self, home: str, away: str) -> float | None:
        h, a = self.share(home), self.share(away)
        if h is None or a is None:
            return None
        return h - a

    def margin_delta(self, home: str, away: str, pts_per_unit: float, cap: float) -> float:
        """Points to add to the *ratings-only* home margin for roster continuity."""
        if pts_per_unit <= 0:
            return 0.0
        gap = self.gap(home, away)
        if gap is None:
            return 0.0
        return max(-cap, min(cap, gap * pts_per_unit))
