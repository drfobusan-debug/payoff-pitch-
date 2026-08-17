"""Who is actually starting, and the one point-swing the rating cannot see.

The rating layer is a team rating: it carries a team's opponent-adjusted offence
forward from the plays that team's offence ran. When the man who ran them is not
the man taking the snaps on Sunday, the rating is describing somebody else.

This module answers that with the only quarterback fact the panel already
carries -- who started -- and separates two things that "the quarterback
changed" had been pooling. Both are measured on the walk-forward rating error
over 3,450 graded games, 2013-2025, ratings from prior weeks only
(``scripts/nfl/september_study.py --qb``), where a *positive* bias means the
rating was too high on the home side:

    a genuine change: this man is the team's own starter now
        home  bias +1.121  t +1.14        away  bias +0.393  t +0.38
    a fill-in: neither last season's starter nor this season's incumbent
        home  bias +4.396  t +5.83        away  bias -3.945  t -5.49

with both sides' own starter in, the rating is unbiased (-0.172, t -0.64 over
2,420 games), which is what makes the fill-in rows readable as an error and not
as a home-field miscalibration.

**An offseason quarterback change is not a rating problem.** A team that signed
or drafted its starter rates correctly from the start, which is worth stating
because it is the opposite of the intuition -- and it is why this module charges
nothing in week 1. What the rating cannot see is a *displaced* incumbent: the
starter is hurt, and the rating is still pricing the offence he ran.

The definition has to survive kickoff
-------------------------------------
"not this season's primary starter" is future information in week 5, because who
ends up with the most starts is not known yet. The shipped test uses only prior
weeks -- neither last season's primary nor the man with the most starts *so far*
this season -- and the effect survives it, weaker and with a sharp split:

    weeks 5+    rating -3.816 (t -8.11)   closing line -1.558 (t -3.41)
    weeks 1-4   rating +1.138 (t +1.22)   closing line +2.593 (t +2.82)

The September rows reverse sign, which is the same finding again: before a team
has an incumbent, "not last year's man" means the new starter, not a backup. So
the charge is conditioned on there being an incumbent to displace, and
``FILL_IN_MIN_WEEK`` is where that becomes true rather than a taste in weeks.

Why the charge is 3 points and not 4
------------------------------------
The slope fitted on weeks 5+ of prior seasons only is stable and slowly decaying
-- about -4.9 as of 2020, -3.90 as of 2025 -- and the shipped constant is 75% of
it, chosen by walk-forward margin MAE over *every* game and not only the 584 it
touches:

    share of fitted charge   25%      50%      75%     100%
    charge as of 2025      -0.98    -1.95    -2.93    -3.90
    margin MAE            10.1542  10.1320  10.1282  10.1418

against 10.1912 uncorrected. The optimum is shallow, which is what a real effect
looks like; charging the full slope measures worse, because the fitted slope
contains the games where the backup was in *because* the game was already lost.

**This is a rating correction, not a bet.** Fading the fill-in side against the
closing line covers 50.55% over 902 games -- under the 52.4% that -110 needs --
and it is 38.76% in weeks 1-4. The line's own miss is real but small (-0.74,
t -1.79 pooled), and ``adjustments.py`` already carries a backup-quarterback
term at zero weight for the same reason. What this fixes is the mean the engine
uses when no market has posted, and the game script the props layer conditions
on, where nobody publishes an answer to check against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# 75% of the -3.90 fitted on weeks 5+ through 2025. See the module docstring for
# the walk-forward table this share comes from.
FILL_IN_MARGIN_POINTS = -3.0
# Below this week a team has too little of its own season for "not the incumbent"
# to mean a displaced starter rather than a new one, and the measured bias
# reverses: +1.14 the other way in weeks 1-4, against -3.82 from week 5 on.
FILL_IN_MIN_WEEK = 5

INCUMBENT = "incumbent"
FILL_IN = "fill_in"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class StarterBook:
    """Who each team's quarterback was, as of every week, from prior weeks only.

    ``prior`` is last season's primary starter keyed by the season it informs, so
    a 2025 lookup returns the man who made the most 2024 starts. ``incumbent`` is
    the man with the most starts *before* the week asked about, which is what
    makes a week-5 answer available in week 5.
    """

    prior: dict[tuple[int, str], str] = field(default_factory=dict)
    incumbent: dict[tuple[int, int, str], str] = field(default_factory=dict)

    def status(self, season: int, week: int, team: str, qb: str | None) -> str:
        """``INCUMBENT``, ``FILL_IN`` or ``UNKNOWN`` for one side of one game.

        ``UNKNOWN`` when the starter is not named, or when the team has neither a
        prior-season starter nor a start this season -- an expansion team or the
        first week of the earliest season in the book. Unknown charges nothing.
        """
        if not qb:
            return UNKNOWN
        last = self.prior.get((season, team))
        held = self.incumbent.get((season, week, team))
        if last is None and held is None:
            return UNKNOWN
        if qb in (last, held):
            return INCUMBENT
        return FILL_IN

    def is_fill_in(self, season: int, week: int, team: str, qb: str | None) -> bool:
        return self.status(season, week, team, qb) == FILL_IN


def build(games: pd.DataFrame) -> StarterBook:
    """Read both books out of the schedule's own ``home_qb_id``/``away_qb_id``."""
    need = {"season", "week", "home_team", "away_team", "home_qb_id", "away_qb_id"}
    if games.empty or not need <= set(games.columns):
        return StarterBook()
    long = pd.concat(
        [
            games[["season", "week", "home_team", "home_qb_id"]].rename(
                columns={"home_team": "team", "home_qb_id": "qb"}
            ),
            games[["season", "week", "away_team", "away_qb_id"]].rename(
                columns={"away_team": "team", "away_qb_id": "qb"}
            ),
        ]
    ).dropna(subset=["team", "qb"])
    if long.empty:
        return StarterBook()
    long = long.astype({"season": int, "week": int}).sort_values(["season", "team", "week"])

    counts = long.groupby(["season", "team", "qb"]).size().rename("n").reset_index()
    primary = counts.sort_values("n", ascending=False).drop_duplicates(["season", "team"])
    # Keyed to the season the rating is *for*, so the lookup needs no arithmetic.
    prior = {(int(row.season) + 1, str(row.team)): str(row.qb) for row in primary.itertuples()}

    incumbent: dict[tuple[int, int, str], str] = {}
    for (season, team), grp in long.groupby(["season", "team"], sort=False):
        seen: dict[str, int] = {}
        for row in grp.itertuples():
            if seen:
                leader = max(seen, key=lambda q: seen[q])
                incumbent[(int(season), int(row.week), str(team))] = leader
            seen[str(row.qb)] = seen.get(str(row.qb), 0) + 1
    return StarterBook(prior=prior, incumbent=incumbent)


def margin_delta(
    book: StarterBook,
    *,
    season: int,
    week: int,
    home: str,
    away: str,
    home_qb: str | None,
    away_qb: str | None,
    points: float = FILL_IN_MARGIN_POINTS,
    min_week: int = FILL_IN_MIN_WEEK,
) -> tuple[float, tuple[str, ...]]:
    """Points to add to the ratings-implied home margin, and why.

    A fill-in on the home side is worth ``points`` (negative) to the home margin
    and a fill-in on the away side the same against the away side, so a game with
    a backup on both sides nets to nothing -- which is correct, and is what
    fitting a single signed slope on ``away - home`` measured.
    """
    if week < min_week:
        return 0.0, ()
    delta = 0.0
    notes: list[str] = []
    if book.is_fill_in(season, week, home, home_qb):
        delta += points
        notes.append(f"{home} starting a fill-in QB {points:+.1f} margin")
    if book.is_fill_in(season, week, away, away_qb):
        delta -= points
        notes.append(f"{away} starting a fill-in QB {-points:+.1f} margin")
    return delta, tuple(notes)
