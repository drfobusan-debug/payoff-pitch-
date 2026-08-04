"""Opponent-adjusted per-play efficiency, fit from CFBD's game-level PPA.

Why this exists
---------------
Points-per-play (CFBD's ``ppa``, the college analogue of EPA/play) is the most
predictive single number in football, but raw per-play numbers are schedule
artefacts: 0.30 PPA against the Sun Belt is not 0.30 PPA against the SEC. So
every team-game is fed into one ridge

    ppa_offence(team, opponent, site) = mu + off[team] - def[opponent] + hfa*site

which is the standard adjusted-plus-minus decomposition. The fit is **leak-free
by construction**: :meth:`EfficiencyProvider.book` only ever reads games played
strictly before the slate week, so a rating never contains the game it prices.

What it is worth, measured
--------------------------
Fit week by week over 2014-2025 (6,513 games with a closing spread):

* as a standalone rating it predicts held-out margins at r 0.56 / MAE 13.1,
  against the closing spread's r 0.63 / MAE 12.4 -- respectable, and worse than
  the market;
* season-to-season it repeats at r 0.69, versus r 0.60 for a ratings fit on
  scoring margin, so per-play efficiency really is the more durable signal;
* **but after the closing spread is in the model it adds nothing**: partial
  correlation with margin -0.001 (season-clustered 95% CI [-0.022, +0.020]),
  held-out MAE 12.218 -> 12.220, and betting the disagreements goes 50.1% ATS
  (-4.4% ROI, i.e. exactly the hold).

So this ships with ``efficiency_blend`` at 0: it is the honest **fallback** when
SP+ is unavailable -- a real model instead of echoing the market back at itself
-- and an opt-in blend otherwise. Raising the blend is not supported by the
measurement above.

Note on defence signs
---------------------
A defence enters the design matrix with a -1, so its fitted coefficient is
already sign-flipped: a stingy defence carries a *positive* value. Net quality is
therefore ``off + def``. Getting this backwards produces ratings that look
plausible and correlate 0.09 with SP+, which is how the sign was caught.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from cfb_engine.data.cfbd import CFBDClient, RatingBook, TeamRating
from cfb_engine.data.teamnames import school_key

log = logging.getLogger(__name__)

# Ridge penalty on the team coefficients. Light: a team has only a handful of
# games early in the season and the response (PPA/play) has an SD near 0.25, so a
# heavy penalty flattens every team onto the intercept.
RIDGE_ALPHA = 0.4

# Points of final margin per unit of adjusted PPA gap, fit on 2014-2025
# (60.7 pts per unit; the gap's SD is ~0.10, so a typical matchup is ~6 points).
POINTS_PER_PPA = 60.7

# Minimum games a team needs before its rating is trusted at all.
MIN_GAMES = 3


@dataclass(frozen=True)
class TeamEfficiency:
    """One team's opponent-adjusted per-play offence and defence."""

    team: str
    offence: float  # PPA/play generated vs an average defence
    defence: float  # PPA/play suppressed vs an average offence (higher = better)
    games: int

    @property
    def net(self) -> float:
        return self.offence + self.defence


@dataclass
class EfficiencyBook:
    ratings: dict[str, TeamEfficiency] = field(default_factory=dict)
    hfa: float = 0.0

    def get(self, team_name: str) -> TeamEfficiency | None:
        eff = self.ratings.get(school_key(team_name))
        if eff is None or eff.games < MIN_GAMES:
            return None
        return eff

    def net_gap_points(self, home: str, away: str) -> float | None:
        """Expected home margin from efficiency alone, before home field."""
        h, a = self.get(home), self.get(away)
        if h is None or a is None:
            return None
        return (h.net - a.net) * POINTS_PER_PPA

    def as_rating_book(self, league_avg: float, points_per_ppa: float = POINTS_PER_PPA) -> RatingBook | None:
        """Express the book on the points scale the pipeline's ratings use.

        Offence and defence are split symmetrically around ``league_avg`` so the
        implied total stays at the league baseline: efficiency informs the margin,
        not scoring pace.
        """
        out: dict[str, TeamRating] = {}
        for key, eff in self.ratings.items():
            if eff.games < MIN_GAMES:
                continue
            half = eff.net * points_per_ppa / 2.0
            out[key] = TeamRating(eff.team, league_avg + half, league_avg - half)
        return RatingBook(ratings=out, league_avg=league_avg) if out else None


def fit_efficiency(rows: list[tuple[str, str, float, float]], alpha: float = RIDGE_ALPHA) -> EfficiencyBook:
    """Ridge-fit offence/defence coefficients.

    ``rows`` are ``(team, opponent, site, ppa)`` where site is +1 at home, -1 on
    the road and 0 at a neutral site.
    """
    import numpy as np

    if not rows:
        return EfficiencyBook()
    teams = sorted({r[0] for r in rows} | {r[1] for r in rows})
    idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    width = 2 * n_teams + 2
    design = np.zeros((len(rows), width))
    target = np.zeros(len(rows))
    played: dict[str, int] = dict.fromkeys(teams, 0)
    for i, (team, opp, site, ppa) in enumerate(rows):
        design[i, idx[team]] = 1.0
        design[i, n_teams + idx[opp]] = -1.0
        design[i, -2] = site
        design[i, -1] = 1.0
        target[i] = ppa
        played[team] += 1

    penalty = np.full(width, alpha)
    penalty[-2:] = 1e-8  # never shrink home field or the intercept
    try:
        coef = np.linalg.solve(design.T @ design + np.diag(penalty), design.T @ target)
    except np.linalg.LinAlgError:  # pragma: no cover - singular is not reachable with a ridge
        log.warning("efficiency ridge failed to solve; returning an empty book")
        return EfficiencyBook()

    ratings = {
        school_key(team): TeamEfficiency(
            team=team,
            offence=float(coef[idx[team]]),
            defence=float(coef[n_teams + idx[team]]),
            games=played[team],
        )
        for team in teams
    }
    return EfficiencyBook(ratings=ratings, hfa=float(coef[-2]))


def blend_efficiency(
    base: RatingBook | None,
    book: EfficiencyBook | None,
    *,
    blend: float,
    league_avg: float,
) -> RatingBook | None:
    """Pull each team's net rating toward its adjusted-efficiency rating.

    With no SP+ base the efficiency book becomes the ratings on its own -- the
    fallback this module exists for. Totals are preserved: only the net moves,
    because efficiency informs margin rather than scoring pace.
    """
    if book is None:
        return base
    if base is None:
        return book.as_rating_book(league_avg)
    weight = min(max(blend, 0.0), 1.0)
    if weight <= 0.0:
        return base
    out: dict[str, TeamRating] = {}
    for key, rating in base.ratings.items():
        eff = book.get(rating.team)
        if eff is None:
            out[key] = rating
            continue
        total = rating.offense + rating.defense
        new_net = (1 - weight) * (rating.offense - rating.defense) + weight * (
            eff.net * POINTS_PER_PPA
        )
        out[key] = TeamRating(rating.team, (total + new_net) / 2, (total - new_net) / 2)
    return RatingBook(ratings=out, league_avg=base.league_avg)


class EfficiencyProvider:
    """Builds an :class:`EfficiencyBook` from CFBD, cached per (season, week)."""

    def __init__(self, cfbd: CFBDClient) -> None:
        self.cfbd = cfbd
        self._cache: dict[tuple[int, int], EfficiencyBook] = {}

    def book(self, season: int, before_week: int) -> EfficiencyBook | None:
        """Ratings from every game played before ``before_week`` this season.

        Falls back to the previous season in full when the current one has too
        few games to fit (preseason and the opening weeks).
        """
        key = (season, before_week)
        if key in self._cache:
            return self._cache[key]
        rows = self._rows(season, before_week)
        if not self._fittable(rows):
            prev = self._rows(season - 1, 99)
            if not self._fittable(prev):
                return None
            log.info(
                "efficiency: only %d team-games so far in %d; using %d from the prior season",
                len(rows),
                season,
                len(prev),
            )
            rows = prev
        book = fit_efficiency(rows)
        self._cache[key] = book
        return book

    @staticmethod
    def _fittable(rows: list[tuple[str, str, float, float]]) -> bool:
        """Enough of the league has played ``MIN_GAMES`` for the fit to be useful.

        A week-2 fit is technically solvable but every team sits below MIN_GAMES,
        so the book would be silently empty; requiring half the league past the
        threshold is what makes the prior-season fallback fire in September.
        """
        if not rows:
            return False
        counts: dict[str, int] = {}
        for team, _opp, _site, _ppa in rows:
            counts[team] = counts.get(team, 0) + 1
        ready = sum(1 for n in counts.values() if n >= MIN_GAMES)
        return ready * 2 >= len(counts)

    def _rows(self, season: int, before_week: int) -> list[tuple[str, str, float, float]]:
        games = self.cfbd.fetch_team_game_ppa(season)
        neutral = self.cfbd.neutral_game_ids(season)
        out: list[tuple[str, str, float, float]] = []
        for row in games:
            if row.week >= before_week and row.season_type == "regular":
                continue
            if row.game_id in neutral:
                site = 0.0
            else:
                site = 1.0 if row.home else -1.0
            out.append((row.team, row.opponent, site, row.offence_ppa))
        return out
