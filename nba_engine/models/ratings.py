"""Nightly team ratings: offence, defence and pace, opponent-adjusted, updated game by game.

Every final gives three observations:

    home points per 100 = league + hca/2 + off(home) + def(away) + noise
    away points per 100 = league - hca/2 + off(away) + def(home) + noise
    possessions per 48  = league pace + pace(home) + pace(away) + noise

``def`` is points allowed above the league, so a good defence is negative.
Each rating is a Kalman estimate (a mean and a variance). A game moves the two
ratings it observes by the share of the surprise that their variances earn, so
a team with few games moves fast, and one with a settled rating moves slowly.
Between games each rating's variance grows by ``drift``, which is the update speed. At a
team's first game of a new season its rating is shrunk toward zero by
``carry`` and its variance reopened by ``season_sd``. The league levels and
home edge are running means of the same residuals.

A game where either team is on the second night of a back-to-back is observed
with ``tired_weight`` of a normal game's precision: the plan's hypothesis was
that tired games say less about a team, so it is a fitted parameter
(``1.0`` means no discount), not an assumption.

Everything is sequential: ``replay`` predicts each game from the state left by
earlier games only, then updates. It is the no-look-ahead record that the fit
and the close comparison grade. The starting values below are what an unfitted
book uses; the fitted ones come from ``params/team_ratings``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, fields

from nba_engine.data import teamnames
from nba_engine.models.schedule import TeamSchedule, schedule, season_of
from nba_engine.schemas import GameResult

NAME = "team_ratings"
REG_MIN = 48.0
OT_MIN = 5.0
TEAMS = frozenset(teamnames.BY_NAME.values())


@dataclass(frozen=True)
class RatingParams:
    obs_sd: float = 11.0  # points per 100, one team-game
    drift_sd: float = 0.5  # per game
    prior_sd: float = 3.0  # spread of team levels before any game
    carry: float = 0.7  # share of last season's rating kept
    season_sd: float = 2.0  # variance reopened at a new season
    tired_weight: float = 1.0  # precision of a game with a team on a back-to-back
    pace_obs_sd: float = 4.0  # possessions per 48
    pace_drift_sd: float = 0.3
    pace_prior_sd: float = 2.0
    pace_carry: float = 0.7
    pace_season_sd: float = 1.0
    league_gain: float = 0.01
    hca_gain: float = 0.01
    b2b_margin: float = 0.0  # points a back-to-back costs, beyond what the rating sees

    @classmethod
    def from_dict(cls, raw: dict) -> RatingParams:
        names = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in raw.items() if k in names})


@dataclass
class Est:
    mean: float
    var: float

    def drift(self, sd: float) -> None:
        self.var += sd * sd


@dataclass
class TeamState:
    off: Est
    dfn: Est
    pace: Est
    season: int
    games: int = 0


@dataclass(frozen=True)
class Prediction:
    espn_id: str
    game_date: str
    away: str
    home: str
    home_ppp: float
    away_ppp: float
    poss: float  # expected possessions for the whole game, overtime included
    home_games: int  # games behind each rating, this season
    away_games: int

    @property
    def margin(self) -> float:
        """Expected home margin."""
        return (self.home_ppp - self.away_ppp) * self.poss / 100.0

    @property
    def total(self) -> float:
        return (self.home_ppp + self.away_ppp) * self.poss / 100.0


@dataclass
class RatingBook:
    params: RatingParams = field(default_factory=RatingParams)
    teams: dict[str, TeamState] = field(default_factory=dict)
    league_ppp: float = 113.0
    league_pace: float = 100.0
    hca: float = 2.5  # points per 100
    ot_minutes: float = 0.3  # running mean of overtime minutes per game

    def _team(self, team: str, season: int) -> TeamState:
        p = self.params
        st = self.teams.get(team)
        if st is None:
            st = TeamState(
                off=Est(0.0, p.prior_sd**2),
                dfn=Est(0.0, p.prior_sd**2),
                pace=Est(0.0, p.pace_prior_sd**2),
                season=season,
            )
            self.teams[team] = st
        elif st.season != season:
            for est, carry, sd in (
                (st.off, p.carry, p.season_sd),
                (st.dfn, p.carry, p.season_sd),
                (st.pace, p.pace_carry, p.pace_season_sd),
            ):
                est.mean *= carry
                est.var = carry * carry * est.var + sd * sd
            st.season, st.games = season, 0
        return st

    def predict(self, g: GameResult, home_b2b: bool = False, away_b2b: bool = False) -> Prediction:
        """Tonight's expectation from the book as it stands; it does not update it."""
        season = season_of(g.game_date)
        h, a = self._team(g.home, season), self._team(g.away, season)
        edge = 0.0 if g.neutral else self.hca / 2.0
        pace48 = self.league_pace + h.pace.mean + a.pace.mean
        poss = pace48 * (REG_MIN + self.ot_minutes) / REG_MIN
        tired = self.params.b2b_margin * (int(away_b2b) - int(home_b2b)) * 100.0 / poss / 2.0
        return Prediction(
            espn_id=g.espn_id,
            game_date=g.game_date.isoformat(),
            away=g.away,
            home=g.home,
            home_ppp=self.league_ppp + edge + h.off.mean + a.dfn.mean + tired,
            away_ppp=self.league_ppp - edge + a.off.mean + h.dfn.mean - tired,
            poss=poss,
            home_games=h.games,
            away_games=a.games,
        )

    def update(self, g: GameResult, tired: bool = False) -> None:
        """Fold one final into the book (it must have both team boxes)."""
        poss = g.possessions
        if poss is None or poss <= 0:
            return
        p = self.params
        season = season_of(g.game_date)
        h, a = self._team(g.home, season), self._team(g.away, season)
        for est in (h.off, h.dfn, a.off, a.dfn):
            est.drift(p.drift_sd)
        for est in (h.pace, a.pace):
            est.drift(p.pace_drift_sd)
        minutes = REG_MIN + OT_MIN * g.overtimes
        pace48 = poss * REG_MIN / minutes
        weight = p.tired_weight if tired else 1.0
        r_pts = p.obs_sd**2 / weight
        edge = 0.0 if g.neutral else self.hca / 2.0
        res_h = 100.0 * g.final_home / poss - (self.league_ppp + edge + h.off.mean + a.dfn.mean)
        res_a = 100.0 * g.final_away / poss - (self.league_ppp - edge + a.off.mean + h.dfn.mean)
        _observe(h.off, a.dfn, res_h, r_pts)
        _observe(a.off, h.dfn, res_a, r_pts)
        res_p = pace48 - (self.league_pace + h.pace.mean + a.pace.mean)
        _observe(h.pace, a.pace, res_p, p.pace_obs_sd**2 / weight)
        self.league_ppp += p.league_gain * (res_h + res_a) / 2.0
        self.league_pace += p.league_gain * res_p
        if not g.neutral:
            self.hca += p.hca_gain * (res_h - res_a)
        self.ot_minutes += p.league_gain * (minutes - REG_MIN - self.ot_minutes)
        h.games += 1
        a.games += 1

    def table(self) -> list[dict[str, float | str | int]]:
        """Current ratings, best net first."""
        rows: list[dict[str, float | str | int]] = []
        for team, st in self.teams.items():
            rows.append(
                {
                    "team": team,
                    "off": round(self.league_ppp + st.off.mean, 2),
                    "def": round(self.league_ppp + st.dfn.mean, 2),
                    "net": round(st.off.mean - st.dfn.mean, 2),
                    "pace": round(self.league_pace + st.pace.mean, 2),
                    "games": st.games,
                }
            )
        return sorted(rows, key=lambda r: -float(r["net"]))


def _observe(x: Est, y: Est, resid: float, r: float) -> None:
    """One noisy reading of ``x + y``: each moves by its share of the surprise."""
    s = x.var + y.var + r
    kx, ky = x.var / s, y.var / s
    x.mean += kx * resid
    y.mean += ky * resid
    x.var -= kx * x.var
    y.var -= ky * y.var


def rateable(g: GameResult) -> bool:
    """A final between two NBA teams, not a preseason or exhibition game."""
    return g.is_final and g.away in TEAMS and g.home in TEAMS and g.season_type != 1


def replay(
    games: Iterable[GameResult],
    params: RatingParams | None = None,
    sched: dict[tuple[str, str], TeamSchedule] | None = None,
) -> tuple[list[Prediction], RatingBook]:
    """Predict every game from earlier games only, then fold it in."""
    ordered = sorted((g for g in games if rateable(g)), key=lambda g: (g.game_date, g.espn_id))
    sched = sched if sched is not None else schedule(ordered)
    book = RatingBook(params=params or RatingParams())
    out: list[Prediction] = []
    for g in ordered:
        home_b2b, away_b2b = (
            (s := sched.get((g.espn_id, t))) is not None and s.b2b for t in (g.home, g.away)
        )
        out.append(book.predict(g, home_b2b=home_b2b, away_b2b=away_b2b))
        book.update(g, tired=home_b2b or away_b2b)
    return out, book


def params_payload(params: RatingParams) -> dict[str, float]:
    return asdict(params)


__all__ = [
    "NAME",
    "Prediction",
    "RatingBook",
    "RatingParams",
    "params_payload",
    "rateable",
    "replay",
]
