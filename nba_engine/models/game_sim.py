"""Game and first-half score distributions, priced for every game market.

The mean comes from the team ratings (``ratings.Prediction``), moved by the
production of regulars who are freshly out:

    home margin = rating margin + lost_margin * (lost(away) - lost(home))
    total       = rating total  + lost_total  * (lost(away) + lost(home))

``lost(team)`` is the sum, over the team's fresh absences (the minutes model's
``MinutesBook.fresh``), of the player's baseline minutes times his points per
regulation minute. A regular out for weeks is already inside the ratings, so
only a fresh absence moves the mean, the same rule the minutes model uses.

The first half takes ``h1_margin_share`` of the margin and ``h1_total_share``
of the total. Each score is an integer draw from a normal with a fitted SD
(per market), so a push on a whole line has its own mass, and a probability is
quoted conditional on no push, which is what a de-vigged two-way price means.
A full-game tie goes to overtime and is excluded the same way.

The halftime drop is split into two dials measured from play-by-play:
``pace_shift`` (second-half possessions over first-half, minus 1) and
``eff_shift`` (second-half points per possession over first-half, minus 1).
The game markets need only the first-half share; the dials are what the props
read, since fewer possessions cost every counting stat and efficiency only
points and makes.

The price that reaches the ledger is ``w * p_model + (1 - w) * p_fair`` with
``w`` fitted per market (``blend``); a market with no fitted weight gets no
model probability and stays ``no_model``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields

from nba_engine.market.board import Selection
from nba_engine.models import minutes, ratings
from nba_engine.models.minutes import MinutesBook, MinutesParams, regulation, team_box
from nba_engine.models.ratings import Prediction, RatingBook, RatingParams
from nba_engine.models.schedule import TeamSchedule, schedule, season_of
from nba_engine.schemas import GameResult

NAME = "game_sim"
SIM_MARKETS: tuple[str, ...] = (
    "game_ml",
    "game_ats",
    "game_total",
    "h1_ml",
    "h1_ats",
    "h1_total",
)
PosKey = tuple[str, str, str, str, float | None]


@dataclass(frozen=True)
class SimParams:
    sd_margin: float = 14.0
    sd_total: float = 18.0
    h1_margin_share: float = 0.55
    h1_total_share: float = 0.5
    sd_h1_margin: float = 11.2
    sd_h1_total: float = 12.0
    lost_margin: float = 0.0  # home-margin points per point of fresh-out production
    lost_total: float = 0.0
    pace_shift: float = 0.0
    eff_shift: float = 0.0
    production_half_life: float = 6.0  # games
    blend: dict[str, float] = field(default_factory=dict)  # market -> model weight

    @classmethod
    def from_dict(cls, raw: dict) -> SimParams:
        names = {f.name for f in fields(cls)} - {"blend"}
        weights = raw.get("blend") or {}
        return cls(
            **{k: float(v) for k, v in raw.items() if k in names},
            blend={str(k): float(v) for k, v in weights.items()},
        )


def params_payload(params: SimParams) -> dict:
    return asdict(params)


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def p_above(mean: float, sd: float, line: float) -> float:
    """P(score > line | score != line) for an integer score ~ rounded normal."""
    win = 1.0 - _phi((math.floor(line) + 0.5 - mean) / sd)
    loss = _phi((math.ceil(line) - 0.5 - mean) / sd)
    return win / (win + loss) if win + loss > 0 else 0.5


@dataclass(frozen=True)
class GameSim:
    """One game's inputs, before the params turn them into distributions."""

    espn_id: str
    game_date: str
    away: str
    home: str
    rating_margin: float
    rating_total: float
    lost_home: float  # points per game the fresh absences took with them
    lost_away: float

    @property
    def matchup(self) -> str:
        return f"{self.away} @ {self.home}"


@dataclass(frozen=True)
class GameDist:
    home: str
    away: str
    margin: float
    total: float
    h1_margin: float
    h1_total: float
    sd_margin: float
    sd_total: float
    sd_h1_margin: float
    sd_h1_total: float

    def prob(self, market: str, side: str, line: float | None) -> float | None:
        """The model's no-push probability for one side of a game or first-half market."""
        h1 = market.startswith("h1_")
        margin, sd_m = (self.h1_margin, self.sd_h1_margin) if h1 else (self.margin, self.sd_margin)
        total, sd_t = (self.h1_total, self.sd_h1_total) if h1 else (self.total, self.sd_total)
        kind = market.split("_", 1)[1]
        if kind == "total":
            if line is None or side not in ("over", "under"):
                return None
            over = p_above(total, sd_t, line)
            return over if side == "over" else 1.0 - over
        if side not in (self.home, self.away):
            return None
        mean = margin if side == self.home else -margin
        if kind == "ml":
            return p_above(mean, sd_m, 0.0)
        if kind == "ats" and line is not None:
            return p_above(mean, sd_m, -line)
        return None


def dist(sim: GameSim, params: SimParams) -> GameDist:
    margin = sim.rating_margin + params.lost_margin * (sim.lost_away - sim.lost_home)
    total = sim.rating_total + params.lost_total * (sim.lost_away + sim.lost_home)
    return GameDist(
        home=sim.home,
        away=sim.away,
        margin=margin,
        total=total,
        h1_margin=params.h1_margin_share * margin,
        h1_total=params.h1_total_share * total,
        sd_margin=params.sd_margin,
        sd_total=params.sd_total,
        sd_h1_margin=params.sd_h1_margin,
        sd_h1_total=params.sd_h1_total,
    )


@dataclass
class ProductionBook:
    """Each player's exponentially weighted points per regulation minute."""

    half_life: float = 6.0
    _pts: dict[str, float] = field(default_factory=dict)
    _min: dict[str, float] = field(default_factory=dict)

    def rate(self, player: str) -> float:
        m = self._min.get(player, 0.0)
        return self._pts.get(player, 0.0) / m if m > 0 else 0.0

    def update(self, g: GameResult) -> None:
        lam = 0.5 ** (1.0 / self.half_life)
        for line in g.players:
            if line.espn_id and line.minutes > 0:
                k = line.espn_id
                self._pts[k] = lam * self._pts.get(k, 0.0) + line.points
                self._min[k] = lam * self._min.get(k, 0.0) + regulation(line.minutes, g.overtimes)


def lost(book: MinutesBook, prod: ProductionBook, out: Iterable[str]) -> float:
    """Points per game the team's fresh absences averaged: baseline minutes x points/minute."""
    return sum(book.baseline(j) * prod.rate(j) for j in book.fresh(out))


def replay(
    games: Iterable[GameResult],
    rating_params: RatingParams | None = None,
    minutes_params: MinutesParams | None = None,
    production_half_life: float = 6.0,
    sched: dict[tuple[str, str], TeamSchedule] | None = None,
) -> list[GameSim]:
    """Every rateable game's inputs from earlier games only, then fold the game in.

    Absences come from the box, as in the minutes replay: live, the injury
    report supplies them.
    """
    ordered = sorted(games, key=lambda x: (x.game_date, x.espn_id))
    sched = sched if sched is not None else schedule([g for g in ordered if ratings.rateable(g)])
    rbook = RatingBook(params=rating_params or RatingParams())
    mbook = MinutesBook(params=minutes_params or MinutesParams())
    prod = ProductionBook(half_life=production_half_life)
    out_sims: list[GameSim] = []
    for g in ordered:
        season = season_of(g.game_date)
        boxes = {t: team_box(g, t) for t in (g.away, g.home)}
        outs = {t: minutes.outs(mbook, t, season, b) for t, b in boxes.items() if b}
        if ratings.rateable(g):
            home_b2b, away_b2b = (
                (s := sched.get((g.espn_id, t))) is not None and s.b2b for t in (g.home, g.away)
            )
            pred: Prediction = rbook.predict(g, home_b2b=home_b2b, away_b2b=away_b2b)
            out_sims.append(
                GameSim(
                    espn_id=g.espn_id,
                    game_date=pred.game_date,
                    away=g.away,
                    home=g.home,
                    rating_margin=pred.margin,
                    rating_total=pred.total,
                    lost_home=lost(mbook, prod, outs.get(g.home, set())),
                    lost_away=lost(mbook, prod, outs.get(g.away, set())),
                )
            )
            rbook.update(g, tired=home_b2b or away_b2b)
        for t, b in boxes.items():
            if b:
                mbook.update(t, season, b, g.overtimes, outs[t])
        prod.update(g)
    return out_sims


def blend(p_model: float, fair: float, weight: float) -> float:
    return weight * p_model + (1.0 - weight) * fair


def model_probs(
    sels: Iterable[Selection],
    dists: Mapping[tuple[str, str], GameDist],
    params: SimParams,
    fair: Mapping[PosKey, float | None] | None = None,
) -> dict[PosKey, float]:
    """Blended probability per ledger position, for markets with a fitted model weight.

    ``dists`` is keyed by ``(slate date, matchup)``; ``fair`` overrides a
    selection's own fair (the Over-bias-adjusted one the ledger uses).
    """
    out: dict[PosKey, float] = {}
    for s in sels:
        w = params.blend.get(s.market, 0.0)
        d = dists.get((s.game_date, s.matchup))
        if w <= 0.0 or d is None:
            continue
        pos = (s.event_id, s.market, s.side, s.entity, s.line)
        f = (fair or {}).get(pos, s.fair)
        p = d.prob(s.market, s.side, s.line)
        if f is None or p is None:
            continue
        out[pos] = blend(p, f, w)
    return out


def by_matchup(sims: Iterable[GameSim], params: SimParams) -> dict[tuple[str, str], GameDist]:
    return {(s.game_date, s.matchup): dist(s, params) for s in sims}
