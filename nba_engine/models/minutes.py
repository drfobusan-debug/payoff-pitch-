"""Pre-game minutes: who plays and for how long, including who takes an absent player's minutes.

Every projection is in regulation minutes (OT minutes are scaled out of the
history and added back by the simulation), and a team's projections sum to 240.

1. **Baseline.** A player's exponentially weighted mean of his regulation
   minutes in games he was available for (a coach's-decision DNP counts as 0),
   with ``half_life`` in games, shrunk toward ``prior_minutes`` by
   ``prior_weight`` pseudo-games. His first game of a new season keeps last
   season's weight; from then on that history is cut to ``carry`` of itself.
2. **Team total.** Available players' baselines rarely sum to 240. A shortfall
   (someone new is out) is handed out by ``baseline ** deficit_power``, no one
   above ``cap``. A surplus (someone is back) is taken by ``baseline ** surplus_power``.
   A regular who has been out for weeks is already missing from everyone's
   baseline, so only a fresh absence moves the shares.
3. **Next man up.** For a regular (baseline >= ``star_minutes``) in his first
   ``fresh_games`` games out, each teammate's past miss on those nights is
   added, shrunk by ``pair_k`` pseudo-games; the team is then rescaled to 240.

Availability in the replay comes from the box: a player absent from it, or
listed with a DNP reason other than a coach's decision, was out before tip.
A coach's-decision DNP was available and played 0. Live, the injury report
supplies the out list. ``replay`` projects each game from earlier games only.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, fields
from datetime import date as Date

from nba_engine.models.allocation import allocate
from nba_engine.models.schedule import season_of
from nba_engine.schemas import GameResult, PlayerLine

NAME = "minutes"
TEAM_MINUTES = 240.0
COACH = "COACH'S DECISION"
RECENT = 10  # games in the naive "recent average" the model is graded against


@dataclass(frozen=True)
class MinutesParams:
    half_life: float = 4.0  # games
    carry: float = 0.1  # share of last season's weight kept after his first new-season game
    prior_minutes: float = 6.0
    prior_weight: float = 1.0  # pseudo-games
    deficit_power: float = 0.5
    surplus_power: float = 0.25
    cap: float = 37.0  # regulation minutes
    pair_k: float = 20.0  # pseudo-games shrinking a teammate's next-man-up history
    fresh_games: float = 4.0
    star_minutes: float = 15.0

    @classmethod
    def from_dict(cls, raw: dict) -> MinutesParams:
        names = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in raw.items() if k in names})


@dataclass(frozen=True)
class Projection:
    espn_id: str
    game_date: Date
    team: str
    player: str
    minutes: float  # projected, regulation
    baseline: float
    actual: float | None  # regulation-scaled box minutes; None live
    recent: float | None  # mean of his last RECENT available games, unnormalised
    games: float  # effective games behind the baseline
    fresh_out: bool  # a regular on his team is freshly out


def regulation(minutes: float, overtimes: int) -> float:
    return minutes * 48.0 / (48.0 + 5.0 * overtimes)


def is_out(line: PlayerLine | None) -> bool:
    """Absent from the box, or a DNP for any reason but the coach's decision."""
    return line is None or (line.dnp and bool(line.dnp_reason) and line.dnp_reason != COACH)


@dataclass
class MinutesBook:
    params: MinutesParams = field(default_factory=MinutesParams)
    _num: dict[str, float] = field(default_factory=dict)
    _den: dict[str, float] = field(default_factory=dict)
    _team: dict[str, str] = field(default_factory=dict)
    _season: dict[str, int] = field(default_factory=dict)
    _streak: dict[str, int] = field(default_factory=dict)
    _recent: dict[str, deque[float]] = field(default_factory=dict)
    _pair_sum: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))
    _pair_n: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))

    def roster(self, team: str, season: int) -> set[str]:
        """Players whose latest game this season was with ``team``."""
        return {p for p, t in self._team.items() if t == team and self._season.get(p) == season}

    def baseline(self, player: str) -> float:
        p = self.params
        num, den = self._num.get(player, 0.0), self._den.get(player, 0.0)
        return (p.prior_weight * p.prior_minutes + num) / (p.prior_weight + den)

    def games(self, player: str) -> float:
        return self._den.get(player, 0.0)

    def recent(self, player: str) -> float | None:
        xs = self._recent.get(player)
        return sum(xs) / len(xs) if xs else None

    def fresh(self, out: Iterable[str]) -> list[str]:
        """Regulars out, but out for fewer than ``fresh_games`` games."""
        p = self.params
        return sorted(
            j
            for j in out
            if self._streak.get(j, 0) < p.fresh_games
            and self._den.get(j, 0.0) > 0
            and self._num[j] / self._den[j] >= p.star_minutes
        )

    def _to_total(self, base: dict[str, float]) -> dict[str, float]:
        p = self.params
        total = sum(base.values())
        if total < TEAM_MINUTES:
            add = allocate(
                TEAM_MINUTES - total,
                {k: v**p.deficit_power for k, v in base.items()},
                {k: max(p.cap - v, 0.0) for k, v in base.items()},
            )
            return {k: v + add.amounts[k] for k, v in base.items()}
        cut = allocate(total - TEAM_MINUTES, {k: v**p.surplus_power for k, v in base.items()}, base)
        return {k: v - cut.amounts[k] for k, v in base.items()}

    def _team_shares(self, available: Iterable[str]) -> dict[str, float]:
        base = {k: self.baseline(k) for k in available}
        return self._to_total(base) if base else {}

    def project(self, available: Iterable[str], out: Iterable[str]) -> dict[str, float]:
        """Regulation minutes for each available player, summing to 240."""
        shares = self._team_shares(available)
        fresh = self.fresh(out)
        if not fresh or not shares:
            return shares
        p = self.params
        moved = {
            k: min(
                max(
                    v
                    + sum(
                        self._pair_sum.get((k, j), 0.0) / (self._pair_n.get((k, j), 0.0) + p.pair_k)
                        for j in fresh
                    ),
                    0.0,
                ),
                p.cap,
            )
            for k, v in shares.items()
        }
        total = sum(moved.values())
        return {k: v * TEAM_MINUTES / total for k, v in moved.items()} if total > 0 else shares

    def update(
        self, team: str, season: int, box: dict[str, PlayerLine], overtimes: int, out: set[str]
    ) -> None:
        """Fold one team-game in. ``box`` is the team's box lines by player id."""
        p = self.params
        available = {k for k, line in box.items() if not is_out(line)}
        shares = self._team_shares(available)
        fresh = self.fresh(out)
        lam = 0.5 ** (1.0 / p.half_life)
        for k in available:
            x = regulation(box[k].minutes, overtimes)
            for j in fresh:
                self._pair_sum[(k, j)] += x - shares[k]
                self._pair_n[(k, j)] += 1.0
            if self._season.get(k) not in (None, season) and k in self._num:
                self._num[k] *= p.carry
                self._den[k] *= p.carry
            self._num[k] = lam * self._num.get(k, 0.0) + x
            self._den[k] = lam * self._den.get(k, 0.0) + 1.0
            self._recent.setdefault(k, deque(maxlen=RECENT)).append(x)
            self._streak[k] = 0
        for k in box.keys() - available:
            if self._season.get(k) not in (None, season) and k in self._num:
                self._num[k] *= p.carry
                self._den[k] *= p.carry
        for k in box:
            self._team[k] = team
            self._season[k] = season
        for j in out:
            self._streak[j] = self._streak.get(j, 0) + 1


def team_box(g: GameResult, team: str) -> dict[str, PlayerLine]:
    return {line.espn_id: line for line in g.players if line.team == team and line.espn_id}


def outs(book: MinutesBook, team: str, season: int, box: dict[str, PlayerLine]) -> set[str]:
    """Rostered players the box shows out before tip."""
    roster = book.roster(team, season)
    return {k for k in roster | box.keys() if is_out(box.get(k))} & roster


def replay(
    games: Iterable[GameResult], params: MinutesParams | None = None
) -> tuple[list[Projection], MinutesBook]:
    """Project every team-game from earlier games only, then fold it in."""
    book = MinutesBook(params=params or MinutesParams())
    out_rows: list[Projection] = []
    for g in sorted(games, key=lambda x: (x.game_date, x.espn_id)):
        season = season_of(g.game_date)
        for team in (g.away, g.home):
            box = team_box(g, team)
            if not box:
                continue
            out = outs(book, team, season, box)
            available = {k for k, line in box.items() if not is_out(line)}
            proj = book.project(available, out)
            fresh = bool(book.fresh(out))
            for k in sorted(available):
                out_rows.append(
                    Projection(
                        espn_id=g.espn_id,
                        game_date=g.game_date,
                        team=team,
                        player=k,
                        minutes=proj[k],
                        baseline=book.baseline(k),
                        actual=regulation(box[k].minutes, g.overtimes),
                        recent=book.recent(k),
                        games=book.games(k),
                        fresh_out=fresh,
                    )
                )
            book.update(team, season, box, g.overtimes, out)
    return out_rows, book


def params_payload(params: MinutesParams) -> dict[str, float]:
    return asdict(params)


__all__ = [
    "COACH",
    "NAME",
    "TEAM_MINUTES",
    "MinutesBook",
    "MinutesParams",
    "Projection",
    "is_out",
    "params_payload",
    "regulation",
    "replay",
    "team_box",
]
