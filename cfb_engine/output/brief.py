"""The reader's context for a game: who these teams are, not only what they price at.

Everything the slate article prints beside the number -- record and streak,
SP+ overall/offense/defense with ranks, the AP poll, the players carrying the
production, who is out, the venue's home-field charge and the kickoff weather.
None of it moves a price: the pipeline has already scored (or deliberately not
scored) each of these; the brief is what a reader needs to *parse* the card.

Built once per slate in the pipeline from feeds the run already pays for
(``/games``, ``/ratings/sp``, ``/ppa/players/season``) plus one ``/rankings``
call, and persisted on every recommendation so ``cfb-engine card`` can rebuild
the article without a network.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date as Date
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from cfb_engine.data.espn import ColorBook, GameColor, color_for
from cfb_engine.data.injuries import InjuryBook, unavailable_for
from cfb_engine.data.teamnames import school_key
from cfb_engine.data.vsin import VSIN_HFA
from cfb_engine.features.context import ContextBook, context_for

if TYPE_CHECKING:
    from cfb_engine.data.cfbd import CFBDClient, GameResult, SPLine
    from cfb_engine.data.ensemble import ModelRatings
    from cfb_engine.schemas import Game, Slate

# Positions whose season PPA says something about a team's identity. Linemen and
# specialists do not appear in the PPA feed at all.
_SKILL = {"QB", "RB", "WR", "TE"}
_TOP_PLAYERS = 3


@dataclass
class TeamBrief:
    name: str
    wins: int = 0
    losses: int = 0
    streak: str | None = None  # "W3" / "L1"
    last: str | None = None  # "beat Kansas 38-21"
    sp_rank: int | None = None
    sp_rating: float | None = None
    off_rating: float | None = None
    off_rank: int | None = None
    def_rating: float | None = None
    def_rank: int | None = None
    poll_rank: int | None = None  # AP Top 25
    tr_rank: int | None = None  # TeamRankings predictive rank (ensemble feed)
    fpi_rank: int | None = None  # ESPN FPI rank (ensemble feed)
    ats: str | None = None  # ESPN season ATS record
    key_players: list[str] = field(default_factory=list)  # "QB Name (+12.3 PPA)"
    leaders: list[str] = field(default_factory=list)  # ESPN stat leaders, "QB Name — 233 YDS, 2 TD"
    out: list[str] = field(default_factory=list)  # "RB Name"

    @property
    def record(self) -> str:
        return f"{self.wins}-{self.losses}"

    @property
    def rated(self) -> bool:
        return self.sp_rank is not None


@dataclass
class GameBrief:
    home: TeamBrief
    away: TeamBrief
    hfa_pts: float | None = None
    hfa_listed: bool = False  # VSiN lists this venue; the charge is venue-specific
    neutral_site: bool = False
    dome: bool = False
    rest_home: int | None = None
    rest_away: int | None = None
    temperature_f: float | None = None
    wind_mph: float | None = None
    precipitation: float | None = None
    venue: str | None = None
    city: str | None = None
    grass: bool | None = None
    conference_game: bool = False
    headline: str | None = None  # ESPN/AP preview headline
    story: str | None = None  # its lede
    tags: list[str] = field(default_factory=list)  # "rivalry", "must-win", ...
    fpi_home: float | None = None  # ESPN matchup predictor, home win %
    precip_pct: float | None = None  # ESPN forecast chance of rain
    gust_mph: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> GameBrief:
        d = dict(d)
        home = d.pop("home")
        away = d.pop("away")
        if not isinstance(home, dict) or not isinstance(away, dict):
            raise ValueError("brief needs home and away")
        return cls(home=TeamBrief(**home), away=TeamBrief(**away), **d)  # type: ignore[arg-type]


def _local_day(stamp: str) -> str:
    """Kickoff's US-Eastern calendar day from CFBD's UTC stamp ("" stays "")."""
    if len(stamp) < 16:
        return stamp[:10]
    try:
        dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return stamp[:10]
    return (dt - timedelta(hours=5)).date().isoformat()


def _form(results: list[GameResult], team: str, before: Date) -> TeamBrief:
    """Record, streak and last result from finals strictly before ``before``."""
    key = school_key(team)
    played = []
    for r in results:
        if _local_day(r.start_date) >= before.isoformat():
            continue
        if school_key(r.home) == key:
            played.append((r.start_date, r.home_points - r.away_points, r.away, r.home_points, r.away_points))
        elif school_key(r.away) == key:
            played.append((r.start_date, r.away_points - r.home_points, r.home, r.away_points, r.home_points))
    played.sort()
    brief = TeamBrief(name=team)
    if not played:
        return brief
    brief.wins = sum(1 for p in played if p[1] > 0)
    brief.losses = sum(1 for p in played if p[1] < 0)
    last_sign = 1 if played[-1][1] > 0 else -1
    run = 0
    for p in reversed(played):
        if (1 if p[1] > 0 else -1) != last_sign or p[1] == 0:
            break
        run += 1
    brief.streak = f"{'W' if last_sign > 0 else 'L'}{run}"
    _, margin, opp, us, them = played[-1]
    verb = "beat" if margin > 0 else "lost to"
    brief.last = f"{verb} {opp} {us}-{them}"
    return brief


def _apply_sp(brief: TeamBrief, sp: dict[str, SPLine]) -> None:
    line = sp.get(school_key(brief.name))
    if line is None:
        return
    brief.sp_rating = line.rating
    brief.sp_rank = line.rank
    brief.off_rating = line.off_rating
    brief.off_rank = line.off_rank
    brief.def_rating = line.def_rating
    brief.def_rank = line.def_rank


def _apply_players(brief: TeamBrief, players: list[tuple[str, str, str, float]]) -> None:
    key = school_key(brief.name)
    mine = [
        (ppa, pos, name)
        for name, pos, team, ppa in players
        if school_key(team) == key and pos in _SKILL
    ]
    mine.sort(reverse=True)
    brief.key_players = [f"{pos} {name} ({ppa:+.1f} PPA)" for ppa, pos, name in mine[:_TOP_PLAYERS]]


def _rank_in(model: ModelRatings | None, team: str) -> int | None:
    if model is None:
        return None
    key = school_key(team)
    if key not in model.net:
        return None
    return 1 + sum(1 for v in model.net.values() if v > model.net[key])


def _apply_out(brief: TeamBrief, injuries: InjuryBook) -> None:
    rows = unavailable_for(injuries, brief.name)
    brief.out = [f"{r.position} {r.player}".strip() for r in rows][:6]


def build_briefs(
    cfbd: CFBDClient,
    season: int,
    slate: Slate,
    *,
    ctx_book: ContextBook,
    injuries: InjuryBook,
    hfa_default: float,
    hfa_enabled: bool,
    color: ColorBook | None = None,
    models: list[ModelRatings] | None = None,
) -> dict[str, GameBrief]:
    """One :class:`GameBrief` per game id. Every feed failure leaves fields ``None``."""
    if not slate.games:
        return {}
    results: list[GameResult] = []
    sp: dict[str, SPLine] = {}
    polls: dict[str, int] = {}
    players: list[tuple[str, str, str, float]] = []
    if cfbd.available():
        results = cfbd.fetch_all_results(season)
        sp = cfbd.fetch_sp_table(season)
        polls = cfbd.fetch_ap_poll(season)
        players = cfbd.fetch_player_ppa_rows(season)

    by_source = {m.source: m for m in models or []}
    out: dict[str, GameBrief] = {}
    for game in slate.games:
        brief = _brief_for(
            game,
            results=results,
            sp=sp,
            polls=polls,
            players=players,
            ctx_book=ctx_book,
            injuries=injuries,
            hfa_default=hfa_default,
            hfa_enabled=hfa_enabled,
        )
        for tb in (brief.home, brief.away):
            tb.tr_rank = _rank_in(by_source.get("teamrankings"), tb.name)
            tb.fpi_rank = _rank_in(by_source.get("fpi"), tb.name)
        if color:
            gc = color_for(color, game.home.name, game.away.name)
            if gc is not None:
                _apply_color(brief, gc)
        out[game.game_id] = brief
    return out


def _apply_color(brief: GameBrief, gc: GameColor) -> None:
    for tb, tc in ((brief.home, gc.home), (brief.away, gc.away)):
        tb.ats = tc.ats
        tb.leaders = list(tc.leaders)
        if tb.wins == 0 and tb.losses == 0 and tc.record:
            parts = tc.record.split("-")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                tb.wins, tb.losses = int(parts[0]), int(parts[1])
    brief.venue = gc.venue
    brief.city = gc.city
    brief.grass = gc.grass
    brief.conference_game = gc.conference_game
    brief.neutral_site = brief.neutral_site or gc.neutral_site
    brief.headline = gc.headline
    brief.story = gc.story
    brief.tags = list(gc.tags)
    brief.fpi_home = gc.fpi_home
    brief.precip_pct = gc.precip_pct
    brief.gust_mph = gc.gust_mph
    if brief.temperature_f is None:
        brief.temperature_f = gc.temperature_f
    if brief.neutral_site:
        brief.hfa_pts = 0.0


def _brief_for(
    game: Game,
    *,
    results: list[GameResult],
    sp: dict[str, SPLine],
    polls: dict[str, int],
    players: list[tuple[str, str, str, float]],
    ctx_book: ContextBook,
    injuries: InjuryBook,
    hfa_default: float,
    hfa_enabled: bool,
) -> GameBrief:
    teams = []
    for info in (game.home, game.away):
        t = _form(results, info.name, game.game_date)
        _apply_sp(t, sp)
        t.poll_rank = polls.get(school_key(info.name))
        _apply_players(t, players)
        if injuries:
            _apply_out(t, injuries)
        teams.append(t)
    home, away = teams
    ctx = context_for(ctx_book, game.home.name, game.away.name)
    listed = VSIN_HFA.get(school_key(game.home.name))
    hfa = listed if (hfa_enabled and listed is not None) else hfa_default
    return GameBrief(
        home=home,
        away=away,
        hfa_pts=0.0 if ctx.neutral_site else hfa,
        hfa_listed=listed is not None,
        neutral_site=ctx.neutral_site,
        dome=ctx.dome,
        rest_home=ctx.rest_home,
        rest_away=ctx.rest_away,
        temperature_f=ctx.temperature_f,
        wind_mph=ctx.wind_mph,
        precipitation=ctx.precipitation,
    )
