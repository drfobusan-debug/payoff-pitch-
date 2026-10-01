"""Phase 3 player props: logs -> projections -> sim-conditioned distributions -> research rows."""

from __future__ import annotations

import json
from datetime import date as Date
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from nhl_engine.audit.grade import settle
from nhl_engine.calibration import Calibrator
from nhl_engine.config import Config, SimParams
from nhl_engine.data.book_rules import BookRules
from nhl_engine.data.capture import QuoteRow
from nhl_engine.data.nhlapi import RosterSpot, parse_result
from nhl_engine.data.skaters import (
    GoalieGame,
    NameIndex,
    SkaterGame,
    parse_current_roster,
    parse_goalie_log,
    parse_skater_log,
)
from nhl_engine.features import props as propf
from nhl_engine.features.starters import Starter
from nhl_engine.market.board import matchups, selections
from nhl_engine.market.pricing import family, is_prop
from nhl_engine.models import props as propm
from nhl_engine.models.goals import game_rates
from nhl_engine.models.periods import simulate
from nhl_engine.pipeline import PropContext, TeamInputs, build_prop_context, price_game
from nhl_engine.schemas import Game, GameResult, PeriodScore, PlayerLine

FIX = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 10, 1, 15, 30, tzinfo=timezone.utc)
LEAGUE = {
    "xgf60_5v5": 2.55,
    "xga60_5v5": 2.55,
    "fin60_5v5": 0.0,
    "pp_xgf60": 7.4,
    "pp_xga60": 0.9,
    "pk_xga60": 7.4,
    "sh_xgf60": 0.9,
    "pen_taken60": 3.4,
    "pen_drawn60": 3.4,
}
SLATE = Date(2026, 10, 1)


def _log_entry(day: str, shots: int, toi: str = "18:00", **kw) -> dict:
    base = {
        "gameId": 2025020001,
        "teamAbbrev": "EDM",
        "opponentAbbrev": "VAN",
        "homeRoadFlag": "H",
        "gameDate": day,
        "shots": shots,
        "goals": 0,
        "assists": 1,
        "points": 1,
        "powerPlayPoints": 0,
        "toi": toi,
    }
    base.update(kw)
    return base


def _skater_games(n: int, shots: int, toi_sec: float, start=Date(2025, 10, 10)) -> list[SkaterGame]:
    out = []
    for i in range(n):
        d = Date.fromordinal(start.toordinal() + i * 2)
        out.append(SkaterGame(1, 1000 + i, d, "EDM", "VAN", True, toi_sec, shots, 0, 1, 1, 0))
    return out


# -- data layer -----------------------------------------------------------------


def test_parse_logs_and_roster():
    data = {
        "gameLog": [
            _log_entry("2025-10-12", 4),
            _log_entry("2025-10-10", 2, toi="19:30", goals=1, points=2, powerPlayPoints=1),
            {"gameDate": "bad", "shots": 1},
        ]
    }
    log = parse_skater_log(8478402, data)
    assert [g.sog for g in log] == [2, 4]  # sorted by date
    assert log[0].toi == 19 * 60 + 30 and log[0].ppp == 1 and log[0].home
    glog = parse_goalie_log(
        1,
        {
            "gameLog": [
                {
                    "gameId": 1,
                    "gameDate": "2025-10-10",
                    "teamAbbrev": "COL",
                    "opponentAbbrev": "LAK",
                    "homeRoadFlag": "R",
                    "toi": "58:00",
                    "shotsAgainst": 30,
                    "goalsAgainst": 2,
                    "gamesStarted": 1,
                }
            ]
        },
    )
    assert glog[0].saves == 28 and glog[0].started and not glog[0].home
    assert parse_skater_log(1, {"gameLog": [{"shotsAgainst": 1}]}) == []

    roster = parse_current_roster(
        "EDM",
        {
            "forwards": [
                {
                    "id": 8478402,
                    "firstName": {"default": "Connor"},
                    "lastName": {"default": "McDavid"},
                    "positionCode": "C",
                }
            ],
            "defensemen": [
                {
                    "id": 1,
                    "firstName": {"default": "Evan"},
                    "lastName": {"default": "Bouchard"},
                    "positionCode": "D",
                }
            ],
            "goalies": [
                {
                    "id": 2,
                    "firstName": {"default": "Stuart"},
                    "lastName": {"default": "Skinner"},
                    "positionCode": "G",
                }
            ],
        },
    )
    assert [r.position for r in roster] == ["C", "D", "G"]
    idx = NameIndex(roster + [RosterSpot(3, "EDM", "C", "Leon Draisaitl")])
    assert idx.find("EDM", "connor mcdavid").player_id == 8478402
    assert idx.find("EDM", "C. McDavid").player_id == 8478402  # unique last name, same initial
    assert idx.find("VAN", "Connor McDavid") is None
    assert idx.find("EDM", "Nobody Here") is None


# -- features -------------------------------------------------------------------


def test_projection_shrinks_toward_prior_and_league():
    league = dict(propf.LEAGUE_F)
    prev = _skater_games(80, 4, 20 * 60, start=Date(2024, 10, 10))  # 12 SOG/60 last year
    none = propf.project_skater(
        player_id=1,
        name="x",
        team="EDM",
        position="C",
        current=[],
        previous=[],
        slate=SLATE,
        league=league,
    )
    assert abs(none.sog60 - league["sog60"]) < 1e-9 and none.games == 0
    assert abs(none.toi - 15.5 * 60) < 1e-6

    with_prev = propf.project_skater(
        player_id=1,
        name="x",
        team="EDM",
        position="C",
        current=[],
        previous=prev,
        slate=SLATE,
        league=league,
    )
    assert league["sog60"] < with_prev.sog60 < 12.0
    assert abs(with_prev.toi - 20 * 60) < 1e-6

    # Current-season games before the slate pull the rate up further; games on/after do not.
    cur = _skater_games(20, 6, 21 * 60, start=Date(2025, 10, 1))
    future = _skater_games(20, 0, 21 * 60, start=SLATE)
    both = propf.project_skater(
        player_id=1,
        name="x",
        team="EDM",
        position="C",
        current=cur + future,
        previous=prev,
        slate=SLATE,
        league=league,
    )
    assert both.sog60 > with_prev.sog60 and both.games == 20
    assert with_prev.toi < both.toi < 21 * 60
    assert both.mean("sog") == both.sog60 * both.toi / 3600

    d = propf.project_skater(
        player_id=1,
        name="x",
        team="EDM",
        position="D",
        current=[],
        previous=[],
        slate=SLATE,
        league=propf.LEAGUE_D,
    )
    assert d.position == "D" and d.sog60 < none.sog60


def test_league_means_pool_only_when_large():
    small = propf.league_means([("F", _skater_games(10, 9, 1200))])
    assert small["F"] == propf.LEAGUE_F
    big = propf.league_means([("F", _skater_games(300, 9, 3600))])
    assert abs(big["F"]["sog60"] - 9.0) < 1e-9 and big["D"] == propf.LEAGUE_D


def test_goalie_projection():
    cur = [
        GoalieGame(1, i, Date(2025, 10, 1 + i), "COL", "X", True, 3600, 30, 1, True)
        for i in range(10)
    ]
    g = propf.project_goalie(
        player_id=1, name="G", team="COL", current=cur, previous=[], slate=SLATE
    )
    assert propf.LEAGUE_SV < g.sv_pct < 30 / 31 and g.shots == 300 and g.games == 10


# -- distributions --------------------------------------------------------------


def test_pmfs_and_over_under():
    pois = propm.poisson_pmf(2.5)
    assert abs(pois.sum() - 1) < 1e-9 and abs((np.arange(len(pois)) * pois).sum() - 2.5) < 1e-6
    nb = propm.negbin_pmf(2.5, 1.5)
    k = np.arange(len(nb))
    mean = (k * nb).sum()
    var = (k**2 * nb).sum() - mean**2
    assert abs(mean - 2.5) < 1e-6 and abs(var - 1.5 * 2.5) < 1e-3
    assert np.allclose(propm.negbin_pmf(2.5, 1.0), pois)
    assert nb[6:].sum() > pois[6:].sum()  # fatter tail
    o = propm.over_under(pois, 2.5, "over")
    u = propm.over_under(pois, 2.5, "under")
    assert abs(o.win + u.win - 1) < 1e-9 and o.push == 0
    p3 = propm.over_under(pois, 3.0, "over")
    assert abs(p3.push - pois[3]) < 1e-12 and 0 < p3.win < 1
    y = propm.anytime(propm.poisson_pmf(0.4), "yes")
    assert abs(y.win - (1 - np.exp(-0.4))) < 1e-6
    assert abs(propm.anytime(propm.poisson_pmf(0.4), "no").win - np.exp(-0.4)) < 1e-6


def test_score_state_from_sim_and_saves_mean():
    strong = {**LEAGUE, "xgf60_5v5": 3.4, "xga60_5v5": 2.0}
    weak = {**LEAGUE, "xgf60_5v5": 2.0, "xga60_5v5": 3.4}
    params = SimParams(draws=4000)
    rates = game_rates(
        strong, weak, LEAGUE, home_goalie_gsax60=0.0, away_goalie_gsax60=0.0, params=params
    )
    sim = simulate(rates, params, seed=3)
    h, a = propm.score_state(sim, home=True), propm.score_state(sim, home=False)
    assert h.leading > h.trailing and a.trailing > a.leading
    assert abs(h.leading - a.trailing) < 1e-9 and abs(h.leading + h.tied + h.trailing - 1) < 1e-9
    assert h.shot_factor < 1 < a.shot_factor
    assert propm.exp_goals_against(sim, home=True) < propm.exp_goals_against(sim, home=False)
    assert propm.saves_mean(30.0, 2.8) == 27.2 and propm.saves_mean(1.0, 5.0) == 0.5
    assert propm.sog_mean(8.0, 1800, shot_factor=0.9, opp_sa_factor=1.1) == 8.0 * 0.5 * 0.9 * 1.1


# -- pricing --------------------------------------------------------------------


def _q(market, side, american, opp, *, line=None, entity="", book="dk", matchup="A @ H"):
    return QuoteRow(
        "2026-10-01T15:00:00Z",
        "2026-10-01",
        matchup,
        "ev1",
        market,
        side,
        entity,
        line,
        book,
        american,
        opp,
    )


def _inputs(code: str, status: str = "confirmed", goalie="G One") -> TeamInputs:
    return TeamInputs(
        code,
        dict(LEAGUE),
        {k: 0.5 for k in LEAGUE},
        10,
        "team_rate",
        Starter(code, 1, goalie, status, "test", None),
        30.0,
        29.0,
    )


def _ctx() -> PropContext:
    ctx = PropContext()
    sk = propf.SkaterProjection(
        1, "Star Player", "H", "F", 20 * 60, 10.0, 1.2, 1.5, 2.7, 0.4, 20, 80
    )
    ctx.skaters[("H", "star player")] = sk
    ctx.goalies[("A", "g one")] = propf.GoalieProjection(2, "G One", "A", 0.91, 300, 10)
    ctx.goalies[("H", "backup guy")] = propf.GoalieProjection(3, "Backup Guy", "H", 0.90, 100, 4)
    return ctx


def test_props_priced_research_only_never_buys():
    rows = [
        _q("game_ml", "H", -150, 130),
        _q("game_ml", "A", 130, -150),
        _q("sk_sog", "over", 100, -120, line=2.5, entity="Star Player"),
        _q("sk_sog", "under", -120, 100, line=2.5, entity="Star Player"),
        _q("sk_g", "over", 200, -260, line=0.5, entity="Star Player"),
        _q("sk_pts", "over", -140, 110, line=0.5, entity="Star Player"),
        _q("sk_a", "over", 120, -150, line=0.5, entity="Star Player"),
        _q("sk_ppp", "over", 150, -190, line=0.5, entity="Star Player"),
        _q("ags", "yes", 150, None, entity="Star Player"),
        _q("g_saves", "over", -110, -110, line=27.5, entity="G One"),
        _q("g_saves", "over", -110, -110, line=25.5, entity="Backup Guy"),  # not the starter
        _q("sk_sog", "over", -110, -110, line=2.5, entity="Unknown Skater"),
        _q("fgs", "yes", 900, None, entity="Star Player"),
    ]
    assert family("sk_sog") == "prop" and family("g_saves") == "prop" and is_prop("ags")
    assert not is_prop("p1_total") and family("p1_total") == "period"
    card = price_game(
        selections(rows),
        home_in=_inputs("H", goalie="Backup Guy"),
        away_in=_inputs("A", goalie="G One"),
        league=LEAGUE,
        cfg=Config(sim=SimParams(draws=3000)),
        slate=SLATE,
        rules=BookRules([]),
        calib=Calibrator(),
        now=NOW,
        priced_at="t",
        tag="test",
        seed=1,
        props=_ctx(),
    )
    by = {(r.market, r.side, r.entity): r for r in card.rows}
    assert ("game_ml", "H", "") in by
    for mk in ("sk_sog", "sk_g", "sk_pts", "sk_a", "sk_ppp", "ags"):
        row = next(r for r in card.rows if r.market == mk)
        assert "research_only" in row.gates and "probation" in row.gates and not row.is_buy
        assert 0 < row.model_prob < 1
    over, under = by[("sk_sog", "over", "Star Player")], by[("sk_sog", "under", "Star Player")]
    assert abs(over.model_prob + under.model_prob - 1) < 1e-9
    # 10 SOG/60 x 20 min ≈ 3.3 expected -> clear over at 2.5
    assert over.model_prob > 0.5
    saves = by[("g_saves", "over", "G One")]
    assert "research_only" in saves.gates and "goalie_unconfirmed" not in saves.gates
    assert ("g_saves", "over", "Backup Guy") in by  # projected starter quoted -> priced
    assert ("sk_sog", "over", "Unknown Skater") not in by  # no projection -> unpriced, not guessed
    assert ("fgs", "yes", "Star Player") not in by
    assert not any(r.is_buy for r in card.rows if is_prop(r.market))


def test_saves_gate_on_unconfirmed_goalie_and_props_skipped_without_context():
    rows = [_q("g_saves", "over", -110, -110, line=27.5, entity="G One")]
    card = price_game(
        selections(rows),
        home_in=_inputs("H", status="projected", goalie="Backup Guy"),
        away_in=_inputs("A", status="projected", goalie="G One"),
        league=LEAGUE,
        cfg=Config(sim=SimParams(draws=2000)),
        slate=SLATE,
        rules=BookRules([]),
        calib=Calibrator(),
        now=NOW,
        priced_at="t",
        tag="t",
        seed=1,
        props=_ctx(),
    )
    assert card.rows and "goalie_unconfirmed" in card.rows[0].gates
    none = price_game(
        selections(rows),
        home_in=_inputs("H"),
        away_in=_inputs("A"),
        league=LEAGUE,
        cfg=Config(sim=SimParams(draws=2000)),
        slate=SLATE,
        rules=BookRules([]),
        calib=Calibrator(),
        now=NOW,
        priced_at="t",
        tag="t",
        seed=1,
    )
    assert none.rows == []


class _FakeLogs:
    def __init__(self):
        self.calls: list[tuple[int, int]] = []

    def current_roster(self, team):
        if team == "H":
            return [
                RosterSpot(10, "H", "C", "Star Player"),
                RosterSpot(11, "H", "D", "Blue Liner"),
                RosterSpot(12, "H", "G", "Net Minder"),
            ]
        return [RosterSpot(20, "A", "L", "Away Guy")]

    def skater_log(self, pid, season, *, final=False):
        self.calls.append((pid, season))
        return _skater_games(30, 3 if pid == 10 else 1, 1100, start=Date(season, 10, 10))

    def goalie_log(self, pid, season, *, final=False):
        self.calls.append((pid, season))
        return [GoalieGame(pid, 1, Date(season, 10, 12), "H", "A", True, 3600, 30, 2, True)]


def test_build_prop_context_resolves_only_quoted_players():
    rows = [
        _q("sk_sog", "over", -110, -110, line=2.5, entity="Star Player"),
        _q("sk_sog", "over", -110, -110, line=1.5, entity="Blue Liner"),
        _q("g_saves", "over", -110, -110, line=27.5, entity="Net Minder"),
        _q("sk_pts", "over", -110, -110, line=0.5, entity="Away Guy"),
        _q("sk_pts", "over", -110, -110, line=0.5, entity="Ghost"),
        _q("game_ml", "H", -150, 130),
    ]
    logs = _FakeLogs()
    ctx = build_prop_context(rows, matchups(rows), slate=SLATE, season=2026, logs=logs)
    assert set(ctx.skaters) == {("H", "star player"), ("H", "blue liner"), ("A", "away guy")}
    assert set(ctx.goalies) == {("H", "net minder")}
    assert ctx.skaters[("H", "blue liner")].position == "D"
    assert ctx.skater("H", "Star Player").sog60 > ctx.skater("A", "Away Guy").sog60
    assert sorted(set(logs.calls)) == [
        (10, 2025),
        (10, 2026),
        (11, 2025),
        (11, 2026),
        (12, 2025),
        (12, 2026),
        (20, 2025),
        (20, 2026),
    ]
    assert ctx.skater("H", "Ghost") is None


# -- grading --------------------------------------------------------------------


def _res(players) -> GameResult:
    return GameResult(
        1,
        SLATE,
        "A",
        "H",
        "OFF",
        periods=(
            PeriodScore(1, "REG", 1, 0),
            PeriodScore(2, "REG", 0, 1),
            PeriodScore(3, "REG", 1, 0),
        ),
        decided="REG",
        players=tuple(players),
    )


def test_prop_settlement_from_boxscore_lines():
    res = _res(
        [
            PlayerLine(
                1, "H", "Star Player", "C", 1200, sog=3, goals=1, assists=0, points=1, blocks=2
            ),
            PlayerLine(2, "H", "Scratched Guy", "D", 0, sog=0),
            PlayerLine(3, "A", "Net Minder", "G", 3600, saves=28, shots_against=30, starter=True),
            PlayerLine(4, "A", "Alexander Ovechkin", "L", 1100, sog=5, goals=0),
        ]
    )

    def s(market, side, entity, line=None):
        return settle(
            market=market, side=side, entity=entity, line=line, ot_rule="incl_ot", res=res
        )

    assert s("sk_sog", "over", "Star Player", 2.5) == "win"
    assert s("sk_sog", "under", "Star Player", 2.5) == "loss"
    assert s("sk_sog", "over", "Star Player", 3.0) == "push"
    assert s("sk_g", "over", "Star Player", 0.5) == "win"
    assert s("sk_a", "over", "Star Player", 0.5) == "loss"
    assert s("sk_pts", "over", "Star Player", 0.5) == "win"
    assert s("sk_blk", "over", "Star Player", 1.5) == "win"
    assert s("ags", "yes", "Star Player") == "win"
    assert s("ags", "no", "Star Player") == "loss"
    assert s("ags", "yes", "Alex Ovechkin") == "loss"  # last-name fallback
    assert s("g_saves", "over", "Net Minder", 27.5) == "win"
    assert s("sk_sog", "over", "Scratched Guy", 0.5) is None  # did not dress -> void
    assert s("sk_sog", "over", "Nobody", 0.5) is None
    assert s("sk_ppp", "over", "Star Player", 0.5) is None  # not in the boxscore
    assert s("fgs", "yes", "Star Player") is None
    # Unrelated settlement untouched.
    assert s("game_total", "over", "", 2.5) == "win"


def test_parse_result_carries_player_lines():
    box = json.loads((FIX / "boxscore.json").read_text())
    rail = json.loads((FIX / "right_rail.json").read_text())
    game = Game(
        game_date=Date(2024, 10, 10),
        away=box["awayTeam"]["abbrev"],
        home=box["homeTeam"]["abbrev"],
        nhl_game_id=box["id"],
    )
    res = parse_result(box, rail, game)
    assert res.players
    benson = next(p for p in res.players if p.name == "Z. Benson")
    assert (
        benson.sog == 3
        and benson.blocks == 1
        and benson.toi == 24 * 60 + 22
        and benson.team == game.home
    )
    goalies = [p for p in res.players if p.position == "G"]
    assert goalies and any(g.starter for g in goalies)
