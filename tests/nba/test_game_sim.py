import math
import random
from datetime import date, timedelta

from nba_engine import cli
from nba_engine.data import boxes, pbp
from nba_engine.market import params
from nba_engine.market.board import Selection
from nba_engine.models import game_sim, sim_fit
from nba_engine.models.game_sim import GameSim, SimParams, p_above
from nba_engine.models.minutes import MinutesParams
from nba_engine.models.sim_fit import BlendPoint
from nba_engine.schemas import GameResult, PlayerLine, TeamBox

ROLE = {"a": 36, "b": 34, "c": 32, "d": 30, "e": 28, "f": 34, "g": 26, "h": 20}


def box(team):
    return TeamBox(team, 42, 88, 13, 36, 17, 22, 10, 33, 13, 19)


def roster(team, out=(), pts=10):
    return tuple(
        PlayerLine(
            f"{team}{k}",
            team,
            f"{team} {k}",
            0 if k in out else m,
            0 if k in out else (30 if k == "a" else pts),
            0,
            0,
            0,
            dnp=k in out,
            dnp_reason="ANKLE" if k in out else "",
        )
        for k, m in ROLE.items()
    )


def final(eid, day, away="NYK", home="BOS", hq=(30, 30, 25, 25), aq=(28, 28, 25, 25), out=()):
    return GameResult(
        eid,
        day,
        away,
        home,
        "STATUS_FINAL",
        aq,
        hq,
        players=roster(home, out) + roster(away),
        away_box=box(away),
        home_box=box(home),
        season_type=2,
    )


def dist(margin=4.0, total=220.0):
    return game_sim.dist(
        GameSim("1", "2025-01-01", "NYK", "BOS", margin, total, 0.0, 0.0), SimParams()
    )


def test_p_above_is_symmetric_and_drops_pushes():
    assert math.isclose(p_above(0.0, 12.0, 0.0), 0.5)
    assert math.isclose(p_above(3.0, 12.0, 3.0), 0.5)  # mean on a whole line: push removed
    assert math.isclose(p_above(3.0, 12.0, 2.5) + p_above(3.0, 12.0, 3.5), 1.0)
    assert p_above(5.0, 12.0, 0.0) > p_above(5.0, 12.0, 3.0) > p_above(5.0, 12.0, 7.0)


def test_both_sides_of_every_market_sum_to_one():
    d = dist()
    assert math.isclose(d.prob("game_ml", "BOS", None) + d.prob("game_ml", "NYK", None), 1.0)
    assert math.isclose(d.prob("game_ats", "BOS", -4.5) + d.prob("game_ats", "NYK", 4.5), 1.0)
    assert math.isclose(d.prob("h1_ats", "BOS", -2.0) + d.prob("h1_ats", "NYK", 2.0), 1.0)
    assert math.isclose(
        d.prob("game_total", "over", 219.5) + d.prob("game_total", "under", 219.5), 1.0
    )
    assert d.prob("game_ml", "BOS", None) > 0.5
    assert d.prob("game_ml", "LAL", None) is None
    assert d.prob("game_total", "BOS", 220.0) is None


def test_first_half_takes_its_share_of_the_mean():
    p = SimParams(h1_margin_share=0.5, h1_total_share=0.5)
    d = game_sim.dist(GameSim("1", "2025-01-01", "NYK", "BOS", 6.0, 230.0, 0.0, 0.0), p)
    assert (d.h1_margin, d.h1_total) == (3.0, 115.0)
    assert math.isclose(d.prob("h1_total", "over", 115.0), 0.5)


def test_fresh_absence_moves_the_mean_by_lost_production():
    s = GameSim("1", "2025-01-01", "NYK", "BOS", 2.0, 220.0, lost_home=20.0, lost_away=0.0)
    d = game_sim.dist(s, SimParams(lost_margin=0.1, lost_total=-0.05))
    assert math.isclose(d.margin, 0.0) and math.isclose(d.total, 219.0)


def season(n=10, start=date(2024, 11, 1), out_from=None, out=()):
    return [
        final(
            str(i),
            start + timedelta(days=2 * i),
            out=out if out_from is not None and i >= out_from else (),
        )
        for i in range(n)
    ]


def test_replay_reads_fresh_absences_only():
    games = season(n=14, out_from=8, out=("a",))
    sims = game_sim.replay(games, minutes_params=MinutesParams(fresh_games=3))
    lost = {s.espn_id: s.lost_home for s in sims}
    assert lost["7"] == 0.0
    assert lost["8"] > 20.0  # his 30 points a game walked out
    assert lost["10"] > 0.0
    assert lost["11"] == 0.0  # out 3+ games: already in the ratings
    assert all(s.lost_away == 0.0 for s in sims)


def test_replay_has_no_look_ahead():
    games = season(n=8)
    changed = games[:-1] + [final("7", games[-1].game_date, hq=(50, 50, 50, 50))]
    a, b = game_sim.replay(games), game_sim.replay(changed)
    assert a == b  # the last game's score never reaches its own prediction


def test_model_probs_only_for_markets_with_a_weight():
    d = dist()
    sel = Selection(
        "2025-01-01",
        "NYK @ BOS",
        "ev",
        "game_ml",
        "BOS",
        "",
        None,
        0.55,
        5,
        5,
        "draftkings",
        -130.0,
        "t",
    )
    tot = Selection(
        "2025-01-01",
        "NYK @ BOS",
        "ev",
        "game_total",
        "over",
        "",
        219.5,
        0.5,
        5,
        5,
        "draftkings",
        -110.0,
        "t",
    )
    out = game_sim.model_probs(
        [sel, tot], {("2025-01-01", "NYK @ BOS"): d}, SimParams(blend={"game_ml": 0.2})
    )
    p = d.prob("game_ml", "BOS", None)
    assert out == {("ev", "game_ml", "BOS", "", None): 0.2 * p + 0.8 * 0.55}
    assert game_sim.model_probs([sel], {("2025-01-01", "NYK @ BOS"): d}, SimParams()) == {}


def test_params_round_trip():
    p = SimParams(sd_margin=13.0, blend={"h1_ml": 0.1})
    assert SimParams.from_dict(game_sim.params_payload(p)) == p


def play(kind, team, period=1, shooting=False, scoring=False, value=0, who=True):
    return {
        "type": {"text": kind},
        "team": {"id": team},
        "period": {"number": period},
        "shootingPlay": shooting,
        "scoringPlay": scoring,
        "scoreValue": value,
        "participants": [{"athlete": {"id": "9"}}] if who else [],
    }


def test_pbp_halves_count_possessions_and_points():
    summary = {
        "header": {
            "competitions": [
                {
                    "competitors": [
                        {"team": {"id": "1"}, "homeAway": "home"},
                        {"team": {"id": "2"}, "homeAway": "away"},
                    ]
                }
            ]
        },
        "plays": [
            play("Jump Shot", "1", shooting=True, scoring=True, value=3),
            play("Jump Shot", "1", shooting=True),
            play("Offensive Rebound", "1"),
            play("Offensive Rebound", "1", who=False),  # team rebound: no new possession
            play("Free Throw - 1 of 2", "1", shooting=True, scoring=True, value=1),
            play("Free Throw - Technical", "1", shooting=True, scoring=True, value=1),
            play("Traveling", "2", period=3),
            play("Bad Pass\nTurnover", "2", period=4),
            play("No Turnover", "2", period=4),
            play("Jump Shot", "2", period=5, shooting=True, scoring=True, value=2),  # overtime
        ],
    }
    lines = {(x.side, x.half): x for x in pbp.halves(summary)}
    assert math.isclose(lines[("home", 1)].possessions, 2 - 1 + 0 + 0.44)
    assert lines[("home", 1)].points == 5
    assert lines[("away", 2)].possessions == 2.0 and lines[("away", 2)].points == 0
    assert pbp.halves({"plays": []}) == []


def test_fit_halves_splits_pace_from_efficiency():
    g = final("1", date(2024, 11, 1))
    halves = [
        pbp.HalfLine("home", 1, 50.0, 55),
        pbp.HalfLine("home", 2, 48.0, 55),
        pbp.HalfLine("away", 1, 50.0, 55),
        pbp.HalfLine("away", 2, 48.0, 55),
    ]
    fit = sim_fit.fit_halves({"1": halves}, {"1": g}, {2024}, draws=10)
    assert math.isclose(fit.pace_shift, -0.04)
    assert math.isclose(fit.eff_shift, 50.0 / 48.0 - 1.0)


def points(rng, n, informative):
    out = []
    for i in range(n):
        truth = rng.uniform(0.2, 0.8)
        won = rng.random() < truth
        p = truth if informative else rng.uniform(0.2, 0.8)
        out.append(BlendPoint("game_ml", 2023, f"d{i % 200}", p, 0.5, won))
    return out


def test_blend_weight_is_zero_for_noise_and_large_for_signal():
    rng = random.Random(3)
    noise = points(rng, 3000, informative=False)
    signal = points(rng, 3000, informative=True)
    assert sim_fit.best_weight(noise) < 0.1
    assert sim_fit.best_weight(signal) > 0.8
    fit = sim_fit.fit_blend(signal, {2023}, set(), draws=20)
    assert fit[0].market == "game_ml" and fit[0].applied > 0.8
    assert sim_fit.fit_blend(noise, {2023}, set(), draws=20)[0].applied < 0.1


def test_points_take_the_main_line_and_skip_pushes():
    g = final("1", date(2025, 1, 1))  # BOS 110-106
    d = dist()
    key = ("2025-01-01", "NYK @ BOS")

    def sel(market, side, line, fair):
        return Selection(
            key[0], key[1], "ev", market, side, "", line, fair, 5, 5, "draftkings", -110.0, "t"
        )

    sels = [
        sel("game_ats", "BOS", -4.0, 0.49),  # main line, but a push
        sel("game_ats", "BOS", -8.5, 0.30),
        sel("game_ats", "NYK", 4.0, 0.51),
        sel("game_total", "over", 215.5, 0.52),
        sel("game_total", "over", 225.5, 0.31),
    ]
    out = sim_fit.points_for_day(sels, {key: d}, {key: g})
    assert [(x.market, x.fair, x.won) for x in out] == [("game_total", 0.52, True)]


def test_fit_sim_cli_writes_versioned_params(tmp_path, monkeypatch):
    monkeypatch.setenv("NBAE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("NBAE_STATE_SYNC", "0")
    rng = random.Random(5)
    start = date(2024, 11, 1)
    for i in range(30):
        h = rng.randint(20, 35)
        g = final(
            str(i),
            start + timedelta(days=2 * i),
            hq=(h, 28, 27, 26),
            aq=(27, rng.randint(20, 35), 27, 26),
            out=("a",) if i >= 20 else (),
        )
        boxes.write_results(tmp_path, g.game_date, [g])
    rc = cli.main(
        [
            "fit-sim",
            "--season",
            "2024-25",
            "--train",
            "2024",
            "--draws",
            "20",
            "--blend-draws",
            "10",
        ]
    )
    assert rc == 0
    held = params.latest(tmp_path, game_sim.NAME)
    p = SimParams.from_dict(held["params"])
    assert p.sd_margin > 0 and p.blend == {}
    assert held["blend"] == []  # no archived closes, so no blend points
