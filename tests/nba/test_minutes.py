import math
from dataclasses import replace
from datetime import date, timedelta

from nba_engine import cli
from nba_engine.data import boxes
from nba_engine.data.espn import parse_box
from nba_engine.market import params
from nba_engine.models import minutes, minutes_fit
from nba_engine.models.minutes import MinutesBook, MinutesParams, is_out
from nba_engine.schemas import GameResult, PlayerLine

COACH = minutes.COACH
TEAM = ("a", "b", "c", "d", "e", "f", "g", "h")
ROLE = {"a": 36, "b": 34, "c": 32, "d": 30, "e": 28, "f": 34, "g": 26, "h": 20}  # sums to 240


def line(pid, mins, team="BOS", dnp_reason=""):
    return PlayerLine(
        pid, team, pid.upper(), mins, 0, 0, 0, 0, dnp=mins == 0, dnp_reason=dnp_reason
    )


def game(eid, day, roster, away="NYK", home="BOS", ot=0):
    opp = tuple(line(f"o{k}", m, away) for k, m in ROLE.items())
    q = (25, 25, 25, 25) + (10,) * ot
    return GameResult(eid, day, away, home, "STATUS_FINAL", q, q, players=roster + opp)


def season(n=12, start=date(2024, 11, 1), out=(), absent=()):
    """``n`` games of the fixed rotation; ``out`` DNP'd injured, ``absent`` missing from the box."""
    gs = []
    for i in range(n):
        roster = tuple(
            line(k, 0, dnp_reason="LEFT ANKLE SPRAIN") if k in out else line(k, m)
            for k, m in ROLE.items()
            if k not in absent
        )
        gs.append(game(str(i), start + timedelta(days=2 * i), roster))
    return gs


def test_availability_rules():
    assert is_out(None)
    assert is_out(line("x", 0, dnp_reason="ILLNESS"))
    assert not is_out(line("x", 0, dnp_reason=COACH))
    assert not is_out(line("x", 12))


def test_parse_box_keeps_the_dnp_reason():
    summary = {
        "boxscore": {
            "players": [
                {
                    "team": {"abbreviation": "BOS"},
                    "statistics": [
                        {
                            "keys": ["minutes", "points"],
                            "athletes": [
                                {"athlete": {"id": "1"}, "stats": ["30", "12"], "reason": COACH},
                                {
                                    "athlete": {"id": "2"},
                                    "stats": [],
                                    "didNotPlay": True,
                                    "reason": "RIGHT KNEE SORENESS",
                                },
                            ],
                        }
                    ],
                }
            ]
        }
    }
    a, b = parse_box(summary)
    assert (a.dnp, a.dnp_reason) == (False, "")
    assert (b.dnp, b.dnp_reason) == (True, "RIGHT KNEE SORENESS")


def test_settled_rotation_projects_its_own_minutes_and_sums_to_240():
    rows, _ = minutes.replay(season(15))
    last = [r for r in rows if r.espn_id == "14" and r.team == "BOS"]
    assert math.isclose(sum(r.minutes for r in last), minutes.TEAM_MINUTES)
    for r in last:
        assert abs(r.minutes - ROLE[r.player]) < 1.0


def test_fresh_absence_hands_minutes_to_teammates_under_the_cap():
    gs = season(12) + season(1, start=date(2024, 12, 1), absent=("a",))
    rows, _ = minutes.replay(gs)
    night = {r.player: r for r in rows if r.game_date == date(2024, 12, 1) and r.team == "BOS"}
    assert "a" not in night
    assert all(r.fresh_out for r in night.values())
    assert math.isclose(sum(r.minutes for r in night.values()), minutes.TEAM_MINUTES)
    assert all(r.minutes > ROLE[p] for p, r in night.items())
    assert all(r.minutes <= MinutesParams().cap + 1e-9 for r in night.values())


def test_long_absence_is_already_in_the_baselines():
    gs = season(12) + season(12, start=date(2024, 12, 1), out=("a",))
    rows, book = minutes.replay(gs)
    last = {
        r.player: r
        for r in rows
        if r.espn_id == "11" and r.game_date.month == 12 and r.team == "BOS"
    }
    assert not any(r.fresh_out for r in last.values())
    assert not book.fresh({"a"})
    assert math.isclose(sum(r.minutes for r in last.values()), minutes.TEAM_MINUTES)


def test_next_man_up_history_moves_the_named_backup():
    """When ``a`` sits, ``h`` takes all his minutes; the pair term learns that."""
    p = MinutesParams(pair_k=1.0, fresh_games=1.0)
    gs = []
    day = date(2024, 11, 1)
    for i in range(40):
        sits = i % 4 == 3
        roster = tuple(
            line(k, (ROLE["h"] + ROLE["a"]) if (sits and k == "h") else m)
            for k, m in ROLE.items()
            if not (sits and k == "a")
        )
        gs.append(game(str(i), day + timedelta(days=2 * i), roster))
    with_pair, _ = minutes.replay(gs, p)
    without, _ = minutes.replay(gs, replace(p, pair_k=1e9))
    last = gs[-1].espn_id

    def miss(rows):
        return sum(abs(r.minutes - (r.actual or 0)) for r in rows if r.espn_id == last)

    assert miss(with_pair) < miss(without)


def test_projection_never_sees_its_own_game():
    gs = season(10)
    rows, _ = minutes.replay(gs)
    changed = gs[:-1] + [
        replace(gs[-1], players=tuple(replace(pl, minutes=48) for pl in gs[-1].players))
    ]
    rows2, _ = minutes.replay(changed)
    a = [r.minutes for r in rows if r.espn_id == "9"]
    b = [r.minutes for r in rows2 if r.espn_id == "9"]
    assert a == b


def test_overtime_minutes_are_scaled_to_regulation():
    assert math.isclose(minutes.regulation(53, 1), 48.0)


def test_coach_dnp_is_available_at_zero():
    book = MinutesBook()
    box = {k: line(k, m) for k, m in ROLE.items()}
    box["z"] = line("z", 0, dnp_reason=COACH)
    book.update("BOS", 2024, box, 0, set())
    assert book.games("z") == 1.0
    assert book.baseline("z") < MinutesParams().prior_minutes


def test_fit_and_grade_and_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("NBAE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("NBAE_STATE_SYNC", "0")
    monkeypatch.setattr(minutes_fit, "GRID", {"half_life": (3.0, 5.0)})
    gs = season(20)
    rows, _ = minutes.replay(gs)
    grades = minutes_fit.grade(rows, {2024}, draws=50)
    assert {g.group for g in grades} >= {"all", "rotation"}
    assert all(g.lo <= g.gain <= g.hi for g in grades)
    for g in gs:
        boxes.write_results(tmp_path, g.game_date, [g])
    rc = cli.main(["fit-minutes", "--season", "2024-25", "--train", "2024", "--draws", "20"])
    assert rc == 0
    held = params.latest(tmp_path, minutes.NAME)
    assert held is not None and "half_life" in held["params"]
    assert MinutesParams.from_dict(held["params"]).half_life in (3.0, 4.0, 5.0)
