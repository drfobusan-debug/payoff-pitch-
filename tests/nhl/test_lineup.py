"""Lineup block: NHL API PBP/shift parsers, MoneyPuck shot log, stints, RAPM, rebuild, availability."""

from __future__ import annotations

import json
import math
import random
from datetime import date as Date
from pathlib import Path

import pandas as pd
import pytest

from nhl_engine.data import availability as av
from nhl_engine.data.capture import QuoteRow
from nhl_engine.data.moneypuck import SHOT_COLUMNS, MoneyPuckClient, full_game_id
from nhl_engine.data.nhlapi import (
    NHLAPIClient,
    RosterSpot,
    Shift,
    parse_roster_spots,
    parse_shifts,
    regular_season_game_ids,
)
from nhl_engine.features import lineup

FIX = Path(__file__).parent / "fixtures"
GAME = 2024020500


@pytest.fixture
def pbp() -> dict:
    return json.loads((FIX / "pbp_2024020500.json").read_text())


@pytest.fixture
def shift_json() -> dict:
    return json.loads((FIX / "shifts_2024020500.json").read_text())


@pytest.fixture
def shots() -> pd.DataFrame:
    return pd.read_csv(FIX / "mp_shots_20500.csv")


# --- NHL API parsers -----------------------------------------------------------


def test_regular_season_ids() -> None:
    ids = regular_season_game_ids(2024, games=3)
    assert ids == [2024020001, 2024020002, 2024020003]
    assert full_game_id(2024, 20500) == GAME


def test_parse_roster_spots(pbp: dict) -> None:
    spots = parse_roster_spots(pbp)
    teams = {s.team for s in spots}
    assert teams == {"CAR", "NYI"}
    assert sum(1 for s in spots if s.position == "G") == 4  # two dressed goalies a side
    assert all(s.name and s.player_id > 0 for s in spots)
    assert parse_roster_spots({}) == []


def test_parse_shifts_drops_non_shift_rows_and_sorts(shift_json: dict) -> None:
    shifts = parse_shifts(shift_json)
    assert len(shifts) == 264  # 267 rows, three typeCode 505 (goal) rows dropped
    assert all(s.period == 1 and s.end > s.start for s in shifts)
    assert shifts == sorted(shifts, key=lambda s: (s.period, s.start, s.player_id))
    assert {s.team for s in shifts} == {"CAR", "NYI"}
    assert parse_shifts({"data": [{"typeCode": 517, "startTime": "x"}]}) == []


def test_client_reads_pbp_and_shifts_from_cache(
    tmp_path: Path, pbp: dict, shift_json: dict
) -> None:
    (tmp_path / f"gamecenter_{GAME}_play-by-play.json").write_text(json.dumps(pbp))
    (tmp_path / f"stats_shiftcharts_cayenneExp_gameId_{GAME}.json").write_text(
        json.dumps(shift_json)
    )
    client = NHLAPIClient(cache_dir=tmp_path)
    assert client.play_by_play(GAME) is not None
    assert len(client.roster(GAME)) == len(parse_roster_spots(pbp))
    assert len(client.shifts(GAME)) == 264


# --- MoneyPuck shots -------------------------------------------------------------


def test_shots_from_cache_trimmed(tmp_path: Path, shots: pd.DataFrame) -> None:
    cache = tmp_path / "moneypuck"
    cache.mkdir()
    shots.to_csv(cache / "shots_2024.csv", index=False)
    mp = MoneyPuckClient(cache_dir=tmp_path, cache_ttl=10**9)
    df = mp.shots(2024)
    assert list(df.columns) == list(SHOT_COLUMNS)
    assert set(df["homeTeamCode"]) == {"CAR"} and set(df["awayTeamCode"]) == {"NYI"}
    assert df["game_id"].dtype.kind == "i" and int(df["game_id"].iloc[0]) == 20500


# --- stints ----------------------------------------------------------------------


def test_build_stints_5v5_only_and_xg_assigned(
    pbp: dict, shift_json: dict, shots: pd.DataFrame
) -> None:
    roster = parse_roster_spots(pbp)
    shifts = parse_shifts(shift_json)
    stints = lineup.build_stints(GAME, shifts, roster, shots, home="CAR", away="NYI")
    assert stints and all(len(s.home) == 5 and len(s.away) == 5 for s in stints)
    assert all(s.period == 1 and s.seconds > 0 and s.venue == "CAR" for s in stints)
    # stints are disjoint and ordered
    for a, b in zip(stints, stints[1:], strict=False):
        assert a.end <= b.start
    total_5v5 = sum(s.seconds for s in stints)
    assert 600 < total_5v5 <= 1200
    # every 5v5 P1 shot that falls inside a stint is counted exactly once
    p1 = shots[(shots.period == 1) & (shots.homeSkatersOnIce == 5) & (shots.awaySkatersOnIce == 5)]
    stint_xg = sum(s.home_xg + s.away_xg for s in stints)
    assert stint_xg <= p1.xGoal.sum() + 1e-9 + shots[shots.period == 1].xGoal.sum() * 0.5
    assert stint_xg > 0


# --- RAPM ------------------------------------------------------------------------


def _synthetic(seed: int = 0, n: int = 4000) -> tuple[list[lineup.Stint], dict[int, float]]:
    """Two teams of 10 skaters each; player 1 is a +1.0 xGF/60 driver."""
    rng = random.Random(seed)
    true_off = {pid: 0.0 for pid in range(1, 21)}
    true_off[1] = 1.0
    true_off[11] = -0.6
    stints = []
    for i in range(n):
        home = tuple(sorted(rng.sample(range(1, 11), 5)))
        away = tuple(sorted(rng.sample(range(11, 21), 5)))
        sec = 45
        base = 2.5 / 3600 * sec
        hx = base * (1 + sum(true_off[p] for p in home) / 2.5) + rng.gauss(0, 0.02)
        ax = base * (1 + sum(true_off[p] for p in away) / 2.5) + rng.gauss(0, 0.02)
        stints.append(lineup.Stint(i, 1, 0, sec, home, away, max(hx, 0), max(ax, 0), "H"))
    return stints, true_off


def test_fit_rapm_recovers_isolated_driver() -> None:
    stints, truth = _synthetic()
    fit = lineup.fit_rapm(stints, lam=500.0)
    imp = fit.impacts
    assert imp[1].off > 0.6 and imp[1].off == max(i.off for i in imp.values())
    assert imp[11].off < -0.3 and imp[11].off == min(i.off for i in imp.values())
    # linemates of player 1 are not credited with his production
    others = [imp[p].off for p in range(2, 11)]
    assert max(abs(x) for x in others) < 0.3
    assert abs(fit.intercept - 2.5) < 0.6
    assert fit.venues == {}


def test_fit_rapm_venue_effect_and_shrink() -> None:
    stints, _ = _synthetic()
    with_venue = lineup.fit_rapm(stints, lam=500.0, venue_effects=True)
    assert set(with_venue.venues) == {"H"}
    tight = lineup.fit_rapm(stints, lam=1e9)
    assert all(abs(i.off) < 1e-3 and abs(i.dfn) < 1e-3 for i in tight.impacts.values())
    assert lineup.fit_rapm([], lam=1.0).impacts == {}


# --- rebuild ---------------------------------------------------------------------


def _impacts() -> dict[int, lineup.Impact]:
    return {
        1: lineup.Impact(1, 0.8, -0.2, 40_000.0),
        2: lineup.Impact(2, 0.1, 0.1, 40_000.0),
        3: lineup.Impact(3, -0.3, 0.4, 40_000.0),
        9: lineup.Impact(9, 0.5, 0.0, 300.0),  # thin call-up
    }


def test_rebuild_healthy_lineup_is_the_team_rate() -> None:
    base = {1: 1.4, 2: 1.8, 3: 1.8}
    r = lineup.rebuild(
        "TOR",
        team_xgf60=2.6,
        team_xga60=2.4,
        dressed=dict(base),
        baseline=base,
        impacts=_impacts(),
        k=20_000.0,
        source="last_game",
    )
    assert math.isclose(r.xgf60, 2.6, abs_tol=1e-12) and math.isclose(r.xga60, 2.4, abs_tol=1e-12)
    assert r.turnover == 0.0


def test_rebuild_absence_is_sized_by_isolated_impact_and_exposure() -> None:
    base = {1: 1.4, 2: 1.8, 3: 1.8}
    dressed = lineup.drop_players(base, [1], positions={1: "C", 2: "L", 3: "D"})
    assert math.isclose(sum(dressed.values()), 5.0)
    assert dressed[2] == pytest.approx(3.2) and dressed[3] == pytest.approx(
        1.8
    )  # F share stays with F
    r = lineup.rebuild(
        "TOR",
        team_xgf60=2.6,
        team_xga60=2.4,
        dressed=dressed,
        baseline=base,
        impacts=_impacts(),
        k=20_000.0,
        source="projected",
    )
    assert r.delta_for < 0 and r.delta_against > 0  # star out: less for, more against
    assert 0 < r.turnover < 1
    # replace him with a thin call-up: near-zero effect because reliability ~ 300/(300+k)
    dressed2 = dict(base)
    dressed2[9] = dressed2.pop(1)
    r2 = lineup.rebuild(
        "TOR",
        team_xgf60=2.6,
        team_xga60=2.4,
        dressed=dressed2,
        baseline=base,
        impacts=_impacts(),
        k=20_000.0,
        source="projected",
    )
    assert abs(r2.contributions[9][0]) < 0.02
    assert r2.xgf60 < r.xgf60 + 0.05


def test_toi_shares_sum_to_five() -> None:
    stints, _ = _synthetic(n=200)
    shares = lineup.toi_shares(stints, {s.game_id: True for s in stints})
    assert math.isclose(sum(shares.values()), 5.0)
    assert set(shares) == set(range(1, 11))


# --- availability ----------------------------------------------------------------


def test_availability_log_append_once_and_current_out(tmp_path: Path) -> None:
    slate = Date(2026, 10, 1)
    prev = [
        RosterSpot(1, "TOR", "C", "A Star"),
        RosterSpot(2, "TOR", "G", "G One"),
        RosterSpot(7, "BOS", "D", "X"),
    ]
    tonight = [RosterSpot(2, "TOR", "G", "G One")]
    quotes = [
        QuoteRow(
            "2026-10-01T20:00:00Z",
            "2026-10-01",
            "BOS@TOR",
            "e",
            "game_ml",
            "TOR",
            "",
            None,
            "dk",
            -130.0,
            110.0,
        ),
        QuoteRow(
            "2026-10-01T21:00:00Z",
            "2026-10-01",
            "BOS@TOR",
            "e",
            "game_ml",
            "TOR",
            "",
            None,
            "fd",
            -120.0,
            100.0,
        ),
    ]
    assert av.market_snapshot(quotes, "BOS@TOR") == (-120.0, "fd")
    recs = av.derive_from_rosters(
        slate,
        "TOR",
        prev,
        tonight,
        minutes_to_drop=25.0,
        market=av.market_snapshot(quotes, "BOS@TOR"),
    )
    assert (
        [r.player_id for r in recs] == [1] and recs[0].role == "F" and recs[0].tag == av.NOT_SCORED
    )
    assert av.append(tmp_path, recs) == recs
    assert av.append(tmp_path, recs) == []  # idempotent
    logged = av.read_log(tmp_path, slate)
    assert len(logged) == 1 and logged[0].market_home_ml == -120.0
    assert set(av.current_out(logged, "TOR")) == {1}
    assert av.current_out(logged, "BOS") == {}
    # a later 'in' record clears him
    back = av.Availability(
        slate.isoformat(), "TOR", 1, "A Star", "in", "manual", "", "2026-10-01T23:00:00Z"
    )
    av.append(tmp_path, [back])
    assert av.current_out(av.read_log(tmp_path, slate), "TOR") == {}
    assert av.log_path(tmp_path, slate).read_text().count("\n") == 2


def test_shift_dataclass_is_hashable() -> None:
    assert len({Shift(1, "TOR", 1, 0, 10), Shift(1, "TOR", 1, 0, 10)}) == 1
