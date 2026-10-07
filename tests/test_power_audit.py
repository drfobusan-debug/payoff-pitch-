"""The power screen's audit article: the day, the whole ledger, and the commentary."""

from __future__ import annotations

from pathlib import Path

from mlb_engine.audit import power_audit, power_roster
from mlb_engine.audit.power_ledger import Position
from mlb_engine.audit.power_roster import ARM, BAT, DROPPED, GATED, HELD, SCREENED, RosterRow
from mlb_engine.data import plays as plays_mod
from mlb_engine.data.plays import PlateAppearance
from mlb_engine.data.results import GameResult, PlayerLine

DAY = "2026-10-06"
GAME = 9001
HOME_SP, AWAY_SP, RELIEVER = 10, 20, 30
BAT_A, BAT_B, BAT_C, BAT_D = 101, 102, 103, 104


def _pos(
    player_id: int,
    *,
    batter: str = "",
    stat: str = "H",
    side: str = "under",
    odds: float = -300.0,
    model: float = 0.6,
    fair: float | None = 0.7,
    rating: str = "ELITE UNDER",
    day: str = DAY,
    run_id: str = "r2",
    tier: str = "Pass",
) -> Position:
    return Position(
        date=day,
        batter=batter or f"Bat {player_id}",
        player_id=player_id,
        game_pk=GAME,
        stat=stat,
        line=0.5,
        side=side,
        book="dk",
        odds=odds,
        model_prob=model,
        fair_prob=fair,
        edge=None,
        ev=None,
        tier=tier,
        rating=rating,
        devigged=fair is not None,
        run_id=run_id,
        arm_tier="elite",
    )


def _bat(pid: int, status: str = HELD, slot: int = 1, side: str = "under") -> RosterRow:
    return RosterRow(
        date=DAY, run_id="r2", arm_tier="elite", kind=BAT, name=f"Bat {pid}",
        player_id=pid, team="ATL", game_pk=GAME, versus="Home Ace", versus_id=HOME_SP,
        slot=slot, projected=True, bucket="ELITE UNDER", side=side, status=status,
    )


def _arm(pid: int, name: str, status: str, tier: str = "elite") -> RosterRow:
    return RosterRow(
        date=DAY, run_id="r2", arm_tier=tier, kind=ARM, name=name, player_id=pid,
        team="LAD", game_pk=GAME, versus=name, versus_id=pid, status=status,
        reason="siera: 2.90" if status == GATED else "", siera=2.9,
    )


def _result() -> GameResult:
    return GameResult(
        game_pk=GAME, final=True, home_runs=3, away_runs=1, f5_home=1, f5_away=0,
        players={
            BAT_A: PlayerLine(batting={"PA": 4, "H": 0, "RBI": 0}, order=100),
            # Projected leadoff, batted third.
            BAT_B: PlayerLine(batting={"PA": 4, "H": 2, "RBI": 1, "1B": 1, "HR": 1}, order=300),
            # BAT_C never played; BAT_D was dropped and still batted.
            BAT_D: PlayerLine(batting={"PA": 3, "H": 1}, order=500),
            HOME_SP: PlayerLine(pitching={"BF": 25, "K": 10, "outs": 21, "H": 4, "BB": 1, "ER": 1}),
            AWAY_SP: PlayerLine(pitching={"BF": 20, "K": 3, "outs": 9, "H": 3, "BB": 2, "ER": 2}),
        },
    )


def _plays() -> list[PlateAppearance]:
    top = [
        PlateAppearance(BAT_A, HOME_SP, "strikeout", "top"),
        PlateAppearance(BAT_B, HOME_SP, "home_run", "top"),
        PlateAppearance(BAT_A, HOME_SP, "walk", "top"),
        PlateAppearance(BAT_B, HOME_SP, "caught_stealing_2b", "top"),  # not a PA
        PlateAppearance(BAT_B, RELIEVER, "single", "top"),  # off the pen
        PlateAppearance(BAT_D, HOME_SP, "double", "top"),
    ]
    bottom = [PlateAppearance(900, AWAY_SP, "field_out", "bottom")]
    return top + bottom


def _build(positions: list[Position], roster: list[RosterRow]):
    return power_audit.build(
        DAY,
        positions,
        roster,
        fetch_result=lambda pk: _result() if pk == GAME else None,
        fetch_plays=lambda pk, final: _plays(),
    )


def test_tally_counts_plate_appearances_not_baserunning_plays() -> None:
    t = plays_mod.tally(_plays()[:4])
    assert (t.pa, t.ab, t.h, t.tb, t.hr, t.bb, t.k) == (3, 2, 1, 4, 1, 1, 1)
    assert plays_mod.starters(_plays()) == {"home": HOME_SP, "away": AWAY_SP}


def test_box_score_order_says_who_started_and_where() -> None:
    assert PlayerLine(order=300).started and PlayerLine(order=300).slot == 3
    assert not PlayerLine(order=301).started
    assert not PlayerLine().started


def test_a_bat_is_graded_against_the_starter_only_and_a_dnp_is_missing() -> None:
    roster = [_bat(BAT_A), _bat(BAT_B), _bat(BAT_C, slot=5)]
    day, _ = _build([_pos(BAT_A)], roster)
    by = {b.row.player_id: b for b in day.bats}
    assert by[BAT_B].vs_starter.h == 1 and by[BAT_B].game.h == 2
    assert by[BAT_B].slot_moved
    assert not by[BAT_C].played and by[BAT_C].game.pa == 0
    assert by[BAT_A].priced and not by[BAT_B].priced
    assert by[BAT_A].probable_started is True


def test_the_day_and_the_ledger_grade_the_last_run_and_void_the_absent() -> None:
    positions = [
        _pos(BAT_A, run_id="r1", side="over", odds=200.0, model=0.3, fair=0.3),
        _pos(BAT_A),
        _pos(BAT_C),  # never played: voided, not lost
        _pos(BAT_A, day="2026-10-05", odds=100.0, model=0.5, fair=0.5),
    ]
    day, total = _build(positions, [])
    assert day.recorded == 2 and len(day.graded) == 1 and day.voided == 1
    assert day.graded[0].result == "win"
    assert total.days == ["2026-10-05", DAY]
    assert total.recorded == 3
    assert not day.roster_recorded


def test_both_sided_pairs_are_flagged_and_removable() -> None:
    positions = [
        _pos(BAT_A, side="over", odds=250.0, model=0.3, fair=0.28, rating="RV NEG WATCH"),
        _pos(BAT_A, side="under", odds=-300.0, model=0.7, fair=0.72, rating="RV NEG WATCH"),
        _pos(BAT_B, side="under"),
    ]
    day, total = _build(positions, [_bat(BAT_A), _bat(BAT_B)])
    assert len(power_audit.two_sided(g.position for g in total.graded)) == 1
    assert len(power_audit.without_pairs(total.graded)) == 1
    notes = power_audit.commentary(day, total)
    assert any("both sides" in p for p in notes.problems)
    assert any("one side per prop" in r for r in notes.recommendations)


def test_commentary_names_lineup_misses_unpriced_bats_and_dropped_positions() -> None:
    roster = [
        _bat(BAT_A), _bat(BAT_B), _bat(BAT_C, slot=5), _bat(BAT_D, status=DROPPED),
        _arm(HOME_SP, "Home Ace", SCREENED), _arm(AWAY_SP, "Away Arm", GATED, tier="soft"),
    ]
    day, total = _build([_pos(BAT_A), _pos(BAT_D, rating="PROD DROP")], roster)
    notes = power_audit.commentary(day, total)
    text = " ".join(notes.problems)
    assert "did not play: Bat 103" in text
    assert "Bat 102 (1→3)" in text
    assert "no recorded price" in text
    assert day.dropped_positions == ["Bat 104"]
    assert any("dropped on production" in p for p in notes.problems)
    assert any("Home Ace" in i and "10 K" in i for i in notes.insights)
    assert notes.recommendations[-1].startswith("This audit changes nothing")


def test_a_small_ledger_is_called_underpowered_not_an_edge() -> None:
    day, total = _build([_pos(BAT_A)], [_bat(BAT_A)])
    cmp_ = power_audit.compare(total.graded)
    assert cmp_ is not None and cmp_.verdict == "underpowered"
    notes = power_audit.commentary(day, total)
    assert any("no pricing change" in r for r in notes.recommendations)
    assert any("Underpowered" in i for i in notes.insights)


def test_a_day_without_a_roster_says_what_is_missing() -> None:
    day, total = _build([_pos(BAT_A)], [])
    notes = power_audit.commentary(day, total)
    assert any("No roster was recorded" in p for p in notes.problems)


def test_the_article_has_the_three_parts_in_order() -> None:
    roster = [_bat(BAT_A), _bat(BAT_B), _arm(HOME_SP, "Home Ace", SCREENED)]
    day, total = _build([_pos(BAT_A)], roster)
    body = power_audit.render_html(day, total, power_audit.commentary(day, total))
    i, j, k = (body.index(h) for h in ("1. The day", "2. The whole record", "3. Commentary"))
    assert i < j < k
    for h in ("Problems", "Insights", "Recommendations", "Every bat", "The arms"):
        assert h in body
    assert "Power screen audit" in power_audit.render_text(
        day, total, power_audit.commentary(day, total)
    )


def test_roster_round_trips_and_a_rerun_replaces_only_its_own_rows(tmp_path: Path) -> None:
    path = tmp_path / power_roster.ROSTER_NAME
    power_roster.record(path, [_bat(BAT_A)], DAY, "r2")
    power_roster.record(path, [_bat(BAT_B)], DAY, "r2")
    other = RosterRow(**{**_bat(BAT_C).__dict__, "run_id": "r1"})
    power_roster.record(path, [other], DAY, "r1")
    rows = power_roster.load(path)
    assert {(r.run_id, r.player_id) for r in rows} == {("r2", BAT_B), ("r1", BAT_C)}
    assert power_roster.for_day(path, DAY)[0].player_id == BAT_B
    assert rows[0].projected is True and rows[0].versus_id == HOME_SP
