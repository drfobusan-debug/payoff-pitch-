"""The context basis and the prop grader: prior weeks only, stamped, never a bet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_engine import props, props_grade
from nfl_engine.data import nflverse
from nfl_engine.features import context, usage
from nfl_engine.models.player import ATTEMPTS, RECEIVING_YARDS, RECEPTIONS, Projection
from tests.nfl.test_props import projection, quote

RECEPTION_TERMS = context.TERMS[RECEPTIONS]


def test_adjusted_mean_is_the_usage_projection_at_a_neutral_game() -> None:
    neutral = context.adjusted_mean(
        context.Terms(0.0, 1.0, 0.0, 0.5, 0.5, 0.5),
        base=6.0,
        share_volume=6.0,
        opp_factor=1.0,
        team_spread=0.0,
        total_line=context.TOTAL_CENTRE,
    )
    assert neutral == pytest.approx(6.0)


def test_adjusted_mean_moves_with_the_defence_and_the_total() -> None:
    soft = context.adjusted_mean(
        RECEPTION_TERMS,
        base=6.0,
        share_volume=6.0,
        opp_factor=1.3,
        team_spread=0.0,
        total_line=45.0,
    )
    stiff = context.adjusted_mean(
        RECEPTION_TERMS,
        base=6.0,
        share_volume=6.0,
        opp_factor=0.7,
        team_spread=0.0,
        total_line=45.0,
    )
    shootout = context.adjusted_mean(
        RECEPTION_TERMS,
        base=6.0,
        share_volume=6.0,
        opp_factor=1.0,
        team_spread=0.0,
        total_line=55.0,
    )
    assert soft > stiff
    assert shootout > context.adjusted_mean(
        RECEPTION_TERMS,
        base=6.0,
        share_volume=6.0,
        opp_factor=1.0,
        team_spread=0.0,
        total_line=45.0,
    )


def test_adjusted_mean_never_goes_negative() -> None:
    assert (
        context.adjusted_mean(
            context.Terms(-5.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            base=1.0,
            share_volume=0.0,
            opp_factor=1.0,
            team_spread=0.0,
            total_line=45.0,
        )
        == 0.0
    )


def test_features_columns_follow_terms_field_order() -> None:
    row = context.features(
        np.array([10.0]), np.array([8.0]), np.array([1.2]), np.array([7.0]), np.array([55.0])
    )[0]
    assert list(row) == pytest.approx([1.0, 10.0, 8.0, 2.0, 10.0, 10.0])


def _weekly_fixture() -> pd.DataFrame:
    rows = []
    for week in range(1, 7):
        for team, opp in (("LA", "SEA"), ("SEA", "LA")):
            rows.append(
                {
                    "player_id": f"qb_{team}",
                    "player_display_name": f"QB {team}",
                    "position": "QB",
                    "season": 2026,
                    "week": week,
                    "team": team,
                    "opponent_team": opp,
                    "attempts": 30.0 if team == "LA" else 40.0,
                    "completions": 20.0,
                    "passing_yards": 250.0,
                    "carries": 2.0,
                    "rushing_yards": 5.0,
                    "targets": None,
                    "receptions": None,
                    "receiving_yards": None,
                }
            )
            rows.append(
                {
                    "player_id": f"wr_{team}",
                    "player_display_name": f"WR {team}",
                    "position": "WR",
                    "season": 2026,
                    "week": week,
                    "team": team,
                    "opponent_team": opp,
                    "attempts": None,
                    "completions": None,
                    "passing_yards": None,
                    "carries": 0.0,
                    "rushing_yards": 0.0,
                    "targets": 9.0 if team == "LA" else 6.0,
                    "receptions": 6.0 if team == "LA" else 4.0,
                    "receiving_yards": 70.0,
                }
            )
    frame = pd.DataFrame(rows)
    frame["season_type"] = "REG"
    return frame


def _games_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2026_07_SEA_LA",
                "season": 2026,
                "week": 7,
                "home_team": "LA",
                "away_team": "SEA",
                "spread_line": 3.0,
                "total_line": 51.0,
            }
        ]
    )


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    weekly = _weekly_fixture()
    monkeypatch.setattr(
        nflverse, "player_week", lambda season: weekly if season == 2026 else pd.DataFrame()
    )
    monkeypatch.setattr(nflverse, "games", lambda: _games_fixture())
    return weekly


def test_context_projections_cover_the_usage_projections_and_read_the_week_line(
    stubbed: pd.DataFrame,
) -> None:
    base = usage.projections(2026, 7)
    ctx = context.projections(2026, 7)
    assert set(ctx) == set(base)
    la = ctx[("wr la", RECEPTIONS)]
    assert la.prior_mean == pytest.approx(base[("wr la", RECEPTIONS)].mean)
    assert la.mean != base[("wr la", RECEPTIONS)].mean
    # A stat without terms keeps the usage projection under this basis.
    assert ctx[("wr la", RECEIVING_YARDS)].mean == base[("wr la", RECEIVING_YARDS)].mean


def test_context_projections_use_only_weeks_before_the_one_priced(stubbed: pd.DataFrame) -> None:
    stubbed.loc[(stubbed.week == 6) & (stubbed.player_id == "qb_LA"), "attempts"] = 90.0
    six = context.projections(2026, 6)
    seven = context.projections(2026, 7)
    assert seven[("qb la", ATTEMPTS)].mean > six[("qb la", ATTEMPTS)].mean


def test_week_context_reads_the_spread_from_each_side(stubbed: pd.DataFrame) -> None:
    ctx = context.week_context(2026, 7)
    assert ctx["LA"].team_spread == 3.0
    assert ctx["SEA"].team_spread == -3.0
    assert ctx["LA"].opponent == "SEA"
    assert ctx["SEA"].total_line == 51.0


def test_priced_rows_carry_the_basis_they_were_priced_under() -> None:
    rows = [quote(), quote(side="under", american=105.0, opposite=-115.0)]
    projections = {("puka nacua", RECEPTIONS): projection()}
    priced = props.price_props(rows, projections, basis=context.BASIS)
    assert priced
    assert {p.basis for p in priced} == {context.BASIS}
    assert all(props.RESEARCH_ONLY in p.screens for p in priced)
    default = props.price_props(rows, projections)
    assert {p.basis for p in default} == {props.BASIS}
    assert context.BASIS != props.BASIS


def _research(tmp_path: Path, basis: str) -> list[props.PricedProp]:
    rows = [
        quote(book="draftkings", american=-115.0, opposite=105.0),
        quote(book="draftkings", side="under", american=105.0, opposite=-115.0),
        quote(book="fanduel", american=-110.0, opposite=-110.0),
        quote(book="fanduel", side="under", american=-110.0, opposite=-110.0),
    ]
    priced = props.price_props(rows, {("puka nacua", RECEPTIONS): projection()}, basis=basis)
    assert props.write_research(priced, season=2026, week=2, root=tmp_path) is not None
    return priced


def test_grade_week_settles_against_the_box_score_and_keeps_the_basis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _research(tmp_path, props.BASIS)
    _research(tmp_path, context.BASIS)
    box = pd.DataFrame(
        [
            {
                "player_display_name": "Puka Nacua",
                "team": "LA",
                "season": 2026,
                "week": 2,
                "season_type": "REG",
                "receptions": 7.0,
            }
        ]
    )
    monkeypatch.setattr(nflverse, "player_week", lambda season: box)
    graded = props_grade.grade_week(2026, 2, root=tmp_path)
    assert graded
    assert {g.basis for g in graded} == {props.BASIS, context.BASIS}
    overs = [g for g in graded if g.side == "over"]
    unders = [g for g in graded if g.side == "under"]
    assert all(g.result == props_grade.WIN for g in overs)
    assert all(g.result == props_grade.LOSS for g in unders)
    assert all(g.pnl == -1.0 for g in unders)
    assert props_grade.graded_path(2026, 2, root=tmp_path).exists()
    assert props_grade.pending_weeks(2026, root=tmp_path) == []
    back = props_grade.read_graded(2026, root=tmp_path)
    assert len(back) == len(graded)
    lines = props_grade.summary(back)
    assert any(props.BASIS in line for line in lines)
    assert any(context.BASIS in line for line in lines)


def test_grade_week_voids_a_player_with_no_box_score_and_pushes_on_the_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _research(tmp_path, props.BASIS)
    monkeypatch.setattr(
        nflverse,
        "player_week",
        lambda season: pd.DataFrame(
            [
                {
                    "player_display_name": "Somebody Else",
                    "team": "LA",
                    "season": 2026,
                    "week": 2,
                    "season_type": "REG",
                    "receptions": 4.0,
                }
            ]
        ),
    )
    graded = props_grade.grade_week(2026, 2, root=tmp_path, write=False)
    assert graded and all(g.result == props_grade.VOID for g in graded)
    assert all(g.reason == props_grade.NO_BOX_SCORE for g in graded)
    assert all(g.pnl == 0.0 for g in graded)

    monkeypatch.setattr(
        nflverse,
        "player_week",
        lambda season: pd.DataFrame(
            [
                {
                    "player_display_name": "Puka Nacua",
                    "team": "LA",
                    "season": 2026,
                    "week": 2,
                    "season_type": "REG",
                    "receptions": 4.5,
                }
            ]
        ),
    )
    graded = props_grade.grade_week(2026, 2, root=tmp_path, write=False)
    assert all(g.result == props_grade.PUSH for g in graded)


def test_grade_week_waits_for_the_box_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _research(tmp_path, props.BASIS)
    monkeypatch.setattr(nflverse, "player_week", lambda season: pd.DataFrame())
    assert props_grade.grade_week(2026, 2, root=tmp_path) == []
    assert not props_grade.graded_path(2026, 2, root=tmp_path).exists()
    assert props_grade.pending_weeks(2026, root=tmp_path) == [2]


def test_read_research_collapses_a_rerun_of_the_same_snapshot(tmp_path: Path) -> None:
    first = _research(tmp_path, props.BASIS)
    _research(tmp_path, props.BASIS)
    rows = props_grade.read_research(props.research_path(2026, 2, root=tmp_path))
    assert len(rows) == len(first)


def test_projection_type_is_shared_between_bases() -> None:
    assert isinstance(projection(), Projection)
