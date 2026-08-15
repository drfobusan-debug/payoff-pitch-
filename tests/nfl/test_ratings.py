"""Opponent adjustment, shrinkage, recency, and the panel that feeds them."""

from __future__ import annotations

import pandas as pd

from nfl_engine.features import panel as panel_mod
from nfl_engine.features import ratings as ratings_mod

# A toy league where offence and defence are deliberately *not* the same teams:
# ATT can move the ball and cannot stop it, WALL is the reverse, MID is average.
# A rating that reads a raw average will call ATT and WALL both mediocre.
OFF_TALENT = {"ATT": 0.30, "WALL": 0.0, "MID": 0.0}
DEF_TALENT = {"ATT": 0.30, "WALL": -0.30, "MID": 0.0}  # positive allows more


def _schedule(weeks: int = 10, home_advantage: float = 0.0) -> pd.DataFrame:
    rows = []
    game = 0
    for week in range(1, weeks + 1):
        for first, second in (("ATT", "MID"), ("WALL", "MID"), ("ATT", "WALL")):
            # Alternate the host each week, or the home term is unidentifiable
            # from the teams themselves -- which is also true of a real schedule.
            home, away = (first, second) if week % 2 else (second, first)
            game += 1
            for off, dfn, is_home in ((home, away, True), (away, home, False)):
                epa = OFF_TALENT[off] + DEF_TALENT[dfn]
                rows.append(
                    {
                        "season": 2024,
                        "week": week,
                        "game_id": f"g{game}",
                        "posteam": off,
                        "defteam": dfn,
                        "is_home": is_home,
                        "epa": epa + (home_advantage if is_home else 0.0),
                        "success": 0.45 + 0.1 * epa,
                        "drives": 11.0,
                    }
                )
    return pd.DataFrame(rows)


def test_ratings_separate_offence_from_defence():
    """The raw average cannot tell ATT from WALL; the adjustment can."""
    frame = _schedule()
    book = ratings_mod.fit(frame, ridge=1.0)
    raw = frame.groupby("posteam").epa.mean()
    assert abs(raw["ATT"] - raw["WALL"]) < 0.2  # raw: near-identical teams
    assert book.rating("ATT").off_epa > book.rating("WALL").off_epa + 0.15
    assert book.rating("WALL").def_epa < book.rating("ATT").def_epa - 0.15


def test_a_good_defence_rates_negative():
    book = ratings_mod.fit(_schedule(), ridge=1.0)
    assert book.rating("WALL").def_epa < 0.0 < book.rating("ATT").def_epa


def test_ridge_shrinks_toward_the_league_mean():
    frame = _schedule()
    loose = ratings_mod.fit(frame, ridge=1.0)
    tight = ratings_mod.fit(frame, ridge=5000.0)
    assert abs(tight.rating("ATT").off_epa) < abs(loose.rating("ATT").off_epa)
    assert abs(tight.rating("ATT").off_epa) < 0.01


def test_recency_weighting_forgets_old_form():
    """Good in week 1, average since: a short memory should rate it average."""
    rows = []
    for week in range(1, 13):
        for off, dfn, is_home in (("HOT", "OPP", True), ("OPP", "HOT", False)):
            hot = 0.40 if week == 1 else 0.0
            rows.append(
                {
                    "season": 2024,
                    "week": week,
                    "game_id": f"w{week}",
                    "posteam": off,
                    "defteam": dfn,
                    "is_home": is_home,
                    "epa": hot if off == "HOT" else 0.0,
                    "success": 0.45,
                    "drives": 11.0,
                }
            )
    frame = pd.DataFrame(rows)
    short = ratings_mod.fit(frame, half_life=2.0, ridge=1.0).rating("HOT").off_epa
    long = ratings_mod.fit(frame, half_life=200.0, ridge=1.0).rating("HOT").off_epa
    assert short < long


def test_a_rating_is_as_of_a_week_and_uses_no_later_game():
    frame = ratings_mod.week_index(_schedule())
    cutoff = 4.0
    book = ratings_mod.fit(frame[frame.week_index < cutoff], asof=cutoff, ridge=1.0)
    later = ratings_mod.fit(frame, ridge=1.0)
    assert book.games_used < later.games_used


def test_unknown_team_rates_exactly_average():
    book = ratings_mod.fit(_schedule(), ridge=1.0)
    unknown = book.rating("NOBODY")
    assert unknown.off_epa == 0.0
    assert unknown.def_epa == 0.0
    assert unknown.net_success() == 0.0


def test_a_thin_book_is_not_usable():
    book = ratings_mod.fit(_schedule(), ridge=1.0)
    assert not book.is_usable()  # 30 games is not 400
    assert ratings_mod.fit(pd.DataFrame()).games_used == 0


def test_home_edge_is_fitted_not_assumed():
    book = ratings_mod.fit(_schedule(home_advantage=0.10), ridge=1.0)
    assert 0.05 < book.home_edge["epa"] < 0.15
    flat = ratings_mod.fit(_schedule(), ridge=1.0)
    assert abs(flat.home_edge["epa"]) < 0.02


def _plays() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024] * 6,
            "week": [1] * 6,
            "game_id": ["g1"] * 6,
            "play_id": [1, 2, 3, 4, 5, 6],
            "posteam": ["A", "A", "A", "B", "B", "B"],
            "defteam": ["B", "B", "B", "A", "A", "A"],
            "play_type": ["pass", "run", "pass", "run", "pass", "kickoff"],
            "epa": [0.5, -0.2, 0.9, 0.1, 0.3, 0.0],
            "success": [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            "pass_oe": [5.0, -3.0, 4.0, -1.0, 2.0, 0.0],
            "fixed_drive": [1, 1, 1, 2, 2, 3],
            "qtr": [1, 1, 1, 2, 2, 2],
            "score_differential": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "game_seconds_remaining": [3600.0, 3570.0, 3540.0, 3000.0, 2970.0, 2900.0],
        }
    )


def test_panel_collapses_to_one_row_per_offence():
    out = panel_mod.collapse(_plays())
    assert len(out) == 2
    row = out[out.posteam == "A"].iloc[0]
    assert row.plays == 3
    assert abs(row.epa - (0.5 - 0.2 + 0.9) / 3) < 1e-9
    assert abs(row.success - 2 / 3) < 1e-9


def test_panel_ignores_kickoffs_and_keeps_pace_neutral():
    out = panel_mod.collapse(_plays())
    # The kickoff is not a play the rating is fitted on.
    assert out.plays.sum() == 5
    assert out.sec_per_play.notna().any()


def test_panel_survives_missing_columns():
    assert panel_mod.collapse(pd.DataFrame({"epa": [0.1]})).empty
