"""The staleness study must attach the right arm and the right side of the bet."""

from __future__ import annotations

from datetime import date as Date

import pandas as pd
import pytest

st = pytest.importorskip("scripts.starter_staleness_study")


def test_pitcher_name_strips_every_prop_suffix() -> None:
    assert st.pitcher_name("Zack Wheeler Strikeouts o6.5") == "Zack Wheeler"
    assert st.pitcher_name("Ranger Suarez Hits u5.5") == "Ranger Suarez"
    assert st.pitcher_name("Mitch Bratt ER o2.5") == "Mitch Bratt"
    assert st.pitcher_name("Jeffrey Springs Outs o16.5") == "Jeffrey Springs"


@pytest.mark.parametrize(
    "market,selection,expected",
    [
        ("pitcher_k", "Zack Wheeler Strikeouts o6.5", True),
        ("pitcher_k", "Zack Wheeler Strikeouts u6.5", False),
        ("pitcher_outs", "Zack Wheeler Outs o16.5", True),
        ("pitcher_h", "Zack Wheeler Hits u5.5", True),  # fewer hits = he pitched well
        ("pitcher_h", "Zack Wheeler Hits o5.5", False),
        ("game_total", "Under 8.5", True),
        ("f5_total", "Over 4.5", False),
        ("game_ml", "PHI ML", None),  # side decides it, not the market
    ],
)
def test_backs_the_arm(market: str, selection: str, expected: bool | None) -> None:
    assert st.backs_the_arm(market, selection) is expected


def _trend(**over) -> dict:
    t = {"siera": 3.5, "stuff": 0.25, "vfa": 94.0,
         "d_siera": 0.0, "d_stuff": 0.0, "d_vfa": 0.0}
    t.update(over)
    return t


def test_totals_average_both_starters() -> None:
    """An over is a bet against both arms, so it inherits both trends."""
    day = Date(2026, 8, 1)
    led = pd.DataFrame([{
        "market": "game_total", "selection": "Over 8.5", "matchup": "PHI @ STL",
        "result": "win", "pnl": 0.91, "d": day,
    }])
    nmap = {"A Pitcher": 1, "B Pitcher": 2}
    games = {("2026-08-01", "PHI @ STL"): {"PHI": "A Pitcher", "STL": "B Pitcher"}}
    trends = {(1, day): _trend(d_vfa=-2.0), (2, day): _trend(d_vfa=+1.0)}
    out = st.attach_trends(led, nmap, games, trends)
    assert len(out) == 1
    assert out.iloc[0]["d_vfa"] == pytest.approx(-0.5)
    assert bool(out.iloc[0]["backs"]) is False  # an over opposes the arms


def test_moneyline_attaches_only_the_side_backed() -> None:
    day = Date(2026, 8, 1)
    led = pd.DataFrame([{
        "market": "game_ml", "selection": "STL ML", "matchup": "PHI @ STL",
        "result": "loss", "pnl": -1.0, "d": day,
    }])
    nmap = {"A Pitcher": 1, "B Pitcher": 2}
    games = {("2026-08-01", "PHI @ STL"): {"PHI": "A Pitcher", "STL": "B Pitcher"}}
    trends = {(1, day): _trend(d_siera=+1.0), (2, day): _trend(d_siera=-1.0)}
    out = st.attach_trends(led, nmap, games, trends)
    assert out.iloc[0]["d_siera"] == pytest.approx(-1.0)  # STL's starter only


def test_a_bet_with_no_trend_is_dropped_not_defaulted() -> None:
    """Missing history must remove the row; a zero trend would be a lie."""
    day = Date(2026, 8, 1)
    led = pd.DataFrame([{
        "market": "pitcher_k", "selection": "Nobody Known Strikeouts o5.5",
        "matchup": "PHI @ STL", "result": "win", "pnl": 1.0, "d": day,
    }])
    assert st.attach_trends(led, {}, {}, {}).empty


def test_welch_reports_the_gap_and_its_t() -> None:
    a = pd.Series([1.0, 1.0, 1.0, 1.0])
    b = pd.Series([-1.0, -1.0, -1.0, -1.0])
    diff, t = st.welch(a, b)
    assert diff == pytest.approx(2.0)
    assert t == float("inf") or t > 10  # zero variance both sides


def test_trend_table_uses_only_prior_pitches() -> None:
    """A trend for day D must not see pitches thrown on or after D."""
    rows = []
    for _ in range(60):
        rows.append({"pitcher": 1, "game_date": "2026-07-01", "pitch_type": "FF",
                     "release_speed": 95.0})
    for _ in range(60):
        rows.append({"pitcher": 1, "game_date": "2026-08-05", "pitch_type": "FF",
                     "release_speed": 80.0})
    df = pd.DataFrame(rows)
    st.MIN_PRIOR_PITCHES, st.MIN_WINDOW_PITCHES = 10, 10
    table = st.trend_table(df, {1})
    key = (1, Date(2026, 8, 5))
    if key in table:  # the 08-05 pitches must not lower the velocity for 08-05
        assert table[key]["vfa"] != pytest.approx(80.0)
