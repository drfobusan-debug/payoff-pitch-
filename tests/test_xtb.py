"""Expected total bases per ball, and the xSLG built from it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_engine.features.regression import build_batter_regression
from mlb_engine.features.xtb import MIN_LEAGUE_BBE, LeagueXTB

rng = np.random.default_rng(3)


def _league(n: int = 8_000) -> pd.DataFrame:
    """A league where barrels leave the park and everything else is an out.

    Balls at 100 mph between 20 and 30 degrees are home runs; the rest are outs.
    Any lookup fitted on this should price the first cell near 4.0 and the rest
    near 0.
    """
    ev = rng.choice([80.0, 100.0], size=n)
    la = rng.choice([5.0, 25.0, 50.0], size=n)
    barrel = (ev > 95) & (la > 20) & (la < 30)
    return pd.DataFrame(
        {
            "launch_speed": ev,
            "launch_angle": la,
            "events": np.where(barrel, "home_run", "field_out"),
            "type": "X",
            "description": "hit_into_play",
        }
    )


def test_the_lookup_learns_where_the_bases_are() -> None:
    xtb = LeagueXTB.from_statcast(_league())
    assert xtb is not None
    hot = pd.DataFrame({"launch_speed": [100.0], "launch_angle": [25.0]})
    cold = pd.DataFrame({"launch_speed": [80.0], "launch_angle": [50.0]})
    assert xtb.expected(hot).iloc[0] > 3.5
    assert xtb.expected(cold).iloc[0] < 0.2


def test_thin_cells_are_shrunk_toward_the_league_ball() -> None:
    """One 110 mph ball that fell in is not worth a single-base expectation."""
    df = _league()
    freak = pd.DataFrame(
        {
            "launch_speed": [112.0],
            "launch_angle": [-40.0],
            "events": ["triple"],
            "type": ["X"],
            "description": ["hit_into_play"],
        }
    )
    xtb = LeagueXTB.from_statcast(pd.concat([df, freak], ignore_index=True))
    assert xtb is not None
    # 3 bases on one ball, pulled most of the way back to the league ball (0.67).
    assert xtb.expected(freak).iloc[0] == pytest.approx(0.76, abs=0.10)


def test_a_league_too_small_to_fit_returns_nothing() -> None:
    assert LeagueXTB.from_statcast(_league(MIN_LEAGUE_BBE - 1)) is None
    assert LeagueXTB.from_statcast(pd.DataFrame({"launch_speed": [95.0]})) is None


def test_expected_slugging_divides_by_at_bats_the_way_slugging_does() -> None:
    xtb = LeagueXTB.from_statcast(_league())
    assert xtb is not None
    balls = pd.DataFrame({"launch_speed": [100.0] * 10, "launch_angle": [25.0] * 10})
    # Ten home-run balls in 40 at-bats: 40 expected bases over 40 at-bats.
    assert xtb.xslg(balls, 40) == pytest.approx(1.0, abs=0.05)
    # The same contact in half the at-bats is twice the slugging.
    assert xtb.xslg(balls, 20) == pytest.approx(2.0, abs=0.10)
    assert np.isnan(xtb.xslg(balls, 0))


def _batter(events: list[str], ev: float, la: float) -> pd.DataFrame:
    """A batter's slice; strikeouts are plate appearances with no ball in play."""
    n = len(events)
    contact = [e != "strikeout" for e in events]
    return pd.DataFrame(
        {
            "events": events,
            "description": ["hit_into_play" if c else "swinging_strike" for c in contact],
            "type": ["X" if c else "S" for c in contact],
            "launch_speed": [ev if c else None for c in contact],
            "launch_angle": [la if c else None for c in contact],
            "launch_speed_angle": [(6 if la > 20 else 3) if c else None for c in contact],
            "bb_type": [("fly_ball" if la > 20 else "ground_ball") if c else None for c in contact],
            "bat_speed": [72.0] * n,
            "zone": [5] * n,
            "woba_value": [0.9] * n,
            "estimated_ba_using_speedangle": [0.4] * n,
            "estimated_woba_using_speedangle": [0.9] * n,
            "inning_topbot": ["Bot"] * n,
        }
    )


def test_the_regression_reads_expected_slugging_off_the_league_lookup() -> None:
    """Strikeouts belong in xSLG's denominator; contact quality alone ignores them."""
    xtb = LeagueXTB.from_statcast(_league())
    assert xtb is not None
    bdf = _batter(["home_run"] * 20 + ["strikeout"] * 20, 100.0, 25.0)
    reg = build_batter_regression(bdf, league_xtb=xtb)
    # 20 home-run balls, 40 at-bats: ~2.0 expected slugging, matching his actual.
    assert reg.xslg == pytest.approx(2.0, abs=0.15)
    assert reg.slg == pytest.approx(2.0)
    # Expected slugging *on contact* has no strikeout in it, so it reads higher
    # than the level -- which is why the false-positive gap is measured off it.
    assert reg.contact_slg > reg.xslg / 2
    assert reg.slg_gap == pytest.approx(reg.slg - reg.contact_slg)


def test_without_a_league_lookup_the_level_falls_back_to_contact_quality() -> None:
    bdf = _batter(["single"] * 20, 95.0, 12.0)
    reg = build_batter_regression(bdf)
    assert reg.xslg == pytest.approx(reg.contact_slg)
