"""Total bases: measured predictors, and the false positives that give bases back.

Every threshold and weight asserted here comes from 2,609 batter-weeks of
trailing-42-day profile against the next seven days of real total bases; see the
weight block in ``mlb_engine.models.selectors`` for the numbers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlb_engine.features.regression import (
    BL_HARD_HIT,
    BL_SPRINT,
    BatterRegression,
    build_batter_regression,
)
from mlb_engine.models.selectors import TBSelector

BASE = dict(
    bbe=50,
    barrel_rate=0.08,
    hard_hit=0.40,
    sweet_spot=0.33,
    bat_speed=71.5,
    max_ev=108.0,
    whiff=0.24,
    zone_contact=0.82,
    xba=0.250,
    xslg=0.400,
    babip=0.290,
    woba=0.320,
    xwoba=0.320,
    sprint_speed=BL_SPRINT,
    pa=200,
)


def reg(**kw: float) -> BatterRegression:
    return BatterRegression(**{**BASE, **kw})  # type: ignore[arg-type]


def tb(**kw: float) -> float:
    return TBSelector().select(reg(**kw)).factor


# --- barrels per plate appearance ------------------------------------------


def test_barrels_are_scored_per_plate_appearance_not_per_batted_ball() -> None:
    """Two bats barrel 8% of batted balls; one puts twice as many balls in play.

    Per batted ball they are identical, which credits the hitter who barrels
    rarely because he rarely makes contact at all.
    """
    contact = tb(bbe=60, pa=200)  # 2.4% of PA end in a barrel
    whiffy = tb(bbe=30, pa=200)  # 1.2%
    assert contact > whiffy


def test_barrel_per_pa_falls_back_to_barrel_rate_when_pas_are_unknown() -> None:
    """A slice with no PA count still gets a barrel term, on the same scale."""
    known = tb(barrel_rate=0.12, bbe=50, pa=100)
    unknown = tb(barrel_rate=0.12, bbe=50, pa=0)
    assert unknown > tb(barrel_rate=0.04, bbe=50, pa=0)
    assert unknown == pytest.approx(known, abs=0.05)


# --- false positives -------------------------------------------------------


def test_slugging_past_expected_slugging_brakes_the_bat() -> None:
    """Total bases is slugging's numerator, so the gap is the market's own luck term."""
    honest = tb(slg=0.400, xslg=0.400)
    lucky = tb(slg=0.470, xslg=0.400)  # +.070, past the +.050 flag
    assert lucky < honest
    assert lucky / honest == pytest.approx(1.0 - 0.09, abs=0.02)  # a 3-point brake


def test_the_gap_brake_is_a_step_and_does_not_keep_deepening() -> None:
    """Bats over +.100 measured no worse than bats over +.050 (-4.3%, p=.53)."""
    assert tb(slg=0.560, xslg=0.400) == pytest.approx(tb(slg=0.460, xslg=0.400))


def test_a_gap_the_other_way_is_not_a_brake() -> None:
    """Slugging *under* expected is the unlucky bat the dxwOBA term already lifts."""
    assert tb(slg=0.330, xslg=0.400) == pytest.approx(tb(slg=0.400, xslg=0.400))


def test_an_unknown_slugging_never_brakes_anything() -> None:
    assert tb(slg=float("nan")) == pytest.approx(tb(slg=0.400))


def test_high_babip_on_soft_contact_brakes_harder_than_high_babip_alone() -> None:
    """A .380 BABIP is a bloop streak only when the contact is below average."""
    hard = tb(babip=0.380, hard_hit=BL_HARD_HIT + 0.05)
    soft = tb(babip=0.380, hard_hit=BL_HARD_HIT - 0.05)
    assert hard == pytest.approx(tb(babip=0.290, hard_hit=BL_HARD_HIT + 0.05))
    assert soft < hard


def test_the_babip_brake_steps_with_how_elevated_it_is() -> None:
    soft = dict(hard_hit=BL_HARD_HIT - 0.05)
    assert tb(babip=0.320, **soft) > tb(babip=0.340, **soft) > tb(babip=0.380, **soft)


def test_the_two_brakes_compound_on_a_bat_that_trips_both() -> None:
    both = tb(slg=0.480, xslg=0.400, babip=0.370, hard_hit=0.35)
    one = tb(slg=0.480, xslg=0.400)
    assert both < one


# --- speed on the extra-base share -----------------------------------------


def test_speed_lifts_doubles_and_triples_and_leaves_home_runs_alone() -> None:
    slow = reg(sprint_speed=25.0).multipliers()
    fast = reg(sprint_speed=29.0).multipliers()
    assert fast["2B"] > slow["2B"]
    assert fast["3B"] > slow["3B"]
    assert fast["HR"] == pytest.approx(slow["HR"])


def test_a_slow_bat_whose_doubles_have_spiked_gets_an_extra_brake() -> None:
    """Bottom-third speed with a top-quarter extra-base rate is the third false positive."""
    surging = reg(sprint_speed=25.5, xbh_per_pa=0.080).multipliers()
    quiet = reg(sprint_speed=25.5, xbh_per_pa=0.030).multipliers()
    assert surging["2B"] < quiet["2B"]
    # A fast bat with the same surge keeps its doubles: it can run them out.
    fast = reg(sprint_speed=29.0, xbh_per_pa=0.080).multipliers()
    assert fast["2B"] > surging["2B"]


def test_the_conjunction_needs_both_halves() -> None:
    fast_surge = reg(sprint_speed=29.0, xbh_per_pa=0.080).multipliers()
    fast_quiet = reg(sprint_speed=29.0, xbh_per_pa=0.030).multipliers()
    assert fast_surge["2B"] == pytest.approx(fast_quiet["2B"])


# --- what is measured, and what is only reported ---------------------------


def test_line_drive_rate_is_reported_but_never_scored() -> None:
    """r=+0.006 against forward TB/PA, and the sign flips between halves."""
    high = TBSelector().select(reg(ld_pct=0.32))
    low = TBSelector().select(reg(ld_pct=0.18))
    assert high.factor == pytest.approx(low.factor)
    assert "ld=0.320" in high.profile


def test_max_ev_outweighs_xslg_now_that_it_is_the_measured_separator() -> None:
    """xSLG's weight used to be 20x max EV's; the backtest has it the other way."""
    ev_edge = tb(max_ev=114.0)
    xslg_edge = tb(xslg=0.560)  # +.160, about 1.7 SD, vs max EV's +6 mph (1.9 SD)
    assert ev_edge > xslg_edge


def test_the_home_road_split_is_reported_but_never_scored() -> None:
    """The sim already prices tonight's venue from the matching half of the splits."""
    biased = TBSelector().select(reg(tb_home_bias=0.42))
    assert biased.factor == pytest.approx(TBSelector().select(reg()).factor)
    assert "home_tb_bias=+42.0%" in biased.profile


def test_thin_samples_score_nothing_at_all() -> None:
    sel = TBSelector().select(reg(bbe=5, slg=0.700, xslg=0.400))
    assert sel.factor == 1.0
    assert sel.signal == "none"


# --- the fields behind all of the above ------------------------------------


def _slice(home_events: list[str], away_events: list[str]) -> pd.DataFrame:
    events = home_events + away_events
    n = len(events)
    return pd.DataFrame(
        {
            "events": events,
            "inning_topbot": ["Bot"] * len(home_events) + ["Top"] * len(away_events),
            "description": ["hit_into_play"] * n,
            "type": ["X"] * n,
            "launch_speed": [95.0] * n,
            "launch_angle": [15.0] * n,
            "launch_speed_angle": [5] * n,
            "bb_type": ["line_drive"] * n,
            "bat_speed": [72.0] * n,
            "zone": [5] * n,
            "woba_value": [0.9] * n,
            "estimated_ba_using_speedangle": [0.4] * n,
            "estimated_woba_using_speedangle": [0.9] * n,
        }
    )


def test_the_venue_split_reads_off_which_half_of_the_inning_he_hits_in() -> None:
    """A hitter bats in the bottom half at home, the top half on the road."""
    home = ["double"] * 20 + ["strikeout"] * 20
    away = ["single"] * 20 + ["strikeout"] * 20
    r = build_batter_regression(_slice(home, away))
    assert r.tb_home_bias == pytest.approx(1.0)  # twice the bases at home


def test_the_venue_split_stays_unknown_until_both_halves_are_real() -> None:
    r = build_batter_regression(_slice(["double"] * 20, ["single"] * 40))
    assert r.tb_home_bias != r.tb_home_bias  # NaN


def test_slugging_and_the_extra_base_rate_come_off_the_events() -> None:
    """SLG counts bases over at-bats, so walks stay out of the denominator."""
    events = ["single", "double", "triple", "home_run", "strikeout", "walk"]
    df = pd.DataFrame(
        {
            "events": events,
            "description": ["hit_into_play"] * 4 + ["swinging_strike", "ball"],
            "type": ["X"] * 4 + ["S", "B"],
            "launch_speed": [100.0] * 4 + [None, None],
            "launch_angle": [15.0] * 4 + [None, None],
            "launch_speed_angle": [6] * 4 + [None, None],
            "bb_type": ["line_drive"] * 4 + [None, None],
            "bat_speed": [72.0] * 6,
            "zone": [5] * 6,
            "woba_value": [0.9, 1.2, 1.6, 2.0, 0.0, 0.7],
            "estimated_ba_using_speedangle": [0.5] * 4 + [None, None],
            "estimated_woba_using_speedangle": [1.0] * 4 + [None, None],
        }
    )
    r = build_batter_regression(df)
    assert r.slg == pytest.approx(10 / 5)  # 1+2+3+4 bases over 5 at-bats
    assert r.xbh_per_pa == pytest.approx(2 / 6)
    assert r.ld_pct == pytest.approx(1.0)
    assert r.slg_gap == pytest.approx(r.slg - r.xslg)
