"""The regression article must keep luck and level as separate claims."""

from __future__ import annotations

from datetime import date as Date

import pytest

art = pytest.importorskip("scripts.regression_article")
swing = pytest.importorskip("mlb_engine.features.swing")


def _pitcher(**over) -> dict:
    p = {
        "name": "Zack Wheeler",
        "siera": 3.10,
        "xk": 0.30,
        "vfa": 95.2,
        "babip": 0.340,
        "woba": 0.360,
        "xwoba": 0.300,
        "dxwoba": -0.060,
        "unlucky_babip": 0.050,
        "d_siera": 0.00,
        "d_xk": 0.00,
        "d_vfa": 0.00,
    }
    p.update(over)
    return p


def _batter(**over) -> dict:
    b = {
        "name": "Gunnar Henderson",
        "woba": 0.252,
        "xwoba": 0.359,
        "dxwoba": 0.107,
        "xslg": 0.484,
        "barrel": 0.09,
        "woba6": 0.252,
        "woba3": 0.250,
    }
    b.update(over)
    return b


def test_declining_arm_is_not_sold_as_a_clean_buy_low() -> None:
    """High BABIP plus a worsening arm must warn, not celebrate."""
    text = art._pitcher_verdict(_pitcher(d_siera=0.60, d_vfa=-1.2), positive=True)
    assert "Read this one carefully" in text
    assert "may not be the one he" in text
    assert "clean version" not in text


def test_stable_arm_with_bad_luck_is_the_clean_case() -> None:
    text = art._pitcher_verdict(_pitcher(d_siera=-0.40, d_vfa=0.3), positive=True)
    assert "clean version" in text


def test_luck_sentence_names_both_luck_terms() -> None:
    text = art._luck_sentence(_pitcher(), positive=True)
    assert ".340" in text and ".290" in text  # BABIP against the norm
    assert "60 points worse" in text  # the wOBA - xwOBA gap


def test_negative_regression_reads_as_borrowed_results() -> None:
    p = _pitcher(babip=0.250, unlucky_babip=-0.040, woba=0.270,
                 xwoba=0.330, dxwoba=0.060)
    text = art._luck_sentence(p, positive=False)
    assert "finding gloves" in text
    assert "flatters" in text


def test_batter_entry_states_the_gap_and_the_power() -> None:
    html = art._batter_entry(_batter(), {"matchup": "BAL @ MIN"}, None, True)
    assert "107-point shortfall" in html
    assert "real power" in html
    assert "passes his props" in html


def test_bet_sentence_lists_only_buys() -> None:
    bets = [
        {"selection": "Wheeler Ks o6.5", "tier": "Strong buy",
         "model_prob": 0.61, "market_american": -115},
        {"selection": "Wheeler Hits u5.5", "tier": "Pass",
         "model_prob": 0.40, "market_american": 100},
    ]
    text = art._bet_sentence(bets, "Wheeler&rsquo;s")
    assert "Ks o6.5" in text
    assert "Hits u5.5" not in text


def test_article_flags_that_the_trend_arrows_are_unproven() -> None:
    html = art.build_html(
        Date(2026, 8, 12), [_pitcher()], [], {}, [_batter()], [], {}, []
    )
    assert "three-week direction</i> of those same" in html
    assert "does not predict the next start" in html
    assert "Part one" in html and "Part two" in html


def _swinging(power: float, **over) -> dict:
    """A hitter dict carrying a readable swing ``power`` SD from league."""
    bmu, bsd = swing.LEAGUE["bat_speed"]
    zmu, zsd = swing.LEAGUE["blast"]
    prof = swing.SwingProfile(
        swings=500, bat_speed=bmu + power * bsd, blast=zmu + power * zsd,
        squared_up=swing.LEAGUE["squared_up"][0], fast=swing.LEAGUE["fast"][0],
        swing_length=swing.LEAGUE["swing_length"][0],
    )
    b = _batter(**over)
    b.update(
        swings=prof.swings, bat_speed=prof.bat_speed, fast=prof.fast,
        squared_up=prof.squared_up, blast=prof.blast, swing_length=prof.swing_length,
        power_z=prof.power_z, contact_z=prof.contact_z,
        stage2=swing.stage_two(-b["dxwoba"], prof),
    )
    return b


def test_the_swing_confirms_a_hitter_the_gap_says_is_due() -> None:
    """Stage two on a positive regressor: the swing agrees, so the rebound is real."""
    html = art._batter_entry(_swinging(1.0), None, None, True)
    assert "swing underneath agrees" in html
    assert "blast rate" in html and "+1.00 standard deviations" in html
    assert "Swing: BatSpd" in html


def test_the_swing_argues_against_a_fade_and_the_entry_says_so() -> None:
    """The false-negative case: lucky and good at once, which the gap alone misses."""
    b = _swinging(1.0, dxwoba=-0.080, woba=0.400, xwoba=0.320)
    assert b["stage2"] == swing.CONTRADICTED
    html = art._batter_entry(b, None, None, False)
    assert "argues against the fade" in html
    assert "out-produce" in html
    assert "swing disagrees" in html  # flagged in the headline, not buried


def test_a_hitter_with_no_tracked_swings_is_unmeasured_not_average() -> None:
    html = art._batter_entry(_batter(), None, None, True)
    assert "not readable at this sample" in html
    assert "Swing: BatSpd" not in html  # nothing to print


def test_the_swing_line_states_its_windows_and_that_attack_angle_is_absent() -> None:
    html = art._swing_line(_swinging(0.5))
    windows = "/".join(
        str(swing.WINDOW[m])
        for m in ("bat_speed", "fast", "squared_up", "blast", "swing_length")
    )
    assert f"{windows} tracked swings" in html
    assert "off 500 in the window" in html  # the sample the levels came out of
    assert "Attack angle is published by no available feed" in html


def test_the_article_prices_no_swing_trend_and_says_why() -> None:
    html = art.build_html(
        Date(2026, 8, 12), [_pitcher()], [], {}, [_swinging(1.0)], [], {}, []
    )
    assert "no swing trend is printed at all" in html
    assert "negatively signed on home runs" in html  # squared-up kept off power
    assert "absent rather than estimated" in html  # attack angle, in the methodology


def test_a_fly_ball_arm_is_told_where_the_correction_lands() -> None:
    """Shape is prose about *what* the correction is, never about whether."""
    text = art._air_sentence(_pitcher(fb=0.44, gb=0.34), positive=True)
    assert "fly-ball arm" in text and "44%" in text
    assert "over the fence" in text
    text = art._air_sentence(_pitcher(fb=0.44, gb=0.34), positive=False)
    assert "ends fastest" in text


def test_a_ground_ball_arm_is_not_sold_home_run_regression() -> None:
    text = art._air_sentence(_pitcher(fb=0.27, gb=0.52), positive=True)
    assert "keeps the ball down" in text and "52%" in text
    assert "singles and double plays" in text
    assert "fence" not in text


def test_shape_is_silent_when_the_batted_ball_data_is_missing() -> None:
    assert art._air_sentence(_pitcher(fb=float("nan")), positive=True) == ""
    assert art._bat_air_sentence(_batter(fb=float("nan")), positive=True) == ""


def test_a_hitters_air_contact_is_discounted_by_his_pop_ups() -> None:
    text = art._bat_air_sentence(_batter(fb=0.45, gb=0.32, iffb=0.30), positive=True)
    assert "45%" in text and "30%" in text
    assert "never going to pay" in text


def test_a_ground_ball_hitter_reads_as_singles_not_power() -> None:
    text = art._bat_air_sentence(_batter(fb=0.26, gb=0.52), positive=True)
    assert "on the ground" in text and "52%" in text
    assert "rather than as home runs" in text


def test_the_pitcher_entry_carries_both_velocity_and_shape() -> None:
    html = art._pitcher_entry(_pitcher(fb=0.44, gb=0.34), None, [], True)
    assert "fastball at 95.2" in html  # vFA level
    assert "vFA +0.0 mph" in html  # vFA three-week trend
    assert "fly-ball arm" in html


def test_the_methodology_defines_fly_ball_rate() -> None:
    html = art.build_html(
        Date(2026, 8, 12), [_pitcher(fb=0.44)], [], {}, [_batter(fb=0.44)], [], {}, []
    )
    assert "Fly-ball rate counts fly balls and pop-ups" in html
