"""Expected home runs: fence geometry, per-ball scoring, and the HR/PA blend."""

from __future__ import annotations

import pandas as pd

from mlb_engine.data.fences import (
    ANCHOR_ANGLES,
    FENCES,
    LEAGUE_FENCE,
    TEAM_VENUE,
    fence_for_team,
    get_fence,
    wall_distance,
)
from mlb_engine.data.parks import PARKS
from mlb_engine.features.rolling import LEAGUE_RATES, OutcomeRates, blend_hr_rate
from mlb_engine.features.xhr import (
    HOME_X,
    HOME_Y,
    batter_xhr,
    hr_probability,
    park_hr_multiplier,
    park_shape_baseline,
    spray_angle,
)

# --- fence geometry ----------------------------------------------------------


def test_every_park_has_fence_geometry() -> None:
    assert set(FENCES) == set(PARKS)


def test_unknown_venue_falls_back_to_a_league_average_park() -> None:
    assert get_fence(None) is LEAGUE_FENCE
    assert get_fence(999999) is LEAGUE_FENCE
    assert fence_for_team("XXX") is LEAGUE_FENCE
    assert fence_for_team(None) is LEAGUE_FENCE


def test_team_abbreviations_map_to_real_parks() -> None:
    for abbrev, venue in TEAM_VENUE.items():
        assert venue in PARKS, abbrev


def test_wall_height_lengthens_the_required_carry() -> None:
    """Fenway's 310ft line is not a short porch behind 37 feet of wall."""
    fenway = get_fence(3)
    lf_line = wall_distance(fenway, -45.0)
    assert fenway.distances[0] == 310
    assert lf_line > 340  # 310 + the Monster
    # A standard 8ft wall is not penalised at all.
    yankee = get_fence(3313)
    assert wall_distance(yankee, 45.0) == yankee.distances[-1]


def test_wall_distance_interpolates_between_anchors() -> None:
    park = get_fence(3313)  # Yankee Stadium: 318 line, 399 left-center
    eff = park.effective()
    mid = wall_distance(park, (ANCHOR_ANGLES[0] + ANCHOR_ANGLES[1]) / 2)
    assert eff[0] < mid < eff[1]
    # Foul territory clamps to the line anchors rather than extrapolating.
    assert wall_distance(park, -80.0) == eff[0]
    assert wall_distance(park, 80.0) == eff[-1]


def test_the_short_porch_is_shorter_than_the_gap() -> None:
    yankee = get_fence(3313)
    assert wall_distance(yankee, 45.0) < wall_distance(yankee, 0.0)


# --- per-ball scoring --------------------------------------------------------


def test_hr_probability_is_a_soft_call_at_the_wall() -> None:
    assert hr_probability(380.0, 380.0) == 0.5
    assert hr_probability(430.0, 380.0) > 0.95
    assert hr_probability(330.0, 380.0) < 0.05
    # Monotone in distance.
    assert hr_probability(390.0, 380.0) > hr_probability(385.0, 380.0)


def test_spray_angle_orients_left_negative_right_positive() -> None:
    x = pd.Series([HOME_X - 50, HOME_X, HOME_X + 50])
    y = pd.Series([HOME_Y - 50] * 3)
    angles = spray_angle(x, y)
    assert angles[0] < 0 < angles[2]
    assert abs(angles[1]) < 1e-9


# --- batter aggregation ------------------------------------------------------


def _ball(
    distance: float,
    la: float = 28.0,
    hc_x: float = HOME_X,
    hc_y: float = HOME_Y - 100,
    team: str = "NYY",
    event: str = "field_out",
) -> dict[str, object]:
    return {
        "batter": 1,
        "events": event,
        "launch_angle": la,
        "launch_speed": 100.0,
        "hit_distance_sc": distance,
        "hc_x": hc_x,
        "hc_y": hc_y,
        "home_team": team,
    }


def test_the_same_fly_ball_is_worth_more_in_a_smaller_park() -> None:
    """330 feet down the line: a home run in the Bronx, an out in Detroit."""
    pull = {"hc_x": HOME_X + 100.0, "hc_y": HOME_Y - 100.0}
    bronx = batter_xhr(pd.DataFrame([_ball(340.0, team="NYY", **pull)]))
    detroit = batter_xhr(pd.DataFrame([_ball(340.0, team="DET", **pull)]))
    assert bronx.xhr > detroit.xhr


def test_xhr_separates_luck_from_power() -> None:
    """Three wall-scrapers that went out are not three home runs of talent."""
    lucky = pd.DataFrame([_ball(345.0, event="home_run") for _ in range(3)] * 20)
    prof = batter_xhr(lucky)
    assert prof.hr == 60
    assert prof.xhr < prof.hr
    assert prof.luck > 0
    assert prof.xhr_per_pa < prof.hr_per_pa


def test_low_and_high_launch_angles_score_zero() -> None:
    """A 400ft line drive and a towering pop-up are not home runs."""
    liner = batter_xhr(pd.DataFrame([_ball(400.0, la=8.0)]))
    popup = batter_xhr(pd.DataFrame([_ball(400.0, la=70.0)]))
    assert liner.xhr == 0.0
    assert popup.xhr == 0.0
    # ... but a 440ft drive in the home-run window clears center field.
    assert batter_xhr(pd.DataFrame([_ball(440.0, la=28.0)])).xhr > 0.9
    # 400ft to dead center at Yankee Stadium (408) is a warning-track out.
    assert batter_xhr(pd.DataFrame([_ball(400.0, la=28.0)])).xhr < 0.5


def test_missing_distance_data_reports_no_data_rather_than_zero() -> None:
    """An old Statcast cache without hit_distance_sc must not zero the prior."""
    rows = [_ball(400.0)]
    bare = pd.DataFrame(rows).drop(columns=["hit_distance_sc"])
    prof = batter_xhr(bare)
    assert not prof.has_data
    assert prof.xhr_per_pa != prof.xhr_per_pa  # NaN
    assert prof.pa == 1  # PAs are still counted


def test_pa_and_batted_ball_counts() -> None:
    prof = batter_xhr(pd.DataFrame([_ball(400.0), _ball(200.0, la=-5.0)]))
    assert prof.pa == 2
    assert prof.batted == 2


# --- the HR/PA blend ---------------------------------------------------------


def _rates(p_hr: float, pa: float) -> OutcomeRates:
    """League-average rates with the HR share overridden, still summing to 1."""
    d = dict(LEAGUE_RATES)
    scale = (1.0 - p_hr) / (1.0 - d["HR"])
    return OutcomeRates(
        pa=pa,
        p_1b=d["1B"] * scale,
        p_2b=d["2B"] * scale,
        p_3b=d["3B"] * scale,
        p_hr=p_hr,
        p_bb=d["BB"] * scale,
        p_k=d["K"] * scale,
        p_out=d["OUT"] * scale,
    )


def test_blend_pulls_a_lucky_hitter_down_and_an_unlucky_one_up() -> None:
    lucky = blend_hr_rate(_rates(0.060, pa=150), xhr_prior=0.030)
    unlucky = blend_hr_rate(_rates(0.010, pa=150), xhr_prior=0.030)
    assert 0.030 < lucky.p_hr < 0.060
    assert 0.010 < unlucky.p_hr < 0.030


def test_blend_respects_sample_size() -> None:
    """A full season of PAs keeps more of the observed rate than a month."""
    thin = blend_hr_rate(_rates(0.060, pa=40), xhr_prior=0.030)
    thick = blend_hr_rate(_rates(0.060, pa=600), xhr_prior=0.030)
    assert thick.p_hr > thin.p_hr


def test_blend_keeps_the_outcome_rates_normalised() -> None:
    out = blend_hr_rate(_rates(0.060, pa=150), xhr_prior=0.020)
    total = out.p_1b + out.p_2b + out.p_3b + out.p_hr + out.p_bb + out.p_k + out.p_out
    assert abs(total - 1.0) < 1e-9
    # Only the HR share is re-aimed; the rest keep their relative shape.
    src = _rates(0.060, pa=150)
    assert abs(out.p_1b / out.p_k - src.p_1b / src.p_k) < 1e-9


def test_blend_is_a_no_op_without_a_prior() -> None:
    src = _rates(0.060, pa=150)
    assert blend_hr_rate(src, xhr_prior=float("nan")) is src
    assert blend_hr_rate(src, xhr_prior=0.030, prior_weight=0.0) is src


def test_blend_is_switchable_and_weighted_from_config() -> None:
    from mlb_engine.config import Config

    assert Config().xhr_blend is True
    assert Config().xhr_prior_weight > 0


# --- tonight's park, applied to this hitter ----------------------------------


def _spray_profile(mean_spray: float, n: int = 250, seed: int = 0) -> pd.DataFrame:
    """A synthetic batted-ball profile centred on a given spray angle."""
    import numpy as np

    rng = np.random.default_rng(seed)
    spray = rng.normal(mean_spray, 12.0, n)
    rad = np.radians(spray)
    return pd.DataFrame({
        "events": "field_out",
        "launch_angle": rng.normal(28.0, 7.0, n),
        "hit_distance_sc": rng.normal(360.0, 40.0, n),
        "hc_x": HOME_X + np.sin(rad) * 100.0,
        "hc_y": HOME_Y - np.cos(rad) * 100.0,
        "home_team": "NYY",
    })


PULL_LHH, PULL_RHH = 30.0, -30.0


def test_the_short_porch_pays_the_pull_hitter_who_can_reach_it() -> None:
    """Yankee Stadium's 314ft right field is worth more to a lefty than Detroit."""
    lhh = _spray_profile(PULL_LHH)
    assert park_hr_multiplier(lhh, 3313) > park_hr_multiplier(lhh, 2394)


def test_the_same_park_is_worth_different_amounts_to_different_hitters() -> None:
    """The point of the whole exercise: a scalar park factor cannot do this."""
    # Oracle Park: 309ft down the right-field line behind a 25ft wall, but a
    # 415ft right-centre. A pull right-hander gains; a pull lefty is buried.
    assert park_hr_multiplier(_spray_profile(PULL_RHH), 2395) > park_hr_multiplier(
        _spray_profile(PULL_LHH), 2395
    )
    # Fenway is the mirror image: the Monster eats the right-handed pull side.
    assert park_hr_multiplier(_spray_profile(PULL_LHH), 3) > park_hr_multiplier(
        _spray_profile(PULL_RHH), 3
    )


def test_coors_is_a_home_run_park_despite_deep_fences() -> None:
    """Geometry alone gets Coors backwards; the carry factor carries the level."""
    assert park_shape_baseline(19) < 1.0  # deepest fences in baseball
    assert park_hr_multiplier(_spray_profile(0.0), 19) > 1.0  # still a HR park
    # A pitcher's park stays a pitcher's park.
    assert park_hr_multiplier(_spray_profile(0.0), 2394) < 1.0  # Comerica


def test_an_average_spray_chart_lands_near_the_park_level() -> None:
    """The batter-specific term is a deviation, not a second park factor."""
    from mlb_engine.data.parks import get_park

    balanced = _spray_profile(0.0, n=400)
    for venue in (22, 2889, 4):  # Dodger, Busch, Rate: unremarkable geometry
        park = get_park(venue)
        assert park is not None
        assert abs(park_hr_multiplier(balanced, venue) - park.carry_factor) < 0.12


def test_park_multiplier_is_bounded_and_degrades_safely() -> None:
    balanced = _spray_profile(0.0)
    for venue in list(FENCES) + [None, 999999]:
        m = park_hr_multiplier(balanced, venue)
        assert 0.80 <= m <= 1.25, venue
    # No distance column, or too few fly balls to mean anything -> neutral.
    assert park_hr_multiplier(balanced.drop(columns=["hit_distance_sc"]), 3313) != (
        park_hr_multiplier(balanced.drop(columns=["hit_distance_sc"]), 3313)
    )
    thin = _spray_profile(0.0, n=3)
    assert park_hr_multiplier(thin, 3313) != park_hr_multiplier(thin, 3313)  # NaN


def test_scale_hr_rate_moves_only_home_runs_and_stays_normalised() -> None:
    from mlb_engine.features.rolling import scale_hr_rate

    src = _rates(0.040, pa=300)
    up = scale_hr_rate(src, 1.20)
    assert abs(up.p_hr - 0.048) < 1e-9
    total = up.p_1b + up.p_2b + up.p_3b + up.p_hr + up.p_bb + up.p_k + up.p_out
    assert abs(total - 1.0) < 1e-9
    assert abs(up.p_1b / up.p_k - src.p_1b / src.p_k) < 1e-9
    # A NaN or nonsense multiplier is a no-op, not a wipeout.
    assert scale_hr_rate(src, float("nan")) is src
    assert scale_hr_rate(src, 0.0) is src
