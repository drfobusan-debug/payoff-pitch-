"""A ground-ball arm allows a different kind of hit, not fewer hits.

Also covers the batter singles re-weight, which moves the contact term's weight
from whiff% onto zone-contact%.
"""

from __future__ import annotations

import pandas as pd

from mlb_engine.config import Config
from mlb_engine.features.regression import (
    BL_BABIP,
    BL_BARREL_ALLOWED,
    BL_GB_ALLOWED,
    BL_WHIFF,
    BL_ZONE_CONTACT,
    SINGLES_WHIFF_SLOPE,
    SINGLES_ZONE_CONTACT_SLOPE,
    BatterRegression,
    PitcherRegression,
    build_pitcher_regression,
)


def _pit(gb: float) -> PitcherRegression:
    """A league-average arm apart from ground-ball rate allowed."""
    return PitcherRegression(
        bbe=100,
        pitches=1000,
        babip_allowed=BL_BABIP,
        woba_allowed=0.320,
        xwoba_allowed=0.320,
        hard_hit_allowed=0.400,
        barrel_allowed=BL_BARREL_ALLOWED,
        csw=0.280,
        k_pct=0.220,
        bb_pct=0.080,
        two_strike_whiff=0.280,
        gb_allowed=gb,
    )


def _bat(*, whiff: float = BL_WHIFF, zone_contact: float = BL_ZONE_CONTACT) -> BatterRegression:
    return BatterRegression(
        bbe=100,
        barrel_rate=0.080,
        hard_hit=0.400,
        sweet_spot=0.330,
        bat_speed=71.5,
        max_ev=108.0,
        whiff=whiff,
        zone_contact=zone_contact,
        xba=0.250,
        xslg=0.400,
        babip=0.290,
        woba=0.320,
        xwoba=0.320,
    )


def test_grounders_shift_hits_from_extra_bases_into_singles() -> None:
    worm = _pit(0.58).allowed_multipliers()
    air = _pit(0.30).allowed_multipliers()
    assert worm["1B"] > air["1B"]
    assert worm["2B"] < air["2B"]
    assert worm["3B"] < air["3B"]
    assert worm["HR"] < air["HR"]


def test_a_league_average_gb_rate_changes_nothing() -> None:
    neutral = _pit(BL_GB_ALLOWED).allowed_multipliers()
    rate_only = _pit(BL_GB_ALLOWED).allowed_multipliers(gb_composition=False)
    assert neutral == rate_only


def test_the_composition_shift_roughly_conserves_hits() -> None:
    """The extra-base hits a grounder arm gives up come back as singles.

    Weighted by league batted-ball shares (~.209 singles, ~.069 extra-base hits
    per ball in play), the singles gain should offset most of the XBH loss --
    the fit put total hits allowed at t=-1.09, i.e. indistinguishable from flat.
    """
    for gb in (0.30, 0.50, 0.58):
        m = _pit(gb).allowed_multipliers()
        base = _pit(gb).allowed_multipliers(gb_composition=False)
        d_hits = 0.209 * (m["1B"] - base["1B"]) + 0.069 * (m["2B"] - base["2B"])
        assert abs(d_hits) < 0.006


def test_composition_is_bounded_and_switchable() -> None:
    assert Config().pitcher_gb_composition is True
    extreme = _pit(1.0).allowed_multipliers()
    assert 0.80 <= extreme["2B"] <= 1.25
    assert 0.85 <= extreme["HR"] <= 1.35
    # Off restores the single shared multiplier for every hit type.
    off = _pit(1.0).allowed_multipliers(gb_composition=False)
    assert off["1B"] == off["2B"] == off["3B"]


def test_gb_allowed_is_read_off_batted_ball_type() -> None:
    rows = [
        {
            "pitcher": 1,
            "launch_speed": 90.0,
            "launch_angle": 5.0,
            "launch_speed_angle": 3,
            "bb_type": "ground_ball" if i < 30 else "fly_ball",
            "description": "hit_into_play",
            "events": "single",
            "zone": 5,
            "type": "X",
            "balls": 0,
            "strikes": 0,
            "stand": "R",
            "pfx_z": 1.0,
            "release_extension": 6.0,
            "release_pos_x": 1.0,
            "release_pos_z": 6.0,
            "release_spin_rate": 2200.0,
        }
        for i in range(40)
    ]
    reg = build_pitcher_regression(pd.DataFrame(rows))
    assert reg.gb_allowed == 0.75

    bare = pd.DataFrame(rows).drop(columns=["bb_type"])
    assert build_pitcher_regression(bare).gb_allowed == BL_GB_ALLOWED


def test_the_contact_terms_stayed_evenly_weighted() -> None:
    """The half-season fit wanted zone-contact% up and whiff% out; the graded
    props would not confirm it (dAUC -.0002, CI [-.0038, +.0033]) and the two
    halves of the season disagreed on the direction, so neither slope moved.
    """
    assert SINGLES_ZONE_CONTACT_SLOPE == SINGLES_WHIFF_SLOPE == 0.30

    d = 0.05
    by_contact = _bat(zone_contact=BL_ZONE_CONTACT + d).multipliers()["1B"]
    by_whiff = _bat(whiff=BL_WHIFF - d).multipliers()["1B"]
    assert by_contact > 1.0 and by_whiff > 1.0
    assert _bat().multipliers()["1B"] == 1.0
