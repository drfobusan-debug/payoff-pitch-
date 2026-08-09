"""Unit tests for the batter hit / H+R+RBI contact gate."""

from __future__ import annotations

from mlb_engine.features.hits_gate import (
    BL_K_PCT,
    BL_XBA_CONTACT,
    BL_ZCONTACT,
    TIER_AVERAGE,
    TIER_ELITE,
    TIER_GOOD,
    TIER_POOR,
    HitsContactGate,
    contact_score,
)
from mlb_engine.features.regression import BatterRegression


def _bat(
    xba: float,
    k_pct: float,
    zone_contact: float = BL_ZCONTACT,
    sprint: float = 27.0,
    bbe: int = 60,
) -> BatterRegression:
    """A batter carrying only the fields the contact composite reads.

    ``xba`` is expected BA over *batted balls*, which averages ~.320 -- not the
    ~.250 league BA per plate appearance.
    """
    return BatterRegression(
        bbe=bbe,
        barrel_rate=0.08,
        hard_hit=0.40,
        sweet_spot=0.33,
        bat_speed=71.5,
        max_ev=108.0,
        whiff=0.24,
        zone_contact=zone_contact,
        xba=xba,
        xslg=0.400,
        babip=0.290,
        woba=0.320,
        xwoba=0.320,
        sprint_speed=sprint,
        k_pct=k_pct,
        pa=200,
    )


ELITE = _bat(xba=0.400, k_pct=0.150, zone_contact=0.920, sprint=28.5)
GOOD = _bat(xba=0.350, k_pct=0.200)
AVERAGE = _bat(xba=0.310, k_pct=BL_K_PCT)
POOR = _bat(xba=0.260, k_pct=0.300, zone_contact=0.800, sprint=26.0)
# Exactly league average on every input.
MEDIAN = _bat(xba=BL_XBA_CONTACT, k_pct=BL_K_PCT)


def test_tiers_order_from_elite_to_poor() -> None:
    gate = HitsContactGate()
    assert gate.tier(ELITE) == TIER_ELITE
    assert gate.tier(GOOD) == TIER_GOOD
    assert gate.tier(AVERAGE) == TIER_AVERAGE
    assert gate.tier(POOR) == TIER_POOR
    assert contact_score(ELITE) > contact_score(GOOD) > contact_score(POOR)


def test_the_league_average_bat_scores_zero_and_is_not_elite() -> None:
    # Regression guard. The composite is centred on the mean of the *batted-ball*
    # distribution; centring it on the .250 league BA per PA instead put the
    # median hitter +2.3 SD high and tiered 81% of the league as elite, which
    # made the gate inert.
    assert abs(contact_score(MEDIAN)) < 0.05
    assert HitsContactGate().tier(MEDIAN) != TIER_ELITE


def test_elite_and_good_clear_without_help_from_the_park() -> None:
    # The thesis: a real bat at a good price is a buy; the park is a modifier,
    # not a veto. Both clear in a below-average hitting environment.
    gate = HitsContactGate()
    for bat in (ELITE, GOOD):
        keep, reason = gate.allows(bat, context=0.94)
        assert keep is True
        assert "OK" in reason


def test_average_bat_needs_the_park_and_weather() -> None:
    gate = HitsContactGate()
    keep, reason = gate.allows(AVERAGE, context=0.95)
    assert keep is False
    assert "average bat" in reason

    keep, reason = gate.allows(AVERAGE, context=1.06)
    assert keep is True
    assert "OK" in reason


def test_poor_bat_is_never_bought_however_good_the_night() -> None:
    gate = HitsContactGate()
    keep, reason = gate.allows(POOR, context=1.20)
    assert keep is False
    assert "poor contact" in reason


def test_poor_bat_in_a_hostile_park_is_an_under() -> None:
    gate = HitsContactGate()
    assert gate.under_reason(POOR, context=1.05) is None  # no-buy, not a fade
    reason = gate.under_reason(POOR, context=0.92)
    assert reason is not None
    assert "UNDER" in reason


def test_platoon_bat_batting_low_is_an_under_on_pa_risk_alone() -> None:
    gate = HitsContactGate()
    reason = gate.under_reason(
        POOR, context=1.05, slot=8, platoon_disadvantage=True
    )
    assert reason is not None
    assert "batting 8th" in reason


def test_platoon_pa_risk_blocks_the_over_from_the_bottom_third() -> None:
    gate = HitsContactGate()
    assert gate.platoon_pa_reason(8, True) is not None
    # A platoon bat high in the order still gets his four plate appearances.
    assert gate.platoon_pa_reason(3, True) is None
    # No platoon disadvantage, no PA risk.
    assert gate.platoon_pa_reason(8, False) is None
    # Unknown lineup spot stays neutral.
    assert gate.platoon_pa_reason(None, True) is None


def test_neutral_on_a_thin_batted_ball_sample() -> None:
    gate = HitsContactGate()
    thin = _bat(xba=0.260, k_pct=0.340, bbe=5)
    assert gate.tier(thin) is None
    keep, reason = gate.allows(thin, context=0.90)
    assert keep is True
    assert "neutral" in reason
    assert gate.under_reason(thin, context=0.90) is None


def test_average_bat_stays_neutral_when_the_park_is_unknown() -> None:
    gate = HitsContactGate()
    keep, reason = gate.allows(AVERAGE, context=None)
    assert keep is True
    assert "neutral" in reason


def test_kill_switch_disables_every_path() -> None:
    gate = HitsContactGate(enabled=False)
    keep, _ = gate.allows(POOR, context=0.80)
    assert keep is True
    assert gate.under_reason(POOR, context=0.80) is None
    assert gate.platoon_pa_reason(9, True) is None


def test_missing_strikeout_rate_does_not_poison_the_score() -> None:
    # k_pct is NaN when the slice has no completed plate appearances.
    bat = _bat(xba=0.350, k_pct=float("nan"))
    score = contact_score(bat)
    assert score == score  # not NaN
