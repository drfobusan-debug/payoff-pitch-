"""Situational feature adjustments and geo helpers."""

from __future__ import annotations

from cfb_engine.config import FeatureParams
from cfb_engine.features.adjustments import compute_adjustment
from cfb_engine.features.context import GameContext, haversine_miles

HFA = 2.4


def _adj(ctx: GameContext, **overrides):
    params = FeatureParams(**overrides) if overrides else FeatureParams()
    return compute_adjustment(ctx, params, HFA, "HOME", "AWAY")


def test_no_context_is_noop():
    adj = _adj(GameContext())
    assert adj.margin_delta == 0.0
    assert adj.total_delta == 0.0
    assert adj.reasons == []


def test_disabled_features_short_circuit():
    ctx = GameContext(neutral_site=True, wind_mph=40.0)
    adj = _adj(ctx, enabled=False)
    assert adj.margin_delta == 0.0 and adj.total_delta == 0.0


def test_neutral_site_removes_hfa():
    adj = _adj(GameContext(neutral_site=True))
    assert adj.margin_delta == -HFA
    assert any("Neutral" in r for r in adj.reasons)


def test_rest_edge_favors_more_rested_home():
    adj = _adj(GameContext(rest_home=10, rest_away=6))
    assert adj.margin_delta > 0
    assert any("Rest" in r for r in adj.reasons)


def test_bye_bonus_applied():
    adj = _adj(GameContext(rest_home=14, rest_away=7))
    assert any("bye" in r for r in adj.reasons)


def test_travel_penalizes_away():
    adj = _adj(GameContext(travel_away_miles=2000.0))
    assert adj.margin_delta > 0  # helps home
    assert any("travels" in r for r in adj.reasons)


def test_travel_below_threshold_ignored():
    adj = _adj(GameContext(travel_away_miles=100.0))
    assert adj.margin_delta == 0.0


def test_weather_is_reported_but_not_priced_by_default():
    """The closing total already contains the wind; the reader still sees it."""
    windy = _adj(GameContext(wind_mph=25.0, precipitation=0.2, temperature_f=20.0))
    assert windy.total_delta == 0.0
    assert any("Wind 25 mph" in r and "not scored" in r for r in windy.reasons)
    assert any("Precipitation" in r for r in windy.reasons)
    assert any("Cold 20F" in r for r in windy.reasons)


def test_weather_cuts_total_when_explicitly_priced():
    windy = _adj(
        GameContext(wind_mph=25.0, precipitation=0.2, temperature_f=20.0),
        wind_total_per_mph=0.45,
        precip_total_pts=2.5,
        cold_total_pts=1.5,
    )
    assert windy.total_delta < 0
    assert any("Wind 25 mph" == r for r in windy.reasons)


def test_weather_skipped_indoors():
    indoors = _adj(
        GameContext(dome=True, wind_mph=25.0, precipitation=0.2, temperature_f=20.0),
        wind_total_per_mph=0.45,
    )
    assert indoors.total_delta == 0.0
    assert indoors.reasons == []


def test_haversine_known_distance():
    # Columbus, OH to Los Angeles, CA ~ 1,990 miles.
    miles = haversine_miles(39.96, -83.00, 34.05, -118.24)
    assert 1900 < miles < 2100
