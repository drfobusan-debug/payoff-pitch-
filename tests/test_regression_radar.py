"""Render-path tests for the Regression Radar report (no network)."""

from __future__ import annotations

from datetime import date

from mlb_engine.output.card import render_pdf
from mlb_engine.output.regression_radar import (
    RegressionTarget,
    build_radar,
    render_html,
    render_markdown,
)

DAY = date(2026, 8, 2)


def _pitcher(name: str, woba: float, xwoba: float) -> RegressionTarget:
    return RegressionTarget(
        name=name, team="BOS", woba=woba, xwoba=xwoba,
        k_pct=0.25, bb_pct=0.06, barrel_pct=0.07, pa=120,
    )


def test_build_radar_sorts_into_four_buckets() -> None:
    pitchers = [_pitcher("A", 0.360, 0.300), _pitcher("B", 0.290, 0.330)]
    batters = [_pitcher("C", 0.300, 0.360), _pitcher("D", 0.360, 0.300)]
    radar = build_radar(pitchers, batters, top_n=10)
    assert set(radar) == {
        "pitchers_positive", "pitchers_negative",
        "batters_positive", "batters_negative",
    }
    # Pitcher w/ woba>>xwoba is the top positive-regression (buy-low) target.
    assert radar["pitchers_positive"][0].name == "A"
    # Batter w/ xwoba>>woba (biggest gap) leads batter positive bucket.
    assert radar["batters_positive"][0].name == "C"


def test_empty_slate_renders_valid_pdf() -> None:
    radar = build_radar([], [], top_n=10)
    html = render_html(radar, DAY)
    md = render_markdown(radar, DAY)
    assert "Regression Radar" in html
    assert "Regression Radar" in md
    assert "No qualified players" in html
    pdf = render_pdf(html)
    assert pdf[:4] == b"%PDF"


def test_populated_radar_includes_player_blurbs() -> None:
    radar = build_radar([_pitcher("Jane Doe", 0.360, 0.300)], [], top_n=10)
    html = render_html(radar, DAY)
    assert "Jane Doe" in html
    assert "buy-low" in html.lower()
