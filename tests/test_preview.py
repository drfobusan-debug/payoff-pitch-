from __future__ import annotations

import datetime as dt

from mlb_engine.output.daily_preview import build_preview_report, game_shape
from mlb_engine.preview import (
    BestBet,
    BullpenLine,
    GamePreview,
    LineupLine,
    RegFlag,
    StarterLine,
    load_previews,
    save_previews,
)


def _starter(name: str) -> StarterLine:
    return StarterLine(
        name=name,
        pitches=1200,
        k_pct=0.26,
        xk_pct=0.24,
        bb_pct=0.07,
        xbb_pct=0.08,
        csw=0.30,
        whiff=0.11,
        swstr=0.12,
        zone_pct=0.48,
        xwoba_allowed=0.305,
        barrel_allowed=0.07,
        dxwoba=0.010,
        spin=2300.0,
    )


def _lineup() -> LineupLine:
    return LineupLine(
        n=9,
        woba=0.330,
        xwoba=0.315,
        dxwoba=-0.015,
        xslg=0.430,
        barrel=0.08,
        hot=[RegFlag(name="A. Hot", points=40.0)],
        cold=[RegFlag(name="B. Cold", points=35.0)],
    )


def _preview(**over) -> GamePreview:
    base = dict(
        game_date="2026-07-28",
        game_pk=777,
        matchup="AAA @ BBB",
        home="BBB",
        away="AAA",
        home_starter=_starter("Home Ace"),
        away_starter=_starter("Away Ace"),
        home_lineup=_lineup(),
        away_lineup=_lineup(),
        home_pen=BullpenLine(xwoba_allowed=0.31, k_pct=0.24, zone_pct=0.45, recent_load=0.9),
        away_pen=BullpenLine(xwoba_allowed=0.33, k_pct=0.22, zone_pct=0.43, recent_load=1.2),
        xrd=0.8,
        xrd_sd=3.2,
        total_mean=8.4,
        p_home_win=0.56,
        p_blowout=0.28,
        p_close=0.33,
        home_ml_prob=0.56,
        away_ml_prob=0.44,
        fav_side="home",
        fav_team="BBB",
        fav_odds=-130.0,
        fav_implied=0.565,
        fav_edge=-0.005,
        best_bets=[
            BestBet(
                selection="BBB ML",
                market="game_ml",
                odds=-130.0,
                model_prob=0.56,
                edge=-0.005,
                ev=0.02,
                tier="strong",
            )
        ],
    )
    base.update(over)
    return GamePreview(**base)


def test_preview_roundtrip(tmp_path):
    previews = [_preview(), _preview(game_pk=778, matchup="CCC @ DDD")]
    path = tmp_path / "previews.json"
    save_previews(previews, path)
    loaded = load_previews(path)

    assert len(loaded) == 2
    p = loaded[0]
    assert isinstance(p, GamePreview)
    assert isinstance(p.home_starter, StarterLine)
    assert isinstance(p.home_lineup, LineupLine)
    assert isinstance(p.home_lineup.hot[0], RegFlag)
    assert isinstance(p.home_pen, BullpenLine)
    assert isinstance(p.best_bets[0], BestBet)
    assert p.home_starter.name == "Home Ace"
    assert p.best_bets[0].selection == "BBB ML"


def test_game_shape_high_scoring_close():
    gp = _preview(total_mean=10.2, p_close=0.4, p_blowout=0.2)
    label, desc = game_shape(gp)
    assert "High-scoring" in label
    assert "coin-flip" in label
    assert "total runs" in desc


def test_game_shape_low_scoring_blowout():
    gp = _preview(total_mean=6.9, p_close=0.15, p_blowout=0.42)
    label, _ = game_shape(gp)
    assert "Low-scoring" in label
    assert "blowout-leaning" in label


def test_report_renders_best_bets_and_edge():
    html, narr = build_preview_report(dt.date(2026, 7, 28), [_preview()])
    # best bet appears in bold list
    assert "BBB ML" in html
    assert "Best bets" in html
    # moneyline implied probability + model probability surface
    assert "56.5%" in html  # implied
    assert "Moneyline" in html
    # narration is sportscaster-style and mentions the matchup
    assert "AAA at BBB" in narr
    assert narr.strip().endswith("Payoff Pitch, out.")


def test_report_handles_no_bets():
    gp = _preview(best_bets=[])
    html, narr = build_preview_report(dt.date(2026, 7, 28), [gp])
    assert "the model passes this game" in html
    assert "No bet here" in narr
