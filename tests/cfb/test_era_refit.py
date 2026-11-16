"""The era refit's close test recovers k, its churn term, and never fits on the test season."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.cfb.era_refit import close_test, close_test_churn, enrich, loso_pred, weight


def _rows(k: float, n: int = 4000, k1: float = 0.0, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        market = rng.normal(0, 14)
        pred = market + rng.normal(0, 4)
        c = rng.normal()
        margin = market + (k + k1 * c) * (pred - market) + rng.normal(0, 3)
        out.append({"pred": pred, "market": market, "margin": margin, "churn": c, "i": i})
    return out


def test_the_close_test_recovers_the_share_of_disagreement_that_lands():
    k, se, n = close_test(_rows(0.3))
    assert n == 4000
    assert k == pytest.approx(0.3, abs=3 * se)
    assert weight(k) == pytest.approx(0.7, abs=3 * se)


def test_the_weight_is_clipped_and_an_unfit_test_falls_back_to_the_market():
    assert weight(1.4) == 0.0
    assert weight(-0.2) == 1.0
    assert weight(float("nan")) == 1.0
    assert np.isnan(close_test(_rows(0.3, n=10))[0])


def test_the_churn_term_picks_up_a_k_that_falls_with_churn():
    (k0, s0), (k1, s1), _ = close_test_churn(_rows(0.3, k1=-0.15))
    assert k0 == pytest.approx(0.3, abs=3 * s0)
    assert k1 == pytest.approx(-0.15, abs=3 * s1)


def test_the_model_margin_for_a_season_is_fitted_without_that_season():
    rows = [
        {"season": s, "gap": g, "margin": (2.0 if s == 2024 else 0.5) * g + 3}
        for s in (2022, 2023, 2024)
        for g in range(-20, 21)
    ]
    pred = {(r["season"], r["gap"]): r["pred"] for r in loso_pred(rows)}
    # 2024's own slope (2.0) is never seen: its fit is on 2022-23 only (0.5).
    assert pred[(2024, 10)] == pytest.approx(0.5 * 10 + 3)
    assert pred[(2022, 10)] != pytest.approx(0.5 * 10 + 3)


def test_enrich_keeps_closed_games_and_zscores_churn_within_the_season():
    rows = [
        {"season": 2024, "home": "a", "away": "b", "spread": -7.0, "margin": 10, "week": 1},
        {"season": 2024, "home": "c", "away": "d", "spread": None, "margin": 3, "week": 1},
    ]
    out = enrich(
        rows,
        prev_sp=lambda s: {"a": 10.0, "b": 4.0},
        churn_of=lambda s: {"a": 30.0, "b": 10.0, "c": 20.0, "d": 20.0},
    )
    assert len(out) == 1
    (r,) = out
    assert r["market"] == 7.0
    assert r["prev_gap"] == 6.0
    assert r["churn"] == pytest.approx(0.0)
