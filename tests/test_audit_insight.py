from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from mlb_engine.output import audit_insight as ai
from mlb_engine.recommendations import Recommendation


def _rec(market: str, model_prob: float, **kw) -> Recommendation:
    return Recommendation(
        game_date=date(2026, 7, 28),
        game_pk=1,
        matchup="AAA @ BBB",
        category=kw.pop("category", "batter"),
        market=market,
        selection="x",
        model_prob=model_prob,
        **kw,
    )


def test_family_of():
    assert ai.family_of("game_ml") == "moneyline"
    assert ai.family_of("f5_rl") == "runline"
    assert ai.family_of("game_total") == "totals"
    assert ai.family_of("batter_hr") == "batter"
    assert ai.family_of("pitcher_k") == "pitcher"


def test_classify_and_counts():
    graded = [
        (_rec("batter_h", 0.7), "win"),  # TP
        (_rec("batter_h", 0.6), "loss"),  # FP
        (_rec("batter_h", 0.3), "win"),  # FN
        (_rec("batter_h", 0.2), "loss"),  # TN
        (_rec("batter_h", 0.9), "push"),  # dropped
    ]
    df = ai.classify(ai.graded_to_frame(graded, date(2026, 7, 28)))
    assert len(df) == 4  # push dropped
    st = ai.whole_engine_stat(df)
    assert (st.tp, st.fp, st.fn, st.tn) == (1, 1, 1, 1)
    assert abs(st.ppv - 0.5) < 1e-9
    assert abs(st.npv - 0.5) < 1e-9


def test_discriminate_finds_separating_metric():
    # winners have high edge, losers have low edge -> should be significant
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(40):
        rows.append({"favored": 1, "won": 1, "edge": rng.normal(0.20, 0.02), "model_prob": 0.7})
        rows.append({"favored": 1, "won": 0, "edge": rng.normal(0.05, 0.02), "model_prob": 0.7})
    df = pd.DataFrame(rows)
    discs = ai._discriminate(df)
    metrics = {d.metric for d in discs}
    assert "edge" in metrics
    edge_d = next(d for d in discs if d.metric == "edge")
    assert edge_d.direction == "higher in winners"
    assert edge_d.p < 0.05


def test_update_store_replaces_date(tmp_path):
    store = tmp_path / "graded_metrics.csv"
    d1 = ai.graded_to_frame([(_rec("batter_h", 0.7), "win")], date(2026, 7, 27))
    ai.update_store(store, d1, date(2026, 7, 27))
    d2 = ai.graded_to_frame([(_rec("batter_h", 0.6), "loss")], date(2026, 7, 27))
    combined = ai.update_store(store, d2, date(2026, 7, 27))
    # same date re-audited -> old rows replaced, not duplicated
    assert len(combined) == 1
    assert combined.iloc[0]["result"] == "loss"
