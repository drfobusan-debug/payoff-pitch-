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


def _fade_all(market: str, wins: int, losses: int) -> ai.PropStat:
    """A market the engine fades 100% of the time, as it does with home runs."""
    graded = [(_rec(market, 0.2), "win")] * wins + [(_rec(market, 0.2), "loss")] * losses
    df = ai.classify(ai.graded_to_frame(graded, date(2026, 7, 28)))
    return ai.whole_engine_stat(df)


def test_npv_lift_is_zero_when_the_engine_fades_everything():
    # 1512 home-run props, every one faded, 90.7% of them lose. Raw NPV reads
    # 0.907 and looks like the best market on the card; it is pure arithmetic.
    st = _fade_all("batter_hr", wins=141, losses=1371)
    assert abs(st.npv - 0.907) < 0.001
    assert abs(st.base_loss - 0.907) < 0.001
    assert abs(st.npv_lift) < 1e-9


def test_reclaimable_fn_flags_the_unexamined_fade_not_the_low_raw_npv():
    # Fading everything means no fade skill at all, so this is exactly where
    # false negatives hide -- even though raw NPV is 0.907.
    assert _fade_all("batter_hr", wins=141, losses=1371).reclaimable_fn is True

    # A fade side genuinely beating its base rate is not a pocket, even though
    # its raw NPV (0.60) is far below the old fixed 0.62 floor.
    graded = [(_rec("game_rl", 0.6), "win")] * 50 + [(_rec("game_rl", 0.6), "loss")] * 50
    graded += [(_rec("game_rl", 0.3), "win")] * 40 + [(_rec("game_rl", 0.3), "loss")] * 60
    st = ai.whole_engine_stat(ai.classify(ai.graded_to_frame(graded, date(2026, 7, 28))))
    assert abs(st.npv - 0.60) < 1e-9
    assert st.npv_lift > 0.02
    assert st.reclaimable_fn is False


def test_below_breakeven_but_above_random_is_a_pricing_problem_not_a_signal_problem():
    # Total bases: buys win 30% against a 21% base rate. Unprofitable at -110,
    # but the selections are 9 points better than random -- switching the market
    # off would delete signal. Only a market under its own base rate has none.
    graded = [(_rec("batter_tb", 0.6), "win")] * 30 + [(_rec("batter_tb", 0.6), "loss")] * 70
    graded += [(_rec("batter_tb", 0.3), "win")] * 12 + [(_rec("batter_tb", 0.3), "loss")] * 88
    st = ai.whole_engine_stat(ai.classify(ai.graded_to_frame(graded, date(2026, 7, 28))))
    assert abs(st.ppv - 0.30) < 1e-9
    assert abs(st.base_win - 0.21) < 1e-9
    assert st.leaks_ppv is True  # loses money
    assert st.picks_below_random is False  # but picks well
    assert st.ppv_lift > 0

    # Genuinely picking below random: buys win 26% against a 30% base rate.
    bad = [(_rec("pitcher_er", 0.6), "win")] * 26 + [(_rec("pitcher_er", 0.6), "loss")] * 74
    bad += [(_rec("pitcher_er", 0.3), "win")] * 34 + [(_rec("pitcher_er", 0.3), "loss")] * 66
    st2 = ai.whole_engine_stat(ai.classify(ai.graded_to_frame(bad, date(2026, 7, 28))))
    assert st2.picks_below_random is True
    assert st2.ppv_lift < 0
