"""The SP/BP scoring backtest must score like the worksheet and grade against the price, offline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

lib = pytest.importorskip("scripts.sp_backtest_lib")


def test_compressed_points_match_worksheet_rule() -> None:
    v = pd.Series([3.0, 2.5, 4.0, np.nan, 3.5, 2.0], index=list("abcdef"))
    pts = lib.compressed_points(v, lower_is_better=True, k=1)
    assert pts["f"] == 2  # lowest xERA is best
    assert pts["d"] == -2  # NaN sorts worst
    assert pts["a"] == 1 and pts["c"] == 1
    high = lib.compressed_points(v, lower_is_better=False, k=1)
    assert high["c"] == 2 and high["d"] == -2


def test_decile_and_z_points_keep_direction() -> None:
    v = pd.Series(np.arange(20, dtype=float))
    dec = lib.decile_points(v, lower_is_better=False)
    assert dec.iloc[-1] == 9 and dec.iloc[0] == 0
    dec_low = lib.decile_points(v, lower_is_better=True)
    assert dec_low.iloc[0] == 9
    z = lib.z_points(v, lower_is_better=False)
    assert z.iloc[-1] > 0 > z.iloc[0]
    assert abs(z.mean()) < 1e-9


def _starters(n: int = 40, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        row = {"date": "2026-06-01", "pitcher": 1000 + i, "name": f"P{i}", "team": "NYY", "IP": 50.0}
        for label in lib.SP_LABELS:
            row[label] = float(rng.normal())
        rows.append(row)
    return pd.DataFrame(rows)


def test_score_starters_totals_and_split() -> None:
    sp = lib.score_starters(_starters())
    assert (sp["k"] == 4).all()
    assert (sp["cmp_total"] == sp["cmp_contact"] + sp["cmp_whiff"]).all()
    assert sp["cmp_total"].between(-22, 22).all()
    assert sp["dec_total"].between(0, 99).all()
    assert set(sp["contact_tercile"]) == {0, 1, 2}
    # per metric, exactly k get +2 and k get -2
    assert (sp["cmp_xERA"] == 2).sum() == 4 and (sp["cmp_xERA"] == -2).sum() == 4


def test_prices_parse_main_lines(tmp_path: Path) -> None:
    closing = tmp_path / "closing"
    closing.mkdir()
    entries = [
        {"matchup": "COL @ NYY", "market": "game_ml", "selection": "COL ML", "american": 250, "no_vig_prob": 0.28},
        {"matchup": "COL @ NYY", "market": "game_ml", "selection": "NYY ML", "american": -280, "no_vig_prob": 0.72},
        {"matchup": "COL @ NYY", "market": "game_rl", "selection": "COL +1.5", "american": 115, "no_vig_prob": 0.46},
        {"matchup": "COL @ NYY", "market": "game_rl", "selection": "NYY -1.5", "american": -128, "no_vig_prob": 0.54},
        {"matchup": "COL @ NYY", "market": "game_total", "selection": "Over 8.5", "american": 100, "no_vig_prob": 0.49},
        {"matchup": "COL @ NYY", "market": "game_total", "selection": "Under 8.5", "american": -109, "no_vig_prob": 0.51},
        {"matchup": "COL @ NYY", "market": "game_total", "selection": "Over 8.0", "american": -120, "no_vig_prob": 0.52},
        {"matchup": "COL @ NYY", "market": "game_total", "selection": "Under 8.0", "american": 100, "no_vig_prob": 0.48},
        {"matchup": "COL @ NYY", "market": "f5_total", "selection": "F5 Over 4.5", "american": -115, "no_vig_prob": 0.51},
        {"matchup": "COL @ NYY", "market": "f5_total", "selection": "F5 Under 4.5", "american": -105, "no_vig_prob": 0.49},
    ]
    (closing / "closing_2026-06-01.json").write_text(json.dumps(entries))
    prices = lib.load_prices(closing)
    g = prices[("2026-06-01", "COL @ NYY")]
    cols = lib.price_columns(pd.Series({"away": "COL", "home": "NYY"}), g)
    assert cols["ml_prob_away"] == 0.28 and cols["ml_home"] == -280
    assert cols["rl_home_m15"] == -128 and cols["rl_away_p15"] == 115
    assert cols["total_line"] == 8.5  # the line closest to a pick
    assert cols["f5_line"] == 4.5 and cols["f5_under_am"] == -105


def test_espn_close_fills_unpriced_dates_only(tmp_path: Path) -> None:
    item = {
        "awayTeamOdds": {"close": {"moneyLine": {"american": "+119"}, "spread": {"american": "-175"},
                                   "pointSpread": {"american": "+1.5"}}},
        "homeTeamOdds": {"close": {"moneyLine": {"american": "-143"}, "spread": {"american": "+144"},
                                   "pointSpread": {"american": "-1.5"}}},
        "close": {"over": {"american": "-118"}, "under": {"american": "-102"},
                  "total": {"alternateDisplayValue": "9"}},
    }
    rows = lib.espn_entries("AZ", "BAL", item)
    assert [r["market"] for r in rows] == ["game_ml"] * 2 + ["game_rl"] * 2 + ["game_total"] * 2
    assert rows[0]["selection"] == "AZ ML" and rows[3]["selection"] == "BAL -1.5" and rows[4]["selection"] == "Over 9.0"
    assert abs(rows[0]["no_vig_prob"] + rows[1]["no_vig_prob"] - 1) < 1e-9
    assert lib.espn_entries("AZ", "BAL", {"awayTeamOdds": {"open": {"moneyLine": {"american": "+135"}}}}) == []
    espn, closing = tmp_path / "espn", tmp_path / "closing"
    espn.mkdir()
    closing.mkdir()
    (espn / "espn_2026-04-15.json").write_text(json.dumps(rows))
    (espn / "espn_2026-08-01.json").write_text(json.dumps(rows))
    (closing / "closing_2026-08-01.json").write_text(json.dumps(
        [{"matchup": "AZ @ BAL", "market": "game_ml", "selection": "AZ ML", "american": 100, "no_vig_prob": 0.5}]))
    prices = lib.load_prices(closing, None, espn)
    assert prices[("2026-04-15", "AZ @ BAL")]["src"] == "espn"
    assert prices[("2026-08-01", "AZ @ BAL")]["src"] == "closing"
    assert prices[("2026-08-01", "AZ @ BAL")]["ml"]["AZ"] == (100.0, 0.5)


def _frame() -> pd.DataFrame:
    """Two dates, four games; away starter clearly better in game 1, worse in game 2."""
    sp = pd.concat([lib.score_starters(_starters(seed=s)) for s in (1, 2)], ignore_index=True)
    sp.loc[sp.index[40:], "date"] = "2026-06-02"
    best = sp[sp["date"] == "2026-06-01"].sort_values("cmp_total")
    lo, hi = int(best.iloc[0]["pitcher"]), int(best.iloc[-1]["pitcher"])
    best2 = sp[sp["date"] == "2026-06-02"].sort_values("cmp_total")
    lo2, hi2 = int(best2.iloc[0]["pitcher"]), int(best2.iloc[-1]["pitcher"])
    bp_rows = []
    for d in ("2026-06-01", "2026-06-02"):
        for t in ("COL", "NYY", "BOS", "SEA"):
            row = {"date": d, "team": t}
            for label in lib.BP_LABELS:
                row[label] = float(hash((d, t, label)) % 100) / 100
            bp_rows.append(row)
    bp = pd.concat([lib.score_bullpens(g) for _, g in pd.DataFrame(bp_rows).groupby("date")], ignore_index=True)
    games = pd.DataFrame([
        {"date": "2026-06-01", "game_pk": 1, "game_number": 1, "doubleheader": False, "away": "COL", "home": "NYY",
         "away_runs": 5, "home_runs": 2, "away_f5": 3, "home_f5": 1, "innings": 9,
         "away_sp_id": hi, "home_sp_id": lo, "away_ra_f5": 1, "home_ra_f5": 3, "away_sp_runs": 1, "home_sp_runs": 4,
         "away_sp_outs": 18, "home_sp_outs": 15, "away_bp_runs": 1, "home_bp_runs": 1, "away_bp_outs": 9, "home_bp_outs": 12},
        {"date": "2026-06-01", "game_pk": 2, "game_number": 1, "doubleheader": False, "away": "BOS", "home": "SEA",
         "away_runs": 1, "home_runs": 6, "away_f5": 0, "home_f5": 4, "innings": 9,
         "away_sp_id": lo, "home_sp_id": hi, "away_ra_f5": 4, "home_ra_f5": 0, "away_sp_runs": 5, "home_sp_runs": 0,
         "away_sp_outs": 12, "home_sp_outs": 21, "away_bp_runs": 1, "home_bp_runs": 1, "away_bp_outs": 12, "home_bp_outs": 6},
        {"date": "2026-06-02", "game_pk": 3, "game_number": 1, "doubleheader": False, "away": "COL", "home": "NYY",
         "away_runs": 4, "home_runs": 3, "away_f5": 2, "home_f5": 2, "innings": 9,
         "away_sp_id": hi2, "home_sp_id": lo2, "away_ra_f5": 2, "home_ra_f5": 2, "away_sp_runs": 2, "home_sp_runs": 3,
         "away_sp_outs": 18, "home_sp_outs": 18, "away_bp_runs": 1, "home_bp_runs": 2, "away_bp_outs": 9, "home_bp_outs": 9},
        {"date": "2026-06-02", "game_pk": 4, "game_number": 1, "doubleheader": True, "away": "BOS", "home": "SEA",
         "away_runs": 2, "home_runs": 3, "away_f5": 1, "home_f5": 1, "innings": 9,
         "away_sp_id": lo2, "home_sp_id": hi2, "away_ra_f5": 1, "home_ra_f5": 1, "away_sp_runs": 3, "home_sp_runs": 2,
         "away_sp_outs": 18, "home_sp_outs": 18, "away_bp_runs": 0, "home_bp_runs": 0, "away_bp_outs": 9, "home_bp_outs": 9},
        {"date": "2026-06-02", "game_pk": 5, "game_number": 2, "doubleheader": True, "away": "BOS", "home": "SEA",
         "away_runs": 7, "home_runs": 3, "away_f5": 4, "home_f5": 1, "innings": 9,
         "away_sp_id": hi2, "home_sp_id": lo2, "away_ra_f5": 1, "home_ra_f5": 4, "away_sp_runs": 1, "home_sp_runs": 5,
         "away_sp_outs": 18, "home_sp_outs": 18, "away_bp_runs": 2, "home_bp_runs": 3, "away_bp_outs": 9, "home_bp_outs": 9},
    ])
    prices = {}
    for d, m, pa in (("2026-06-01", "COL @ NYY", 0.45), ("2026-06-01", "BOS @ SEA", 0.40),
                     ("2026-06-02", "COL @ NYY", 0.55), ("2026-06-02", "BOS @ SEA", 0.50)):
        a, h = m.split(" @ ")
        prices[(d, m)] = {
            "src": "closing",
            "ml": {a: (120.0, pa), h: (-130.0, 1 - pa)},
            "rl": {(a, 1.5): (-150.0, 0.6), (a, -1.5): (170.0, 0.37), (h, -1.5): (140.0, 0.4), (h, 1.5): (-170.0, 0.63)},
            "total": {8.5: {"over": (-105.0, 0.5), "under": (-115.0, 0.5)}},
            "f5_ml": {a: (130.0, pa), h: (-140.0, 1 - pa)},
            "f5_total": {4.5: {"over": (-110.0, 0.5), "under": (-110.0, 0.5)}},
        }
    return lib.build_frame(games, sp, bp, prices)


def test_build_frame_joins_and_excludes_doubleheaders() -> None:
    df = _frame()
    assert len(df) == 5
    assert df["both_scored"].all()
    assert df.loc[df["game_pk"] == 1, "sp_gap_cmp"].iloc[0] > 0
    assert df.loc[df["game_pk"] == 2, "sp_gap_cmp"].iloc[0] < 0
    assert df.loc[df["game_pk"] == 1, "ml_prob_away"].iloc[0] == 0.45
    dh = df[df["dh_collision"]]
    assert len(dh) == 2 and dh["ml_prob_away"].isna().all()
    assert df["gap_sp_xERA"].notna().all()


def test_side_outcomes_grade_against_recorded_price() -> None:
    df = _frame()
    priced = df[df["ml_prob_away"].notna()]
    o = lib.side_outcomes(priced, priced["sp_gap_cmp"])
    g1 = o.loc[priced["game_pk"] == 1].iloc[0]
    assert g1["won"] == 1 and g1["mkt"] == 0.45
    assert g1["ml_units"] == pytest.approx(1.2)  # away +120 won
    assert g1["residual"] == pytest.approx(0.55)
    assert g1["rl_line"] == 1.5 and g1["rl_cover"] == 1  # dog side takes +1.5
    g2 = o.loc[priced["game_pk"] == 2].iloc[0]
    assert g2["won"] == 1 and g2["rl_line"] == -1.5 and g2["rl_cover"] == 1
    assert g2["rl_units"] == pytest.approx(1.4)


def test_boot_ci_and_power_helpers() -> None:
    m, lo, hi = lib.boot_mean_ci(np.array([1.0, 1.0, 1.0, 1.0]), boot=200)
    assert m == lo == hi == 1.0
    assert lib.n_for_mean(0.03) > lib.n_for_mean(0.05)
    assert lib.n_for_corr(0.15) > lib.n_for_corr(0.25)
    reg = lib.ols(np.array([1.0, 2.0, 3.0, 4.1]), np.array([[1.0], [2.0], [3.0], [4.0]]), ["x"])
    assert reg.loc["x", "coef"] == pytest.approx(1.03, abs=0.01)


def test_strategy_table_grades_five_candidates() -> None:
    df = _frame()
    priced = df[df["both_scored"] & df["ml_prob_away"].notna()]
    strat = lib.strategy_table(priced, boot=50)
    assert len(strat) == 5 and strat["strategy"].str.match(r"S[1-5] ").all()
    assert set(strat["verdict"]) <= {"supported", "rejected", "underpowered", "not observed"}
    assert (strat["need_n"] > 0).all()
    split = lib.f5_contact_split(df, boot=50)
    assert list(split["cell"]) == ["both top", "one top", "neither top", "both bottom"]
    assert split["n"].sum() >= len(df[df["both_scored"]]) - split.loc[3, "n"]
    assert lib.strategy_verdict(0.05, 0.01, 0.09, 100, 2000) == "supported"
    assert lib.strategy_verdict(-0.05, -0.09, -0.01, 100, 2000) == "rejected"
    assert lib.strategy_verdict(0.01, -0.02, 0.04, 100, 2000) == "underpowered"
    assert lib.strategy_verdict(0.01, -0.02, 0.04, 2500, 2000) == "rejected"
    assert lib.strategy_verdict(float("nan"), float("nan"), float("nan"), 0, 2000) == "not observed"


def test_write_report_runs_offline(tmp_path: Path) -> None:
    df = _frame()
    sp = pd.DataFrame()
    text = lib.write_report(df, sp, tmp_path, boot=50)
    assert "## Candidate strategies" in text and "## H1" in text and "## H6" in text
    assert (tmp_path / "strategies.csv").exists() and (tmp_path / "f5_contact_split.csv").exists()
    assert (tmp_path / "sp_scoring_backtest.md").exists()
    assert (tmp_path / "metric_table_sp.csv").exists()
    assert (tmp_path / "h5_schemes.csv").exists()
