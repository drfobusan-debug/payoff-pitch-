"""Synthetic, no-network checks for the out-of-sample worksheet study scripts.

Builds a small fake priced-game sample in which the bullpen HardHit% gap is
genuinely predictive and the starter K-BB% gap is pure noise, runs the whole
analysis pipeline into a temp dir and checks the verdict machinery, the price
parsing and the Holm correction.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

analysis = pytest.importorskip("scripts.oos_worksheet_analysis")
odds = pytest.importorskip("scripts.oos_odds_history")

from mlb_engine.output.daily_worksheet import BAT_COLS, BP_COLS, SP_COLS  # noqa: E402

TEAMS = ["NYY", "BOS", "LAD", "SF", "CHC", "STL", "HOU", "TEX", "ATL", "NYM"]


def _synthetic(n_games: int, seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    games, prices = [], []
    day0 = datetime(2024, 4, 1, tzinfo=timezone.utc)
    for i in range(n_games):
        home, away = rng.choice(TEAMS, 2, replace=False)
        commence = day0 + timedelta(days=i // 8, hours=23, minutes=int(rng.integers(0, 60)))
        p_home_true = float(np.clip(rng.normal(0.54, 0.06), 0.3, 0.75))
        hh_gap = rng.choice([-4, -3, -1, 0, 1, 3, 4])  # bp HardHit% pts gap, predictive
        p_home = float(np.clip(p_home_true + 0.04 * hh_gap, 0.05, 0.95))
        home_win = rng.random() < p_home
        row: dict = {
            "game_pk": 700000 + i, "date": commence.date().isoformat(), "commence": commence.isoformat(),
            "dh": "N", "home": home, "away": away,
            "home_runs": int(rng.integers(0, 9)) + (3 if home_win else 0),
            "away_runs": int(rng.integers(0, 9)) + (0 if home_win else 3),
            "home_r15": int(rng.integers(0, 4)), "away_r15": int(rng.integers(0, 4)),
            "home_r6": int(rng.integers(0, 3)), "away_r6": int(rng.integers(0, 3)),
            "innings": 9, "sp_pool": 60, "sp_k": 6,
        }
        for side in ("home", "away"):
            row[f"{side}_sp_id"] = int(rng.integers(500000, 700000))
            row[f"{side}_sp_hand"] = rng.choice(["L", "R"])
            row[f"{side}_opp_hand"] = rng.choice(["L", "R"])
            for prefix, cols in (("sp", [c[0] for c in SP_COLS]), ("bp", [c[0] for c in BP_COLS])):
                total = 0
                for label in cols:
                    pts = int(rng.choice([-2, 1, 1, 1, 2]))
                    if prefix == "bp" and label == "HardHit%":
                        pts = 1 + (hh_gap if side == "home" else 0)
                    row[f"{side}_{prefix}_{label}_pts"] = pts
                    row[f"{side}_{prefix}_{label}_val"] = float(rng.normal())
                    row[f"{side}_{prefix}_{label}_pct"] = float(rng.random())
                    row[f"{side}_{prefix}_{label}_z"] = float(rng.normal())
                    total += pts
                row[f"{side}_{prefix}_total"] = total
            for split in ("ovr", "6", "vsl", "vsr", "hand"):
                row[f"{side}_bat_{split}_total"] = int(rng.integers(0, 30))
                row[f"{side}_bat_{split}_rank"] = int(rng.integers(1, 31))
            for label in [c[0] for c in BAT_COLS]:
                row[f"{side}_bat_hand_{label}_pts"] = int(rng.choice([-2, 1, 2]))
                row[f"{side}_bat_hand_{label}_pct"] = float(rng.random())
            row[f"{side}_bat_hand_pa60"] = int(rng.integers(400, 1400))
        games.append(row)
        # fair no-vig price with 2% vig each side
        pa = 1 - p_home_true
        ml_home = -100 * p_home_true / (1 - p_home_true) * 1.04 if p_home_true >= 0.5 else 100 * (1 - p_home_true) / p_home_true / 1.04
        ml_away = -100 * pa / (1 - pa) * 1.04 if pa >= 0.5 else 100 * (1 - pa) / pa / 1.04
        prices.append({
            "home": home, "away": away, "commence": commence.isoformat(),
            "snapshot": (commence - timedelta(hours=1)).isoformat(), "hours_to_pitch": 1.0,
            "n_books_ml": 5, "book": "pinnacle", "ml_home": round(ml_home), "ml_away": round(ml_away),
            "p_home_book": p_home_true, "ml_home_med": round(ml_home), "ml_away_med": round(ml_away),
            "p_home_cons": p_home_true, "rl_book": "pinnacle", "rl_home_pt": -1.5 if p_home_true >= 0.5 else 1.5,
            "rl_home": -110, "rl_away": -110, "rl_home_med": -110, "rl_away_med": -110,
            "tot_book": "pinnacle", "total_pt": float(rng.choice([7.5, 8.0, 8.5, 9.0, 9.5])),
            "over": -110, "under": -110, "total_med": 8.5,
        })
    return pd.DataFrame(games), pd.DataFrame(prices)


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    games, prices = _synthetic(2400)
    gp, pp = tmp_path / "games.csv", tmp_path / "prices.csv"
    games.to_csv(gp, index=False)
    prices.to_csv(pp, index=False)
    out = tmp_path / "out"
    report = analysis.run([gp], pp, out)
    assert report.exists()
    priced = pd.read_csv(out / "oos_priced_games.csv")
    assert len(priced) == 2400, "every synthetic game must join to its price row"
    hyp = pd.read_csv(out / "oos_hypotheses.csv")
    assert list(hyp["id"][:9]) == ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8a", "H8b"]
    h1 = hyp.set_index("id").loc["H1"]
    assert h1["point"] > 0 and h1["lo"] > 0, "planted BP HardHit% edge must be recovered"
    assert h1["verdict"] == "confirmed OOS"
    h5 = hyp.set_index("id").loc["H5"]
    assert h5["lo"] < 0 < h5["hi"], "noise metric must not show a residual excluding zero"
    tables = pd.read_csv(out / "oos_metric_tables.csv")
    assert set(tables["scoring"]) == {"pts", "dec", "z"}
    pooled_pts = tables[tables["stratum"].eq("pooled") & tables["scoring"].eq("pts")]
    assert len(pooled_pts) == 2 * (11 + 1 + 2), "11 metrics + TOTAL + 2 families for SP and BP"
    bat = pd.read_csv(out / "oos_batting_hypotheses.csv")
    assert list(bat["id"]) == [f"H-V{i}" for i in range(7)]
    assert bat["holm_p"].notna().all()
    text = report.read_text()
    assert "pre-game" in text and "closing" in text
    for f in ("oos_rolling_400.csv", "oos_batting_tables.csv"):
        assert (out / f).exists()


def test_holm_and_verdicts() -> None:
    assert analysis.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert analysis.verdict(+1, 3.0, 0.5, 5.5, 1000, 800) == "confirmed OOS"
    assert analysis.verdict(+1, 3.0, -0.5, 6.5, 500, 2000) == "sign replicates but underpowered"
    assert analysis.verdict(+1, 1.0, -2.0, 4.0, 5000, 2000) == "does not replicate"
    assert analysis.verdict(+1, -3.0, -5.0, -1.0, 1000, 800) == "reversed"
    assert analysis.verdict(-1, -3.0, -5.0, -1.0, 1000, 800) == "confirmed OOS"


def test_power_n_matches_two_sample_formula() -> None:
    # 3-point residual with sd 50 pts -> ((1.96+0.8416)*50/3)^2 ~ 2180
    assert abs(analysis.power_n(3.0, 50.0) - 2180.3) < 1


def test_odds_history_parse_event_no_vig_and_slots() -> None:
    snap = datetime(2024, 6, 14, 22, 55, tzinfo=timezone.utc)
    ev = {
        "commence_time": "2024-06-14T23:10:00Z", "home_team": "New York Yankees", "away_team": "Boston Red Sox",
        "bookmakers": [
            {"key": "pinnacle", "markets": [
                {"key": "h2h", "outcomes": [{"name": "New York Yankees", "price": -150}, {"name": "Boston Red Sox", "price": 140}]},
                {"key": "spreads", "outcomes": [{"name": "New York Yankees", "price": 110, "point": -1.5},
                                                {"name": "Boston Red Sox", "price": -130, "point": 1.5}]},
                {"key": "totals", "outcomes": [{"name": "Over", "price": -105, "point": 8.5},
                                               {"name": "Under", "price": -115, "point": 8.5}]},
            ]},
            {"key": "fanduel", "markets": [
                {"key": "h2h", "outcomes": [{"name": "New York Yankees", "price": -160}, {"name": "Boston Red Sox", "price": 145}]},
            ]},
        ],
    }
    row = odds.parse_event(ev, snap)
    assert row is not None
    assert row["home"] == "NYY" and row["away"] == "BOS" and row["book"] == "pinnacle"
    assert row["ml_home"] == -150 and row["rl_home_pt"] == -1.5 and row["total_pt"] == 8.5
    p_h, p_a = odds.no_vig(odds.american_to_prob(-150), odds.american_to_prob(140))
    assert abs(p_h + p_a - 1) < 1e-12 and abs(row["p_home_book"] - p_h) < 1e-9
    assert 0.2 < row["hours_to_pitch"] < 0.3
    # a late snapshot (after first pitch) must not be used
    assert odds.parse_event(ev, snap + timedelta(hours=1)) is None
    slots = odds.slots_for_day([datetime(2024, 6, 14, 17, 10, tzinfo=timezone.utc), datetime(2024, 6, 14, 23, 10, tzinfo=timezone.utc)])
    assert all(s < datetime(2024, 6, 14, 23, 10, tzinfo=timezone.utc) for s in slots)
    assert len(slots) == 2


def test_closing_extract_parse_snapshot_assigns_rl_by_favourite() -> None:
    closing = pytest.importorskip("scripts.oos_closing_extract")
    rows = [
        {"matchup": "BOS @ NYY", "market": "game_ml", "selection": "NYY ML", "no_vig_prob": 0.58, "american": -140},
        {"matchup": "BOS @ NYY", "market": "game_ml", "selection": "BOS ML", "no_vig_prob": 0.42, "american": 120},
        {"matchup": "BOS @ NYY", "market": "game_rl", "selection": "NYY -1.5", "american": 150},
        {"matchup": "BOS @ NYY", "market": "game_rl", "selection": "BOS +1.5", "american": -175},
        {"matchup": "BOS @ NYY", "market": "f5_ml", "selection": "NYY ML", "no_vig_prob": 0.6, "american": -150},
        {"matchup": "SF @ LAD", "market": "game_ml", "selection": "LAD ML", "no_vig_prob": 0.55, "american": -125},
    ]
    out = closing.parse_snapshot(rows, "2026-08-04")
    assert len(out) == 1  # SF @ LAD lacks a second ML side and is dropped
    g = out[0]
    assert (g["home"], g["away"], g["date"]) == ("NYY", "BOS", "2026-08-04")
    assert g["p_home_close"] == 0.58 and g["away_ml_close"] == 120
    assert g["home_rl_line_close"] == -1.5 and g["home_rl_price_close"] == 150
    assert g["away_rl_line_close"] == 1.5 and g["away_rl_price_close"] == -175
