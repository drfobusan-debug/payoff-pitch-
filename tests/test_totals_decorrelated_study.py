"""Synthetic, no-network checks for scripts/totals_decorrelated_study.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import totals_decorrelated_study as tds

ARMS = ["SIERA", "xERA", "CSW%", "K-BB%"]


def _games(n: int, rng: np.random.Generator) -> pd.DataFrame:
    dates = [(pd.Timestamp("2026-07-01") + pd.Timedelta(days=i // 4)).strftime("%Y-%m-%d") for i in range(n)]
    teams = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
    rows = []
    for i, d in enumerate(dates):
        a, h = teams[(2 * i) % 8], teams[(2 * i + 1) % 8]
        rows.append({
            "game_pk": 1000 + i, "date": d, "away": a, "home": h,
            "away_sp_id": 10 + (2 * i) % 8, "home_sp_id": 10 + (2 * i + 1) % 8,
            "away_runs": int(rng.poisson(4.2)), "home_runs": int(rng.poisson(4.5)),
            "doubleheader": i % 17 == 0,
        })
    return pd.DataFrame(rows)


def _arm_table(games: pd.DataFrame, key: str, ids: list, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for d in games["date"].unique():
        for k in ids:
            r = {"date": d, key: k}
            for m in ARMS:
                r[m] = float(rng.normal(4, 0.5)) if m in ("SIERA", "xERA") else float(rng.normal(0.2, 0.05))
                r[f"z_{m}"] = float(rng.normal())
            rows.append(r)
    return pd.DataFrame(rows)


def _pitches(games: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for g in games.itertuples(index=False):
        for _ in range(30):
            rows.append({
                "game_date": g.date, "inning_topbot": rng.choice(["Top", "Bot"]),
                "away_team": g.away, "home_team": g.home,
                "woba_value": float(rng.choice([0.0, 0.9, 1.25, 2.0])), "woba_denom": 1,
                "launch_speed_angle": int(rng.integers(1, 7)), "bb_type": "fly_ball" if rng.random() < 0.5 else None,
            })
    df = pd.DataFrame(rows)
    df["woba_value"] = df["woba_value"].astype("Float64")
    df["woba_denom"] = df["woba_denom"].astype("Int64")
    df["launch_speed_angle"] = df["launch_speed_angle"].astype("Int64")
    return df


@pytest.fixture
def frame(tmp_path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    games = _games(240, rng)
    closes = {(g.date, f"{g.away} @ {g.home}"): float(rng.choice([7.5, 8.0, 8.5, 9.0])) for g in games.itertuples(index=False)}
    starters = _arm_table(games, "pitcher", list(range(10, 18)), rng)
    pens = _arm_table(games, "team", ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"], rng)
    batting = tds.team_batting_asof(_pitches(games, rng))
    venues = {str(pk): {"venue_id": 1} for pk in games["game_pk"]}
    boxes = [{"pk": int(g.game_pk), "date": g.date, "hp": f"ump{i % 5}", "runs": int(g.away_runs + g.home_runs)}
             for i, g in enumerate(games.itertuples(index=False))]
    return tds.build_frame(games, starters, pens, closes, batting, venues, tds.umpire_asof(boxes))


def test_closing_totals_picks_even_money_and_drops_in_play(tmp_path: Path) -> None:
    quotes = [
        {"matchup": "A @ B", "market": "game_total", "selection": "Over 8.5", "no_vig_prob": 0.46},
        {"matchup": "A @ B", "market": "game_total", "selection": "Over 8.0", "no_vig_prob": 0.51},
        {"matchup": "A @ B", "market": "game_total", "selection": "Over 3.5", "no_vig_prob": 0.50},
        {"matchup": "A @ B", "market": "moneyline", "selection": "A", "no_vig_prob": 0.50},
    ]
    (tmp_path / "closing_2026-08-01.json").write_text(json.dumps(quotes))
    assert tds.closing_totals(tmp_path) == {("2026-08-01", "A @ B"): 8.0}


def test_team_batting_is_strictly_before_the_date() -> None:
    rng = np.random.default_rng(1)
    games = _games(8, rng)
    table = tds.team_batting_asof(_pitches(games, rng))
    first = pd.Timestamp(games["date"].min())
    assert np.isnan(tds.batting_before(table, games.iloc[0].away, first)[0])
    woba, barrel = tds.batting_before(table, games.iloc[0].away, first + pd.Timedelta(days=1))
    assert 0 <= woba <= 2 and 0 <= barrel <= 100


def test_umpire_uses_only_prior_games() -> None:
    boxes = [{"pk": i, "date": f"2026-08-{i:02d}", "hp": "u", "runs": 10} for i in range(1, 4)]
    u = tds.umpire_asof(boxes).set_index("game_pk")
    assert u.loc[1].ump_n == 0 and np.isnan(u.loc[1].ump_diff)
    assert u.loc[3].ump_n == 2 and u.loc[3].ump_diff == 0


def test_frame_drops_doubleheaders_and_sums_sheet_columns(frame: pd.DataFrame) -> None:
    assert not frame.empty
    assert (frame["sum_like"] == frame[tds.SHEET_COLS].sum(axis=1)).all()
    assert (frame["resid"] == frame["runs"] - frame["close"]).all()
    assert set(frame["result"]) <= {"over", "under", "push"}
    assert frame[tds.FACTORS].notna().all().all()


def test_walk_forward_is_out_of_sample(frame: pd.DataFrame) -> None:
    wf = tds.walk_forward(frame, min_train=40)
    scored = wf[wf["ols"].notna()]
    assert wf.loc[:39, "ols"].isna().all()
    assert not scored.empty and scored["orth"].notna().all()


def test_factor_partial_r_strips_the_close() -> None:
    n = 400
    rng = np.random.default_rng(0)
    close = rng.normal(8, 1, n)
    df = pd.DataFrame({"close": close, "resid": rng.normal(0, 3, n)})
    df["sp"] = close + rng.normal(0, 0.1, n)  # pure restatement of the close
    for f in ("pen", "off", "park", "ump", "sum_like"):
        df[f] = 0.0
    t = tds.factor_table(df).set_index("factor")
    assert t.loc["sp", "r_close"] > 0.99
    assert abs(t.loc["sp", "partial_r_resid"]) < 0.15


def test_report_writes_all_artifacts(frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tds, "MIN_TRAIN", 40)
    ledger = frame[["date", "game"]].copy()
    ledger["sum_pts"] = frame["sum_like"]
    ledger["result"] = frame["result"]
    path = tds.write_report(frame, tmp_path, boot=50, ledger=ledger)
    text = path.read_text()
    for section in ("## 1.", "## 2.", "## 3.", "## 4.", "## 4b.", "## 5."):
        assert section in text
    for name in ("factor_table.csv", "sign_records.csv", "bands.csv", "games_scored.csv", "ols_in_sample.csv"):
        assert (tmp_path / name).exists()
