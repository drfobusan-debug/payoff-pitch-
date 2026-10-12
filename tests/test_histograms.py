"""Phase 2a: raw simulator histograms, per-game prediction hash, late-pass replacement."""

from __future__ import annotations

import inspect
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlb_engine.audit import histograms as hg
from mlb_engine.models.montecarlo import GameSimResult
from mlb_engine.models.props import p_over
from mlb_engine.recommendations import Recommendation, load_json, save_json

D = date(2026, 10, 11)
N = 5000


def _res(seed: int = 3) -> GameSimResult:
    rng = np.random.default_rng(seed)
    pois = lambda lam, shape=N: rng.poisson(lam, shape).astype(np.int16)  # noqa: E731
    bat = {t: {s: pois(0.3, (N, 9)) for s in ("H", "1B", "2B", "3B", "HR", "BB", "K", "R", "RBI")}
           for t in ("home", "away")}
    pit = {t: {s: pois(4.0) for s in ("outs", "H", "BB", "K", "ER")} for t in ("home", "away")}
    return GameSimResult(N, pois(4.6), pois(4.2), pois(2.4), pois(2.2), bat, pit)


def _game(pk: int = 1, start: str = "2026-10-11T23:10:00Z"):
    def team(abbrev: str, base: int):
        lineup = [SimpleNamespace(player=SimpleNamespace(mlbam_id=base + i)) for i in range(9)]
        return SimpleNamespace(abbrev=abbrev, lineup=lineup,
                               probable_pitcher=SimpleNamespace(mlbam_id=base + 99))
    g = SimpleNamespace(game_pk=pk, game_datetime_utc=start, home=team("MIL", 100), away=team("LAD", 200))
    g.matchup = lambda: "LAD @ MIL"
    return g


def _rec(pk: int, sel: str, p: float) -> Recommendation:
    return Recommendation(game_date=D, game_pk=pk, matchup="LAD @ MIL", category="game",
                          market="game_total", selection=sel, model_prob=p)


def test_histograms_reproduce_raw_sim_probabilities() -> None:
    res = _res()
    h = hg.build_game_hists(res, _game())  # type: ignore[arg-type]
    full = hg.dense(h["full"])
    hi, ai = np.indices(full.shape)
    tot, mar = hi + ai, hi - ai
    total = res.home_runs_full.astype(float) + res.away_runs_full
    margin = res.home_runs_full.astype(float) - res.away_runs_full
    for line in (7.5, 8.0, 9.5):
        assert full[tot > line].sum() / N == pytest.approx(p_over(total, line), abs=1e-12)
    assert full[mar > 0].sum() / N == pytest.approx(float((margin > 0).mean()), abs=1e-12)
    assert full[mar > 1.5].sum() / N == pytest.approx(float((margin > 1.5).mean()), abs=1e-12)
    f5 = hg.dense(h["f5"])
    assert f5.sum() == N
    k = hg.dense(h["props"][hg.prop_key("batter", 103, "H")])
    kk = np.indices(k.shape)[0]
    arr = res.bat["home"]["H"][:, 3].astype(float)
    assert k[kk > 0.5].sum() / N == pytest.approx(p_over(arr, 0.5), abs=1e-12)
    ks = hg.dense(h["props"][hg.prop_key("pitcher", 299, "K")])
    assert ks.sum(axis=0).tolist()[: full.shape[0]] is not None
    # the prop x total histogram's total margin is the game's total
    assert np.allclose(ks.sum(axis=0), np.bincount(total.astype(int), minlength=ks.shape[1]))


def test_pred_hash_survives_json_round_trip(tmp_path: Path) -> None:
    recs = [_rec(1, "Over 8.5", 0.55), _rec(1, "Under 8.5", 0.45)]
    save_json(recs, tmp_path / "p.json")
    back = load_json(tmp_path / "p.json")
    assert hg.pred_hashes(recs) == hg.pred_hashes(list(reversed(back)))
    changed = [_rec(1, "Over 8.5", 0.56), recs[1]]
    assert hg.pred_hashes(changed)[1] != hg.pred_hashes(recs)[1]


def test_late_pass_replaces_only_repriced_games(tmp_path: Path) -> None:
    g1, g2 = _game(1, "2026-10-11T17:05:00Z"), _game(2, "2026-10-11T23:10:00Z")
    recs = [_rec(1, "Over 8.5", 0.5), _rec(2, "Over 8.5", 0.5)]
    hists = {1: hg.build_game_hists(_res(1), g1), 2: hg.build_game_hists(_res(2), g2)}  # type: ignore[arg-type]
    assert hg.save_histograms(tmp_path, D, hists, recs, [g1, g2], {1: "evA"}) == 2  # type: ignore[list-item]
    before = hg.load_histograms(hg.hist_path(tmp_path, D))
    # doubleheader: same matchup, two keys -- one by event id, one by matchup|start
    assert set(before) == {"evA", "LAD @ MIL|2026-10-11T23:10Z"}
    late = [_rec(2, "Over 8.5", 0.6)]
    hg.save_histograms(tmp_path, D, {2: hg.build_game_hists(_res(9), g2)}, recs[:1] + late, [g2])  # type: ignore[arg-type,list-item]
    after = hg.load_histograms(hg.hist_path(tmp_path, D))
    assert after["evA"] == before["evA"]
    assert after["LAD @ MIL|2026-10-11T23:10Z"]["pred_hash"] == hg.pred_hash(late)


def test_cli_writes_histograms_after_card_and_before_push() -> None:
    from mlb_engine import cli
    src = inspect.getsource(cli.cmd_run)
    i_wb, i_js = src.index("write_workbook("), src.index("save_json(")
    i_h, i_push = src.index("_write_histograms("), src.index("_state_push(")
    assert i_wb < i_js < i_h < i_push


def test_merge_hist_files(tmp_path: Path) -> None:
    g1, g2 = _game(1), _game(2, "2026-10-11T17:05:00Z")
    a, b = tmp_path / "a", tmp_path / "b"
    hg.save_histograms(a, D, {1: hg.build_game_hists(_res(1), g1)}, [_rec(1, "Over 8.5", 0.5)], [g1], {1: "e1"})  # type: ignore[arg-type,list-item]
    hg.save_histograms(b, D, {2: hg.build_game_hists(_res(2), g2)}, [_rec(2, "Over 8.5", 0.5)], [g2], {2: "e2"})  # type: ignore[arg-type,list-item]
    assert hg.merge_hist_files(hg.hist_path(a, D), hg.hist_path(b, D))
    assert set(hg.load_histograms(hg.hist_path(b, D))) == {"e1", "e2"}
