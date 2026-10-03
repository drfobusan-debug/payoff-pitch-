"""League-rate shrink, sim goal level, market anchor on game markets, ledger sync."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date as Date
from datetime import datetime, timezone
from pathlib import Path

import pytest

from nhl_engine import state
from nhl_engine.audit.ledger import row_from
from nhl_engine.calibration import Calibrator
from nhl_engine.config import Config, GateParams, SimParams
from nhl_engine.data.book_rules import BookRules
from nhl_engine.market.board import Selection, selections
from nhl_engine.market.pricing import Tier, anchor, stamp
from nhl_engine.models.goals import side_rates
from nhl_engine.models.markets import Prob
from nhl_engine.pipeline import price_game, shrink_league
from tests.nhl.test_phase2 import LEAGUE as P2_LEAGUE
from tests.nhl.test_phase2 import NOW as P2_NOW
from tests.nhl.test_phase2 import STRONG, WEAK, _inputs, _q

LEAGUE = {
    "xgf60_5v5": 2.5,
    "xga60_5v5": 2.5,
    "fin60_5v5": 0.0,
    "pp_xgf60": 7.4,
    "pp_xga60": 0.9,
    "pk_xga60": 7.4,
    "sh_xgf60": 0.9,
    "pen_taken60": 3.4,
    "pen_drawn60": 3.4,
}
NOW = datetime(2026, 10, 2, 22, 0, tzinfo=timezone.utc)


def test_shrink_league_weights_by_team_games():
    live = {"fin60_5v5": 0.35, "xgf60_5v5": 2.3, "pen_taken60": 4.2}
    prior = {"fin60_5v5": 0.0, "xgf60_5v5": 2.5, "pen_taken60": 3.4}
    ks = {"fin60_5v5": 400.0, "xgf60_5v5": 100.0, "pen_taken60": float("inf")}
    out = shrink_league(live, prior, 16.0, ks)
    assert out["fin60_5v5"] == pytest.approx(0.35 * 16 / 416)
    assert out["xgf60_5v5"] == pytest.approx(2.3 * 16 / 116 + 2.5 * 100 / 116)
    assert out["pen_taken60"] == pytest.approx(3.4)
    assert shrink_league(live, prior, 0.0, ks) == prior
    assert shrink_league({"x": 1.0}, {"x": float("nan")}, 5.0, {})["x"] == 1.0


def test_default_league_k_covers_every_rate_the_sim_reads():
    ks = Config().shrink.league_k_games
    assert set(LEAGUE) <= set(ks)
    assert ks["pen_taken60"] == float("inf") and ks["fin60_5v5"] == 400.0


def test_goal_level_scales_goal_rates_not_penalties_or_en_strength():
    base = SimParams(goal_level=1.0)
    up = replace(base, goal_level=1.1)
    kw = dict(is_home=True, opp_goalie_gsax60=0.0, own_goalie_gsax60=0.0)
    a = side_rates(LEAGUE, LEAGUE, LEAGUE, params=base, **kw)
    b = side_rates(LEAGUE, LEAGUE, LEAGUE, params=up, **kw)
    assert b.g5 == pytest.approx(a.g5 * 1.1)
    assert b.pp == pytest.approx(a.pp * 1.1)
    assert b.sh == pytest.approx(a.sh * 1.1)
    assert b.en_for == pytest.approx(a.en_for)
    assert b.minors == pytest.approx(a.minors)


def _sel(market: str = "game_ml", consensus: float = 0.55, american: float = -110) -> Selection:
    return Selection(
        matchup="A @ H",
        market=market,
        side="H",
        entity="",
        line=None,
        consensus=consensus,
        books=3,
        two_sided=3,
        best_american=american,
        best_book="dk",
        best_captured_at=NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
        per_book={"dk": american},
    )


def test_anchor_blends_win_and_keeps_push():
    p = anchor(Prob(0.70, 0.08), 0.50, 0.9)
    assert p.win == pytest.approx(0.52) and p.push == 0.08
    assert anchor(Prob(0.7), 0.5, 1.5).win == pytest.approx(0.5)


def test_anchored_row_grades_sim_but_prices_bet():
    thr = GateParams(market_anchor=0.9)
    sel = _sel(consensus=0.55, american=110)
    bet = anchor(Prob(0.65), sel.consensus, thr.market_anchor)
    pr = stamp(
        sel,
        bet,
        ot_rule="incl_ot_so",
        thr=thr,
        quote_age_minutes=5,
        goalie_confirmed=True,
        settlement_gate=None,
        sim_prob=0.65,
    )
    # sim 10 pts over the market -> bet edge 0.01: below min_edge, no buy
    assert pr.edge == pytest.approx(0.01) and pr.tier is Tier.PASS and "min_edge" in pr.gates
    row = row_from(
        pr,
        slate_date=Date(2026, 10, 2),
        home="H",
        away="A",
        home_goalie="",
        away_goalie="",
        goalie_status="",
        lineup_source="",
        priced_at="t",
        card_tag="initial",
    )
    assert row.model_prob == pytest.approx(0.65) and row.bet_prob == pytest.approx(0.56)


def test_price_game_anchors_game_markets_only():
    rows = [
        _q("game_ml", "H", -150, 130),
        _q("game_ml", "A", 130, -150),
        _q("p1_total", "over", -120, 100, line=1.5),
    ]
    card = price_game(
        selections(rows),
        home_in=_inputs("H", STRONG, "confirmed"),
        away_in=_inputs("A", WEAK, "confirmed"),
        league=P2_LEAGUE,
        cfg=Config(sim=SimParams(draws=3000)),
        slate=Date(2026, 10, 1),
        rules=BookRules([]),
        calib=Calibrator(),
        now=P2_NOW,
        priced_at="t",
        tag="initial",
        seed=1,
    )
    by = {(r.market, r.side): r for r in card.rows}
    ml = by[("game_ml", "H")]
    assert ml.bet_prob == pytest.approx(0.1 * ml.model_prob + 0.9 * ml.consensus)
    assert ml.edge == pytest.approx(ml.bet_prob - ml.consensus)
    assert any(r.startswith("anchored 0.90") for r in ml.reasons)
    p1 = by[("p1_total", "over")]
    assert p1.bet_prob == pytest.approx(p1.model_prob)


def test_ledger_sync_pushes_changed_pulls_missing(tmp_path: Path):
    local, branch = tmp_path / "local", tmp_path / "branch"
    local.mkdir()
    (local / "predictions_2026-10-02.json").write_text("[1]")
    (local / "graded_2026-10-02.json").write_text(json.dumps([{"v": 1}]))
    (local / "notes.json").write_text("{}")
    pushed = state._copy_ledger(local, branch, overwrite=True)
    assert sorted(pushed) == ["ledger/graded_2026-10-02.json", "ledger/predictions_2026-10-02.json"]
    assert state._copy_ledger(local, branch, overwrite=True) == []
    (local / "graded_2026-10-02.json").write_text(json.dumps([{"v": 2}]))
    assert state._copy_ledger(local, branch, overwrite=True) == ["ledger/graded_2026-10-02.json"]

    other = tmp_path / "other"
    other.mkdir()
    (other / "graded_2026-10-02.json").write_text("stale-local")
    pulled = state._copy_ledger(branch, other, overwrite=False)
    assert pulled == ["ledger/predictions_2026-10-02.json"]
    assert (other / "graded_2026-10-02.json").read_text() == "stale-local"
