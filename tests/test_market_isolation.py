"""The market-accuracy isolation: grading, metrics, splits and the two ledger readers."""

from __future__ import annotations

import io
from datetime import date as Date
from pathlib import Path

import pytest

from mlb_engine.audit.grade import LOSS, PUSH, WIN
from mlb_engine.audit.market_isolation import (
    FOCUS_MARKETS,
    Row,
    brier,
    build_report,
    calibration,
    edge_terciles,
    last_run_per_day,
    log_loss,
    markets_present,
    paired_bootstrap,
    price_band,
    record,
    render_markdown,
    rows_from_engine_ledger,
    rows_from_power,
    selection_ladder,
    summary_rows,
    write_csv,
)
from mlb_engine.audit.power_ledger import Position, grade_positions
from mlb_engine.data.results import GameResult, PlayerLine

POWER_LEDGER = """date,batter,player_id,game_pk,stat,line,side,book,odds,model_prob,bet_prob,fair_prob,edge,ev,tier,rating,devigged,delivery,run_id,rank,points,fit_pts,fit_rv,category,gate,arm_tier
2026-09-01,Hitter One,1,100,H,0.5,over,dk,-130.0,0.62,0.60,0.55,0.07,0.04,Moderate buy,HOLD,True,,r1,,,,,batter,,
2026-09-01,Hitter Two,2,100,H,1.5,over,dk,200.0,0.40,0.38,0.33,0.07,0.05,Pass,HOLD,True,,r1,,,,,batter,ev_floor,
2026-09-01,Hitter Three,3,100,BB,0.5,over,dk,150.0,0.45,0.44,,,,Pass,HOLD,False,,r1,,,,,batter,one_way_quote,
2026-09-01,Hitter Four,4,100,BB,0.5,under,dk,-110.0,0.60,0.58,0.52,0.08,0.03,Pass,HOLD,True,,r1,,,,,batter,thin_edge,
2026-09-01,Arm One,10,100,outs,15.5,over,dk,-115.0,0.55,0.54,0.50,0.05,0.02,Pass,,True,,r1,,,,,pitcher,ev_floor,
2026-09-01,Arm One,10,100,BB,1.5,under,dk,120.0,0.50,0.49,0.45,0.05,0.04,Pass,,True,,r1,,,,,pitcher,ev_floor,
2026-09-01,Scratched,5,100,H,0.5,over,dk,-120.0,0.60,0.59,0.54,0.06,0.03,Pass,HOLD,True,,r1,,,,,batter,ev_floor,
2026-09-01,Early Row,6,100,H,0.5,over,dk,-120.0,0.60,0.59,0.54,0.06,0.03,Pass,HOLD,True,,r0,,,,,batter,ev_floor,
"""

ENGINE_LEDGER = """date,matchup,category,market,selection,line,book,odds,under_odds,tier,model_prob,ev,result,pnl,margin,veto_gate,pass_gate,raw_prob,fair_prob,bet_prob,close_odds,close_prob,clv,clv_ev,lineup_status,hours_to_first_pitch,batx_prob,source
2026-09-01,A @ B,Batter Props,batter_h,Some Guy H o0.5,0.5,dk,-125.0,-103.0,Pass,0.5779,0.04,loss,-1.0,,,no_buy,0.59,0.5271,0.5779,,,,,posted,2.4,,engine
2026-09-01,A @ B,Batter Props,batter_h,Some Guy H u0.5,0.5,dk,150.0,-201.0,Pass,0.3642,-0.03,win,1.5,,,ev_floor,0.28,0.3872,0.3849,,,,,posted,2.4,,engine
2026-09-01,A @ B,Batter Props,batter_bb,Other Guy BB o0.5,0.5,dk,180.0,,Pass,0.40,0.05,win,1.8,,,one_way_quote,0.40,0.36,0.40,,,,,posted,2.4,,engine
2026-09-01,A @ B,Pitcher Props,pitcher_outs,Arm Outs o16.5,16.5,dk,-110.0,-110.0,Strong buy,0.58,0.10,push,0.0,,,,0.58,0.50,0.56,,,,,posted,2.4,,engine
2026-09-01,A @ B,Pitcher Props,pitcher_bb,Arm BB u1.5,1.5,,,,Pass,0.55,,win,0.91,,,unpriced,0.55,,,,,,,posted,2.4,,engine
2026-08-01,A @ B,Batter Props,batter_h,Old Row H o0.5,0.5,dk,-125.0,-103.0,Pass,0.57,0.04,win,0.8,,,no_buy,0.59,0.5271,0.5779,,,,,posted,2.4,,engine
2026-09-01,A @ B,Totals,game_total,Under 9.5,9.5,dk,-110.0,-110.0,Pass,0.49,,win,0.91,,,,0.44,0.50,0.49,,,,,posted,2.3,,engine
"""


def _box() -> GameResult:
    return GameResult(
        game_pk=100,
        final=True,
        home_runs=4,
        away_runs=2,
        f5_home=2,
        f5_away=1,
        players={
            1: PlayerLine(batting={"PA": 4, "H": 2}),
            2: PlayerLine(batting={"PA": 4, "H": 1}),
            3: PlayerLine(batting={"PA": 4, "BB": 1}),
            4: PlayerLine(batting={"PA": 4, "BB": 0}),
            6: PlayerLine(batting={"PA": 4, "H": 1}),
            10: PlayerLine(pitching={"BF": 24, "outs": 18, "BB": 1}),
        },
    )


def _power_rows(tmp_path: Path) -> list[Row]:
    from mlb_engine.audit.power_ledger import load

    path = tmp_path / "power_screen_ledger.csv"
    path.write_text(POWER_LEDGER, encoding="utf-8")
    positions = last_run_per_day(load(path))
    graded, voided = grade_positions(positions, {100: _box()})
    assert voided == 1  # the scratch
    return rows_from_power(graded)


def _row(**kw: object) -> Row:
    base: dict[str, object] = dict(
        market="H",
        date="2026-09-01",
        side="over",
        line=0.5,
        odds=-110.0,
        model_prob=0.55,
        bet_prob=0.54,
        fair_prob=0.5,
        tier="Pass",
        gate="",
        result=WIN,
        units=0.91,
        one_way=False,
    )
    base.update(kw)
    return Row(**base)  # type: ignore[arg-type]


def test_power_rows_grade_and_keep_last_run(tmp_path: Path) -> None:
    rows = _power_rows(tmp_path)
    by_name = {(r.market, r.line_label): r for r in rows}
    assert len(rows) == 6  # eight recorded, one earlier run dropped, one scratch voided
    assert by_name[("H", "o0.5")].result == WIN and by_name[("H", "o0.5")].is_buy
    assert by_name[("H", "o1.5")].result == LOSS
    assert by_name[("BB", "o0.5")].one_way and by_name[("BB", "o0.5")].result == WIN
    assert by_name[("BB", "u0.5")].result == WIN
    assert by_name[("SP outs", "o15.5")].result == WIN
    assert by_name[("SP BB", "u1.5")].result == WIN
    assert not by_name[("H", "o0.5")].one_way
    assert by_name[("H", "o0.5")].edge == pytest.approx(0.07)


def test_engine_rows_keep_priced_props_in_range() -> None:
    rows = rows_from_engine_ledger(
        io.StringIO(ENGINE_LEDGER), start=Date(2026, 8, 18), end=Date(2026, 9, 30)
    )
    markets = sorted(r.market for r in rows)
    # the unpriced arm, the game total and the August row are left out
    assert markets == ["BB", "H", "H", "SP outs"]
    under = next(r for r in rows if r.side == "under")
    assert under.market == "H" and under.units == 1.5 and under.result == WIN
    one_way = next(r for r in rows if r.market == "BB")
    assert one_way.one_way and not one_way.two_sided
    push = next(r for r in rows if r.market == "SP outs")
    assert push.result == PUSH and not push.decided and push.is_buy and push.units == 0.0


def test_last_run_per_day_keeps_the_latest_run_only() -> None:
    def pos(day: str, run: str) -> Position:
        return Position(
            date=day, batter="x", player_id=1, game_pk=1, stat="H", line=0.5, side="over",
            book="", odds=None, model_prob=0.5, fair_prob=None, edge=None, ev=None,
            tier="Pass", rating="", devigged=False, run_id=run,
        )

    kept = last_run_per_day([pos("d1", "a"), pos("d1", "b"), pos("d2", "a")])
    assert [(p.date, p.run_id) for p in kept] == [("d1", "b"), ("d2", "a")]


def test_brier_and_log_loss() -> None:
    assert brier([], []) is None and log_loss([], []) is None
    assert brier([1.0, 0.0], [1, 0]) == 0.0
    assert brier([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    assert log_loss([0.5, 0.5], [1, 0]) == pytest.approx(0.6931, abs=1e-4)
    # a certain wrong call is clipped, not infinite
    clipped = log_loss([1.0], [0])
    assert clipped is not None and clipped < 20


def test_paired_bootstrap_verdicts() -> None:
    outcomes = [1, 0] * 50
    sharp = [0.9, 0.1] * 50
    dull = [0.5, 0.5] * 50
    ci = paired_bootstrap(sharp, dull, outcomes, resamples=500, seed=1)
    assert ci is not None and ci.hi < 0 and ci.verdict == "model beats market"
    assert ci.p_model_better == 1.0
    ci2 = paired_bootstrap(dull, sharp, outcomes, resamples=500, seed=1)
    assert ci2 is not None and ci2.lo > 0 and ci2.verdict == "market beats model"
    ci3 = paired_bootstrap(dull, dull, outcomes, resamples=500, seed=1)
    assert ci3 is not None and ci3.verdict == "indistinguishable"
    assert ci3.lo == 0.0 and ci3.hi == 0.0
    assert paired_bootstrap([0.5], [0.5], [1]) is None


def test_record_units_hit_rate_and_scored_rows() -> None:
    rows = [
        _row(result=WIN, units=0.91),
        _row(result=LOSS, units=-1.0),
        _row(result=PUSH, units=0.0),
        _row(result=WIN, units=1.5, one_way=True, fair_prob=None),
    ]
    rec = record("x", rows)
    assert (rec.n, rec.wins, rec.losses, rec.pushes) == (4, 2, 1, 1)
    assert rec.hit_rate == pytest.approx(2 / 3)
    assert rec.units == pytest.approx(1.41)
    assert rec.roi == pytest.approx(1.41 / 4)
    assert rec.scored == 2  # the push and the one-way row are not scored
    expected_model = ((0.55 - 1) ** 2 + (0.55 - 0) ** 2) / 2
    assert rec.brier_model == pytest.approx(expected_model)
    assert rec.brier_market == pytest.approx(0.25)
    assert rec.brier_gap == pytest.approx(expected_model - 0.25)


def test_price_bands() -> None:
    assert price_band(-150) == "<= -150"
    assert price_band(-200) == "<= -150"
    assert price_band(-149) == "-149..+149"
    assert price_band(100) == "-149..+149"
    assert price_band(150) == "+150..+299"
    assert price_band(299) == "+150..+299"
    assert price_band(300) == ">= +300"
    assert price_band(None) == "(no price)"


def test_edge_terciles_cut_marked_rows_lowest_first() -> None:
    rows = [_row(model_prob=0.5 + e / 100) for e in (-2, -1, 0, 1, 2, 3, 4, 5, 6)]
    rows.append(_row(fair_prob=None, one_way=True))
    cuts = edge_terciles(rows)
    assert [len(g) for _, g in cuts] == [3, 3, 3]
    assert cuts[0][0].startswith("low edge (-2.0..+0.0pts)")
    assert cuts[2][0].startswith("high edge (+4.0..+6.0pts)")


def test_selection_ladder_thins_by_edge() -> None:
    rows = [_row(model_prob=0.5 + e / 100, result=WIN if e > 1 else LOSS) for e in (-1, 0, 1, 2, 3)]
    rows.append(_row(one_way=True, fair_prob=None))
    ladder = selection_ladder(rows)
    assert [r.label for r in ladder] == [
        "every two-sided row, blind",
        "edge >= 0 pt",
        "edge >= 1 pt",
        "edge >= 2 pt",
        "edge >= 3 pt",
    ]
    assert [r.n for r in ladder] == [5, 4, 3, 2, 1]
    assert ladder[3].hit_rate == 1.0 and ladder[0].hit_rate == pytest.approx(0.4)


def test_calibration_bins_predicted_against_realised() -> None:
    rows = [
        _row(model_prob=0.31, result=WIN),
        _row(model_prob=0.39, result=LOSS),
        _row(model_prob=0.75, result=WIN),
        _row(model_prob=0.72, result=PUSH),
    ]
    bins = calibration(rows, lambda r: r.model_prob)
    assert [(b.label, b.n) for b in bins] == [("0.3-0.4", 2), ("0.7-0.8", 1)]
    assert bins[0].predicted == pytest.approx(0.35) and bins[0].realised == 0.5
    assert calibration(rows, lambda r: None) == []


def test_build_report_splits_and_summary(tmp_path: Path) -> None:
    rows = _power_rows(tmp_path)
    assert markets_present(rows) == list(FOCUS_MARKETS)
    rep = build_report("H", rows, resamples=200)
    assert rep.all_rows.n == 2 and rep.two_sided.scored == 2 and rep.one_way.n == 0
    assert rep.thin and rep.bootstrap is not None
    assert [r.label for r in rep.splits["tier"]] == ["Pass", "bought"]
    assert {r.label for r in rep.splits["line"]} == {"o0.5", "o1.5"}
    assert {r.label for r in rep.splits["side"]} == {"over"}
    assert [r.label for r in rep.splits["price band"]] == ["-149..+149", "+150..+299"]
    bb = build_report("BB", rows, resamples=200)
    assert bb.one_way.n == 1 and bb.two_sided.scored == 1
    assert bb.bootstrap is None and bb.verdict == "no two-sided rows"

    reports = [build_report(m, rows, resamples=200) for m in markets_present(rows)]
    summ = summary_rows(reports)
    assert [s["market"] for s in summ][: len(FOCUS_MARKETS)] == list(FOCUS_MARKETS)
    assert all(s["verdict"].endswith("(thin)") or "no two-sided" in s["verdict"] for s in summ)

    md = render_markdown(reports, "t")
    assert "## Four-market summary" in md and "| H |" in md and "### by price band" in md
    out = tmp_path / "out.csv"
    write_csv(reports, out)
    text = out.read_text().splitlines()
    assert text[0].startswith("market,section,label,n,")
    assert any(line.startswith("H,selection,edge >= 3 pt,") for line in text)
