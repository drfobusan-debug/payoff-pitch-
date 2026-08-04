"""Workbook generation and CLI wiring."""

from __future__ import annotations

from datetime import date

import pytest
from openpyxl import load_workbook

from cfb_engine.audit.ledger import daily_rollup, entries_from_graded, overall_metrics
from cfb_engine.cli import _build_parser, main
from cfb_engine.market.tiers import Tier
from cfb_engine.output.excel import write_ledger_workbook, write_workbook
from cfb_engine.recommendations import Recommendation, load_json, save_json

DAY = date(2025, 11, 1)


def _rec(tier: Tier, market: str = "game_ml") -> Recommendation:
    return Recommendation(
        game_date=DAY,
        game_id="g1",
        matchup="Alabama vs Georgia",
        market=market,
        selection="Georgia ML",
        model_prob=0.6,
        market_american=-120,
        ev=0.05,
        edge=0.04,
        fair_prob=0.55,
        tier=tier,
        home_abbrev="Georgia",
        away_abbrev="Alabama",
        team_side="home",
        side="win",
    )


def test_write_workbook_has_expected_tabs(tmp_path):
    recs = [_rec(Tier.STRONG), _rec(Tier.PASS, "game_total")]
    out = write_workbook(recs, tmp_path / "cfb.xlsx", DAY)
    wb = load_workbook(out)
    for tab in ("Strong Buys", "Moderate Buys", "Fades", "All"):
        assert tab in wb.sheetnames


def test_write_ledger_workbook(tmp_path):
    graded = [(_rec(Tier.STRONG), "win"), (_rec(Tier.MODERATE), "loss")]
    entries = entries_from_graded(graded, DAY)
    out = write_ledger_workbook(
        entries, overall_metrics(entries), daily_rollup(entries), tmp_path / "ledger.xlsx"
    )
    wb = load_workbook(out)
    assert "Overall" in wb.sheetnames
    assert "Bets" in wb.sheetnames


def test_predictions_json_roundtrip(tmp_path):
    recs = [_rec(Tier.STRONG), _rec(Tier.MODERATE, "game_ats")]
    path = tmp_path / "preds.json"
    save_json(recs, path)
    back = load_json(path)
    assert len(back) == 2
    assert back[0].tier == Tier.STRONG
    assert back[0].game_date == DAY


def test_cli_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_cli_parser_has_all_commands():
    parser = _build_parser()
    ns = parser.parse_args(["run"])
    assert ns.command == "run"
    for cmd in ("card", "close", "audit", "report", "calibrate"):
        assert parser.parse_args([cmd]).command == cmd
