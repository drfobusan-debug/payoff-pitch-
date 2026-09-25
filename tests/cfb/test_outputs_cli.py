"""Workbook generation and CLI wiring."""

from __future__ import annotations

from datetime import date

import pytest
from openpyxl import load_workbook

from cfb_engine.audit.ledger import daily_rollup, entries_from_graded, overall_metrics
from cfb_engine.audit.priced import engine_priced_stat
from cfb_engine.audit.probation import market_probation
from cfb_engine.cli import _build_parser, main
from cfb_engine.market.tiers import Tier
from cfb_engine.output.audit_report import build_audit_article
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


def test_money_and_probation_reach_the_workbook(tmp_path):
    graded = [(_rec(Tier.STRONG), "win"), (_rec(Tier.MODERATE), "loss")]
    entries = entries_from_graded(graded, DAY)
    out = write_ledger_workbook(
        entries,
        overall_metrics(entries),
        daily_rollup(entries),
        tmp_path / "ledger.xlsx",
        money_rows=[engine_priced_stat(entries)],
        probation_rows=market_probation(entries),
    )
    wb = load_workbook(out)
    assert "Money (priced buys)" in wb.sheetnames
    assert "Probation" in wb.sheetnames
    money = wb["Money (priced buys)"]
    header = [c.value for c in money[1]]
    assert "Needs" in header and "ROI" in header
    assert money.cell(row=2, column=header.index("N") + 1).value == 2
    verdict = wb["Probation"]
    assert verdict.cell(row=2, column=1).value == "WATCHING"


def test_the_audit_article_says_what_the_prices_demanded():
    """A reader who only hears the win rate cannot tell a winner from a loser."""
    graded = [(_rec(Tier.STRONG), "win"), (_rec(Tier.MODERATE), "loss")]
    entries = entries_from_graded(graded, DAY)
    stat = engine_priced_stat(entries)
    html, narration = build_audit_article(
        DAY,
        overall_metrics(entries),
        [],
        2,
        money_rows=[stat],
        probation=["game_ml: shut game_ml until the refit"],
    )
    assert "What the prices did" in html
    assert "shut game_ml until the refit" in html
    assert "where they needed" in narration


def test_the_audit_lead_keeps_the_slate_apart_from_the_ledger():
    """Cumulative buys must not be read out as what tonight's slate did."""
    graded = [(_rec(Tier.STRONG), "win"), (_rec(Tier.MODERATE), "loss")]
    ledger = entries_from_graded(graded, date(2025, 10, 25)) + entries_from_graded(graded, DAY)
    slate = [e for e in ledger if e.date == DAY.isoformat()]
    slate[1].result, slate[1].pnl = "win", 0.9
    slate_buy = next(m for m in overall_metrics(slate) if m.tier == "Buy (S+M)")
    html, narration = build_audit_article(DAY, overall_metrics(ledger), [], 2, slate_buy=slate_buy)
    assert "the slate's buys went <b>2-0</b>" in html
    assert "Ledger to date: buys <b>3-1</b>" in html
    assert "The slate's buys went 2 and 0" in narration
    assert "Ledger to date, the model's buys are 3 and 1" in narration

    html, narration = build_audit_article(DAY, overall_metrics(ledger), [], 0)
    assert "slate's buys" not in html and "Ledger to date: buys <b>3-1</b>" in html
    assert "No plays cleared" not in narration


def test_the_audit_lead_counts_a_pushed_buy_as_placed():
    graded = [(_rec(Tier.STRONG), "push"), (_rec(Tier.MODERATE), "push")]
    slate = entries_from_graded(graded, DAY)
    slate_buy = next(m for m in overall_metrics(slate) if m.tier == "Buy (S+M)")
    html, narration = build_audit_article(DAY, overall_metrics(slate), [], 2, slate_buy=slate_buy)
    assert "the slate's buys went <b>0-0-2</b>" in html
    assert "no buys cleared" not in html
    assert "0 and 0 and 2 pushed" in narration


def test_the_audit_article_says_so_when_nothing_crossed_the_bar():
    html, _ = build_audit_article(DAY, [], [], 0)
    assert "no market or screen is on probation" in html


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
    for cmd in ("card", "close", "audit", "report", "calibrate", "probation"):
        assert parser.parse_args([cmd]).command == cmd
    assert parser.parse_args(["probation", "--since", "2025-09-01"]).since == "2025-09-01"


def test_the_scheduled_audit_grades_yesterday_not_today(monkeypatch):
    """At 03:00 with no --date, today's slate has no predictions; yesterday's does."""
    from datetime import date

    from cfb_engine import cli

    monkeypatch.setattr(cli, "_today", lambda: date(2026, 9, 12))
    args = _build_parser().parse_args(["audit"])
    assert cli._audit_day(args) == date(2026, 9, 11)
    assert cli._day(args) == date(2026, 9, 12)
    explicit = _build_parser().parse_args(["audit", "--date", "2026-09-04"])
    assert cli._audit_day(explicit) == date(2026, 9, 4)


def test_open_baselines_the_week_ahead_without_overwriting_an_earlier_board(tmp_path, monkeypatch):
    """``cfb-engine open`` seeds the first-seen board for future slates, first quote wins."""
    from datetime import date

    from cfb_engine import cli
    from cfb_engine.audit import snapshot
    from cfb_engine.config import Config
    from cfb_engine.data.oddsapi import OddsAPIClient
    from cfb_engine.schemas import Game, Slate, TeamGameInfo

    cfg = Config(data_dir=tmp_path, state_sync=False)
    saturday = date(2026, 9, 19)
    cfg.audit_dir.mkdir(parents=True)
    snapshot.save({"A@B|game_ats|B": snapshot.SideQuote(-110, 0.5, -6.5)}, cfg.board_file(saturday))

    def fake_fetch(self, day):
        if day != saturday:
            return Slate(slate_date=day), {}
        game = Game(
            game_id="g1",
            game_date=day,
            home=TeamGameInfo(name="B", abbrev="B", is_home=True),
            away=TeamGameInfo(name="A", abbrev="A", is_home=False),
        )
        return Slate(slate_date=day, games=[game]), {}

    monkeypatch.setattr(OddsAPIClient, "fetch_board", fake_fetch)
    monkeypatch.setattr(
        snapshot,
        "board_quotes",
        lambda slate, board: {
            "A@B|game_ats|B": snapshot.SideQuote(-115, 0.52, -7.5),
            "A@B|game_ats|A": snapshot.SideQuote(-105, 0.48, 7.5),
        },
    )
    args = _build_parser().parse_args(["open", "--date", "2026-09-13", "--days", "7"])
    assert cli.cmd_open(cfg, args) == 0

    board = snapshot.load(cfg.board_file(saturday))
    assert board["A@B|game_ats|B"].line == -6.5  # earliest quote kept
    assert board["A@B|game_ats|A"].line == 7.5  # newly posted side added
    assert not cfg.board_file(date(2026, 9, 14)).exists()
