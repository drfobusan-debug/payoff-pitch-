"""Workbook generation and CLI wiring."""

from __future__ import annotations

from datetime import date

import pytest
from openpyxl import load_workbook

from cfb_engine.audit.grade import build_result_index
from cfb_engine.audit.ledger import daily_rollup, entries_from_graded, overall_metrics
from cfb_engine.audit.priced import engine_priced_stat
from cfb_engine.audit.probation import WATCHING, Probation, market_probation
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


def test_a_lean_the_line_has_come_to_is_bold_with_a_legend(tmp_path):
    agrees = _rec(Tier.PASS, "game_ats")
    agrees.drift = 0.03
    flat = _rec(Tier.PASS, "game_total")
    flat.drift = 0.0
    out = write_workbook([agrees, flat], tmp_path / "cfb.xlsx", DAY)
    ws = load_workbook(out)["All"]
    market = [c.value for c in ws[1]].index("Market")
    bold = [
        ws.cell(row=r, column=market + 1).value for r in (2, 3) if ws.cell(row=r, column=1).font.bold
    ]
    assert bold == [agrees.display_category]
    assert str(ws.cell(row=5, column=1).value).startswith("Bold: the model leans")


def test_the_audit_article_lists_each_graded_only_rule_before_its_verdict():
    """A WATCHING rule still prints its record, so the PDF shows what is being tracked."""
    rule = Probation(
        "ats_upgrade_top25_game", "upgrade", 38, 0.432, 0.141, 0.629, 0.235, WATCHING, ""
    )
    html, _ = build_audit_article(DAY, [], [], 0, tracked=[rule])
    assert "Nothing has crossed the volume" in html
    assert "ats_upgrade_top25_game</td><td>WATCHING</td><td>38</td>" in html
    assert "+43.2%" in html and "+62.9%" in html and "+23.5%" in html
    bare, _ = build_audit_article(DAY, [], [], 0)
    assert "Graded-only rule" not in bare


def _audit_fixture(tmp_path, monkeypatch, slates):
    """A data dir with saved predictions for ``slates`` and a ledger graded through Oct 9."""
    from dataclasses import replace

    from cfb_engine import cli
    from cfb_engine.audit.ledger import update_ledger
    from cfb_engine.config import load_config
    from cfb_engine.data.cfbd import GameResult

    monkeypatch.setenv("CFBE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CFBE_STATE_SYNC", "0")
    cfg = load_config()
    cfg.audit_dir.mkdir(parents=True, exist_ok=True)
    for d in slates:
        save_json([replace(_rec(Tier.STRONG), game_date=d)], cfg.predictions_file(d))
    oct9 = date(2026, 10, 9)
    update_ledger(cfg.ledger_file, entries_from_graded([(_rec(Tier.STRONG), "win")], oct9), oct9)
    final = GameResult(home="Georgia", away="Alabama", home_points=27, away_points=20)
    monkeypatch.setattr(cli, "_today", lambda: date(2026, 10, 14))
    monkeypatch.setattr(cli, "_result_index", lambda c, d: build_result_index([final]))
    monkeypatch.setattr(cli, "_grade_podcasts", lambda c, d, i: 0)
    emitted: list[date] = []
    monkeypatch.setattr(cli, "_emit_ledger", lambda c, e, d, **kw: emitted.append(d))
    return cfg, emitted


def test_the_scheduled_audit_catches_up_a_slate_a_missed_night_left_ungraded(tmp_path, monkeypatch):
    from cfb_engine.audit.ledger import load_ledger

    oct10, oct13, stale = date(2026, 10, 10), date(2026, 10, 13), date(2026, 9, 1)
    cfg, emitted = _audit_fixture(tmp_path, monkeypatch, [stale, oct10, oct13])
    assert main(["audit", "--no-email"]) == 0
    dates = sorted({e.date for e in load_ledger(cfg.ledger_file)})
    assert dates == ["2026-10-09", "2026-10-10", "2026-10-13"]
    assert emitted == [oct13]


def test_an_audit_with_a_date_grades_only_that_slate(tmp_path, monkeypatch):
    from cfb_engine.audit.ledger import load_ledger

    oct10, oct13 = date(2026, 10, 10), date(2026, 10, 13)
    cfg, emitted = _audit_fixture(tmp_path, monkeypatch, [oct10, oct13])
    assert main(["audit", "--date", "2026-10-13", "--no-email"]) == 0
    assert sorted({e.date for e in load_ledger(cfg.ledger_file)}) == ["2026-10-09", "2026-10-13"]
    assert emitted == [oct13]


def test_a_missed_slate_is_caught_up_even_when_yesterday_had_no_card(tmp_path, monkeypatch):
    from cfb_engine.audit.ledger import load_ledger

    oct10 = date(2026, 10, 10)
    cfg, emitted = _audit_fixture(tmp_path, monkeypatch, [oct10])
    assert main(["audit", "--no-email"]) == 0
    assert "2026-10-10" in {e.date for e in load_ledger(cfg.ledger_file)}
    assert emitted == [oct10]
