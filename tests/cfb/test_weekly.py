"""Weekly audit: the week and the ledger, each with its sample size attached."""

from __future__ import annotations

from datetime import date

import pytest

from cfb_engine import cli
from cfb_engine.audit.ledger import LedgerEntry, update_ledger
from cfb_engine.audit.weekly import game_briers, segment, weekly_audit
from cfb_engine.output.weekly_report import build_weekly_article

END = date(2026, 10, 3)


def _e(
    day: str,
    result: str,
    *,
    market: str = "game_ats",
    category: str = "Spread (ATS)",
    selection: str = "Troy -3.5",
    line: float | None = -3.5,
    odds: float = -110,
    tier: str = "Strong buy",
    model_prob: float = 0.55,
    fair_prob: float | None = 0.5,
    close_prob: float | None = None,
    clv: float | None = None,
    clv_ev: float | None = None,
    ev: float | None = 0.05,
) -> LedgerEntry:
    pnl = {"win": round(100 / 110, 4), "loss": -1.0, "push": 0.0}[result]
    return LedgerEntry(
        date=day,
        matchup="A @ B",
        category=category,
        market=market,
        selection=selection,
        line=line,
        book="b",
        odds=odds,
        tier=tier,
        model_prob=model_prob,
        ev=ev,
        result=result,
        pnl=pnl,
        fair_prob=fair_prob,
        close_prob=close_prob,
        clv=clv,
        clv_ev=clv_ev,
    )


def test_segment_reports_record_roi_range_and_breakeven():
    rows = [_e("2026-10-01", "win"), _e("2026-10-01", "loss"), _e("2026-10-01", "push")]
    s = segment("x", rows)
    assert (s.n, s.wins, s.losses, s.pushes, s.record) == (2, 1, 1, 1, "1-1-1")
    assert s.units == pytest.approx(100 / 110 - 1, abs=1e-3)
    assert s.roi_lo < s.roi < s.roi_hi
    assert s.breakeven == pytest.approx(110 / 210, abs=1e-4)
    assert s.verdict.startswith("underpowered")


def test_a_big_consistent_edge_is_called_and_a_small_one_is_not():
    big = segment("big", [_e("2026-10-01", "win")] * 300 + [_e("2026-10-01", "loss")] * 200)
    assert big.roi_lo > 0 and big.verdict == "winning beyond noise"
    small = segment("small", [_e("2026-10-01", "win")] * 6 + [_e("2026-10-01", "loss")] * 4)
    assert small.roi > 0 and small.roi_lo < 0
    assert small.need is not None and small.need > small.n
    assert f"{small.need:,}" in small.verdict


def test_week_window_is_the_seven_slates_ending_the_date_and_the_ledger_stops_there():
    entries = [
        _e("2026-09-26", "win"),  # previous week
        _e("2026-09-27", "loss"),  # first day of the week
        _e("2026-10-03", "win"),  # last day
        _e("2026-10-06", "win"),  # after the report date
    ]
    rep = weekly_audit(entries, END)
    assert rep.start == date(2026, 9, 27)
    assert rep.week.slates == ["2026-09-27", "2026-10-03"]
    assert rep.week.buys.record == "1-1"
    assert rep.ledger.slates == ["2026-09-26", "2026-09-27", "2026-10-03"]
    assert rep.ledger.buys.record == "2-1"
    assert [p.label for p in rep.week.periods] == ["2026-09-27", "2026-10-03"]
    assert [p.label for p in rep.ledger.periods] == ["Sep 20-Sep 26", "Sep 27-Oct 3"]


def test_pass_rows_are_graded_but_kept_out_of_the_buys():
    entries = [_e("2026-10-01", "win"), _e("2026-10-01", "loss", tier="Pass")]
    rep = weekly_audit(entries, END)
    assert rep.week.buys.record == "1-0"
    assert rep.week.passes.record == "0-1"


def test_splits_separate_sides_favorites_and_dogs():
    entries = [
        _e(
            "2026-10-01",
            "loss",
            market="game_total",
            category="Totals",
            selection="Under 50.0",
            line=50,
        ),
        _e(
            "2026-10-01",
            "win",
            market="game_total",
            category="Totals",
            selection="Over 50.0",
            line=50,
        ),
        _e("2026-10-01", "win", selection="USM +10.5", line=10.5),
        _e(
            "2026-10-01",
            "win",
            market="game_ml",
            category="Moneyline",
            selection="Troy ML",
            line=None,
            odds=-300,
        ),
    ]
    splits = {s.label: s.record for s in weekly_audit(entries, END).week.splits}
    assert splits["Totals: Under"] == "0-1"
    assert splits["Totals: Over"] == "1-0"
    assert splits["ATS: underdog"] == "1-0"
    assert splits["ATS spread 8-14.5"] == "1-0"
    assert splits["ML: favorite"] == "1-0"
    assert "ATS: favorite" not in splits


def test_briers_score_each_game_once_on_the_favorite_or_the_over():
    rows = [
        # ML: favorite won; the dog row of the same game is ignored.
        _e(
            "2026-10-01",
            "win",
            market="game_ml",
            selection="A ML",
            line=None,
            model_prob=0.6,
            fair_prob=0.7,
            close_prob=0.8,
        ),
        _e(
            "2026-10-01",
            "loss",
            market="game_ml",
            selection="B ML",
            line=None,
            model_prob=0.3,
            fair_prob=0.3,
            close_prob=0.2,
        ),
        # Spread: only the dog row exists, so the favorite is read off it, flipped.
        _e("2026-10-01", "win", selection="B +3.5", line=3.5, model_prob=0.6, fair_prob=0.5),
        # Total: the over lost.
        _e(
            "2026-10-01",
            "loss",
            market="game_total",
            selection="Over 50.0",
            line=50.0,
            model_prob=0.4,
            fair_prob=0.5,
        ),
        _e(
            "2026-10-01",
            "win",
            market="game_total",
            selection="Under 50.0",
            line=50.0,
            model_prob=0.6,
            fair_prob=0.5,
        ),
    ]
    ml, ats, tot = game_briers(rows)
    assert ml.n == 1 and ml.model == pytest.approx(0.16) and ml.market == pytest.approx(0.09)
    assert ml.n_close == 1 and ml.close == pytest.approx(0.04)
    assert ats.n == 1 and ats.model == pytest.approx(0.16) and ats.market == pytest.approx(0.25)
    assert ats.n_close == 0
    assert tot.n == 1 and tot.model == pytest.approx(0.16) and tot.market == pytest.approx(0.25)


def test_close_ev_is_read_against_the_juice_the_prices_carried_at_bet_time():
    rows = [_e("2026-10-01", "win", fair_prob=0.5, clv=0.01, clv_ev=-0.02)] * 3
    s = segment("x", rows)
    assert s.bet_ev == pytest.approx(0.5 * (1 + 100 / 110) - 1)
    assert s.close_ev == pytest.approx(-0.02)
    assert s.close_ev - s.bet_ev > 0


def test_slates_that_move_together_widen_the_range():
    spread = [_e(d, r) for d in ("2026-09-01", "2026-09-02") for r in ("win", "loss")] * 5
    lumped = [_e("2026-09-01", "win")] * 10 + [_e("2026-09-02", "loss")] * 10
    a, b = segment("spread", spread), segment("lumped", lumped)
    assert a.roi == pytest.approx(b.roi)
    assert b.roi_hi - b.roi_lo > a.roi_hi - a.roi_lo


def test_findings_say_when_the_market_is_better_and_the_profit_outruns_the_close():
    rows = [
        _e(
            "2026-10-01",
            r,
            model_prob=0.55,
            fair_prob=0.6 if r == "win" else 0.4,
            clv=0.01,
            clv_ev=-0.02,
        )
        for r in ["win"] * 7 + ["loss"] * 4
    ]
    text = " ".join(weekly_audit(rows, END).week.findings)
    assert "not proven" in text
    assert "bet-time market was more accurate than the model" in text
    assert "likely variance" in text
    assert text.index("closing price") < text.index("Brier") < text.index("Buys are")


def test_article_has_both_halves_with_the_same_sections():
    rows = [_e("2026-09-20", "win"), _e("2026-10-01", "loss")]
    html = build_weekly_article(weekly_audit(rows, END))
    assert "This week (Sep 27-Oct 3)" in html and "Ledger to date" in html
    for section in ("Summary", "By market", "Price at the close", "Forecast accuracy (Brier)"):
        assert html.count(f"<h3>{section}</h3>") == 2
    assert (
        html.index("Price at the close")
        < html.index("Forecast accuracy")
        < html.index("<h3>Buys</h3>")
    )
    assert "By slate" in html and "By week" in html


def test_weekly_command_writes_the_article_without_email(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CFBE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CFBE_STATE_SYNC", "0")
    monkeypatch.setattr(cli, "generate_weekly_report", _fake_report(tmp_path))
    cfg = cli.load_config()
    update_ledger(cfg.ledger_file, [_e("2026-10-01", "win"), _e("2026-10-01", "loss")], END)
    assert cli.main(["weekly", "--date", END.isoformat(), "--no-email"]) == 0
    assert (tmp_path / "weekly.html").exists()
    assert "Buys are 1-1" in capsys.readouterr().out


def test_weekly_command_needs_a_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("CFBE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CFBE_STATE_SYNC", "0")
    assert cli.main(["weekly", "--date", END.isoformat(), "--no-email"]) == 1


def _fake_report(tmp_path):
    def fake(report, cfg, *, email, to):
        assert email is False
        path = tmp_path / "weekly.html"
        path.write_text(build_weekly_article(report))
        return {"html": path, "pdf": None}

    return fake


def test_a_short_perfect_run_is_not_called_beyond_noise():
    s = segment("hot", [_e("2026-10-01", "win")] * 7)
    assert s.roi_hi - s.roi_lo > 1.0
    assert s.verdict.startswith("underpowered")
