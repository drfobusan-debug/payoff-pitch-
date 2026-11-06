"""Nightly goal level: fit to the market on earlier nights, capped, stamped and reported."""

from __future__ import annotations

import math
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import pytest

from nhl_engine import closetest, level
from nhl_engine.audit.ledger import LedgerRow, save_rows
from nhl_engine.audit.ledger_report import build, render_md
from nhl_engine.closetest import Obs
from nhl_engine.config import SimParams
from nhl_engine.level import MainTotal
from tests.nhl.test_level_anchor import LEAGUE

BASE = 1.035


def _row(night: str, matchup: str, line: float, sim: float, mkt: float, lvl: float | None):
    return LedgerRow(
        slate_date=night,
        matchup=matchup,
        home="BUF",
        away="BOS",
        market="game_total",
        side="over",
        entity="",
        line=line,
        ot_rule="incl_ot_so",
        book="draftkings",
        american=-110,
        books=8,
        consensus=mkt,
        model_prob=sim,
        push_prob=0.0,
        edge=0.0,
        ev=0.0,
        tier="Pass",
        goal_level=lvl,
    )


def _nights(n_nights: int, games: int, gap: float, lvl: float | None = None) -> list[MainTotal]:
    out = []
    for d in range(n_nights):
        night = (Date(2026, 10, 9) + timedelta(days=d)).isoformat()
        for g in range(games):
            out.append(MainTotal(night, f"G{g}", 6.5, 0.47, 0.47 + gap, lvl or BASE))
    return out


def test_main_totals_takes_the_line_nearest_a_coin_flip_and_defaults_the_level():
    rows = [
        _row("2026-10-10", "BOS @ BUF", 5.5, 0.6, 0.62, None),
        _row("2026-10-10", "BOS @ BUF", 6.5, 0.45, 0.49, None),
        _row("2026-10-10", "BOS @ BUF", 6.0, 0.5, 0.58, 1.05),
    ]
    (m,) = level.main_totals(rows, base=BASE)
    assert m.line == 6.5 and m.p_market == 0.49 and m.level == BASE


def test_fit_starts_from_the_fixed_level_and_steps_at_most_the_cap():
    obs = _nights(5, 10, 0.03)
    f = level.fit(obs, slate=Date(2026, 10, 14), base=BASE, sens=1.0)
    assert f.previous == BASE and f.n == 50
    assert f.gap == pytest.approx(0.03)
    assert f.target == pytest.approx(BASE * math.exp(0.03))
    assert f.level == pytest.approx(BASE * math.exp(level.MAX_STEP))
    assert "capped" in f.reason


def test_fit_moves_all_the_way_inside_the_cap_and_puts_rows_on_one_level():
    # rows priced 1% higher already read 1 pt higher; the gap left is 0.5 pt
    obs = _nights(5, 10, 0.005, lvl=BASE * math.exp(0.01))
    f = level.fit(obs, slate=Date(2026, 10, 14), base=BASE, sens=1.0)
    assert f.previous == pytest.approx(BASE * math.exp(0.01))
    assert f.level == pytest.approx(f.previous * math.exp(0.005)) and f.reason == ""


def test_fit_uses_only_earlier_nights_this_season_and_holds_when_thin():
    obs = _nights(5, 10, 0.03)
    assert level.fit(obs, slate=Date(2026, 10, 9), base=BASE, sens=1.0).n == 0
    early = level.fit(obs, slate=Date(2026, 10, 11), base=BASE, sens=1.0)
    assert early.n == 20 and early.level == BASE and "held" in early.reason
    later_season = level.fit(obs, slate=Date(2027, 10, 9), base=BASE, sens=1.0)
    assert later_season.n == 0 and later_season.level == BASE
    many = _nights(25, 2, 0.0)
    assert len(level.window(many, Date(2026, 11, 30))) == 2 * level.WINDOW_NIGHTS


def test_fit_never_drifts_past_the_bound():
    obs = _nights(5, 10, 0.5, lvl=BASE * math.exp(level.MAX_DRIFT))
    f = level.fit(obs, slate=Date(2026, 10, 14), base=BASE, sens=1.0)
    assert f.level == pytest.approx(BASE * math.exp(level.MAX_DRIFT))


def test_sensitivity_is_positive_and_plausible():
    s = level.sensitivity(LEAGUE, SimParams(goal_level=BASE, draws=5000), line=6.0)
    assert 0.5 < s < 1.5


def test_prior_rows_skip_versioned_copies_and_tonight(tmp_path: Path):
    led = tmp_path / "ledger"
    save_rows(
        [_row("2026-10-10", "A @ B", 6.5, 0.4, 0.5, None)], led / "predictions_2026-10-10.json"
    )
    save_rows(
        [_row("2026-10-10", "A @ B", 6.5, 0.4, 0.5, None)], led / "predictions_2026-10-10.v2.json"
    )
    save_rows(
        [_row("2026-10-11", "A @ B", 6.5, 0.4, 0.5, None)], led / "predictions_2026-10-11.json"
    )
    rows = level.prior_rows(tmp_path, Date(2026, 10, 11))
    assert [r.slate_date for r in rows] == ["2026-10-10"]


def test_audit_prints_the_level_per_night():
    rows = [
        _row("2026-10-10", "A @ B", 6.5, 0.45, 0.50, None),
        _row("2026-10-11", "A @ B", 6.5, 0.48, 0.50, 1.045),
    ]
    a = build(rows)
    assert [(n.night, round(n.level, 3)) for n in a.levels] == [
        ("2026-10-10", 1.035),
        ("2026-10-11", 1.045),
    ]
    md = render_md(a, as_of="2026-10-12")
    assert (
        "Goal level vs the market" in md
        and "| 2026-10-11 | 1.045 | 1 | 48.0% | 50.0% | -2.0 |" in md
    )


def _o(market: str, p_sim: float, p_close: float, result: float, d: int) -> Obs:
    return Obs(
        f"2024-10-{d:02d}", "BOS @ BUF", market, 6.5, p_sim, p_close, result, 6.0, 7, "c", 1.0
    )


def test_compare_keeps_the_fit_only_when_both_markets_are_no_worse(tmp_path: Path):
    fixed = [_o(m, 0.45, 0.5, d % 2, d) for m in ("game_total", "game_pl") for d in range(1, 21)]
    walk = [_o(m, 0.49, 0.5, d % 2, d) for m in ("game_total", "game_pl") for d in range(1, 21)]
    assert "nightly fit: keep" in closetest.compare(fixed, walk)
    worse_pl = [o if o.market == "game_total" else _o("game_pl", 0.4, 0.5, 1, 1) for o in walk]
    out = closetest.compare(fixed, worse_pl)
    assert "take it back out" in out and "stays zero" in out
    path = tmp_path / "w.csv"
    closetest.write_csv(walk, path)
    assert closetest.read_csv(path)[0].level == 1.0
