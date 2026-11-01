"""Margin shape: fit, pricing of spreads / moneylines / first halves, first-half grading, spread CLV."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from cbb_engine import audit, card
from cbb_engine.market import consensus
from cbb_engine.models import margin as mg
from engine_runtime import ledger, quotes
from engine_runtime.quotes import Quote
from scripts.cbb import fit_margin

DAY = "2026-12-10"
NOW = datetime(2026, 12, 10, 18, 5, tzinfo=timezone.utc)
MW = mg.MarginWidth(
    sd_by_phase={"all": 11.0, "dec": 11.0}, key_ratio={1: 0.8, 4: 1.2}, ml_scale=0.95,
    h1_margin_share=0.55, h1_margin_sd=9.0, h1_total_a=-3.7, h1_total_b=0.5, h1_total_sd=9.8,
)


def _rows(n: int, seed: int = 1) -> list[dict[str, str]]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        spread = rng.choice([-12.5, -6.5, -3.5, 2.5, 7.5])
        m = 0
        while m == 0:
            m = round(rng.gauss(-spread, 11))
        h1 = round(rng.gauss(-0.55 * spread, 9))
        total = rng.choice([130.0, 140.0, 150.0])
        h1t = round(rng.gauss(0.5 * total - 4, 9.5))
        out.append({
            "game_date": f"2025-0{1 + i % 3}-15", "home_spread": str(spread), "close_total": str(total),
            "home_pts": str(70 + m), "away_pts": "70", "h1_home": str((h1t + h1) // 2), "h1_away": str((h1t - h1) // 2),
        })
    return out


def test_fit_recovers_width_share_and_half_total() -> None:
    f = fit_margin.fit(_rows(4000))
    assert abs(f["sd_by_phase"]["all"] - 11) < 0.5  # type: ignore[index]
    assert abs(float(f["h1_margin_share"]) - 0.55) < 0.05  # type: ignore[arg-type]
    assert abs(float(f["h1_total_b"]) - 0.5) < 0.1  # type: ignore[arg-type]
    assert abs(float(f["h1_total_a"]) + 0.5 * 140 - float(f["h1_total_b"]) * 140 - (-4.0)) < 1.5  # type: ignore[arg-type]
    assert abs(float(f["h1_total_sd"]) - 9.5) < 0.6  # type: ignore[arg-type]


def test_key_ratio_fires_on_a_spike_only() -> None:
    base = _rows(4000)
    assert fit_margin.fit(base)["key_ratio"] == {}
    spiked = base + [{**r, "home_pts": str(int(r["away_pts"]) + 4), "home_spread": "-3.5"} for r in base[:300]]
    assert "4" in fit_margin.fit(spiked)["key_ratio"]  # type: ignore[operator]


def test_full_game_has_no_ties_half_does() -> None:
    assert 0 not in mg.pmf(0.0, 11.0, {}, tie=False)
    assert mg.pmf(0.0, 9.0, {}, tie=True)[0] > 0


def test_solve_mu_inverts_the_cover_price() -> None:
    for side, line, p in (("home", -3.5, 0.52), ("away", 7.5, 0.47)):
        mu = mg.solve_mu(side, line, p, 11.0, MW.key_ratio)
        assert abs((mg.p_side(mg.pmf(mu, 11.0, MW.key_ratio, tie=False), side, line) or 0) - p) < 1e-4


def test_price_moneyline_from_spread_and_first_half() -> None:
    p_home = mg.price("game_ml", "home", None, home_mu=7.0, full_total=None, day=DAY, mw=MW)
    p_away = mg.price("game_ml", "away", None, home_mu=7.0, full_total=None, day=DAY, mw=MW)
    assert p_home is not None and p_away is not None and abs(p_home + p_away - 1) < 1e-9 and p_home > 0.7
    h1 = mg.price("h1_ml", "home", None, home_mu=7.0, full_total=None, day=DAY, mw=MW)
    assert h1 is not None and 0.5 < h1 < p_home
    over = mg.price("h1_total", "over", 66.5, home_mu=None, full_total=140.0, day=DAY, mw=MW)
    assert over is not None and abs(over - 0.5) < 0.05
    assert mg.price("game_ats", "home", -3.5, home_mu=None, full_total=None, day=DAY, mw=MW) is None


def _q(book: str, market: str, side: str, line: float | None, am: float, opp: float, pid: str = "b1", kind: str = "board") -> Quote:
    return Quote(game_date=DAY, event_id="e1", tip_utc=f"{DAY}T23:00:00Z", market=market, side=side, line=line, book=book,
                 entity="", american=am, opposite_american=opp, captured_at=f"{DAY}T18:00:00Z", pass_id=pid, pass_kind=kind)


def _board(pid: str = "b1", kind: str = "board") -> list[Quote]:
    rows = []
    for b in ("pinnacle", "fanduel", "caesars", "betrivers", "draftkings"):
        rows += [_q(b, "game_ats", "home", -6.5, -110, -110, pid, kind), _q(b, "game_ats", "away", 6.5, -110, -110, pid, kind)]
        rows += [_q(b, "game_ml", "home", None, -280, 230, pid, kind), _q(b, "game_ml", "away", None, 230, -280, pid, kind)]
        rows += [_q(b, "game_total", "over", 145.5, -110, -110, pid, kind), _q(b, "game_total", "under", 145.5, -110, -110, pid, kind)]
    return rows


def _write(root: Path, rows: list[Quote], pid: str = "b1", kind: str = "board") -> None:
    quotes.write_shard(root, "cbb", DAY, rows, pass_id=pid, pass_kind=kind)


def test_card_prices_research_sides_and_leaves_live_buys_alone(tmp_path: Path) -> None:
    from cbb_engine.models import distribution as dist

    width = dist.Width(sd_by_bucket={"all": 16.0})
    _write(tmp_path, _board())
    plain = card.price(tmp_path, DAY, now=NOW, pass_id="c0", pass_kind="featured", width=width)
    priced = card.price(tmp_path, DAY, now=NOW, pass_id="c1", pass_kind="featured", width=width, margin=MW)
    assert plain is not None and priced is not None
    sides = [r for r in priced.rows if r.market in ("game_ats", "game_ml")]
    assert sides and all(r.p_model is not None and r.status == "research" for r in sides)
    ml = next(r for r in sides if r.market == "game_ml" and r.side == "home")
    assert 0.7 < (ml.p_model or 0) < 0.85
    ats = next(r for r in sides if r.market == "game_ats" and r.side == "home")
    assert abs((ats.p_model or 0) - (ats.p_fair or 0)) < 0.01
    assert all(r.p_model is None for r in plain.rows if r.market != "game_total")
    live = lambda p: [(r.side, r.p_model, r.tier) for r in p.rows if r.status == "live"]  # noqa: E731
    assert live(plain) == live(priced)


def test_first_half_grading_and_tie_push() -> None:
    row = ledger.Row(slate_date=DAY, event_id="e1", tip_utc="", market="h1_ats", side="home", line=-2.5, book="draftkings",
                     pass_id="p", pass_kind="featured", bet_of_record=True, status="shadow")
    final, h1 = (60, 75), (30, 34)
    assert audit.outcome(row, final, h1) == "win"
    assert audit.outcome(ledger.Row(**{**row.__dict__, "market": "h1_ml", "line": None}), final, (32, 32)) == "push"
    assert audit.outcome(ledger.Row(**{**row.__dict__, "market": "h1_total", "side": "under", "line": 64.5}), final, h1) == "win"
    assert audit.outcome(row, final) == ""


def test_spread_close_fair_reads_a_line_move() -> None:
    close = consensus.build([q for q in _board("cl", "close") if q.book != "draftkings"], shard="cl")
    row = ledger.Row(slate_date=DAY, event_id="e1", tip_utc="", market="game_ats", side="home", line=-5.5, book="draftkings",
                     pass_id="p", pass_kind="featured", bet_of_record=True, status="research")
    assert audit._close_fair(row, close, None) is None  # no close at -5.5 without the margin shape
    p = audit._close_fair(row, close, None, MW)
    assert p is not None and p > 0.52  # home -5.5 is better than the -6.5 close


def test_margin_json_loads() -> None:
    mw = mg.load()
    assert mw is not None and mw.sd(DAY) and mw.h1_margin_sd > 0
    raw = json.loads(mg.PARAMS.read_text())
    assert raw["n"] >= 5000
