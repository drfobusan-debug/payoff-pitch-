"""Phase 2c: the shadow CLI reads only the disk and writes only its own ledgers."""

from __future__ import annotations

import csv
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlb_engine.audit import histograms as hg
from mlb_engine.audit import shadow, tape
from mlb_engine.models.montecarlo import GameSimResult
from mlb_engine.recommendations import Recommendation, save_json

D = date(2026, 10, 11)
T1 = datetime(2026, 10, 11, 23, 10, tzinfo=timezone.utc)
N = 6000


def _res() -> GameSimResult:
    rng = np.random.default_rng(5)
    def pois(lam: float, shape: int | tuple[int, int] = N) -> np.ndarray:
        return rng.poisson(lam, shape).astype(np.int16)
    bat = {t: {s: pois(0.35, (N, 9)) for s in ("H", "1B", "2B", "3B", "HR", "BB", "K", "R", "RBI")}
           for t in ("home", "away")}
    pit = {t: {s: pois(4.0) for s in ("outs", "H", "BB", "K", "ER")} for t in ("home", "away")}
    return GameSimResult(N, pois(4.7), pois(4.3), pois(2.4), pois(2.2), bat, pit)


def _game():
    def team(base: int):
        lineup = [SimpleNamespace(player=SimpleNamespace(mlbam_id=base + i)) for i in range(9)]
        return SimpleNamespace(lineup=lineup, probable_pitcher=SimpleNamespace(mlbam_id=base + 99))
    g = SimpleNamespace(game_pk=1, game_datetime_utc="2026-10-11T23:10:00Z",
                        home=team(100), away=team(200))
    g.matchup = lambda: "LAD @ MIL"
    return g


def _recs(p: float = 0.55) -> list[Recommendation]:
    return [
        Recommendation(game_date=D, game_pk=1, matchup="LAD @ MIL", category="game",
                       market="game_total", selection="Over 8.5", model_prob=p),
        Recommendation(game_date=D, game_pk=1, matchup="LAD @ MIL", category="batter",
                       market="batter_h", selection="José Ramírez H o0.5", model_prob=0.62,
                       player_id=103, stat="H"),
    ]


def _cap(market: str, sel: str, book: str, am: float, opp: float) -> tape.QuoteCapture:
    return tape.QuoteCapture("evA", T1, "LAD @ MIL", market, sel, book, am, opp)


def _setup(d: Path) -> None:
    recs = _recs()
    save_json(recs, d / f"predictions_{D.isoformat()}.json")
    hg.save_histograms(d, D, {1: hg.build_game_hists(_res(), _game())}, recs, [_game()], {1: "evA"})  # type: ignore[arg-type,list-item]
    caps = [
        _cap("game_total", "Over 8.5", "dk", -110, -110),
        _cap("game_total", "Over 8.5", "fd", -105, -115),
        _cap("game_ml", "MIL ML", "dk", -130, 110),
        _cap("game_ml", "LAD ML", "dk", 110, -130),
        _cap("game_rl", "MIL -1.5", "dk", 140, -165),
        _cap("batter_h", "Jose Ramirez H o0.5", "dk", -190, 150),
        _cap("batter_h", "Jose Ramirez H o0.5", "mgm", -185, 145),
        _cap("batter_h", "Jose Ramirez H o0.5", "fd", -180, 140),  # same line: no cross-book row
        _cap("batter_h", "Jose Ramirez H o1.5", "fd", 230, -300),  # different line: one row
    ]
    tape.append_tape(d, D, caps, tape.RUN, now=T1 - timedelta(hours=3), game_pks={"evA": 1})


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    import smtplib

    from mlb_engine.data import oddsapi
    def boom(*a: object, **k: object) -> None:
        raise AssertionError("the shadow must not touch the network")
    monkeypatch.setattr(oddsapi.OddsAPIClient, "__init__", boom)
    monkeypatch.setattr(smtplib.SMTP, "__init__", boom)


def test_shadow_pins_reads_and_writes_only_its_ledgers(tmp_path: Path, no_network: None) -> None:
    _setup(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    n_shadow, n_cross = shadow.run_shadow(tmp_path, D)
    after = {p.name for p in tmp_path.iterdir()}
    assert after - set(before) == {shadow.SHADOW_LEDGER, shadow.CROSSBOOK_LEDGER}
    assert all((tmp_path / k).read_bytes() == v for k, v in before.items())
    rows = _rows(tmp_path / shadow.SHADOW_LEDGER)
    assert len(rows) == n_shadow
    by_market = {r["market"]: r for r in rows}
    assert {"game_total", "game_ml", "game_rl", "batter_h"} <= set(by_market)
    assert all(r["game_key"] == "evA" and r["game_pk"] == "1" for r in rows)
    tot = by_market["game_total"]
    assert tot["pin_status"] == tilt_pinned() and abs(float(tot["p_tilt_1pass"]) - float(tot["fair_prob"])) < 1e-4
    assert tot["hyp_total_078"] and tot["rl_residual"] and tot["total_holdout_status"]
    prop = by_market["batter_h"]  # the accented prediction row resolved through canonical()
    assert prop["live_model_prob"] == "0.62"
    assert abs(float(prop["p_tilt_1pass"]) - float(prop["fair_prob"])) < 1e-6
    cross = _rows(tmp_path / shadow.CROSSBOOK_LEDGER)
    assert n_cross == len(cross) == 1
    assert cross[0]["read_line"] == "1.5" and cross[0]["pin_line"] == "0.5" and cross[0]["read_book"] == "fd"
    assert cross[0]["player_id"] == "103"


def test_shadow_rerun_replaces_the_date(tmp_path: Path, no_network: None) -> None:
    _setup(tmp_path)
    shadow.run_shadow(tmp_path, D)
    n, _ = shadow.run_shadow(tmp_path, D)
    assert len(_rows(tmp_path / shadow.SHADOW_LEDGER)) == n


def test_stale_histogram_is_skipped(tmp_path: Path, no_network: None) -> None:
    _setup(tmp_path)
    save_json(_recs(0.57), tmp_path / f"predictions_{D.isoformat()}.json")  # a later re-price
    shadow.run_shadow(tmp_path, D)
    rows = _rows(tmp_path / shadow.SHADOW_LEDGER)
    assert [r["pin_status"] for r in rows] == [shadow.HIST_STALE]
    assert _rows(tmp_path / shadow.CROSSBOOK_LEDGER) == []


def tilt_pinned() -> str:
    from mlb_engine.audit import tilt
    return tilt.PINNED


def _result(home: int, away: int, hits: int) -> object:
    from mlb_engine.data.results import GameResult, PlayerLine
    pl = PlayerLine()
    pl.batting.update({"H": hits, "PA": 4})
    return GameResult(1, True, home, away, 0, 0, {103: pl})


def test_grading_uses_the_read_line_and_the_canonical_name(tmp_path: Path, no_network: None) -> None:
    from mlb_engine.audit import shadow_grade as sg
    _setup(tmp_path)
    shadow.run_shadow(tmp_path, D)
    sg.grade_shadow(tmp_path, D, {1: _result(5, 3, 1)}, _recs())  # type: ignore[dict-item]
    rows = {r["market"]: r for r in _rows(tmp_path / shadow.SHADOW_LEDGER)}
    assert rows["game_total"]["result"] == sg.NO  # 8 runs under 8.5
    assert rows["game_ml"]["result"] == sg.YES
    assert rows["game_rl"]["result"] == sg.YES  # home by 2 covers -1.5
    assert rows["batter_h"]["result"] == sg.YES  # 1 hit over the 0.5 pin
    cross = _rows(tmp_path / shadow.CROSSBOOK_LEDGER)
    assert cross[0]["result"] == sg.NO  # the same hit is under the 1.5 read line
    block = sg.morning_block(tmp_path)
    assert block[-1] == "Gate shut: no alternate keys on the tape. 200-row rule is a 2027 rule."
    assert any("sign violations: 0" in ln for ln in block)
    assert not any("ml_holdout_gap" in ln for ln in block)


def test_audit_email_carries_the_shadow_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mlb_engine.output import audit_insight
    _setup(tmp_path)
    shadow.run_shadow(tmp_path, D)
    sent: list[tuple[str, str]] = []
    def fake_send(cfg: object, *, subject: str, html_body: str, text_body: str, **_: object) -> str:
        sent.append((html_body, text_body))
        return "me@example.com"
    monkeypatch.setattr(audit_insight, "send_card_email", fake_send)
    cfg = SimpleNamespace(audit_dir=tmp_path)
    audit_insight.send_audit_emails(cfg, D, [("a.pdf", b"%PDF")], [], to=None)
    html_body, text_body = sent[0]
    gate = "Gate shut: no alternate keys on the tape. 200-row rule is a 2027 rule."
    assert gate in html_body and text_body.rstrip().endswith(gate)
    empty = SimpleNamespace(audit_dir=tmp_path / "none")
    sent.clear()
    audit_insight.send_audit_emails(empty, D, [("a.pdf", b"%PDF")], [], to=None)
    assert "Gate shut" not in sent[0][0]
