"""Phase 1: consensus → p_fair, stale flag, v0 distribution, gates, card pass, grading."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cbb_engine import audit, card
from cbb_engine.data.espn import Game
from cbb_engine.market import consensus
from cbb_engine.market.gates import Thresholds, tier
from cbb_engine.models import distribution as dist
from engine_runtime import ledger, quotes
from engine_runtime.quotes import Quote

SPORT = "cbb"
DAY = "2026-11-10"


NOW = datetime(2026, 11, 10, 18, 5, tzinfo=timezone.utc)


def _q(book: str, side: str, am: float, opp: float | None, *, line: float = 145.5, market: str = "game_total", pid: str = "p1", kind: str = "board", eid: str = "e1", cap: str = f"{DAY}T18:00:00Z") -> Quote:
    return Quote(
        game_date=DAY, event_id=eid, tip_utc=f"{DAY}T23:00:00Z", market=market, side=side, line=line, book=book, entity="",
        american=am, opposite_american=opp, captured_at=cap, pass_id=pid, pass_kind=kind,
    )


def _board(pid: str = "p1", *, dk_over: float = -110, kind: str = "board") -> list[Quote]:
    rows = []
    for b, over, under in (("pinnacle", -115, -105), ("fanduel", -110, -110), ("caesars", -112, -108), ("betrivers", -110, -110)):
        rows += [_q(b, "over", over, under, pid=pid, kind=kind), _q(b, "under", under, over, pid=pid, kind=kind)]
    rows += [_q("draftkings", "over", dk_over, -110, pid=pid, kind=kind), _q("draftkings", "under", -110, dk_over, pid=pid, kind=kind)]
    return rows


CLOSE_AT = f"{DAY}T22:50:00Z"
WIDTH = dist.Width(sd_by_bucket={"all": 11.0}, key_mass={}, parity_shift=0.0, fitted_on="test", n=999)


def test_consensus_excludes_execution_books_and_counts_pairs() -> None:
    (a,) = consensus.build(_board())
    assert a.paired_books == 4 and a.books == 5
    assert a.sides == ("over", "under")
    assert abs(a.p_fair["over"] + a.p_fair["under"] - 1) < 1e-9
    assert [e.book for e in a.exec_quotes] == ["draftkings", "draftkings"]
    # a +200 DK over must not move p_fair
    (b,) = consensus.build(_board(dk_over=200))
    assert b.p_fair == a.p_fair


def test_stale_flag_needs_consensus_move_and_held_quote() -> None:
    prev = consensus.build(_board("p0"))
    cur_rows = [q for q in _board("p1") if q.book != "draftkings"]
    cur_rows = [Quote(**{**q.__dict__, "line": 146.5}) for q in cur_rows] + [q for q in _board("p1") if q.book == "draftkings"]
    cur = consensus.build(cur_rows)
    consensus.flag_stale(cur, prev)
    dk = [e for a in cur for e in a.exec_quotes]
    assert dk and all(e.stale for e in dk)
    same = consensus.build(_board("p2"))
    consensus.flag_stale(same, prev)
    assert not any(e.stale for a in same for e in a.exec_quotes)


def test_width_unfit_without_params(tmp_path: Path) -> None:
    assert dist.load_width(tmp_path / "missing.json") is None


def test_solve_hits_anchor_and_one_push_mass() -> None:
    d = dist.solve_total(145.0, 11.0, WIDTH, tol=0.1)
    assert abs(d.median - 145.0) <= 0.1
    assert abs(sum(d.pmf.values()) - 1) < 1e-9
    assert d.push_mass(145.0) == pytest.approx(d.pmf[145])
    assert d.push_mass(145.5) == 0.0
    assert d.p_over(145.5) + d.p_under(145.5) == pytest.approx(1.0)
    assert d.paths == 0 and d.se == 0.0


def test_tier_requires_edge_and_price() -> None:
    thr = Thresholds()
    assert tier(0.50, -110, 0.50, thr)[0] == "Pass"
    t, edge, ev, gates = tier(0.58, -110, 0.52, thr)
    assert t == "Strong buy" and gates == [] and ev > 0
    assert tier(0.70, -110, 0.50, thr)[3] == ["max_edge"]


def _write_board(root: Path, pid: str, rows: list[Quote], kind: str = "board") -> None:
    quotes.write_shard(root, SPORT, DAY, rows, pass_id=pid, pass_kind=kind, extra={"teams": {"e1": ["Duke", "Kansas"]}})


def test_card_pass_is_bet_of_record_and_close_is_archive(tmp_path: Path) -> None:
    _write_board(tmp_path, "b1", _board("b1"))
    priced = card.price(tmp_path, DAY, now=NOW, pass_id="card1", pass_kind="featured")
    assert priced is not None
    rows = priced.rows
    assert rows and all(r.bet_of_record and r.status == "live" for r in rows)
    assert all("width_unfit" in r.gates for r in rows)
    assert all(r.model_version == "market_only" and r.p_model is None for r in rows)
    ledger.write_once(tmp_path, SPORT, DAY, "card1", rows)
    closed = card.price(tmp_path, DAY, now=NOW, pass_id="close1", pass_kind="close")
    assert closed is not None and not any(r.bet_of_record for r in closed.rows)
    with pytest.raises(ValueError):
        ledger.write_once(tmp_path, SPORT, DAY, "close1", [ledger.Row(**{**closed.rows[0].__dict__, "bet_of_record": True})])


def test_card_prices_model_channel_with_width_and_thin_market_gate(tmp_path: Path) -> None:
    _write_board(tmp_path, "b1", _board("b1", dk_over=130))
    priced = card.price(tmp_path, DAY, now=NOW, pass_id="card1", pass_kind="featured", width=WIDTH)
    assert priced is not None
    over = next(r for r in priced.rows if r.side == "over")
    assert over.p_model is not None and over.sim_median is not None and over.target_median is not None
    assert abs(over.sim_median - over.target_median) <= 0.1
    assert over.push_mass == 0.0 and over.parity_mass is not None
    assert over.model_version == "v0_shape" and "v0_shape" in over.edge_by_rung
    assert over.is_buy and over.status == "live"
    thin = [q for q in _board("b2") if q.book in ("pinnacle", "draftkings")]
    _write_board(tmp_path, "b2", thin)
    priced2 = card.price(tmp_path, DAY, now=NOW, pass_id="card2", pass_kind="featured", width=WIDTH)
    assert priced2 is not None and all("thin_market" in r.gates for r in priced2.rows)


def test_grade_sets_clv_outcome_and_supersedes(tmp_path: Path) -> None:
    _write_board(tmp_path, "b1", _board("b1", dk_over=130))
    priced = card.price(tmp_path, DAY, now=NOW, pass_id="card1", pass_kind="featured", width=WIDTH)
    assert priced is not None
    ledger.write_once(tmp_path, SPORT, DAY, "card1", priced.rows)
    close = []
    for b, over, under in (("pinnacle", -125, 105), ("fanduel", -120, 100), ("caesars", -122, 102), ("betrivers", -120, 100)):
        close += [_q(b, "over", over, under, pid="c1", kind="close", cap=CLOSE_AT), _q(b, "under", under, over, pid="c1", kind="close", cap=CLOSE_AT)]
    close += [_q("draftkings", "over", -115, -105, pid="c1", kind="close", cap=CLOSE_AT), _q("draftkings", "under", -105, -115, pid="c1", kind="close", cap=CLOSE_AT)]
    _write_board(tmp_path, "c1", close, kind="close")
    game = Game(espn_id="x", start_utc="", away="Kansas", home="Duke", state="STATUS_FINAL", neutral=False, conference_game=False, away_halves=(40, 40), home_halves=(35, 36))
    written = audit.grade(tmp_path, DAY, [game], {"e1": ("Duke", "Kansas")}, pass_id="audit1")
    over = next(r for r in written if r.side == "over")
    assert over.supersedes and over.close_status == "pre_tip"
    assert over.clv_consensus is not None and over.clv_consensus > 0
    assert over.clv_exec is not None and over.clv_exec > 0
    assert over.outcome == "win" and over.pnl == pytest.approx(1.3)
    rec = ledger.record(tmp_path, SPORT, DAY)
    assert all(r.outcome for r in rec.values())
    rep = audit.report(list(rec.values()), ["v0_shape"])
    assert rep["record"]["w"] == 1 and rep["v_clv"]["n"] >= 1
    json.dumps(rep, default=str)


def _close_board(pid: str, cap: str, *, line: float = 145.5, over: float = -125, under: float = 105) -> list[Quote]:
    rows = []
    for b in ("pinnacle", "fanduel", "caesars", "betrivers", "draftkings"):
        rows += [_q(b, "over", over, under, line=line, pid=pid, kind="close", cap=cap), _q(b, "under", under, over, line=line, pid=pid, kind="close", cap=cap)]
    return rows


def _priced(tmp_path: Path) -> None:
    _write_board(tmp_path, "b1", _board("b1", dk_over=130))
    priced = card.price(tmp_path, DAY, now=NOW, pass_id="card1", pass_kind="featured", width=WIDTH)
    assert priced is not None
    ledger.write_once(tmp_path, SPORT, DAY, "card1", priced.rows)


def test_early_or_after_tip_close_gets_no_clv(tmp_path: Path) -> None:
    _priced(tmp_path)
    _write_board(tmp_path, "c_early", _close_board("c_early", f"{DAY}T21:30:00Z"), kind="close")
    _write_board(tmp_path, "c_late", _close_board("c_late", f"{DAY}T23:05:00Z"), kind="close")
    written = audit.grade(tmp_path, DAY, [], {}, pass_id="audit1")
    assert written and all(r.close_status == "post_tip" and r.clv_consensus is None for r in written)
    _write_board(tmp_path, "c_ok", _close_board("c_ok", f"{DAY}T22:50:00Z"), kind="close")
    assert audit._closes(quotes.read_day(tmp_path, SPORT, DAY))[0]["e1"][0].pass_id == "c_ok"


def test_total_clv_is_read_at_the_bets_own_line(tmp_path: Path) -> None:
    _priced(tmp_path)
    # the close moved to 148.5 at even odds: no anchor at the bet's 145.5 line
    _write_board(tmp_path, "c1", _close_board("c1", CLOSE_AT, line=148.5, over=-110, under=-110), kind="close")
    over = next(r for r in audit.grade(tmp_path, DAY, [], {}, pass_id="a1", width=WIDTH) if r.side == "over")
    assert over.close_status == "pre_tip" and over.clv_exec is None
    # the close median moved 3 points past the bet's line: the Over was worth more at the close
    assert over.close_consensus == pytest.approx(audit.pilot.p_close(148.5, 11.0, "over", 145.5))
    assert over.clv_consensus is not None and over.clv_consensus > 5


def test_spread_and_moneyline_grade_from_the_side_and_line() -> None:
    row = ledger.Row(slate_date=DAY, event_id="e1", tip_utc="", market="game_ats", side="away", line=3.5, book="draftkings",
                     pass_id="p", pass_kind="featured", bet_of_record=True, status="research")
    final = (70, 73)  # away, home
    assert audit.outcome(row, final) == "win"
    assert audit.outcome(ledger.Row(**{**row.__dict__, "line": -3.0, "side": "home"}), final) == "push"
    assert audit.outcome(ledger.Row(**{**row.__dict__, "market": "game_ml", "line": None}), final) == "loss"
    assert audit.outcome(ledger.Row(**{**row.__dict__, "market": "h1_ats"}), final) == ""


def test_price_leaves_out_games_that_have_tipped(tmp_path: Path) -> None:
    _write_board(tmp_path, "b1", _board("b1"))
    after = datetime(2026, 11, 10, 23, 1, tzinfo=timezone.utc)
    priced = card.price(tmp_path, DAY, now=after, pass_id="card1", pass_kind="featured")
    assert priced is not None and priced.rows == []


def test_state_copy_unions_manifest_lines_and_keeps_existing_files(tmp_path: Path) -> None:
    from cbb_engine import state

    a, b = tmp_path / "a", tmp_path / "b"
    _write_board(a, "b1", _board("b1"))
    _write_board(b, "b2", _board("b2"))
    src, dest = state.local_trees(a, a / "data"), state.local_trees(b, b / "data")
    copied = state._copy(src, dest, ("quotes",))
    assert f"quotes/{DAY}/b1.parquet" in copied and f"quotes/{DAY}/manifest.jsonl" in copied
    assert sorted(m["pass_id"] for m in quotes.read_manifest(b, SPORT, DAY)) == ["b1", "b2"]
    assert quotes.verify(b, SPORT, DAY) == [] and state._copy(src, dest, ("quotes",)) == []
