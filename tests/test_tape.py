"""Phase 1: the quote tape, event-aware captures, close quarantine, header-union merge."""

from __future__ import annotations

import csv
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from mlb_engine.audit import clv, tape
from mlb_engine.audit.ledger import LedgerEntry
from mlb_engine.data.oddsapi import OddsAPIClient, _Event
from mlb_engine.market import keys
from mlb_engine.state import merge_dated_csv

D = date(2026, 10, 11)
T0 = datetime(2026, 10, 11, 17, 5, tzinfo=timezone.utc)
T1 = datetime(2026, 10, 11, 23, 10, tzinfo=timezone.utc)


def _cap(sel="Over 8.5", market="game_total", book="dk", am=-110.0, opp=-110.0,
         eid="e1", commence=T1, matchup="LAD @ MIL"):
    return tape.QuoteCapture(eid, commence, matchup, market, sel, book, am, opp)


def test_round_trip_dedup_and_hours(tmp_path: Path) -> None:
    now = T1 - timedelta(hours=2)
    caps = [_cap(), _cap(book="fd")]
    assert tape.append_tape(tmp_path, D, caps, tape.RUN, now=now) == 2
    assert tape.append_tape(tmp_path, D, caps, tape.RUN, now=now) == 0
    assert tape.append_tape(tmp_path, D, caps, tape.CLOSE, now=now) == 2
    rows = tape.load_tape(tape.tape_path(tmp_path, D))
    assert len(rows) == 4
    assert all(r.hours_to_first_pitch == 2.0 for r in rows)
    assert rows[0].line == 8.5 and rows[0].side == "over"


def test_unknown_version_skipped(tmp_path: Path) -> None:
    import gzip
    import json
    tape.append_tape(tmp_path, D, [_cap()], tape.RUN, now=T0)
    p = tape.tape_path(tmp_path, D)
    with gzip.open(p, "at") as f:
        f.write(json.dumps({"tape_version": 99}) + "\n")
    assert len(tape.load_tape(p)) == 1


@pytest.mark.parametrize(
    ("market", "sel", "line", "side"),
    [
        ("game_ml", keys.game_ml("LAD"), None, "LAD"),
        ("game_rl", keys.game_rl("LAD", -1.5), -1.5, "LAD"),
        ("game_rl", keys.game_rl("MIL", 1.5), 1.5, "MIL"),
        ("f5_rl", keys.f5_rl("MIL", 0.5), 0.5, "MIL"),
        ("game_total", keys.game_total(False, 8.0), 8.0, "under"),
        ("f5_total", keys.f5_total(True, 4.5), 4.5, "over"),
        ("f5_ml", keys.f5_ml("LAD"), None, "LAD"),
        ("batter_bb", keys.batter_prop("Shohei Ohtani", "BB", 0.5), 0.5, "over"),
        ("batter_hrr", keys.batter_prop("Mookie Betts", "H+R+RBI", 2.5, "under"), 2.5, "under"),
        ("pitcher_k", keys.pitcher_prop("Tarik Skubal", "Ks", 7.0), 7.0, "over"),
    ],
)
def test_parse_line(market: str, sel: str, line: float | None, side: str) -> None:
    assert tape.parse_line(market, sel) == line
    assert tape.parse_side(market, sel) == side


def test_vsin_rows_have_no_event_id() -> None:
    from mlb_engine.market.ev import MarketQuote
    quotes = {("LAD @ MIL", "game_ml", "LAD ML"): [MarketQuote("circa", -130, opposite_american=110)],
              ("LAD @ MIL", "game_total", "Over 8.5"): [MarketQuote("dk", -110, opposite_american=-110)]}
    rows = tape.vsin_captures(quotes, {("LAD @ MIL", "game_total", "Over 8.5")}, [])
    assert [r.selection for r in rows] == ["LAD ML"] and rows[0].event_id is None
    assert tape.game_key(None, "LAD @ MIL", None) == "LAD @ MIL|"


def _dh_raw(eid: str, commence: str, price: int) -> dict:
    return {
        "id": eid, "home_team": "Milwaukee Brewers", "away_team": "Los Angeles Dodgers",
        "commence_time": commence,
        "bookmakers": [{"key": "dk", "markets": [{"key": "totals", "outcomes": [
            {"name": "Over", "point": 8.5, "price": price},
            {"name": "Under", "point": 8.5, "price": -price - 20},
        ]}]}],
    }


def test_doubleheader_captures_stay_apart_but_quotes_blend() -> None:
    client = OddsAPIClient(None)
    out: dict = {}
    caps: list[tape.QuoteCapture] = []
    for eid, start, price in (("g1", T0, -105), ("g2", T1, -115)):
        ev = _Event(eid, "MIL", "LAD", "milwaukee brewers", "los angeles dodgers", start)
        client._parse_game(_dh_raw(eid, start.isoformat(), price), ev, out, f5=False, tape=caps)
    assert {c.event_id for c in caps if c.selection == "Over 8.5"} == {"g1", "g2"}
    # Known limit, not fixed here: the live Quotes key still blends both games.
    assert len(out[("LAD @ MIL", "game_total", "Over 8.5")]) == 2

    class _T:
        def __init__(self, abbrev: str) -> None:
            self.abbrev = abbrev

    class _G:
        def __init__(self, pk: int, start: datetime) -> None:
            self.game_pk, self.game_datetime_utc = pk, start.isoformat()
            self.away, self.home = _T("LAD"), _T("MIL")

    pks = tape.match_game_pks(caps, [_G(1, T0), _G(2, T1)])  # type: ignore[list-item]
    assert pks == {"g1": 1, "g2": 2}
    mid = T0 + (T1 - T0) / 2
    tie = tape.match_game_pks([_cap(eid="x", commence=mid)], [_G(1, T0), _G(2, T1)])  # type: ignore[list-item]
    assert tie == {"x": None}


def _rows(caps, source, now):
    return tape.to_rows(caps, slate_date=D, source=source, now=now)


def _entry(sel="Over 8.5", market="game_total", odds=-110.0, fair=0.5, matchup="LAD @ MIL"):
    return LedgerEntry(date=D.isoformat(), matchup=matchup, category="game", market=market,
                       selection=sel, line=None, book="dk", odds=odds, tier="pass",
                       model_prob=0.5, ev=0.0, result="win", pnl=0.0, fair_prob=fair)


def test_close_is_consensus_at_last_pregame_capture() -> None:
    early = _rows([_cap(am=-150, opp=130)], tape.CLOSE, T1 - timedelta(hours=3))
    late = _rows([_cap(book=b, am=-120, opp=100) for b in ("dk", "fd", "mgm")],
                 tape.CLOSE, T1 - timedelta(hours=1))
    inplay = _rows([_cap(am=-400, opp=300)], tape.CLOSE, T1 + timedelta(hours=1))
    tc = clv.close_from_tape(early + late + inplay)
    want = clv.evaluate(0.5, [r.quote() for r in late]).fair_prob
    assert tc.reason is None and tc.quote is not None
    assert tc.quote.no_vig_prob == pytest.approx(want, abs=1e-6)


def test_close_falls_back_to_run_then_no_pregame() -> None:
    run = _rows([_cap(am=-120, opp=100)], tape.RUN, T1 - timedelta(hours=4))
    inplay = _rows([_cap(am=-400, opp=300)], tape.CLOSE, T1 + timedelta(hours=1))
    assert clv.close_from_tape(run + inplay).quote is not None
    assert clv.close_from_tape(inplay).reason == clv.NO_PREGAME
    one_sided = _rows([_cap(opp=None)], tape.CLOSE, T1 - timedelta(hours=1))
    assert clv.close_from_tape(one_sided).reason == clv.NO_PAIR


def test_overround_logs_until_band_then_drops_pair() -> None:
    rows = _rows([_cap(book="dk", am=-110, opp=-110), _cap(book="junk", am=-200, opp=-200)],
                 tape.CLOSE, T1 - timedelta(hours=1))
    assert clv.close_from_tape(rows, band=None).reason is None
    kept = clv.close_from_tape(rows, band=(-0.01, 0.15))
    assert kept.reason is None and kept.quote is not None
    assert kept.quote.no_vig_prob == pytest.approx(0.5)
    junk = [r for r in rows if r.book == "junk"]
    assert clv.close_from_tape(junk, band=(-0.01, 0.15)).reason == clv.OVERROUND


def _closing(sel, am, prob, market="game_total", matchup="LAD @ MIL"):
    q = clv.ClosingQuote(matchup, market, sel, am, prob)
    return {q.key: q}


def test_ath_and_turner_fixtures() -> None:
    ath = _entry("ATH ML", "game_ml", 242.0, 0.28, "ATH @ BOS")
    turner = _entry("Trea Turner H o1.5", "batter_h", 183.0, 0.34)
    clean = _entry()
    n = clv.attach_clv([ath], _closing("ATH ML", -2000, 0.95, "game_ml", "ATH @ BOS"))
    n += clv.attach_clv([turner], _closing("Trea Turner H o1.5", -185, 0.63, "batter_h"))
    n += clv.attach_clv([clean], _closing("Over 8.5", -110, 0.52))
    assert n == 1
    assert ath.clv_quarantine == clv.IMPLAUSIBLE and ath.clv is None
    assert turner.clv_quarantine == clv.MOVE_GT_15PP and turner.close_prob is None
    assert clean.clv_quarantine == clv.CLEAN and clean.clv == pytest.approx(0.02)
    assert clv.clv_rows([ath, turner, clean]) == [("game_total", clean.clv, clean.clv_ev)]


def test_move_boundary() -> None:
    a, b = _entry(fair=0.5), _entry(fair=0.5)
    clv.attach_clv([a], _closing("Over 8.5", -110, 0.649))
    clv.attach_clv([b], _closing("Over 8.5", -110, 0.651))
    assert a.clv_quarantine == clv.CLEAN and b.clv_quarantine == clv.MOVE_GT_15PP


def test_unjudged_stays_blank_and_precedence() -> None:
    e = _entry(sel="Under 8.5")
    clv.attach_clv([e], _closing("Over 8.5", -110, 0.5))
    assert e.clv_quarantine == ""
    # In play AND two events for the matchup: no_pregame outranks event_mismatch.
    rows = _rows([_cap(eid="g1", am=-400, opp=300), _cap(eid="g2", am=-400, opp=300)],
                 tape.CLOSE, T1 + timedelta(hours=1))
    x = _entry()
    clv.attach_clv([x], {}, rows)
    assert x.clv_quarantine == clv.NO_PREGAME


def test_join_rule_one_event_vs_doubleheader() -> None:
    now = T1 - timedelta(hours=1)
    one = _rows([_cap(eid="g1", am=-120, opp=100)], tape.CLOSE, now)
    two = one + _rows([_cap(eid="g2", am=-130, opp=110)], tape.CLOSE, now)
    a, b = _entry(fair=0.52), _entry(fair=0.52)
    clv.attach_clv([a], {}, one)
    clv.attach_clv([b], {}, two)
    assert a.clv_quarantine == clv.CLEAN
    assert b.clv_quarantine == clv.EVENT_MISMATCH
    assert "1 quarantined" in clv.judged_line([a, b])


def test_tape_union_merge(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    tape.append_tape(a, D, [_cap(book="dk")], tape.RUN, now=T0)
    tape.append_tape(b, D, [_cap(book="fd")], tape.RUN, now=T0)
    pa, pb = tape.tape_path(a, D), tape.tape_path(b, D)
    assert tape.merge_tape_files(pa, pb)
    assert tape.merge_tape_files(pa, pb)
    assert sorted(r.book for r in tape.load_tape(pb)) == ["dk", "fd"]


def _write_csv(p: Path, fields: list[str], rows: list[dict]) -> None:
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_header_union_at_equal_width(tmp_path: Path) -> None:
    remote, local = tmp_path / "r.csv", tmp_path / "l.csv"
    _write_csv(remote, ["date", "k", "remote_only"], [{"date": "1", "k": "a", "remote_only": "R"}])
    _write_csv(local, ["date", "k", "clv_quarantine"], [{"date": "2", "k": "b", "clv_quarantine": "clean"}])
    merge_dated_csv(remote, local, ("date", "k"))
    with local.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert reader.fieldnames == ["date", "k", "clv_quarantine", "remote_only"]
    assert {r["k"]: (r["clv_quarantine"], r["remote_only"]) for r in rows} == {
        "a": ("", "R"), "b": ("clean", ""),
    }


def test_tape_hook_swallows_oserror(tmp_path: Path, monkeypatch) -> None:
    from mlb_engine import pipeline

    def boom(*a, **k):
        raise OSError("disk")

    monkeypatch.setattr(tape, "append_tape", boom)

    class _P:
        deps = type("D", (), {"oddsapi": None})()
        cfg = type("C", (), {"audit_dir": tmp_path})()

    slate = type("S", (), {"games": []})()
    pipeline.Pipeline._record_tape(_P(), slate, D, {})  # type: ignore[arg-type]
