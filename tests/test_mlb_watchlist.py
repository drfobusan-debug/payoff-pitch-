"""MLB "What we're watching": pre-registered schedule bands, engine screens, worksheet and totals."""

from __future__ import annotations

from pathlib import Path

from mlb_engine.audit import watchlist as wl
from mlb_engine.audit.ledger import LedgerEntry
from mlb_engine.data.parks import PARKS
from mlb_engine.output.card import render_html, render_markdown
from mlb_engine.output.daily_worksheet import WEIGHTS, WEIGHTS_VERSION

V1, V2 = sorted(PARKS)[:2]


def _g(pk, day, dn, sgn, gis, home="HOM", away="AWY", home_id=1, away_id=2, venue=V1, hh="23"):
    return {"pk": pk, "date": day, "start": f"{day}T{hh}:05:00Z", "dn": dn, "sgn": sgn, "gis": gis,
            "home": home, "away": away, "home_id": home_id, "away_id": away_id, "venue": venue}


SERIES = [
    _g(1, "2027-04-01", "night", 1, 3),
    _g(2, "2027-04-02", "night", 2, 3),
    _g(3, "2027-04-03", "day", 3, 3, hh="17"),
]


def test_getaway_and_day_after_night_come_from_the_schedule() -> None:
    ctx = wl.context(SERIES)
    assert ctx[3].getaway and ctx[3].day_after_night
    assert not ctx[2].getaway and not ctx[2].day_after_night
    assert [ctx[k].home_streak for k in (1, 2, 3)] == [1, 2, 3]


def test_series_opener_measures_how_far_the_visitor_came() -> None:
    prev = _g(10, "2027-03-31", "night", 3, 3, home="XXX", home_id=9, away="AWY", away_id=2, venue=V2)
    ctx = wl.context([prev, *SERIES])
    assert ctx[1].opener_visitor_miles is not None and ctx[1].opener_visitor_miles > 0
    assert ctx[2].opener_visitor_miles is None  # not an opener


def _e(day: str, market: str, sel: str, won: bool, **kw: object) -> LedgerEntry:
    base: dict[str, object] = dict(date=day, matchup="AWY @ HOM", category="game", market=market,
                                   selection=sel, line=None, book="draftkings", odds=-110.0, tier="Pass",
                                   model_prob=0.5, ev=0.0, result="win" if won else "loss",
                                   pnl=0.909 if won else -1.0, fair_prob=0.5)
    base.update(kw)
    return LedgerEntry(**base)  # type: ignore[arg-type]


def test_bands_read_only_registered_rows_at_the_market() -> None:
    ctx = wl.context(SERIES)
    rows = {r.name: r for r in wl.band_rows([
        _e("2027-04-03", "game_total", "Under 8.5", True),
        _e("2027-04-03", "game_total", "Over 8.5", False),
        _e("2026-09-03", "game_total", "Under 8.5", True),  # before registration
        _e("2027-04-03", "pitcher_bb", "X Walks o1.5", True, tier="Strong buy"),
        _e("2027-04-03", "batter_h", "Y H o0.5", False, pass_gate="batter_over_elite_arm"),
    ], ctx)}
    getaway = rows["under_getaway_day"].read
    assert getaway.n == 1 and getaway.units > 0
    assert getaway.excess == getaway.units - (0.5 * (1 + 100 / 110) - 1)
    assert rows["under_day_after_night"].read.n == 1
    assert rows["home_ml_10_straight_days"].read.n == 0
    assert rows["pitcher_bb_buys"].read.n == 1
    assert rows["batter_over_elite_arm"].wants == wl.LOSES and rows["batter_over_elite_arm"].read.n == 1


def test_a_doubleheader_is_not_guessed_at() -> None:
    dh = [*SERIES, _g(4, "2027-04-03", "night", 3, 3, hh="23")]
    rows = {r.name: r for r in wl.band_rows([_e("2027-04-03", "game_total", "Under 8.5", True)], wl.context(dh))}
    assert rows["under_getaway_day"].read.n == 0


def test_worksheet_bullpen_side_bets_the_better_pen() -> None:
    row = {"date": "2027-04-03", "game": "AWY @ HOM", "away": "AWY", "home": "HOM", "fav": "HOM",
           "result": "dog", "away_bp": "8", "home_bp": "-2", "away_ml": "140", "home_ml": "-160",
           "weights": WEIGHTS_VERSION}
    r = wl.bullpen_row([row, {**row, "away_bp": "1"}, {**row, "weights": "old"}], WEIGHTS_VERSION).read
    assert r.n == 1 and r.pnls[0] == 1.4


def test_totals_sum_call_is_shown_against_always_over() -> None:
    rows = [{"date": "2027-04-0" + str(i), "sum_pts": s, "result": res}
            for i, (s, res) in enumerate([("2", "over"), ("-1", "over"), ("0", "under"), ("3", "push")], 1)]
    t = wl.totals_benchmark(rows)
    assert (t.n, t.sum_calls, t.sum_wins, t.over_wins) == (3, 2, 1, 2)


def test_worksheet_weights_are_frozen() -> None:
    assert WEIGHTS == {"bat vs hand": 1.0, "bp": 1.5, "sp": 1.0}
    assert WEIGHTS_VERSION == "w1-1.5-1-dec-spz-bpz"


def test_the_card_prints_the_block(tmp_path: Path) -> None:
    w = wl.build(tmp_path, SERIES, WEIGHTS_VERSION)
    from datetime import date
    md = render_markdown([], date(2027, 4, 3), watching=w)
    assert "## What we're watching" in md and "Under, getaway day game" in md
    assert "always-Over" in md and f"{wl.MIN_N} more games" in md
    assert "What we're watching" in render_html([], date(2027, 4, 3), watching=w)
    assert "Schedule unavailable" in wl.markdown(wl.build(tmp_path, [], WEIGHTS_VERSION))[2]


def test_a_later_card_reads_every_season_since_registration(tmp_path: Path, monkeypatch) -> None:
    from datetime import date
    from types import SimpleNamespace

    from mlb_engine import cli

    seen: list[int] = []
    monkeypatch.setattr(wl, "fetch_schedule", lambda get, season, d: seen.append(season) or [])
    assert cli._watchlist(SimpleNamespace(audit_dir=tmp_path), date(2028, 5, 1)) is not None  # type: ignore[arg-type]
    assert seen == [2027, 2028]
