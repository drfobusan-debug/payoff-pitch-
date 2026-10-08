"""The daily worksheet: scoring, weights, the ledger and its grading."""

from __future__ import annotations

import math
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import pytest
from openpyxl import load_workbook

from mlb_engine.config import Config
from mlb_engine.data.vsin import SideSplit, Split
from mlb_engine.market.ev import MarketQuote
from mlb_engine.output import daily_worksheet as dw
from mlb_engine.output.daily_worksheet import (
    WEIGHTS,
    LedgerRow,
    Prices,
    Ranking,
    StarterTable,
    TeamLine,
    band_of,
    build_rows,
    clean_ledger,
    day_incomplete,
    decile_points,
    grade,
    implied,
    load_ledger,
    merge,
    rescale_batting,
    save_ledger,
    score,
    score_deciles,
    score_z,
    tally,
    write_workbook,
)
from mlb_engine.schemas import Game, Hand, Pitcher, Slate, TeamGameInfo, Venue

# --- scoring -----------------------------------------------------------------------


def test_score_top_bottom_middle_and_nan_last() -> None:
    vals = {"A": 1.0, "B": 2.0, "C": 3.0, "D": math.nan, "E": 5.0}
    rk = score(list(vals), [("m", 1, False, True)], lambda e, _l: vals[e], 1, lambda e: 0.0)
    assert rk.pts["A"]["m"] == 2  # best (lower is better)
    assert rk.pts["D"]["m"] == -2  # missing sorts worst
    assert rk.pts["B"]["m"] == rk.pts["C"]["m"] == rk.pts["E"]["m"] == 1
    assert rk.ranked[0] == "A" and rk.ranked[-1] == "D"
    assert math.isnan(rk.vals["D"]["m"])


def test_score_higher_is_better_and_percent_display() -> None:
    vals = {"A": 0.10, "B": 0.30, "C": 0.20}
    rk = score(list(vals), [("k", 1, True, False)], lambda e, _l: vals[e], 1, lambda e: 0.0)
    assert rk.pts["B"]["k"] == 2 and rk.pts["A"]["k"] == -2
    assert rk.vals["B"]["k"] == 30.0


def test_score_z_standardises_signs_clips_and_fills_nan_with_worst() -> None:
    vals = {"A": 1.0, "B": 2.0, "C": 3.0, "D": math.nan, "E": 100.0}
    rk = score_z(list(vals), [("m", 1, False, True)], lambda e, _l: vals[e], lambda e: 0.0)
    assert rk.pts["A"]["m"] >= rk.pts["B"]["m"] >= rk.pts["C"]["m"] > rk.pts["E"]["m"]  # lower is better
    assert rk.pts["E"]["m"] >= -dw.Z_CLIP and rk.pts["A"]["m"] <= dw.Z_CLIP
    assert rk.pts["D"]["m"] == rk.pts["E"]["m"]  # missing takes the table worst
    assert rk.ranked[0] == "A" and set(rk.ranked[-2:]) == {"D", "E"}
    assert rk.k == 0 and math.isnan(rk.vals["D"]["m"])


def test_score_z_higher_is_better_sums_metrics_and_handles_constant_column() -> None:
    vals = {"A": {"k": 0.10, "c": 5.0}, "B": {"k": 0.30, "c": 5.0}, "C": {"k": 0.20, "c": 5.0}}
    rk = score_z(list(vals), [("k", 1, True, False), ("c", 1, False, False)],
                 lambda e, label: vals[e][label], lambda e: 0.0)
    assert rk.pts["B"]["k"] > 0 > rk.pts["A"]["k"] and abs(rk.pts["C"]["k"]) < 1e-9
    assert all(rk.pts[e]["c"] == 0.0 for e in vals)  # zero spread -> no points
    assert rk.total("B") == round(rk.pts["B"]["k"] + rk.pts["B"]["c"], 1)
    assert rk.ranked == ["B", "C", "A"] and rk.vals["B"]["k"] == 30.0


def test_decile_points_ties_nan_and_direction() -> None:
    vals = {f"T{i}": float(i) for i in range(1, 31)}  # 30 teams, 1..30
    hi = decile_points(vals, lower=False)
    assert hi["T30"] == 10 and hi["T28"] == 10 and hi["T27"] == 9 and hi["T1"] == 1 and hi["T3"] == 1
    assert sorted(hi.values()) == sorted([d for d in range(1, 11) for _ in range(3)])
    lo = decile_points(vals, lower=True)
    assert lo["T1"] == 10 and lo["T30"] == 1 and all(lo[t] == 11 - hi[t] for t in vals)
    # NaN is the bottom decile; ties share one decile.
    vals["T30"] = math.nan
    vals["T28"] = vals["T29"]
    hi = decile_points(vals, lower=False)
    assert hi["T30"] == 1 and hi["T28"] == hi["T29"] == 10
    assert decile_points({"A": 1.0, "B": 1.0}, lower=False) == {"A": 10, "B": 10}


def test_score_deciles_is_uncompressed_and_ranked_by_sum() -> None:
    ents = [f"T{i}" for i in range(1, 31)]
    vals = {e: {"a": float(i), "b": 31.0 - i} for i, e in enumerate(ents, 1)}
    rk = score_deciles(ents, [("a", 1, False, False), ("b", 1, False, True)], lambda e, m: vals[e][m], lambda e: -vals[e]["a"])
    assert rk.deciles and rk.pts["T30"] == {"a": 10, "b": 10} and rk.pts["T1"] == {"a": 1, "b": 1}
    assert len({rk.raw_total(e) for e in ents}) == 10  # ten distinct sums, not three
    assert rk.ranked[0] == "T30" and rk.total("T30") == rescale_batting(20)


def test_rescale_batting_keeps_the_old_mean_and_range() -> None:
    assert rescale_batting(60) + rescale_batting(61) == 2 * dw.BAT_OLD_MEAN
    assert rescale_batting(11) == -6.05 and rescale_batting(110) == 23.65  # ~ old 5..22 plus the tails


# --- weights -----------------------------------------------------------------------


def test_weighted_total_uses_run_share_weights_and_skips_missing() -> None:
    t = TeamLine("SD", "X", "R", 10.5, 6, 8)
    assert t.raw_total() == 24.5
    assert t.weighted() == 10.5 + 6 * 1.5 + 8 * 1
    assert WEIGHTS == {"bat vs hand": 1.0, "bp": 1.5, "sp": 1.0}
    missing = TeamLine("SD", "", "", 10.5, 6, None)
    assert missing.weighted() == 10.5 + 9  # nothing counted for the absent starter, not zero
    assert missing.label() == "sd (TBD)"


# --- slate rows --------------------------------------------------------------------


def _rank(pts: dict[str, int]) -> Ranking:
    return Ranking({t: {"m": v} for t, v in pts.items()}, {t: {"m": 0.0} for t in pts},
                   {t: set() for t in pts}, sorted(pts, key=lambda t: -pts[t]), 3)


def _dec(sums: dict[str, int]) -> Ranking:
    """A decile batting table whose single metric carries the 11-metric sum."""
    return Ranking({t: {"m": v} for t, v in sums.items()}, {t: {"m": 0.0} for t in sums},
                   {t: set() for t in sums}, sorted(sums, key=lambda t: -sums[t]), 0, deciles=True)


def _team(abbrev: str, home: bool, sp: str | None, hand: Hand = Hand.R) -> TeamGameInfo:
    pp = Pitcher(mlbam_id=hash(sp) % 10000, name=sp, throws=hand) if sp else None
    return TeamGameInfo(team_id=1, name=abbrev, abbrev=abbrev, is_home=home, probable_pitcher=pp)


def _game(pk: int, away: TeamGameInfo, home: TeamGameInfo) -> Game:
    return Game(game_pk=pk, game_date=Date(2026, 9, 14), status="S",
                venue=Venue(venue_id=1, name="v"), home=home, away=away)


def _fixture() -> tuple[Slate, dict[str, Ranking], Ranking, StarterTable, dict[tuple[int, str], Prices]]:
    bats = {
        "Overall": _dec({"SD": 90, "COL": 30, "NYY": 70, "MIN": 70}),
        "vs LHP": _dec({"SD": 95, "COL": 25, "NYY": 60, "MIN": 60}),
        "vs RHP": _dec({"SD": 90, "COL": 35, "NYY": 65, "MIN": 70}),
    }
    pens = _rank({"SD": 14, "COL": -10, "NYY": 10, "MIN": 10})
    sps = StarterTable(_rank({"1": 15, "2": -7, "3": 9}), {1: "Ace", 2: "Scrub", 3: "Arm"},
                       {1: "SD", 2: "COL", 3: "NYY"}, {1: 100.0, 2: 50.0, 3: 90.0},
                       {"ace": 1, "scrub": 2, "arm": 3})
    slate = Slate(slate_date=Date(2026, 9, 14), games=[
        _game(1, _team("SD", False, "Ace", Hand.L), _team("COL", True, "Scrub")),
        _game(2, _team("NYY", False, "Arm"), _team("MIN", True, "Unknown Guy")),
    ])
    prices = {(g.game_pk, t.abbrev): Prices() for g in slate.games for t in (g.away, g.home)}
    prices[(1, "SD")] = Prices(ml=-175, ml_book="lowvig", rl_line=-1.5, rl=+105, rl_book="dk",
                               dk_ml_handle=70, dk_ml_bets=60)
    prices[(1, "COL")] = Prices(ml=+160, ml_book="dk", rl_line=1.5, rl=-125, rl_book="dk")
    return slate, bats, pens, sps, prices


def test_build_rows_reads_the_split_by_opposing_hand_and_flags_missing_starter() -> None:
    slate, bats, pens, sps, prices = _fixture()
    rows = build_rows(slate, bats, pens, sps, prices)
    (_, a, h, r1), (_, a2, h2, r2) = rows
    # COL faces the lefty Ace -> vs LHP; SD faces the righty Scrub -> vs RHP; both rescaled from the decile sum.
    assert h.bat_vs_hand == rescale_batting(25) == -1.85 and a.bat_vs_hand == rescale_batting(90) == 17.65
    assert a.sp == 15 and h.sp == -7 and r1.complete
    assert a.weighted() == round(17.65 + 14 * 1.5 + 15 * 1, 1) and r1.away_bat == 17.65
    assert r1.gap == round(a.weighted() - h.weighted(), 1) and r1.fav == "SD"
    assert r1.away_ml == -175 and r1.home_ml == 160 and r1.away_rl_line == -1.5
    assert r1.fav_implied is not None and 0.6 < r1.fav_implied < 0.65
    assert r1.dk_ml_handle_away == 70 and r1.dk_ml_handle_home is None
    prices[(1, "COL")].circa_rl_handle = 33
    prices[(1, "COL")].circa_rl_bets = 44
    (_, _, _, r1b), _ = build_rows(slate, bats, pens, sps, prices)
    assert r1b.circa_rl_handle_home == 33 and r1b.circa_rl_bets_home == 44
    # MIN's probable is not in the starter table: no points, row marked incomplete.
    assert h2.sp is None and a2.sp == 9 and not r2.complete
    assert r2.home_sp_pts is None


class _Board:
    def __init__(self, quotes: dict, fail: bool = False) -> None:
        self.quotes, self.fail = quotes, fail

    def fetch(self, slate: Slate, **_: object) -> dict:
        if self.fail:
            raise RuntimeError("401")
        return self.quotes


class _VSIN:
    def __init__(self, sides: dict) -> None:
        self.sides = sides

    def fetch_side_splits(self, slate: Slate) -> dict:
        return self.sides


def _patch_feeds(monkeypatch: pytest.MonkeyPatch, board: _Board, vsin: _VSIN) -> None:
    monkeypatch.setattr(dw, "OddsAPIClient", lambda *a, **k: board)
    monkeypatch.setattr(dw, "VSINClient", lambda *a, **k: vsin)


_STALE_VSIN = {
    ("SD @ COL", "SD", "draftkings"): SideSplit(ml_american=-205, ml=Split(62, 72), rl_line=-1.5, rl=Split(24, 65)),
}


def test_fetch_prices_keys_by_game_pk_and_reads_board_and_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    slate, *_ = _fixture()
    board = {
        ("SD @ COL", "game_ml", "SD ML"): [MarketQuote("dk", -170), MarketQuote("lowvig", -165)],
        ("SD @ COL", "game_ml", "COL ML"): [MarketQuote("dk", 150)],
        ("SD @ COL", "game_rl", "SD -1.5"): [MarketQuote("dk", 110)],
    }
    _patch_feeds(monkeypatch, _Board(board), _VSIN(_STALE_VSIN))
    book = dw.fetch_prices(Config(), slate)
    assert book.board_ok and book.doubleheaders == [] and book.note() == ""
    sd = book.prices[(1, "SD")]
    assert (sd.ml, sd.ml_book, sd.rl_line, sd.rl) == (-165, "lowvig", -1.5, 110)
    assert (sd.dk_ml_handle, sd.dk_ml_bets, sd.dk_rl_handle) == (62, 72, 24)
    assert book.prices[(1, "COL")].ml == 150 and book.prices[(2, "NYY")].ml is None
    assert set(book.prices) == {(1, "SD"), (1, "COL"), (2, "NYY"), (2, "MIN")}


def test_no_board_means_no_price_not_a_vsin_carry_over(monkeypatch: pytest.MonkeyPatch) -> None:
    slate, bats, pens, sps, _ = _fixture()
    for board in (_Board({}), _Board({}, fail=True)):
        _patch_feeds(monkeypatch, board, _VSIN(_STALE_VSIN))
        book = dw.fetch_prices(Config(), slate)
        assert not book.board_ok and book.note().startswith("NO BOARD")
        sd = book.prices[(1, "SD")]
        assert sd.ml is None and sd.rl_line is None and sd.ml_text() == ""
        assert sd.dk_ml_handle == 62  # the splits are still VSIN's to give
        (_, _, _, r1), _ = build_rows(slate, bats, pens, sps, book.prices)
        assert r1.away_ml is None and r1.fav_implied is None and r1.away_rl_line is None


def test_doubleheader_games_are_not_priced_off_a_shared_matchup_key(monkeypatch: pytest.MonkeyPatch) -> None:
    slate, bats, pens, sps, _ = _fixture()
    g1 = slate.games[0]
    g2 = _game(3, _team("SD", False, "Arm"), _team("COL", True, "Scrub"))
    slate = Slate(slate_date=slate.slate_date, games=[g1, g2, slate.games[1]])
    board = {
        ("SD @ COL", "game_ml", "SD ML"): [MarketQuote("dk", -170)],
        ("SD @ COL", "game_ml", "COL ML"): [MarketQuote("dk", 150)],
        ("NYY @ MIN", "game_ml", "NYY ML"): [MarketQuote("dk", -120)],
        ("NYY @ MIN", "game_ml", "MIN ML"): [MarketQuote("dk", 105)],
    }
    _patch_feeds(monkeypatch, _Board(board), _VSIN(_STALE_VSIN))
    book = dw.fetch_prices(Config(), slate)
    assert book.board_ok and book.doubleheaders == ["SD @ COL"]
    assert "DOUBLEHEADER: the board quotes SD @ COL once" in book.note()
    assert all(book.prices[(pk, t)].ml is None for pk in (1, 3) for t in ("SD", "COL"))
    assert book.prices[(1, "SD")].dk_ml_handle is None
    assert book.prices[(2, "NYY")].ml == -120
    rows = build_rows(slate, bats, pens, sps, book.prices)
    by = {r.game_pk: r for _, _, _, r in rows}
    assert by[1].fav_implied is None and by[3].fav_implied is None and by[2].fav_implied is not None


def test_implied_strips_the_vig() -> None:
    assert implied(-110, -110) == 0.5
    p = implied(-175, 160)
    assert p is not None and abs(p - 0.6229) < 0.002
    assert implied(None, 160) is None


# --- ledger ------------------------------------------------------------------------


def _row(pk: int, gap: float, fav: str = "SD", result: str = "", complete: bool = True, day: str = "2026-09-14") -> LedgerRow:
    return LedgerRow(day, "SD @ COL", pk, "SD", "COL", "Ace", "Scrub", 9, 14, 15, -8, -10, -7,
                     80.0, 80.0 - gap, gap, fav, complete, away_ml=-175, home_ml=160,
                     away_rl_line=-1.5, fav_implied=0.62, result=result)


def test_ledger_round_trips_with_none_and_bool(tmp_path: Path) -> None:
    path = tmp_path / "l.csv"
    rows = [_row(1, 30.5), _row(2, -4.0, fav="COL", complete=False)]
    rows[1].away_ml = None
    save_ledger(path, rows)
    back = load_ledger(path)
    assert back == rows
    assert back[1].away_ml is None and back[1].complete is False and back[0].complete is True


def test_old_ledger_rows_with_bat6_columns_still_load_and_stay_graded(tmp_path: Path) -> None:
    path = tmp_path / "old.csv"
    path.write_text(
        "date,game,game_pk,away,home,away_sp,home_sp,away_bat,away_bat6,away_bp,away_sp_pts,home_bat,home_bat6,"
        "home_bp,home_sp_pts,away_w,home_w,gap,fav,complete,away_ml,home_ml,away_rl_line,away_rl,home_rl,"
        "fav_implied,weights,away_runs,home_runs,result,rl_result\n"
        "2026-09-01,SD @ COL,7,SD,COL,Ace,Scrub,9,8,14,15,-8,-6,-10,-7,83.0,-52.0,135.0,SD,True,-175,160,-1.5,105,"
        "-125,0.62,w1-1-1.5-3,5,3,fav,away\n"
    )
    (r,) = load_ledger(path)
    assert r.game_pk == 7 and r.away_bat == 9 and r.home_bat == -8 and r.complete
    assert r.graded and r.result == "fav" and r.rl_result == "away" and r.away_runs == 5
    assert r.weights == "w1-1-1.5-3" != dw.WEIGHTS_VERSION
    assert not hasattr(r, "away_bat6")
    # The old weights tag keeps the row out of the band tally; a fresh row is counted.
    assert all(t.n == 0 for t in tally([r]))
    assert {t.band: t.n for t in tally([r, _row(8, 30.5, result="fav")])}["all"] == 1
    # Re-saving writes the current header and the row round-trips.
    save_ledger(path, [r])
    assert "bat6" not in path.read_text().splitlines()[0] and load_ledger(path) == [r]


def test_merge_rewrites_ungraded_rows_and_keeps_graded_ones() -> None:
    graded = _row(1, 30.5, result="fav")
    ungraded = _row(2, 5.0)
    fresh = [_row(1, 99.0), _row(2, 7.0), _row(3, 1.0)]
    out = merge([graded, ungraded], fresh)
    by = {r.game_pk: r for r in out}
    assert by[1].gap == 30.5 and by[1].result == "fav"  # never touched
    assert by[2].gap == 7.0  # re-run replaces the ungraded price/gap
    assert by[3].gap == 1.0 and len(out) == 3


def test_grade_fills_finals_favourite_and_runline() -> None:
    rows = [_row(1, 30.5), _row(2, -10.0, fav="COL"), _row(3, 2.0)]
    n = grade(rows, {1: ("SD @ COL", 5, 3), 2: ("SD @ COL", 6, 1)})
    assert n == 2
    assert rows[0].result == "fav" and rows[0].rl_result == "away"  # 5-1.5 > 3
    assert rows[1].result == "dog" and rows[1].rl_result == "away"
    assert rows[2].result == "" and rows[2].away_runs is None
    # A second pass never regrades.
    assert grade(rows, {1: ("SD @ COL", 0, 9)}) == 0 and rows[0].result == "fav"
    # A level final is neither a hit nor a miss and stays out of the tally.
    assert grade(rows, {3: ("SD @ COL", 4, 4)}) == 1 and rows[2].result == "tie"
    assert all(t.n == 0 for t in dw.tally([rows[2]]))


def test_grade_skips_rows_with_a_missing_starter() -> None:
    rows = [_row(1, 30.5, complete=False), _row(2, 12.0)]
    finals = {1: ("SD @ COL", 5, 3), 2: ("SD @ COL", 2, 6)}
    assert grade(rows, finals) == 1
    assert rows[0].result == "" and rows[0].away_runs is None and not rows[0].graded
    assert rows[1].result == "dog"
    # a day whose only ungraded rows lack a starter is never re-fetched
    assert dw.grade_pending([rows[0]], Date(2026, 9, 30)) == 0


def test_starter_override_replaces_the_listed_probable_on_that_day_only() -> None:
    cws = _team("CWS", False, "Sean Newcomb", Hand.L)
    cle = _team("CLE", True, "Gavin Williams")
    slate = dw.apply_starter_overrides(Slate(slate_date=Date(2026, 9, 14), games=[_game(1, cws, cle)]))
    pp = slate.games[0].away.probable_pitcher
    assert pp is not None and pp.name == "Sean Burke" and pp.throws == Hand.R
    later = _team("CWS", False, "Sean Newcomb", Hand.L)
    slate = dw.apply_starter_overrides(Slate(slate_date=Date(2026, 9, 15), games=[_game(2, later, cle)]))
    pp = slate.games[0].away.probable_pitcher
    assert pp is not None and pp.name == "Sean Newcomb"


def test_tally_by_gap_band_excludes_incomplete_and_compares_to_market() -> None:
    rows = [
        _row(1, 30.5, result="fav"), _row(2, 22.0, result="dog"),
        _row(3, 3.0, result="fav"), _row(4, 40.0, result="fav", complete=False), _row(5, 12.0),
    ]
    t = {b.band: b for b in tally(rows)}
    assert t["20-34"].n == 2 and t["20-34"].fav_wins == 1 and t["20-34"].win_rate == 0.5
    assert t["0-9"].n == 1 and t["35+"].n == 0 and t["10-19"].n == 0  # incomplete and ungraded skipped
    assert t["all"].n == 3 and t["all"].market_rate is not None and abs(t["all"].market_rate - 0.62) < 1e-9
    assert t["all"].implied_n == 3 and t["all"].priced_win_rate == t["all"].win_rate
    assert t["all"].edge is not None and abs(t["all"].edge - (2 / 3 - 0.62)) < 1e-9
    assert band_of(-9.9) == "0-9" and band_of(35) == "35+"


def test_edge_compares_the_priced_games_only() -> None:
    # Two unpriced wins (a no-board day) must not inflate the edge over the one priced loss.
    unpriced = [_row(1, 30.5, result="fav"), _row(2, 22.0, result="fav")]
    for r in unpriced:
        r.fav_implied = r.away_ml = r.home_ml = None
    priced = _row(3, 25.0, result="dog")
    t = {b.band: b for b in tally(unpriced + [priced])}["20-34"]
    assert (t.n, t.fav_wins, t.implied_n, t.priced_fav_wins) == (3, 2, 1, 0)
    assert t.win_rate is not None and abs(t.win_rate - 2 / 3) < 1e-9
    assert t.priced_win_rate == 0.0 and t.edge is not None and abs(t.edge - (0.0 - 0.62)) < 1e-9
    text = dw.summary_text(unpriced + [priced], None)
    assert "20-34: 2-1 (67% vs mkt 62% on 0-1 priced)" in text
    # With no priced game at all there is no market and no edge.
    t0 = {b.band: b for b in tally(unpriced)}["20-34"]
    assert t0.market_rate is None and t0.edge is None and t0.priced_win_rate is None


# --- workbook ----------------------------------------------------------------------


def test_workbook_layout_and_ranked_block(tmp_path: Path) -> None:
    slate, bats, pens, sps, prices = _fixture()
    rows = build_rows(slate, bats, pens, sps, prices)
    ledger = [r for _, _, _, r in rows] + [_row(9, 25.0, result="fav", day="2026-09-13")]
    out = write_workbook(tmp_path / "w.xlsx", Date(2026, 9, 14), Date(2026, 9, 13), rows, prices,
                         pens, bats, sps, ledger, Date(2026, 9, 13))
    wb = load_workbook(out)
    assert wb.sheetnames == ["Matchups", "Audit", "Ledger", "Bullpens", "Bat Overall", "Bat vs LHP",
                             "Bat vs RHP", "Starters"]
    ws = wb["Matchups"]
    hdr = [c.value for c in ws[1]]
    assert hdr[:7] == ["game", "team", "bat vs hand", "bp", "sp", "raw TOTAL", "wTOTAL"]
    assert "bat 6+" not in hdr and "sp k-bb%" not in hdr and "BES rp" not in hdr
    assert ws["A2"].value == "SD @ COL" and ws["B2"].value == "sd (Ace, L)"
    assert ws["C2"].value == 17.65 and ws["G2"].value == round(17.65 + 14 * 1.5 + 15 * 1, 1)  # rescaled bat in wTOTAL
    assert ws["H2"].value == "-175 lowvig" and ws["I2"].value == "-1.5 +105 dk" and ws["L2"].value == "70/60"
    assert ws["B4"].value == "away - home"
    assert ws["G4"].value == round(ws["G2"].value - ws["G3"].value, 1)
    legend = " ".join(str(c.value) for row in ws.iter_rows() for c in row if isinstance(c.value, str))
    assert "decile sum rescaled" in legend and "bat 6+" not in legend and "Innings 6+" not in legend
    assert "wTOTAL = bat x1.0 + bp x1.5 + sp x1.0" in legend
    bo = wb["Bat Overall"]
    assert [c.value for c in bo[1]][:4] == ["Rk", "Team", "PTS", "dec sum"]
    assert bo["B2"].value == "SD" and bo["C2"].value == 17.65 and bo["D2"].value == 90
    notes = " ".join(str(c.value) for row in bo.iter_rows() for c in row if isinstance(c.value, str))
    assert "decile among the 30" in notes and "d10 = best" in notes and "best 3 +2" not in notes
    assert ws["B8"].value == "away - home *"  # MIN's starter missing
    # Ranked block: biggest |gap| first.
    c0 = hdr.index("rank") + 1
    gaps = [ws.cell(row=r, column=c0 + 6).value for r in (2, 3)]
    assert gaps == sorted(gaps, reverse=True)
    assert ws.cell(row=2, column=c0 + 1).value == "SD @ COL"
    assert ws.cell(row=3, column=c0 + 1).value == "NYY @ MIN *"
    wa = wb["Audit"]
    assert [c.value for c in wa[1]] == ["gap band", "games", "fav W", "fav L", "fav win %", "priced",
                                        "priced fav win %", "market implied %", "edge (pts)", "fav RL cover %"]
    assert [c.value for c in wa[6]][:9] == ["all", 1, 1, 0, 100.0, 1, 100.0, 62.0, 38.0]
    assert wb["Ledger"]["A2"].value == "2026-09-14"
    assert "NO BOARD" not in legend


def test_workbook_and_note_carry_the_no_board_stamp(tmp_path: Path) -> None:
    slate, bats, pens, sps, _ = _fixture()
    book = dw.PriceBook({(g.game_pk, t.abbrev): Prices() for g in slate.games for t in (g.away, g.home)},
                        board_ok=False, doubleheaders=[])
    rows = build_rows(slate, bats, pens, sps, book.prices)
    out = write_workbook(tmp_path / "w.xlsx", Date(2026, 9, 14), Date(2026, 9, 13), rows, book.prices,
                         pens, bats, sps, [r for _, _, _, r in rows], None, book.note())
    ws = load_workbook(out)["Matchups"]
    assert ws["H2"].value in (None, "") and ws["I2"].value in (None, "")
    legend = " ".join(str(c.value) for row in ws.iter_rows() for c in row if isinstance(c.value, str))
    assert dw.NO_BOARD_NOTE in legend
    page = dw.worksheet_html(out, Date(2026, 9, 14))
    assert "NO BOARD" in page


def test_pdf_pages_are_the_worksheet_first_then_the_tables_in_the_same_fills(tmp_path: Path) -> None:
    slate, bats, pens, sps, prices = _fixture()
    rows = build_rows(slate, bats, pens, sps, prices)
    ledger = [r for _, _, _, r in rows]
    out = write_workbook(tmp_path / "worksheet_2026-09-14.xlsx", Date(2026, 9, 14), Date(2026, 9, 13),
                         rows, prices, pens, bats, sps, ledger, None)
    page = dw.worksheet_html(out, Date(2026, 9, 14))
    titles = [t for _, t in dw.PDF_SHEETS]
    positions = [page.index(t) for t in titles]
    assert positions == sorted(positions) and positions[0] < page.index("Matchups ranked by weighted gap")
    assert "Audit - 2026" not in page and "Ledger - 2026" not in page
    assert "SD @ COL" in page and "background:#" in page
    pdf = dw.write_pdf(out, Date(2026, 9, 14))
    assert pdf == tmp_path / "worksheet_2026-09-14.pdf" and pdf.read_bytes()[:4] == b"%PDF"


def test_summary_text_reports_yesterday_and_bands() -> None:
    ledger = [_row(1, 30.5, result="fav", day="2026-09-13"), _row(2, 3.0, result="dog", day="2026-09-13")]
    text = dw.summary_text(ledger, Date(2026, 9, 13))
    assert text.startswith("Worksheet 2026-09-13: favoured side 1-1")
    assert "20-34: 1-0 (100% vs mkt 62%)" in text
    assert dw.summary_text([], None) == ""


def test_summary_text_lists_today_by_gap_with_incomplete_starred() -> None:
    ledger = [_row(1, 3.0), _row(2, -30.5, fav="COL", complete=False)]
    text = dw.summary_text(ledger, None, Date(2026, 9, 14))
    assert text == "Worksheet 2026-09-14 by weighted gap: SD @ COL COL 30.5*; SD @ COL SD 3.0  (* starter missing)"


def test_vsin_splits_are_skipped_when_the_sheet_is_not_for_today(monkeypatch: pytest.MonkeyPatch) -> None:
    slate, *_ = _fixture()
    board = {("SD @ COL", "game_ml", "SD ML"): [MarketQuote("dk", -170)]}
    _patch_feeds(monkeypatch, _Board(board), _VSIN(_STALE_VSIN))
    other_day = slate.slate_date - timedelta(days=2)
    book = dw.fetch_prices(Config(), slate, today=other_day)
    assert book.prices[(1, "SD")].ml == -170
    assert book.prices[(1, "SD")].dk_ml_handle is None
    book = dw.fetch_prices(Config(), slate, today=slate.slate_date)
    assert book.prices[(1, "SD")].dk_ml_handle == 62


def test_merge_drops_the_same_game_pk_filed_under_an_older_date() -> None:
    moved = _row(7, 4.0, result="fav", day="2026-09-22")
    out = merge([moved], [_row(7, 6.0, day="2026-09-23")])
    assert [(r.date, r.gap) for r in out] == [("2026-09-23", 6.0)]


def test_clean_ledger_dedupes_pk_blanks_doubleheader_splits_and_ungrades_incomplete() -> None:
    dup_old = _row(1, 2.0, result="fav", day="2026-09-22")
    dup_new = _row(1, 2.0, result="fav", day="2026-09-23")
    dh1 = _row(2, 3.0, day="2026-09-25")
    dh2 = _row(3, -1.0, day="2026-09-25")
    dh1.dk_ml_handle_away = dh2.dk_ml_handle_away = 58.0
    dh1.circa_rl_bets_home = dh2.circa_rl_bets_home = 67.0
    inc = _row(4, 1.0, result="dog", complete=False, day="2026-09-20")
    inc.away_runs, inc.home_runs, inc.rl_result = 4, 8, "home"
    lone = _row(5, 9.0, day="2026-09-26")
    lone.dk_ml_handle_away = 40.0
    out = clean_ledger([dup_old, dup_new, dh1, dh2, inc, lone])
    by = {r.game_pk: r for r in out}
    assert len(out) == 5 and by[1].date == "2026-09-23" and by[1].result == "fav"
    assert by[2].dk_ml_handle_away is None and by[3].circa_rl_bets_home is None
    assert by[5].dk_ml_handle_away == 40.0
    assert by[4].result == "" and by[4].rl_result == "" and by[4].away_runs is None
    assert tally(out)[-1].n == 1


def test_a_day_is_incomplete_while_an_ungraded_game_lacks_a_starter() -> None:
    day = Date(2026, 9, 14)
    assert not day_incomplete([_row(1, 5.0)], day)
    assert day_incomplete([_row(1, 5.0), _row(2, 3.0, complete=False)], day)
    assert not day_incomplete([_row(2, 3.0, complete=False, day="2026-09-13")], day)
