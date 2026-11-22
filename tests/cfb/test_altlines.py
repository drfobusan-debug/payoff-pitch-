"""Alternate spreads and totals: tracked beside the card, graded both ways."""

from __future__ import annotations

from pathlib import Path

from cfb_engine.audit import altlines

T0 = "2026-10-24T15:00:00Z"


def _event() -> dict:
    return {
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "alternate_spreads",
                        "outcomes": [
                            {"name": "California", "point": 16.5, "price": 130},
                            {"name": "California", "point": 20.5, "price": -150},
                            {"name": "Miami (OH)", "point": -16.5, "price": -160},
                        ],
                    },
                    {
                        "key": "alternate_totals",
                        "outcomes": [
                            {"name": "Under", "point": 50.5, "price": 120},
                            {"name": "Over", "point": 50.5, "price": -140},
                        ],
                    },
                ],
            },
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "alternate_spreads",
                        "outcomes": [{"name": "California", "point": 16.5, "price": 140}],
                    }
                ],
            },
        ]
    }


def _rows() -> list[dict]:
    return altlines.parse_event(_event(), "MIAOH", "CAL", "miami oh", "california", T0)


def test_the_event_reply_parses_to_one_row_per_book_side_and_number() -> None:
    rows = _rows()
    assert len(rows) == 6
    assert {r["side"] for r in rows if r["market"] == "game_ats"} == {"home", "away"}
    assert all(r["matchup"] == "CAL @ MIAOH" for r in rows)


def test_both_directions_grade_at_the_best_price() -> None:
    graded = altlines.grade(
        date="2026-10-24",
        matchup="CAL @ MIAOH",
        market="game_ats",
        selection="California +18.5",
        main_line=18.5,
        tier="Strong buy",
        side="away",
        home_pts=35,
        away_pts=17,  # lost by 18: +18.5 covers, +16.5 loses, +20.5 covers
        rows=_rows(),
    )
    by = {r.alt_line: r for r in graded}
    assert by[16.5].favour == -2.0 and by[16.5].result == "loss" and by[16.5].book == "fanduel"
    assert by[20.5].favour == 2.0 and by[20.5].result == "win"
    assert round(by[20.5].pnl, 3) == 0.667


def test_totals_count_points_in_our_favour_for_each_side() -> None:
    assert altlines.favour("game_total", "over", 52.5, 50.5) == 2.0
    assert altlines.favour("game_total", "under", 52.5, 50.5) == -2.0
    assert altlines.outcome("game_total", "under", 50.5, 28, 21) == "win"


def test_quotes_after_kickoff_are_ignored() -> None:
    graded = altlines.grade(
        date="2026-10-24",
        matchup="CAL @ MIAOH",
        market="game_total",
        selection="Under 52.5",
        main_line=52.5,
        tier="Pass",
        side="under",
        home_pts=28,
        away_pts=21,
        rows=_rows(),
        kickoff_utc="2026-10-24T14:00:00Z",
    )
    assert graded == []


def test_only_spread_buys_and_graded_only_totals_are_tracked() -> None:
    assert altlines.tracked("game_ats", "Strong buy", None)
    assert not altlines.tracked("game_ats", "Pass", None)
    assert altlines.tracked("game_total", "Pass", "graded_only_game_total")
    assert not altlines.tracked("game_ml", "Strong buy", None)


def test_a_regrade_replaces_its_slate_and_the_report_splits_directions(tmp_path: Path) -> None:
    path = tmp_path / "alt_ledger.csv"
    kw = dict(
        date="2026-10-24",
        matchup="CAL @ MIAOH",
        market="game_ats",
        selection="California +18.5",
        main_line=18.5,
        tier="Strong buy",
        side="away",
        home_pts=35,
        away_pts=17,
        rows=_rows(),
    )
    rows = altlines.grade(**kw)  # type: ignore[arg-type]
    altlines.save_day(path, "2026-10-24", rows)
    altlines.save_day(path, "2026-10-24", rows)
    report = {r[0]: r for r in altlines.report(path)}
    assert report["game_ats sold (all)"][1:4] == (1, 0, 1)
    assert report["game_ats bought (all)"][1:4] == (1, 1, 0)


def _priced_rows() -> list[dict]:
    def q(side: str, point: float, price: float) -> dict:
        return {"matchup": "A @ H", "captured_at": T0, "market": "game_ats", "side": side,
                "point": point, "book": "draftkings", "american": price}

    return [
        q("away", 4.5, 150), q("home", -4.5, -180),
        q("away", 5.5, 130), q("home", -5.5, -155),
        q("away", 10.5, -250), q("home", -10.5, 200),
        q("away", 0.5, 300), q("home", -0.5, -400),
    ]


def _engine_graded(main_edge: float) -> list[altlines.AltRow]:
    from cfb_engine.config import EVThresholds

    return altlines.grade(
        date="2026-10-24", matchup="A @ H", market="game_ats", selection="A +7.5", main_line=7.5,
        tier="Strong buy", side="away", home_pts=24, away_pts=20, rows=_priced_rows(),
        p_main=0.56, mean=6.0, sd=16.0, thr=EVThresholds(min_ev=0.0, min_edge=0.02, max_edge=0.15),
        main_edge=main_edge, main_result="win", main_american=-110,
    )


def test_the_engine_flags_the_one_rung_that_beats_the_main_line(tmp_path: Path) -> None:
    import pytest

    assert altlines.side_prob("game_ats", "away", 31.5, 29.05, 15.97) == pytest.approx(0.561, abs=0.002)
    graded = _engine_graded(main_edge=0.03)
    by = {r.alt_line: r for r in graded}
    assert 0.5 not in by  # 7 points sold: outside the window
    sold3, sold2, bought = by[4.5], by[5.5], by[10.5]
    assert sold3.favour == -3.0 and sold3.p_engine == pytest.approx(0.485, abs=0.003)
    assert sold3.fair_prob == pytest.approx(0.3836, abs=0.001) and sold3.flagged
    assert sold2.edge is not None and sold2.edge > 0.03 and not sold2.flagged  # a candidate, not the best
    assert bought.edge is not None and bought.edge < 0 and not bought.flagged
    path = tmp_path / "alt_ledger.csv"
    altlines.save_day(path, "2026-10-24", graded)
    labels = {r[0] for r in altlines.report(path)}
    assert "game_ats sold engine-flagged" in labels and "game_ats bought engine-flagged" not in labels
    assert [a.favour for a in altlines.alts(path, flagged_only=True)] == [-3.0]
    main = altlines.alts(path, main_line=True)
    assert [(a.favour, a.outcome, round(a.pnl, 4)) for a in main] == [(-3.0, "win", 0.9091)]
    same = {r[0]: r for r in altlines.report(path)}["game_ats sold engine-flagged: main line"]
    assert same[1:5] == (1, 1, 0, 0.91)


def test_no_rung_is_flagged_when_none_beats_the_main_line() -> None:
    assert not any(r.flagged for r in _engine_graded(main_edge=0.12))
    # The best rung's edge is ~0.102: beating a 0.095 main line by under a point is not enough.
    assert not any(r.flagged for r in _engine_graded(main_edge=0.095))
    assert any(r.flagged for r in _engine_graded(main_edge=0.09))


def test_rows_without_the_sim_stay_unpriced() -> None:
    graded = altlines.grade(
        date="2026-10-24", matchup="A @ H", market="game_ats", selection="A +7.5", main_line=7.5,
        tier="Strong buy", side="away", home_pts=24, away_pts=20, rows=_priced_rows(),
    )
    assert graded and all(r.p_engine is None and not r.flagged for r in graded)
