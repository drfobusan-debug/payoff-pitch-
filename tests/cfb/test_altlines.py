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


def _swapped_game(tmp_path: Path) -> tuple[object, object]:
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        creds=SimpleNamespace(odds_api_key="x"),
        odds_regions="us",
        odds_cache_dir=tmp_path,
        altlines_file=lambda d: tmp_path / "alt.jsonl",
        alt_ledger_file=tmp_path / "alt_ledger.csv",
    )
    event = {
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "alternate_spreads",
                        "outcomes": [{"name": "Buffalo", "point": 5.5, "price": -150}],
                    }
                ],
            }
        ]
    }
    altlines.append(
        altlines.parse_event(event, "UMASS", "BUFF", "umass", "buffalo", T0),
        cfg.altlines_file(None),
    )
    return cfg, event


def test_a_reversed_cfbd_result_with_an_alias_grades_the_right_team(tmp_path: Path) -> None:
    from datetime import date

    from cfb_engine import cli
    from cfb_engine.audit.grade import build_result_index
    from cfb_engine.data.cfbd import GameResult
    from cfb_engine.market.tiers import Tier
    from cfb_engine.recommendations import Recommendation

    cfg, _ = _swapped_game(tmp_path)
    rec = Recommendation(
        game_date=date(2026, 10, 24),
        game_id="g1",
        matchup="BUFF @ UMASS",
        market="game_ats",
        selection="Buffalo +3.5",
        line=3.5,
        model_prob=0.6,
        market_american=-110,
        opposite_american=-110,
        ev=0.05,
        edge=0.04,
        fair_prob=0.55,
        tier=Tier.STRONG,
        team_side="away",
        side="cover",
        home_abbrev="UMass",
        away_abbrev="Buffalo",
        kickoff_utc="2026-10-24T16:00:00Z",
    )
    final = GameResult(home="Buffalo", away="Massachusetts", home_points=30, away_points=10)
    index = build_result_index([final])
    assert cli._grade_alt_lines(cfg, date(2026, 10, 24), [(rec, "win")], index) == 1  # type: ignore[arg-type]
    report = {r[0]: r for r in altlines.report(cfg.alt_ledger_file)}  # type: ignore[attr-defined]
    assert report["game_ats bought (all)"][1:4] == (1, 1, 0)  # Buffalo +5.5 won by 20


def test_a_failed_alternate_capture_never_stops_the_card(tmp_path: Path, monkeypatch) -> None:
    from datetime import date
    from types import SimpleNamespace

    from cfb_engine import cli

    class Down:
        def __init__(self, *a: object, **k: object) -> None:
            raise RuntimeError("quota exhausted")

    monkeypatch.setattr(cli, "OddsAPIClient", Down)
    monkeypatch.delenv("CFBE_ALT_LINES", raising=False)
    cfg, _ = _swapped_game(tmp_path)
    rec = SimpleNamespace(
        matchup="BUFF @ UMASS", market="game_ats", tier="Strong buy", pass_gate=None
    )
    cli._capture_alt_lines(cfg, date(2026, 10, 24), [rec])  # type: ignore[arg-type,list-item]
