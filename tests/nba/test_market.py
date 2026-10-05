from datetime import date

import pytest

from nba_engine import cli
from nba_engine.data import boxes
from nba_engine.data.capture import QuoteRow
from nba_engine.market import overbias, params
from nba_engine.market.board import ev_per_unit, novig, selections
from nba_engine.schemas import GameResult, PlayerLine

DAY = "2025-12-10"
M = "SAS @ LAL"


def q(market, side, line, book, am, opp, entity="", at="2025-12-11T02:55:00Z"):
    return QuoteRow(at, DAY, M, "ev1", market, side, entity, line, book, am, opp)


def game(pts=30, total=(110, 105)):
    player = PlayerLine("1", "LAL", "Luka Dončić", 36, pts, 8, 9, 4)
    dnp = PlayerLine("2", "LAL", "LeBron James", 0, 0, 0, 0, 0, dnp=True)
    return GameResult(
        "401",
        date(2025, 12, 10),
        "SAS",
        "LAL",
        "STATUS_FINAL",
        away_q=(30, 25, 30, total[0] - 85),
        home_q=(25, 25, 25, total[1] - 75),
        players=(player, dnp),
    )


def test_power_devig_sums_to_one_and_keeps_the_favourite():
    p_fav, p_dog = novig(-200, 170), novig(170, -200)
    assert p_fav + p_dog == pytest.approx(1.0)
    assert p_fav > 0.6


def test_selection_consensus_uses_paired_books_and_best_exec_price():
    rows = [
        q("game_ml", "LAL", None, "draftkings", -150, 130),
        q("game_ml", "LAL", None, "betmgm", -140, 120),
        q("game_ml", "LAL", None, "fanduel", -160, 135),
        q("game_ml", "LAL", None, "bovada", -170, None),
        q("game_ml", "LAL", None, "draftkings", -155, 135, at="2025-12-11T02:00:00Z"),
    ]
    (sel,) = selections(rows)
    assert (sel.books, sel.paired_books) == (4, 3)
    assert sel.exec_book == "betmgm" and sel.exec_american == -140
    assert sel.exec_books == 2 and sel.buyable
    assert sel.exec_prices["draftkings"] == -150


def test_a_side_neither_exec_book_posts_cannot_be_bought():
    (sel,) = selections([q("pl_pts", "over", 28.5, "fanduel", -110, -110, "Luka Doncic")])
    assert sel.fair == pytest.approx(0.5)
    assert sel.exec_book is None and not sel.buyable and sel.exec_books == 0


def test_outcomes_settle_totals_halves_props_and_void_dnp():
    g = game()
    assert overbias.outcome("game_total", "", g) == 215
    assert overbias.outcome("h1_total", "", g) == 105
    assert overbias.outcome("pl_pts", "Luka Doncic", g) == 30
    assert overbias.outcome("pl_pra", "Luka Doncic", g) == 47
    assert overbias.outcome("pl_pts", "LeBron James", g) is None
    assert overbias.outcome("pl_pts", "Nobody", g) is None


def test_grade_rows_consensus_and_exec_books_skip_pushes():
    finals = boxes.by_matchup([game(pts=30)])
    rows = [
        q("pl_pts", "over", 28.5, "draftkings", -110, -110, "Luka Doncic"),
        q("pl_pts", "under", 28.5, "draftkings", -110, -110, "Luka Doncic"),
        q("pl_pts", "over", 28.5, "fanduel", -120, 100, "Luka Doncic"),
        q("pl_pts", "over", 30.0, "betmgm", -110, -110, "Luka Doncic"),
    ]
    graded = overbias.grade(rows, finals)
    assert {(g.book, g.hit) for g in graded} == {("consensus", 1), ("draftkings", 1)}


def test_a_rematch_on_another_day_is_its_own_game():
    second = GameResult(**{**game(pts=20).__dict__, "game_date": date(2026, 1, 5)})
    finals = boxes.by_matchup([game(pts=30), second])
    later = [
        QuoteRow(
            "2026-01-06T00:55:00Z",
            "2026-01-05",
            M,
            "ev2",
            "pl_pts",
            "over",
            "Luka Doncic",
            28.5,
            "draftkings",
            -110,
            -110,
        )
    ]
    rows = [q("pl_pts", "over", 28.5, "draftkings", -110, -110, "Luka Doncic"), *later]
    assert len(selections(rows)) == 2
    graded = overbias.grade(rows, finals)
    assert sorted((g.game[0], g.book, g.hit) for g in graded) == [
        ("2025-12-10", "consensus", 1),
        ("2025-12-10", "draftkings", 1),
        ("2026-01-05", "consensus", 0),
        ("2026-01-05", "draftkings", 0),
    ]


def test_fit_applies_a_gap_only_when_its_interval_excludes_zero():
    graded = [
        overbias.Graded((DAY, f"G{i}"), "pl_pts", "consensus", 0.5, int(i % 4 == 0))
        for i in range(400)
    ] + [overbias.Graded((DAY, f"H{i}"), "pl_ast", "consensus", 0.5, i % 2) for i in range(40)]
    gaps = {g.market: g for g in overbias.fit(graded, draws=200)}
    assert gaps["pl_pts"].gap == pytest.approx(-0.25)
    assert gaps["pl_pts"].applied == pytest.approx(-0.25)
    assert gaps["pl_ast"].applied == 0.0


def test_params_versions_round_trip_and_shift_both_sides(tmp_path):
    gap = overbias.Gap("pl_pts", "consensus", 100, 50, 0.5, 0.48, -0.02, -0.03, -0.01)
    params.write(tmp_path, overbias.NAME, overbias.to_payload([gap], seasons=["2025-26"]))
    loaded = params.latest(tmp_path, overbias.NAME)
    assert loaded is not None and loaded["version"]
    shift = overbias.shifts(loaded)
    assert shift == {"pl_pts": pytest.approx(-0.02)}
    over, under = selections(
        [
            q("pl_pts", "over", 28.5, "draftkings", -110, -110, "Luka Doncic"),
            q("pl_pts", "under", 28.5, "draftkings", -110, -110, "Luka Doncic"),
        ]
    )
    assert overbias.adjusted_fair(over, shift) == pytest.approx(0.48)
    assert overbias.adjusted_fair(under, shift) == pytest.approx(0.52)


def test_results_archive_round_trips(tmp_path):
    day = date(2025, 12, 10)
    boxes.write_results(tmp_path, day, [game()])
    (back,) = boxes.read_results(tmp_path, day) or []
    assert back == game()
    assert boxes.read_results(tmp_path, date(2025, 12, 11)) is None
    assert boxes.norm_name("Luka Dončić Jr.") == "luka doncic"
    assert boxes.norm_name("Karl-Anthony Towns") == boxes.norm_name("Karl Anthony Towns")


def test_a_line_missing_from_the_newest_capture_is_withdrawn():
    early, late = "2025-12-11T01:00:00Z", "2025-12-11T02:55:00Z"
    rows = [
        q("pl_pts", "over", 28.5, "betmgm", +105, -125, "Luka Doncic", at=early),
        q("pl_pts", "over", 28.5, "draftkings", -110, -110, "Luka Doncic", at=early),
        q("pl_pts", "over", 28.5, "draftkings", -115, -105, "Luka Doncic", at=late),
        q("game_ml", "away", None, "betmgm", +150, -175, "SAS", at=early),
    ]
    by_market = {s.market: s for s in selections(rows)}
    assert by_market["pl_pts"].exec_prices == {"draftkings": -115}
    assert by_market["game_ml"].exec_prices == {"betmgm": 150}


class _NoBoxClient:
    def results(self, day):
        return [GameResult(**{**game().__dict__, "players": ()})]


def test_a_final_without_its_box_is_not_archived(tmp_path):
    day = date(2025, 12, 10)
    assert len(boxes.ensure_results(tmp_path, day, _NoBoxClient())) == 1
    assert boxes.read_results(tmp_path, day) is None


def test_an_empty_fit_writes_no_params_version(tmp_path, monkeypatch):
    monkeypatch.setenv("NBAE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("NBAE_STATE_SYNC", "0")
    argv = [
        "fit-over-bias",
        "--season",
        "2025-26",
        "--since",
        "2025-12-10",
        "--until",
        "2025-12-11",
    ]
    assert cli.main(argv) == 1
    assert params.latest(tmp_path, overbias.NAME) is None


def test_ev_per_unit_at_even_money():
    assert ev_per_unit(0.55, 100) == pytest.approx(0.10)
