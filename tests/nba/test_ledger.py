from dataclasses import asdict, replace
from datetime import date

import pytest

from nba_engine import cli
from nba_engine.audit import ledger, scorecard
from nba_engine.audit.settle import (
    ABSENT,
    DNP,
    LOSS,
    NOT_PLAYED,
    PUSH,
    VOID,
    WIN,
    pnl,
    settle,
    settle_row,
)
from nba_engine.data import boxes, capture
from nba_engine.data.capture import QuoteRow
from nba_engine.market import overbias, params
from nba_engine.market.board import novig, selections
from nba_engine.schemas import GameResult, PlayerLine

DAY = "2025-12-10"
M = "SAS @ LAL"
TIP = "2025-12-11T03:00:00Z"
TIPS = {"ev1": TIP}


def q(market, side, line, book, am, opp, entity="", at="2025-12-11T02:00:00Z"):
    return QuoteRow(at, DAY, M, "ev1", market, side, entity, line, book, am, opp)


def game(away_q=(30, 25, 30, 25), home_q=(25, 25, 25, 30), state="STATUS_FINAL", players=None):
    if players is None:
        players = (
            PlayerLine("1", "LAL", "Luka Dončić", 36, 30, 8, 9, 4),
            PlayerLine("2", "LAL", "LeBron James", 0, 0, 0, 0, 0, dnp=True),
            PlayerLine("3", "SAS", "Nic Claxton", 24, 10, 7, 2, 0),
            PlayerLine("4", "LAL", "Bub Carrington", 20, 8, 2, 3, 2),
        )
    return GameResult("401", date(2025, 12, 10), "SAS", "LAL", state, away_q, home_q, players)


# -- settlement --------------------------------------------------------------
def test_game_markets_include_overtime_and_spread_sign():
    g = game(away_q=(30, 25, 30, 25, 12), home_q=(25, 25, 25, 35, 10))  # 110-110 reg, SAS +2 in OT
    assert settle("game_ml", "SAS", "", None, g) == WIN
    assert settle("game_ml", "LAL", "", None, g) == LOSS
    assert settle("game_ats", "LAL", "", 1.5, g) == LOSS  # -2 + 1.5
    assert settle("game_ats", "LAL", "", 2.5, g) == WIN
    assert settle("game_ats", "SAS", "", -2.0, g) == PUSH
    assert settle("game_total", "over", "", 241.5, g) == WIN  # 122 + 120 = 242
    assert settle("game_total", "under", "", 242.0, g) == PUSH


def test_first_half_uses_two_quarters_and_ties_push():
    g = game(away_q=(30, 20, 40, 10), home_q=(25, 25, 20, 20))  # half 50-50
    assert settle("h1_ml", "SAS", "", None, g) == PUSH
    assert settle("h1_ats", "LAL", "", -0.5, g) == LOSS
    assert settle("h1_total", "over", "", 99.5, g) == WIN
    assert settle("h1_total", "under", "", 100.0, g) == PUSH


def test_props_settle_dnp_and_absent_are_void_never_loss():
    g = game()
    assert settle("pl_pts", "over", "Luka Doncic", 29.5, g) == WIN
    assert settle("pl_pra", "under", "Luka Doncic", 47.0, g) == PUSH  # 30+8+9
    assert settle("pl_3pm", "under", "Luka Doncic", 3.5, g) == LOSS
    assert settle_row("pl_pts", "over", "LeBron James", 20.5, g) == (VOID, DNP)
    assert settle_row("pl_pts", "over", "Anthony Davis", 20.5, g) == (VOID, ABSENT)
    assert pnl(VOID, -110) == 0.0


def test_book_spellings_reach_the_box():
    g = game()
    assert settle("pl_reb", "over", "Nicolas Claxton", 6.5, g) == WIN  # surname + initial
    assert settle("pl_pts", "under", "Carlton Carrington", 8.5, g) == WIN  # alias
    other = game(players=(PlayerLine("9", "MIN", "Justin Edwards", 10, 4, 1, 1, 0),))
    assert settle_row("pl_pts", "over", "Anthony Edwards", 25.5, other) == (VOID, ABSENT)


def test_unsettled_games():
    assert settle_row("game_ml", "SAS", "", None, game(state="STATUS_POSTPONED")) == (
        VOID,
        NOT_PLAYED,
    )
    assert settle("game_ml", "SAS", "", None, game(state="STATUS_SCHEDULED")) is None
    assert settle("pl_pts", "over", "Luka Doncic", 1.5, game(players=())) is None  # boxless final


def test_pnl_american():
    assert pnl(WIN, 150) == pytest.approx(1.5)
    assert pnl(WIN, -125) == pytest.approx(0.8)
    assert pnl(LOSS, 150) == -1.0
    assert pnl(PUSH, -110) == 0.0
    assert pnl(None, -110) is None


# -- pricing, record, close --------------------------------------------------
def board(at="2025-12-11T02:00:00Z"):
    return [
        q("game_ml", "LAL", None, "draftkings", -150, 130, at=at),
        q("game_ml", "SAS", None, "draftkings", 130, -150, at=at),
        q("game_ml", "LAL", None, "betmgm", -145, 125, at=at),
        q("game_ml", "SAS", None, "betmgm", 125, -145, at=at),
        q("game_ml", "LAL", None, "fanduel", -155, 135, at=at),
        q("game_ml", "SAS", None, "fanduel", 135, -155, at=at),
        q("pl_pts", "over", 29.5, "draftkings", -110, -110, "Luka Doncic", at=at),
        q("pl_pts", "under", 29.5, "draftkings", -110, -110, "Luka Doncic", at=at),
        q("pl_pts", "over", 30.5, "fanduel", 100, -120, "Luka Doncic", at=at),
        q("pl_pts", "under", 30.5, "fanduel", -120, 100, "Luka Doncic", at=at),
    ]


def priced(rows, tag="t1", model=None):
    pre = ledger.pregame(rows, TIPS)
    return ledger.price(
        selections(pre),
        ledger.quote_index(pre),
        tips=TIPS,
        shift={},
        versions="over_bias=v1",
        pass_tag=tag,
        model=model,
    )


def by(rows, market, side, line=None):
    return next(r for r in rows if r.market == market and r.side == side and r.line == line)


def test_price_records_refused_rows_and_one_vs_two_book():
    rows = priced(board())
    assert len(rows) == 6 and all(r.mode == "paper" and r.tier == "Pass" for r in rows)
    sas = by(rows, "game_ml", "SAS")
    assert (sas.book, sas.american, sas.opposite_american, sas.exec_books) == (
        "draftkings",
        130,
        -150,
        2,
    )
    assert sas.exec_prices == "betmgm:125;draftkings:130"
    assert sas.gates == "no_model" and not sas.is_buy
    assert by(rows, "pl_pts", "over", 29.5).exec_books == 1
    alt = by(rows, "pl_pts", "over", 30.5)
    assert alt.book == "" and alt.american is None and "no_exec_book" in alt.gates
    bought = priced(board(), model={("ev1", "game_ml", "SAS", "", None): 0.47})
    sas = by(bought, "game_ml", "SAS")
    assert sas.gates == "" and sas.edge == pytest.approx(0.47 - sas.fair, abs=1e-6)


def test_quotes_at_or_after_tip_are_never_priced_or_closed():
    late = board(at=TIP) + board(at="2025-12-11T03:10:00Z")
    assert ledger.pregame(late, TIPS) == []
    assert ledger.pregame(board(), {}) == []  # unknown tip keeps nothing


def test_of_record_takes_earliest_buy_else_latest_pass():
    a = by(priced(board(), "a"), "game_ml", "SAS")
    b = replace(
        a, pass_tag="b", priced_at="2025-12-11T02:30:00Z", tier="Lean", gates="", american=120
    )
    c = replace(a, pass_tag="c", priced_at="2025-12-11T02:50:00Z")
    assert ledger.of_record([c, b, a]) == [b]
    assert ledger.of_record([c, a]) == [c]


def test_grade_close_clv_pulled_and_moved():
    open_ = priced(board())
    close = [
        q("game_ml", "LAL", None, "draftkings", -170, 145, at="2025-12-11T02:55:00Z"),
        q("game_ml", "SAS", None, "draftkings", 145, -170, at="2025-12-11T02:55:00Z"),
        q("game_ml", "LAL", None, "betmgm", -165, 140, at="2025-12-11T02:55:00Z"),
        q("game_ml", "SAS", None, "betmgm", 140, -165, at="2025-12-11T02:55:00Z"),
        q(
            "pl_pts",
            "over",
            30.5,
            "draftkings",
            -105,
            -115,
            "Luka Doncic",
            at="2025-12-11T02:55:00Z",
        ),
        q(
            "pl_pts",
            "under",
            30.5,
            "draftkings",
            -115,
            -105,
            "Luka Doncic",
            at="2025-12-11T02:55:00Z",
        ),
        q("game_ml", "SAS", None, "draftkings", 400, -500, at="2025-12-11T03:05:00Z"),  # in play
    ]
    finals = {M: game()}
    graded = ledger.grade(open_, finals, board() + close, shift={}, graded_at="x")
    lal = by(graded, "game_ml", "LAL")
    assert (
        lal.outcome == LOSS and lal.pnl == -1.0 and (lal.away_score, lal.home_score) == (110, 105)
    )
    assert lal.book == "betmgm" and lal.close_status == "close" and lal.close_american == -165
    assert lal.clv == pytest.approx(novig(-165, 140) - novig(-145, 125), abs=1e-6)
    sas = by(graded, "game_ml", "SAS")
    assert sas.close_american == 145 and sas.clv < 0 and sas.pnl == pytest.approx(1.3)
    over = by(graded, "pl_pts", "over", 29.5)
    assert over.close_status == "moved" and over.clv is None
    assert over.pre_pull_american == -110 and over.pre_pull_clv == pytest.approx(0.0)
    assert over.espn_id == "401"


def test_pulled_prop_keeps_its_result_and_pre_pull_clv():
    rows = [r for r in priced(board()) if r.market == "pl_pts" and r.american is not None]
    later = [
        q(
            "pl_pts",
            "over",
            29.5,
            "draftkings",
            -130,
            110,
            "Luka Doncic",
            at="2025-12-11T02:30:00Z",
        ),
        q(
            "pl_pts",
            "under",
            29.5,
            "draftkings",
            110,
            -130,
            "Luka Doncic",
            at="2025-12-11T02:30:00Z",
        ),
        q("game_ml", "LAL", None, "draftkings", -150, 130, at="2025-12-11T02:55:00Z"),
    ]
    pulled = [
        q("pl_pts", "over", 29.5, "fanduel", -110, -110, "Luka Doncic", at="2025-12-11T02:55:00Z")
    ]
    graded = ledger.grade(rows, {M: game()}, board() + later + pulled, shift={}, graded_at="x")
    over = by(graded, "pl_pts", "over", 29.5)
    assert over.close_status == "pulled" and over.clv is None and over.outcome == WIN
    assert over.pre_pull_american == -130
    assert over.pre_pull_clv == pytest.approx(novig(-130, 110) - novig(-110, -110), abs=1e-6)


def test_grading_does_not_rewrite_the_price_of_record():
    rows = priced(board())
    win = ledger.grade(rows, {M: game()}, board(), shift={}, graded_at="x")
    lose = ledger.grade(rows, {M: game(away_q=(0, 0, 0, 0))}, board(), shift={}, graded_at="x")
    keep = ("american", "fair", "model_prob", "edge", "ev", "gates", "tier", "priced_at", "book")
    for a, b in zip(win, lose, strict=True):
        da, db = asdict(a), asdict(b)
        assert {k: da[k] for k in keep} == {k: db[k] for k in keep}
    assert ledger.grade(rows, {}, board(), shift={}, graded_at="x") == rows


def test_csv_round_trip_and_write_once(tmp_path):
    rows = ledger.grade(priced(board()), {M: game()}, board(), shift={}, graded_at="x")
    path = ledger.graded_path(tmp_path, date(2025, 12, 10))
    assert ledger.write_once(rows, path) == path
    assert ledger.load(path) == rows
    assert ledger.write_once(rows[:1], path) is None
    assert len(ledger.load(path)) == len(rows)


# -- scorecard -----------------------------------------------------------------
def row(
    outcome, american=-110.0, buy=False, event="e", market="game_ats", book="draftkings", n=2, **kw
):
    r = ledger.LedgerRow(
        DAY,
        TIP,
        event,
        "",
        M,
        market,
        "incl_ot",
        "LAL",
        "",
        -3.5,
        book,
        american,
        -110.0,
        n,
        "",
        5,
        5,
        0.5,
        None,
        None,
        None,
        "Lean" if buy else "Pass",
        "" if buy else "no_model",
        "2025-12-11T02:00:00Z",
        "t",
        "",
        outcome=outcome,
        **kw,
    )
    r.pnl = pnl(outcome, american)
    return r


def test_ppv_npv_against_base_rate_and_false_negatives():
    rows = [
        row(WIN, buy=True, event="a"),
        row(WIN, buy=True, event="b"),
        row(LOSS, buy=True, event="c"),
        row(WIN, event="d"),
        row(LOSS, event="e"),
        row(LOSS, event="f"),
        row(LOSS, event="g"),
        row(PUSH, buy=True, event="h"),
        row(VOID, buy=True, event="i"),
    ]
    m = scorecard.metrics(rows, lambda r: r.is_buy, "buys", draws=50)
    assert (m.n, m.wins, m.losses, m.pushes, m.voids) == (4, 2, 1, 1, 1)
    assert m.base_rate == pytest.approx(3 / 7, abs=1e-4)
    assert m.ppv == pytest.approx(2 / 3, abs=1e-4) and m.npv == pytest.approx(3 / 4, abs=1e-4)
    assert m.ppv_lift == pytest.approx(2 / 3 - 3 / 7, abs=1e-3)
    assert m.units == pytest.approx(2 * 100 / 110 - 1, abs=1e-3)
    gate = next(
        t for t in scorecard.tables(rows, draws=50)["gate"] if t.label == "refused:no_model"
    )
    assert (gate.wins, gate.losses) == (1, 3)  # one false negative


def test_tables_split_by_book_and_exec_books_and_clv_kinds():
    rows = [
        row(
            WIN,
            buy=True,
            event="a",
            book="draftkings",
            n=1,
            close_status="close",
            clv=0.02,
            clv_ev=0.01,
        ),
        row(
            LOSS, buy=True, event="b", book="betmgm", n=2, close_status="pulled", pre_pull_clv=0.05
        ),
    ]
    t = scorecard.tables(rows, draws=50)
    assert [m.label for m in t["book"]] == ["buys@betmgm", "buys@draftkings"]
    one, two = t["exec_books"]
    assert (one.n, two.n) == (1, 1)
    buys = t["buys"][0]
    assert (buys.clv_n, buys.mean_clv, buys.pulled, buys.pre_pull_n) == (1, 0.02, 1, 1)
    assert buys.mean_pre_pull_clv == 0.05


def test_integrity_flags_look_ahead_and_duplicates():
    good = row(WIN)
    late = replace(row(LOSS, event="z"), priced_at=TIP)
    dup = replace(good)
    i = scorecard.integrity([good, late, dup])
    assert (i.priced_after_tip, i.duplicates, i.paper) == (1, 1, 3)


def test_calibration_bins():
    rows = [replace(row(WIN, event=str(k)), fair=0.62) for k in range(3)] + [
        replace(row(LOSS, event="x"), fair=0.62)
    ]
    assert scorecard.calibration(rows, lambda r: r.fair) == [("0.60-0.70", 4, 0.62, 0.75)]


# -- CLI ---------------------------------------------------------------------
@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("NBAE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("NBAE_STATE_SYNC", "0")
    return tmp_path


def test_record_then_grade_then_audit(root, monkeypatch, capsys):
    day = date(2025, 12, 10)
    capture.write_snapshot(board(), root, day, label="events", captured_at="2025-12-11T02:00:00Z")
    params.write(root, overbias.NAME, {"shifts": {}})
    path = cli._record_pass(root, day, TIPS, "main", "2025-12-11T02:00:00Z")
    assert path is not None and path.exists()
    assert cli._record_pass(root, day, TIPS, "late", "2025-12-11T03:30:00Z") is None  # tipped
    monkeypatch.setattr(boxes, "ensure_results", lambda d, dd, client=None: [game()])
    assert cli.main(["grade", "--date", "2025-12-10", "--no-sync"]) == 0
    graded = ledger.load(ledger.graded_path(root, day))
    assert len(graded) == 6 and all(r.versions.startswith("over_bias=") for r in graded)
    assert cli.main(["audit", "--no-sync", "--draws", "20"]) == 0
    out = capsys.readouterr().out
    assert "board:game_ml" in out and "priced_after_tip': 0" in out


def test_grade_waits_for_unfinished_games(root, monkeypatch):
    day = date(2025, 12, 10)
    capture.write_snapshot(board(), root, day, label="events", captured_at="2025-12-11T02:00:00Z")
    cli._record_pass(root, day, TIPS, "main", "2025-12-11T02:00:00Z")
    monkeypatch.setattr(
        boxes, "ensure_results", lambda d, dd, client=None: [game(state="STATUS_IN_PROGRESS")]
    )
    assert cli.main(["grade", "--date", "2025-12-10", "--no-sync"]) == 1
    assert not ledger.graded_path(root, day).exists()
