"""Novig's quotes captured at card time and every graded row re-valued at them."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from cfb_engine.audit import novig, snapshot
from cfb_engine.audit.ledger import LedgerEntry, load_ledger, update_ledger
from cfb_engine.audit.snapshot import SideQuote
from cfb_engine.audit.weekly import weekly_audit
from cfb_engine.data import oddsapi
from cfb_engine.data.cfbd import GameResult
from cfb_engine.market.board import GameOdds
from cfb_engine.market.ev import MarketQuote, ev_per_dollar
from cfb_engine.market.linevalue import prob_per_point
from cfb_engine.output.weekly_report import build_weekly_article
from cfb_engine.pipeline import Pipeline
from cfb_engine.recommendations import Recommendation
from cfb_engine.schemas import Game, Slate, TeamGameInfo
from cfb_engine.state import pull_state, push_state
from tests.cfb.test_state import _repo_pair

DAY = date(2026, 11, 21)
NOW = datetime(2026, 11, 21, 15, 0, tzinfo=timezone.utc)
MATCHUP = "AUB @ ALA"


@pytest.fixture(autouse=True)
def _scratch_dirs_may_push(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CFBE_STATE_PUSH", "1")


def _game(home: str, away: str, kick: str) -> Game:
    return Game(
        game_id=f"{away}-{home}",
        game_date=DAY,
        commence_time_utc=kick,
        home=TeamGameInfo(name=home, abbrev=home, is_home=True),
        away=TeamGameInfo(name=away, abbrev=away, is_home=False),
    )


def _odds(matchup: str, home: str, away: str) -> GameOdds:
    odds = GameOdds(matchup=matchup)
    for book, (spread, ml_home, ml_away) in {
        "draftkings": (-7.0, -280.0, 230.0),
        "novig": (-6.5, -260.0, 240.0),
    }.items():
        odds.add_ml(home, MarketQuote(book, ml_home, ml_away))
        odds.add_ml(away, MarketQuote(book, ml_away, ml_home))
        odds.add_spread(spread, home, MarketQuote(book, 100.0, -100.0))
        odds.add_spread(spread, away, MarketQuote(book, -100.0, 100.0))
        odds.add_total(55.5, True, MarketQuote(book, -102.0, 102.0))
        odds.add_total(55.5, False, MarketQuote(book, 102.0, -102.0))
    return odds


def _slate() -> tuple[Slate, dict[str, GameOdds]]:
    later = _game("ALA", "AUB", "2026-11-21T20:30:00Z")
    started = _game("UGA", "UF", "2026-11-21T14:00:00Z")
    board = {
        later.matchup(): _odds(later.matchup(), "ALA", "AUB"),
        started.matchup(): _odds(started.matchup(), "UGA", "UF"),
    }
    return Slate(slate_date=DAY, games=[later, started]), board


def test_capture_keeps_only_novig_and_only_games_not_yet_kicked() -> None:
    slate, board = _slate()
    quotes = novig.capture(slate, board, NOW)
    assert {k.split("|")[0] for k in quotes} == {MATCHUP}
    ats = quotes[snapshot.key(MATCHUP, "game_ats", "ALA -6.5")]
    assert (ats.line, ats.american) == (-6.5, 100.0)
    assert quotes[snapshot.key(MATCHUP, "game_ml", "AUB ML")].american == 240.0
    assert ats.captured_at == "2026-11-21T15:00:00Z"
    assert ats.kickoff_utc == "2026-11-21T20:30:00Z"


def test_a_later_pull_wins_per_side_and_keeps_what_it_did_not_see(tmp_path: Path) -> None:
    path = tmp_path / "novig.json"
    slate, board = _slate()
    assert novig.save_capture(slate, board, path, NOW) == 6
    board[MATCHUP].ml.clear()
    board[MATCHUP].add_ml("ALA", MarketQuote("novig", -300.0, 250.0))
    later = NOW.replace(hour=16)
    novig.save_capture(slate, board, path, later)
    saved = snapshot.load(path)
    assert saved[snapshot.key(MATCHUP, "game_ml", "ALA ML")].american == -300.0
    assert saved[snapshot.key(MATCHUP, "game_ml", "AUB ML")].american == 240.0
    assert novig.save_capture(Slate(slate_date=DAY), {}, path, later) == 0


def _rec(**kw: object) -> Recommendation:
    base: dict[str, object] = dict(
        game_date=DAY,
        game_id="1",
        matchup=MATCHUP,
        market="game_ats",
        selection="ALA -7.0",
        model_prob=0.56,
        line=-7.0,
        team_side="home",
        side="cover",
        home_abbrev="ALA",
        away_abbrev="AUB",
    )
    base.update(kw)
    return Recommendation(**base)  # type: ignore[arg-type]


def _entry(**kw: object) -> LedgerEntry:
    base: dict[str, object] = dict(
        date=DAY.isoformat(),
        matchup=MATCHUP,
        category="ATS",
        market="game_ats",
        selection="ALA -7.0",
        line=-7.0,
        book="draftkings",
        odds=-110.0,
        tier="Moderate",
        model_prob=0.56,
        ev=0.04,
        result="push",
        pnl=0.0,
    )
    base.update(kw)
    return LedgerEntry(**base)  # type: ignore[arg-type]


def test_a_row_is_regraded_and_valued_at_novigs_own_number() -> None:
    quotes = {snapshot.key(MATCHUP, "game_ats", "ALA -6.5"): SideQuote(100.0, 0.5, -6.5)}
    closing = {snapshot.key(MATCHUP, "game_ats", "ALA -7.0"): SideQuote(-110.0, 0.5, -7.0)}
    final = GameResult(home="ALA", away="AUB", home_points=30, away_points=23)
    e = _entry()
    assert novig.stamp(e, _rec(), quotes, closing, final, margin_sd=16.0, total_sd=13.0)
    assert (e.novig_line, e.novig_odds) == (-6.5, 100.0)
    assert (e.novig_result, e.novig_pnl) == ("win", 1.0)  # a push at -7 covers at -6.5
    close_at_novig = 0.5 + 0.5 * prob_per_point(16.0)
    assert e.novig_clv_ev == pytest.approx(ev_per_dollar(close_at_novig, 100.0), abs=1e-4)

    bare = _entry()
    assert not novig.stamp(bare, _rec(), {}, closing, final, margin_sd=16.0, total_sd=13.0)
    assert bare.novig_odds is None and bare.novig_clv_ev is None


def test_the_columns_survive_the_ledger_file(tmp_path: Path) -> None:
    path = tmp_path / "ledger.csv"
    e = _entry(novig_line=-6.5, novig_odds=100.0, novig_clv_ev=0.031, novig_result="win")
    e.novig_pnl = 1.0
    update_ledger(path, [e, _entry(selection="AUB +7.0")], DAY)
    back = {r.selection: r for r in load_ledger(path)}
    got = back["ALA -7.0"]
    assert (got.novig_line, got.novig_odds, got.novig_clv_ev) == (-6.5, 100.0, 0.031)
    assert (got.novig_result, got.novig_pnl) == ("win", 1.0)
    assert back["AUB +7.0"].novig_result is None and back["AUB +7.0"].novig_odds is None


def _paired(i: int, book: float, at_novig: float, tier: str = "Moderate") -> LedgerEntry:
    return _entry(
        matchup=f"G{i} @ H{i}",
        tier=tier,
        clv_ev=book,
        novig_clv_ev=at_novig,
        novig_odds=100.0,
        novig_result="push",
        novig_pnl=0.0,
    )


def test_the_comparison_pairs_rows_and_calls_only_past_the_bar() -> None:
    rows = [_paired(i, -0.03, -0.01 + (0.004 if i % 2 else -0.004)) for i in range(20)]
    rows.append(_entry(matchup="X @ Y", clv_ev=-0.02))  # no Novig quote
    rows.append(_paired(99, -0.02, 0.0, tier="Pass"))
    buys = novig.compare(rows)[0]
    assert (buys.label, buys.n, buys.of) == ("Buys", 20, 21)
    assert buys.book_ev == pytest.approx(-0.03)
    assert buys.gain == pytest.approx(0.02)
    assert buys.verdict == f"needs {novig.BAR_BUYS - 20} more"
    assert novig.compare(rows)[-1].n == 21  # every graded side counts the Pass row

    many = [_paired(i, -0.03, -0.01 + (0.004 if i % 2 else -0.004)) for i in range(160)]
    assert novig.compare(many)[0].verdict == "Novig ahead by > 2 se"
    same = [_paired(i, -0.03, -0.03 + (0.01 if i % 2 else -0.01)) for i in range(160)]
    assert novig.compare(same)[0].verdict == "within 2 se"
    assert novig.compare([_entry()])[0].verdict == "no Novig quotes yet"


def test_the_weekly_audit_prints_the_novig_table() -> None:
    rows = [_paired(i, -0.03, -0.01) for i in range(3)]
    html = build_weekly_article(weekly_audit(rows, DAY), novig_rows=novig.compare(rows))
    assert "Novig: would the exchange" in html and "3 of 3" in html
    assert "Novig:" not in build_weekly_article(weekly_audit(rows, DAY))


class _Resp:
    headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> list[object]:
        return []


def test_the_board_asks_for_named_books_only_when_told(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict[str, str]] = []
    monkeypatch.setattr(oddsapi.credits, "debit_for", lambda *a, **k: True)
    monkeypatch.setattr(
        oddsapi.http, "get", lambda url, params, timeout: sent.append(params) or _Resp()
    )
    client = oddsapi.OddsAPIClient("key")
    client.fetch_board(DAY)
    client.fetch_board(DAY, bookmakers="novig")
    assert "bookmakers" not in sent[0]
    assert sent[1]["bookmakers"] == "novig"
    assert sent[0]["markets"] == sent[1]["markets"] == "h2h,spreads,totals"


def test_the_card_run_saves_novig_and_shrugs_off_a_failed_pull(tmp_path: Path) -> None:
    slate, board = _slate()
    path = tmp_path / "novig.json"
    cfg = SimpleNamespace(novig=True, novig_file=lambda day: path)

    class Odds:
        def __init__(self, fail: bool) -> None:
            self.fail = fail

        def fetch_board(self, day: date, *, bookmakers: str | None = None) -> object:
            if self.fail:
                raise RuntimeError("feed down")
            assert bookmakers == "novig"
            return slate, board

    pipe = object.__new__(Pipeline)
    pipe.cfg, pipe.odds = cfg, Odds(fail=True)  # type: ignore[assignment]
    pipe._capture_novig(DAY, slate)
    assert not path.exists()
    pipe.odds = Odds(fail=False)  # type: ignore[assignment]
    pipe._capture_novig(DAY, Slate(slate_date=DAY, games=[slate.games[0]]))
    assert all(
        q.american in (100.0, -100.0, 240.0, -260.0, 102.0, -102.0)
        for q in snapshot.load(path).values()
    )
    cfg.novig = False
    path.unlink()
    pipe._capture_novig(DAY, slate)
    assert not path.exists()


def test_novig_quotes_travel_on_the_state_branch(tmp_path: Path) -> None:
    repo = _repo_pair(tmp_path)
    mac, box = tmp_path / "mac", tmp_path / "box"
    (mac / "audit").mkdir(parents=True)
    key = snapshot.key(MATCHUP, "game_ml", "ALA ML")
    snapshot.save({key: SideQuote(-260.0, 0.71, None)}, mac / "audit" / "novig_2026-11-21.json")
    push_state(mac, "card", repo=repo, branch="engine-state")
    pull_state(box, repo=repo, branch="engine-state")
    assert snapshot.load(box / "audit" / "novig_2026-11-21.json")[key].american == -260.0
