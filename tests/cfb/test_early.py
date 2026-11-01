"""The early card: Saturday priced on Tuesday, recorded beside the day's card."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from datetime import date
from pathlib import Path

import pytest

from cfb_engine import cli
from cfb_engine import pipeline as pipeline_mod
from cfb_engine.audit import early
from cfb_engine.audit.ledger import LedgerEntry
from cfb_engine.config import Config
from cfb_engine.data.cfbd import CFBDClient
from cfb_engine.market.board import GameOdds
from cfb_engine.market.ev import MarketQuote
from cfb_engine.market.tiers import Tier
from cfb_engine.pipeline import Pipeline
from cfb_engine.schemas import Game, Slate, TeamGameInfo
from cfb_engine.state import PREFIX, pull_state, push_state

SAT = date(2026, 11, 7)


def test_next_saturday_is_the_coming_one_and_a_week_out_on_saturday() -> None:
    assert early.next_saturday(date(2026, 11, 3)) == SAT  # Tuesday
    assert early.next_saturday(date(2026, 11, 6)) == SAT  # Friday
    assert early.next_saturday(SAT) == date(2026, 11, 14)


def _e(
    selection: str,
    line: float | None,
    result: str,
    *,
    market: str = "game_ats",
    tier: str = Tier.STRONG.value,
    clv_pts: float | None = None,
    day: str = SAT.isoformat(),
) -> LedgerEntry:
    pnl = {"win": 0.9091, "loss": -1.0}.get(result, 0.0)
    return LedgerEntry(
        date=day,
        matchup="UGA @ ALA",
        category="x",
        market=market,
        selection=selection,
        line=line,
        book="draftkings",
        odds=-110,
        tier=tier,
        model_prob=0.56,
        ev=0.05,
        result=result,
        pnl=pnl,
        clv_pts=clv_pts,
    )


def test_points_gained_is_signed_from_the_bettors_side() -> None:
    assert early.points_gained(_e("UGA +7.0", 7.0, "win"), _e("UGA +6.0", 6.0, "win")) == 1.0
    assert early.points_gained(_e("ALA -6.0", -6.0, "win"), _e("ALA -7.0", -7.0, "win")) == 1.0
    assert (
        early.points_gained(
            _e("Over 50.5", 50.5, "win", market="game_total"),
            _e("Over 52.5", 52.5, "win", market="game_total"),
        )
        == 2.0
    )
    assert (
        early.points_gained(
            _e("Under 50.5", 50.5, "win", market="game_total"),
            _e("Under 52.5", 52.5, "win", market="game_total"),
        )
        == -2.0
    )
    assert (
        early.points_gained(
            _e("ALA ML", None, "win", market="game_ml"), _e("ALA ML", None, "win", market="game_ml")
        )
        is None
    )


def test_compare_pairs_the_same_side_across_a_moved_line_and_ignores_other_slates() -> None:
    tuesday = [
        _e("UGA +7.0", 7.0, "win", clv_pts=1.0),
        _e("Over 50.5", 50.5, "loss", market="game_total"),
        _e("ALA ML", None, "win", market="game_ml", tier=Tier.PASS.value),
    ]
    saturday = [
        _e("UGA +6.0", 6.0, "push", clv_pts=0.0),
        _e("Under 49.5", 49.5, "win", market="game_total"),
        _e("UGA +3.0", 3.0, "loss", day="2026-10-31"),  # no early card that day
    ]
    rows = {r.label: r for r in early.compare(tuesday, saturday)}
    assert (rows["Tuesday buys"].n, rows["Saturday buys, same slates"].n) == (2, 2)
    both = rows["Bought both days, at Tuesday's number"]
    assert (both.n, both.wins, both.pts_gained, both.clv_pts) == (1, 1, 1.0, 1.0)
    assert rows["Bought both days, at Saturday's number"].pushes == 1
    assert rows["Tuesday only"].losses == 1
    assert rows["Saturday only"].wins == 1


def _fake_slate() -> Slate:
    game = Game(
        game_id="1",
        game_date=SAT,
        home=TeamGameInfo(name="Alabama", abbrev="ALA", is_home=True),
        away=TeamGameInfo(name="Georgia", abbrev="UGA", is_home=False),
    )
    return Slate(slate_date=SAT, games=[game])


@pytest.mark.parametrize("is_early", [True, False])
def test_an_early_run_writes_none_of_the_slates_own_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, is_early: bool
) -> None:
    monkeypatch.setenv("CFBE_DATA_DIR", str(tmp_path))
    for flag in ("ENSEMBLE", "EFFICIENCY", "ESPN_COLOR", "VSIN_SPLITS", "INJURY_FEED", "KALSHI"):
        monkeypatch.setenv(f"CFBE_{flag}", "0")
    monkeypatch.setattr(pipeline_mod, "build_briefs", lambda *a, **k: {})
    monkeypatch.setattr(pipeline_mod, "fetch_watch_book", lambda *a, **k: None)
    cfg = Config()
    cfg.ensure_dirs()
    pipe = Pipeline(cfg, cfbd=CFBDClient(None))
    slate = _fake_slate()
    odds = GameOdds(matchup=slate.games[0].matchup())
    quote = MarketQuote(book="draftkings", american=-110, opposite_american=-110)
    odds.add_spread(-7.0, "ALA", quote)
    odds.add_spread(7.0, "UGA", quote)
    board = {odds.matchup: odds}
    monkeypatch.setattr(pipe.odds, "fetch_board", lambda day: (slate, board))
    pipe.run(SAT, early=is_early)
    assert cfg.board_file(SAT).exists() is not is_early
    if is_early:
        assert not cfg.books_file(SAT).exists()


def _args(day: str | None) -> argparse.Namespace:
    return argparse.Namespace(date=day)


def test_early_refuses_a_slate_that_is_not_ahead_and_keeps_the_first_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CFBE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CFBE_STATE_SYNC", "0")
    monkeypatch.setattr(cli, "_today", lambda: date(2026, 11, 3))
    cfg = Config()
    cfg.ensure_dirs()
    assert cli.cmd_early(cfg, _args("2026-11-03")) == 1
    cfg.early_file(SAT).write_text("[]")
    monkeypatch.setattr(cli, "Pipeline", lambda *a, **k: pytest.fail("repriced"))
    assert cli.cmd_early(cfg, _args(None)) == 0
    assert cfg.early_file(SAT).read_text() == "[]"


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _repo_pair(tmp_path: Path) -> Path:
    origin, repo = tmp_path / "origin.git", tmp_path / "repo"
    _git(["init", "--bare", "-q", str(origin)], tmp_path)
    _git(["init", "-q", str(repo)], tmp_path)
    (repo / "README").write_text("x")
    _git(["add", "README"], repo)
    _git(["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"], repo)
    _git(["remote", "add", "origin", str(origin)], repo)
    _git(["push", "-q", "origin", "HEAD:main"], repo)
    return repo


def test_the_early_card_and_its_ledger_travel_on_the_state_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CFBE_STATE_PUSH", "1")
    repo = _repo_pair(tmp_path)
    mac, box = tmp_path / "mac", tmp_path / "box"
    (mac / "audit").mkdir(parents=True)
    (mac / "audit" / "early_2026-11-07.json").write_text('[{"who": "tue"}]')
    (mac / "audit" / "early_ledger.csv").write_text(
        "date,matchup,category,market,selection,line,result\n"
        "2026-11-07,A @ B,x,game_ats,B -3.0,-3.0,win\n"
    )
    push_state(mac, "mac", repo=repo, branch="engine-state")
    state = repo.parent / f".{repo.name}-engine-state"
    with gzip.open(state / PREFIX / "early" / "early_2026-11-07.json.gz") as fz:
        assert json.load(fz) == [{"who": "tue"}]
    assert not list((state / PREFIX / "predictions").glob("early_*"))

    pulled = pull_state(box, repo=repo, branch="engine-state", dates=("2026-11-07",))
    assert {"early_2026-11-07.json", "early_ledger.csv"} <= set(pulled.pulled)
    assert (box / "audit" / "early_2026-11-07.json").read_text() == '[{"who": "tue"}]'
