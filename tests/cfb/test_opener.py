"""The opener capture: a baseline days before the card, graded and never gating."""

from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path

import pytest

from cfb_engine.audit import snapshot
from cfb_engine.audit.ledger import LedgerEntry, load_ledger, update_ledger
from cfb_engine.audit.probation import (
    CANDIDATE_SCREENS,
    CANDIDATE_UPGRADES,
    PROMOTE,
    WATCHING,
    CandidateUpgrade,
    probation_rows,
    upgrade_probation,
)
from cfb_engine.audit.snapshot import SideQuote
from cfb_engine.config import Config
from cfb_engine.data.cfbd import CFBDClient
from cfb_engine.data.oddsapi import OddsAPIClient
from cfb_engine.market import keys
from cfb_engine.market.board import GameOdds
from cfb_engine.market.ev import MarketQuote
from cfb_engine.market.tiers import Tier
from cfb_engine.pipeline import Pipeline
from cfb_engine.schemas import Game, Slate, TeamGameInfo
from cfb_engine.state import pull_state, push_state

MATCHUP = "UGA @ ALA"
DAY = date(2026, 9, 5)


def _event(eid: str, home: str, away: str, kick: str) -> dict[str, object]:
    return {
        "id": eid,
        "home_team": home,
        "away_team": away,
        "commence_time": kick,
        "bookmakers": [],
    }


def test_one_request_files_each_game_under_its_eastern_kickoff_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, str]] = []
    payload = [
        # 03:30 UTC Sunday is 23:30 Saturday in New York: a Saturday game.
        _event("1", "Alabama Crimson Tide", "Georgia Bulldogs", "2026-09-06T03:30:00Z"),
        _event("2", "Ohio State Buckeyes", "Michigan Wolverines", "2026-09-05T16:00:00Z"),
        _event("3", "Texas Longhorns", "Oklahoma Sooners", "2026-09-12T19:30:00Z"),
    ]
    client = OddsAPIClient("key")

    def fake(url: str, **params: str) -> object:
        calls.append(params)
        return payload

    monkeypatch.setattr(client, "_get_json", fake)
    boards = client.fetch_boards(DAY, 8)
    assert len(calls) == 1
    assert sorted(boards) == [DAY, date(2026, 9, 12)]
    slate, board = boards[DAY]
    assert [g.game_id for g in slate.games] == ["2", "1"]  # kickoff order
    assert all(g.game_date == DAY for g in slate.games)
    assert len(board) == 2


def test_no_key_means_no_request() -> None:
    assert OddsAPIClient(None).fetch_boards(DAY, 7) == {}


def _slate_and_board(point: float) -> tuple[Slate, dict[str, GameOdds]]:
    game = Game(
        game_id="1",
        game_date=DAY,
        home=TeamGameInfo(name="Alabama", abbrev="ALA", is_home=True),
        away=TeamGameInfo(name="Georgia", abbrev="UGA", is_home=False),
    )
    odds = GameOdds(matchup=game.matchup())
    quote = MarketQuote(book="b", american=-110, opposite_american=-110)
    odds.add_spread(point, "ALA", quote)
    odds.add_spread(point, "UGA", quote)
    return Slate(slate_date=DAY, games=[game]), {game.matchup(): odds}


def test_opener_drift_reads_the_week_while_the_gate_keeps_the_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The opener was ALA -6; the day's first board and the bet are both -8. The
    gate's drift sees no movement; the opener drift sees two points toward ALA."""
    monkeypatch.setenv("CFBE_DATA_DIR", str(tmp_path))
    cfg = Config()
    pipe = Pipeline(cfg, cfbd=CFBDClient(None))
    opener_slate, opener_board = _slate_and_board(-6.0)
    snapshot.save(snapshot.board_quotes(opener_slate, opener_board), cfg.opener_file(DAY))

    slate, board = _slate_and_board(-8.0)
    pipe._baseline_board(DAY, slate, board)
    sel = keys.game_ats("ALA", -8.0)
    day_drift = pipe._drift(MATCHUP, "game_ats", sel, "cover", -8.0, 0.5)
    week_drift = pipe._drift(MATCHUP, "game_ats", sel, "cover", -8.0, 0.5, pipe._opener)
    assert day_drift == pytest.approx(0.0)
    assert week_drift is not None and week_drift > 0.04
    # The opener is read, never written, by the run.
    assert snapshot.load(cfg.board_file(DAY))[snapshot.key(MATCHUP, "game_ats", sel)].line == -8.0
    assert snapshot.load(cfg.opener_file(DAY))[snapshot.key(MATCHUP, "game_ats", sel)].line == -6.0


def test_a_slate_without_an_opener_has_no_opener_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CFBE_DATA_DIR", str(tmp_path))
    pipe = Pipeline(Config(), cfbd=CFBDClient(None))
    slate, board = _slate_and_board(-8.0)
    pipe._baseline_board(DAY, slate, board)
    sel = keys.game_ats("ALA", -8.0)
    assert pipe._drift(MATCHUP, "game_ats", sel, "cover", -8.0, 0.5, pipe._opener) is None


def _entry(
    result: str,
    day: int,
    *,
    drift: float | None = None,
    open_drift: float | None = None,
    tier: str = Tier.MODERATE.value,
) -> LedgerEntry:
    return LedgerEntry(
        date=f"2026-09-{day:02d}" if day <= 30 else f"2026-10-{day - 30:02d}",
        matchup=f"G{day} @ H{day}",
        category="Spread (ATS)",
        market="game_ats",
        selection=f"H{day} -3.0",
        line=-3.0,
        book="b",
        odds=100.0,
        tier=tier,
        model_prob=0.55,
        ev=0.05,
        result=result,
        pnl={"win": 1.0, "loss": -1.0}.get(result, 0.0),
        drift=drift,
        open_drift=open_drift,
    )


def test_the_ledger_round_trips_opener_drift_and_reads_old_files(tmp_path: Path) -> None:
    path = tmp_path / "ledger.csv"
    update_ledger(path, [_entry("win", 5, open_drift=0.031)], DAY)
    assert load_ledger(path)[0].open_drift == 0.031
    old = tmp_path / "old.csv"
    lines = path.read_text().splitlines()
    head = lines[0].split(",")
    i = head.index("open_drift")
    old.write_text(
        "\n".join(",".join(c for j, c in enumerate(r.split(",")) if j != i) for r in lines) + "\n"
    )
    assert load_ledger(old)[0].open_drift is None


def _winners_and_losers(pattern: str, **kw: float) -> list[LedgerEntry]:
    return [_entry("win" if ch == "w" else "loss", i + 1, **kw) for i, ch in enumerate(pattern)]


def test_an_upgrade_whose_buys_win_in_both_halves_is_promoted() -> None:
    agrees = _winners_and_losers("w" * 22 + "l" * 6 + "w" * 22 + "l" * 6, drift=0.03)
    against = _winners_and_losers("l" * 40, drift=-0.03)
    upgrade = CandidateUpgrade("agrees", lambda e: (e.drift or 0) >= 0.02, "line came to us")
    (verdict,) = upgrade_probation(agrees + against, (upgrade,), min_n=50)
    assert verdict.status == PROMOTE
    assert verdict.n == 56
    assert "promote agrees" in verdict.finding and verdict.finding.endswith("(line came to us)")


def test_an_upgrade_whose_halves_disagree_is_not_promoted() -> None:
    rows = _winners_and_losers("w" * 28 + "l" * 28, drift=0.03)
    upgrade = CandidateUpgrade("agrees", lambda e: (e.drift or 0) >= 0.02, "line came to us")
    (verdict,) = upgrade_probation(rows, (upgrade,), min_n=50)
    assert verdict.status != PROMOTE


def test_the_registered_upgrades_only_read_buys_with_a_measured_move() -> None:
    passes = _winners_and_losers("w" * 60, drift=0.05, open_drift=0.05)
    for e in passes:
        e.tier = Tier.PASS.value
    unmeasured = _winners_and_losers("w" * 60)
    verdicts = upgrade_probation(passes + unmeasured, min_n=10)
    assert {v.name for v in verdicts} == {u.name for u in CANDIDATE_UPGRADES}
    assert all((v.status, v.n) == (WATCHING, 0) for v in verdicts)


def test_upgrades_and_the_opener_veto_appear_in_the_probation_table() -> None:
    names = {p.name for p in probation_rows(_winners_and_losers("wl" * 10))}
    assert {u.name for u in CANDIDATE_UPGRADES} <= names
    assert "open_drift_refuse_adverse_2pct" in {c.name for c in CANDIDATE_SCREENS}
    assert "open_drift_refuse_adverse_2pct" in names


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def test_the_opener_travels_on_the_state_branch_earliest_quote_winning(tmp_path: Path) -> None:
    origin = tmp_path / "origin.git"
    _git(["init", "--bare", "-q", str(origin)], tmp_path)
    repo = tmp_path / "repo"
    _git(["init", "-q", str(repo)], tmp_path)
    (repo / "README").write_text("x")
    _git(["add", "README"], repo)
    _git(["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"], repo)
    _git(["remote", "add", "origin", str(origin)], repo)
    _git(["push", "-q", "origin", "HEAD:main"], repo)

    mac, box = tmp_path / "mac", tmp_path / "box"
    k = "A @ B|game_ats|B"
    snapshot.save({k: SideQuote(-110, 0.5, -6.0)}, mac / "audit" / "opener_2026-09-05.json")
    push_state(mac, "mac", repo=repo, branch="engine-state")
    snapshot.save({k: SideQuote(-110, 0.5, -8.0)}, box / "audit" / "opener_2026-09-05.json")
    pulled = pull_state(box, repo=repo, branch="engine-state")
    assert "opener_2026-09-05.json" in pulled.pulled
    assert snapshot.load(box / "audit" / "opener_2026-09-05.json")[k].line == -6.0
