"""Conference tier and AP-ranked splits of the spread buys: graded only, never applied."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from cfb_engine.audit.ledger import LedgerEntry, entries_from_graded, load_ledger, update_ledger
from cfb_engine.audit.probation import (
    CANDIDATE_SCREENS,
    CANDIDATE_UPGRADES,
    WATCHING,
    candidate_probation,
    probation_rows,
    upgrade_probation,
)
from cfb_engine.data.cfbd import GameResult, _conferences
from cfb_engine.market import atscontext
from cfb_engine.market.tiers import Tier
from cfb_engine.output.brief import GameBrief, TeamBrief
from cfb_engine.recommendations import Recommendation

DAY = date(2026, 10, 10)
(SCREEN,) = (c for c in CANDIDATE_SCREENS if c.name == atscontext.G5_ONLY_RULE)
(UPGRADE,) = (u for u in CANDIDATE_UPGRADES if u.name == atscontext.TOP25_RULE)


def _entry(
    result: str,
    conf_tier: str | None = "G5-G5",
    top25: bool | None = False,
    market: str = "game_ats",
    tier: str = Tier.STRONG.value,
    day: str = "2026-10-10",
) -> LedgerEntry:
    return LedgerEntry(
        date=day,
        matchup="Away @ Home",
        category="Spread (ATS)",
        market=market,
        selection="Home -3.5",
        line=-3.5,
        book="draftkings",
        odds=-110,
        tier=tier,
        model_prob=0.56,
        ev=0.05,
        result=result,
        pnl=0.9091 if result == "win" else -1.0,
        conf_tier=conf_tier,
        top25=top25,
    )


def test_tiers_follow_the_conference_and_classification() -> None:
    assert atscontext.team_tier("Georgia", "SEC", "fbs") == "P4"
    assert atscontext.team_tier("Notre Dame", "FBS Independents", "fbs") == "P4"
    assert atscontext.team_tier("Boise State", "Pac-12", "fbs") == "G5"
    assert atscontext.team_tier("UConn", "FBS Independents", "fbs") == "G5"
    assert atscontext.team_tier("Montana", "Big Sky", "fcs") == "FCS"
    assert atscontext.team_tier("Georgia", "SEC", "") == ""
    assert atscontext.game_tier("P4", "G5") == "G5-P4"
    assert atscontext.game_tier("P4", "") is None


def test_a_cfbd_row_carries_both_conferences_onto_the_result() -> None:
    row: dict[str, object] = {
        "homeConference": "Sun Belt",
        "homeClassification": "fbs",
        "awayConference": "Big Ten",
        "awayClassification": "fbs",
    }
    res = GameResult("Troy", "Iowa", 10, 20, **_conferences(row))
    assert atscontext.result_tier(res) == "G5-P4"
    assert atscontext.result_tier(GameResult("Troy", "Iowa", 10, 20)) is None


def test_a_ranked_side_on_the_card_marks_the_game() -> None:
    def brief(home: int | None, away: int | None) -> GameBrief:
        return GameBrief(home=TeamBrief("H", poll_rank=home), away=TeamBrief("A", poll_rank=away))

    assert atscontext.top25_game(brief(None, 12)) is True
    assert atscontext.top25_game(brief(None, None)) is False
    assert atscontext.top25_game(None) is None


def test_both_splits_round_trip_through_the_ledger(tmp_path: Path) -> None:
    rec = Recommendation(
        game_date=DAY,
        game_id="g1",
        matchup="Away @ Home",
        market="game_ats",
        selection="Home -3.5",
        model_prob=0.56,
        line=-3.5,
        market_american=-110,
        tier=Tier.STRONG,
        home_abbrev="Home",
        away_abbrev="Away",
        brief=GameBrief(home=TeamBrief("Home", poll_rank=7), away=TeamBrief("Away")),
    )
    (entry,) = entries_from_graded([(rec, "win")], DAY)
    assert entry.top25 is True
    entry.conf_tier = "P4-P4"
    unknown = _entry("loss", conf_tier=None, top25=None)
    unknown.matchup = "Other @ Game"
    path = tmp_path / "ledger.csv"
    update_ledger(path, [entry, unknown], DAY)
    loaded = {e.matchup: e for e in load_ledger(path)}
    assert (loaded["Away @ Home"].conf_tier, loaded["Away @ Home"].top25) == ("P4-P4", True)
    assert (loaded["Other @ Game"].conf_tier, loaded["Other @ Game"].top25) == (None, None)


def test_the_screen_grades_only_g5_only_spread_buys() -> None:
    rows = [
        _entry("loss"),
        _entry("win", conf_tier="G5-P4"),
        _entry("loss", market="game_total"),
        _entry("loss", tier=Tier.PASS.value),
        _entry("loss", conf_tier=None),
    ]
    (verdict,) = candidate_probation(rows, (SCREEN,), min_n=50)
    assert verdict.n == 1 and verdict.status == WATCHING


def test_the_upgrade_grades_only_ranked_spread_buys() -> None:
    rows = [
        _entry("win", top25=True),
        _entry("win", top25=True, day="2026-10-03"),
        _entry("loss", top25=False),
        _entry("loss", top25=None),
        _entry("loss", top25=True, market="game_ml"),
    ]
    (verdict,) = upgrade_probation(rows, (UPGRADE,), min_n=50)
    assert verdict.n == 2 and verdict.status == WATCHING


def test_both_are_asked_of_every_audit() -> None:
    names = {p.name for p in probation_rows([_entry("win", top25=True)])}
    assert {atscontext.G5_ONLY_RULE, atscontext.TOP25_RULE} <= names
