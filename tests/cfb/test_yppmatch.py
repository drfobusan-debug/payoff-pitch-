"""YPP Under rules: pregame team YPP, game tags, ledger stamps and graded-only probation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from cfb_engine.audit.ledger import LedgerEntry, load_ledger, update_ledger
from cfb_engine.audit.probation import (
    CANDIDATE_RULES,
    PROMOTE,
    WATCHING,
    YPP_RULES,
    probation_rows,
    rule_probation,
)
from cfb_engine.market import yppmatch
from cfb_engine.market.tiers import Tier


def _stats(
    team: str, yards: float, allowed: float, plays: float = 100.0
) -> list[dict[str, object]]:
    half = plays / 2
    return [
        {"team": team, "statName": "totalYards", "statValue": yards},
        {"team": team, "statName": "totalYardsOpponent", "statValue": allowed},
        {"team": team, "statName": "rushingAttempts", "statValue": half},
        {"team": team, "statName": "passAttempts", "statValue": half},
        {"team": team, "statName": "rushingAttemptsOpponent", "statValue": half},
        {"team": team, "statName": "passAttemptsOpponent", "statValue": half},
    ]


# YPP for / allowed: two elite offenses, two poor ones, and a middle tier.
_BOOK_ROWS = [
    *_stats("Ohio State", 750, 450),
    *_stats("Oregon", 720, 480),
    *_stats("Navy", 450, 520),
    *_stats("Rice", 420, 900),
    *_stats("Tulsa", 560, 560),
    *_stats("Army", 540, 540),
    *_stats("Temple", 500, 600),
    *_stats("Kent State", 380, 950),
]
BOOK = yppmatch.team_ypp(_BOOK_ROWS)


def test_team_ypp_divides_yards_by_plays_and_skips_partial_rows() -> None:
    assert BOOK["Ohio State"] == (7.5, 4.5)
    partial = yppmatch.team_ypp([{"team": "X", "statName": "totalYards", "statValue": 300}])
    assert partial == {}


def test_both_top_needs_both_offenses_in_the_top_quartile() -> None:
    assert yppmatch.BOTH_TOP in (yppmatch.tags("Oregon @ Ohio State", BOOK) or "").split()
    assert yppmatch.BOTH_TOP not in (yppmatch.tags("Navy @ Ohio State", BOOK) or "").split()


def test_matchup_tag_uses_offense_against_the_yards_the_defense_allows() -> None:
    # Two leaky defenses lift the matchup index even without elite offenses.
    assert yppmatch.MATCHUP_TOP in (yppmatch.tags("Rice @ Kent State", BOOK) or "").split()
    assert yppmatch.tags("Navy @ Army", BOOK) == yppmatch.NONE


def test_unjudged_games_stamp_none_not_a_miss() -> None:
    assert yppmatch.tags("Holy Cross @ Navy", BOOK) is None
    assert yppmatch.stamp("game_total", "Navy @ Army", {}) is None
    assert yppmatch.stamp("game_ats", "Oregon @ Ohio State", BOOK) is None
    assert yppmatch.cutoffs(yppmatch.team_ypp(_stats("A", 500, 500))) is None


def test_picks_only_the_under_row_of_a_tagged_game() -> None:
    tags = "both_top matchup_top"
    assert yppmatch.picks(yppmatch.BOTH_TOP, "game_total", "Under 60.5", tags)
    assert not yppmatch.picks(yppmatch.BOTH_TOP, "game_total", "Over 60.5", tags)
    assert not yppmatch.picks(yppmatch.BOTH_TOP, "game_total", "Under 60.5", yppmatch.NONE)
    assert not yppmatch.picks(yppmatch.BOTH_TOP, "game_total", "Under 60.5", None)


def _entry(day: int, selection: str, result: str, tags: str | None) -> LedgerEntry:
    return LedgerEntry(
        date=f"2026-09-{day:02d}",
        matchup="Oregon @ Ohio State",
        category="Totals",
        market="game_total",
        selection=selection,
        line=60.5,
        book="pinnacle",
        odds=100.0,
        under_odds=-110,
        tier=Tier.PASS.value,
        model_prob=0.5,
        ev=0.0,
        result=result,
        pnl=1.0 if result == "win" else -1.0,
        ypp_tags=tags,
    )


def test_rules_accrue_on_every_graded_under_and_stay_off_the_live_rules() -> None:
    names = {r.name for r in YPP_RULES}
    assert names == set(yppmatch.RULE_NAMES.values())
    assert not names & {r.name for r in CANDIDATE_RULES}
    assert names <= {p.name for p in probation_rows([])}
    rows = [_entry(d, "Under 60.5", "win", "both_top") for d in range(1, 21)]
    rows += [_entry(d, "Over 60.5", "loss", "both_top") for d in range(1, 21)]
    rows.append(_entry(21, "Under 60.5", "win", "matchup_top"))
    by = {p.name: p for p in rule_probation(rows, YPP_RULES, min_n=10)}
    both = by[yppmatch.RULE_NAMES[yppmatch.BOTH_TOP]]
    assert both.n == 20 and both.status == PROMOTE and both.kind == "rule"
    mm = by[yppmatch.RULE_NAMES[yppmatch.MATCHUP_TOP]]
    assert mm.n == 1 and mm.status == WATCHING


def test_ledger_round_trips_tags_and_keeps_blank_as_unstamped(tmp_path: Path) -> None:
    path = tmp_path / "ledger.csv"
    update_ledger(path, [_entry(1, "Under 60.5", "win", "both_top matchup_top")], date(2026, 9, 1))
    update_ledger(path, [_entry(2, "Under 60.5", "win", yppmatch.NONE)], date(2026, 9, 2))
    update_ledger(path, [_entry(3, "Under 60.5", "win", None)], date(2026, 9, 3))
    got = {e.date: e.ypp_tags for e in load_ledger(path)}
    assert got == {
        "2026-09-01": "both_top matchup_top",
        "2026-09-02": yppmatch.NONE,
        "2026-09-03": None,
    }
