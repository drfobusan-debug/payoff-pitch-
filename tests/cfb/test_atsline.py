"""Spread-size splits of the ATS buys: graded only, never applied."""

from __future__ import annotations

from cfb_engine.audit.ledger import LedgerEntry
from cfb_engine.audit.probation import (
    CANDIDATE_SCREENS,
    CANDIDATE_UPGRADES,
    WATCHING,
    candidate_probation,
    probation_rows,
    upgrade_probation,
)
from cfb_engine.market import atsline
from cfb_engine.market.tiers import Tier

(MID_DOG,) = (c for c in CANDIDATE_SCREENS if c.name == atsline.MID_DOG_RULE)
(MID_FAV,) = (c for c in CANDIDATE_SCREENS if c.name == atsline.MID_FAV_RULE)
(BIG_DOG,) = (u for u in CANDIDATE_UPGRADES if u.name == atsline.BIG_DOG_RULE)


def _entry(
    line: float | None,
    result: str = "win",
    market: str = "game_ats",
    tier: str = Tier.STRONG.value,
) -> LedgerEntry:
    return LedgerEntry(
        date="2026-10-17",
        matchup="Away @ Home",
        category="Spread (ATS)",
        market=market,
        selection="Home",
        line=line,
        book="draftkings",
        odds=-110,
        tier=tier,
        model_prob=0.55,
        ev=0.05,
        result=result,
        pnl=0.9091 if result == "win" else -1.0,
    )


def test_the_ranges_are_inclusive_at_both_ends_and_do_not_overlap() -> None:
    assert [atsline.big_dog("game_ats", x) for x in (14.0, 14.5, 30.0)] == [False, True, True]
    assert [atsline.mid_dog("game_ats", x) for x in (7.0, 7.5, 14.0, 14.5)] == [
        False,
        True,
        True,
        False,
    ]
    assert [atsline.mid_favourite("game_ats", x) for x in (-3.0, -3.5, -7.0, -7.5)] == [
        False,
        True,
        True,
        False,
    ]
    assert not atsline.mid_favourite("game_ats", 5.0)
    assert not atsline.mid_dog("game_ats", -10.0)


def test_only_spread_rows_with_a_line_count() -> None:
    assert not atsline.big_dog("game_total", 55.5)
    assert not atsline.mid_dog("game_ml", 10.0)
    assert not atsline.mid_favourite("game_ats", None)


def test_each_split_grades_only_the_buys_in_its_range() -> None:
    rows = [
        _entry(10.5, "loss"),
        _entry(-5.5, "loss"),
        _entry(17.0),
        _entry(3.0),
        _entry(10.5, "loss", tier=Tier.PASS.value),
        _entry(None, "loss"),
    ]
    (dog,) = candidate_probation(rows, (MID_DOG,), min_n=50)
    (fav,) = candidate_probation(rows, (MID_FAV,), min_n=50)
    (big,) = upgrade_probation(rows, (BIG_DOG,), min_n=50)
    assert (dog.n, fav.n, big.n) == (1, 1, 1)
    assert {dog.status, fav.status, big.status} == {WATCHING}


def test_all_three_are_asked_of_every_audit() -> None:
    names = {p.name for p in probation_rows([_entry(17.0)])}
    assert {atsline.BIG_DOG_RULE, atsline.MID_DOG_RULE, atsline.MID_FAV_RULE} <= names
