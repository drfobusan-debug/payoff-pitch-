"""Top-25 unit monitor: point-in-time ranks, ledger stamps and the graded-only rules."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from cfb_engine.audit.ledger import LedgerEntry, load_ledger, update_ledger
from cfb_engine.audit.probation import PROMOTE, UNIT_RULES, WATCHING, probation_rows, rule_probation
from cfb_engine.data.cfbd import GameMeta
from cfb_engine.market import unitranks
from cfb_engine.market.tiers import Tier


def _row(team: str, off: float, deff: float, pass_off: float = 0.3) -> dict[str, object]:
    return {
        "team": team,
        "offense": {"ppa": off, "passingPlays": {"ppa": pass_off}, "rushingPlays": {"ppa": off}},
        "defense": {"ppa": deff, "passingPlays": {"ppa": deff}, "rushingPlays": {"ppa": deff}},
    }


def _meta(week: int, day: str) -> GameMeta:
    return GameMeta(
        home="A",
        away="B",
        neutral_site=False,
        venue_id=None,
        start_date=f"{day}T19:00:00Z",
        week=week,
        season_type="regular",
    )


def test_ranks_order_each_unit_the_right_way_round() -> None:
    ranks = unitranks.rank_teams([_row("Ohio State", 0.4, -0.1), _row("Rice", 0.1, 0.2)])
    assert ranks["Ohio State"]["off"] == 1 and ranks["Rice"]["off"] == 2
    assert ranks["Ohio State"]["def"] == 1 and ranks["Rice"]["def"] == 2
    assert unitranks.rank_teams([_row("A", 0.2, 0.0), _row("B", 0.2, 0.0)])["B"]["off"] == 1


def test_prior_week_stops_before_the_slate() -> None:
    schedule = [_meta(6, "2026-10-03"), _meta(7, "2026-10-08"), _meta(7, "2026-10-10")]
    assert unitranks.prior_week(schedule, date(2026, 10, 10)) == 6
    assert unitranks.prior_week(schedule, date(2026, 10, 3)) == 5
    assert unitranks.prior_week([_meta(1, "2026-08-29")], date(2026, 8, 29)) == 0


def test_side_and_opponent_come_from_the_ledger_labels() -> None:
    assert unitranks.side_team("game_ats", "Jacksonville S -3.0") == "Jacksonville S"
    assert unitranks.side_team("game_ml", "Navy ML") == "Navy"
    assert unitranks.side_team("game_total", "Over 47.5") is None
    assert unitranks.opponent("Tulsa @ Navy", "Navy") == "Tulsa"
    assert unitranks.opponent("Tulsa @ Navy", "Army") is None


def test_stamp_encodes_both_sides_and_an_unranked_opponent_reads_empty() -> None:
    ranks = unitranks.rank_teams([_row("Navy", 0.4, -0.1), _row("Tulsa", 0.1, 0.2)])
    mine, opp = unitranks.stamp("game_ml", "Navy ML", "Tulsa @ Navy", ranks) or ("", "")
    assert unitranks.decode(mine)["def"] == 1 and unitranks.decode(opp)["def"] == 2
    assert unitranks.stamp("game_ats", "Navy -3.0", "Holy Cross @ Navy", ranks) == (mine, "")
    assert unitranks.stamp("game_total", "Over 47.5", "Tulsa @ Navy", ranks) is None
    assert unitranks.stamp("game_ml", "Navy ML", "Tulsa @ Navy", {}) is None


def test_picks_needs_a_top_unit_against_an_opponent_outside_it() -> None:
    assert unitranks.picks("def", "game_ats", "game_ats", "def:3", "def:40")
    assert unitranks.picks("def", "game_ats", "game_ats", "def:3", "")
    assert not unitranks.picks("def", "game_ats", "game_ats", "def:3", "def:12")
    assert not unitranks.picks("def", "game_ats", "game_ml", "def:3", "def:40")
    assert not unitranks.picks("def", "game_ats", "game_ats", None, None)


def _entry(day: int, result: str, market: str = "game_ats") -> LedgerEntry:
    selection = "Navy -3.0" if market == "game_ats" else "Navy ML"
    return LedgerEntry(
        date=f"2026-09-{day:02d}",
        matchup="Tulsa @ Navy",
        category="ATS",
        market=market,
        selection=selection,
        line=-3.0 if market == "game_ats" else None,
        book="pinnacle",
        odds=100.0,
        under_odds=-110,
        tier=Tier.PASS.value,
        model_prob=0.5,
        ev=0.0,
        result=result,
        pnl=1.0 if result == "win" else -1.0,
        unit_ranks="off:60 def:4 pass_def:30 rush_def:2",
        opp_unit_ranks="def:70",
    )


def test_each_unit_and_market_accrues_as_its_own_rule() -> None:
    names = {r.name for r in probation_rows([])}
    for unit in unitranks.UNITS:
        assert {f"{unit}_top25_ats", f"{unit}_top25_ml"} <= names
    rows = [_entry(d, "win") for d in range(1, 21)] + [_entry(21, "loss", "game_ml")]
    by = {p.name: p for p in rule_probation(rows, UNIT_RULES, min_n=10)}
    assert by["def_top25_ats"].n == 20 and by["def_top25_ats"].status == PROMOTE
    assert by["rush_def_top25_ats"].n == 20
    assert by["off_top25_ats"].n == 0 and by["pass_def_top25_ats"].n == 0
    assert by["def_top25_ml"].n == 1 and by["def_top25_ml"].status == WATCHING


def test_ledger_round_trips_the_stamps(tmp_path: Path) -> None:
    path = tmp_path / "ledger.csv"
    stamped = _entry(1, "win")
    blank = _entry(2, "win")
    blank.unit_ranks, blank.opp_unit_ranks = "", ""
    update_ledger(path, [stamped], date(2026, 9, 1))
    update_ledger(path, [blank], date(2026, 9, 2))
    got = {e.date: e for e in load_ledger(path)}
    assert got["2026-09-01"].unit_ranks == stamped.unit_ranks
    assert got["2026-09-02"].unit_ranks == ""
