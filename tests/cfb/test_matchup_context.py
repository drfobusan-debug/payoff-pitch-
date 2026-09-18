"""Injury roles, unit matchups and the player-watch line on the CFB card."""

from __future__ import annotations

from pathlib import Path

from cfb_engine.data.advanced import UNITS, parse_advanced
from cfb_engine.data.depth import (
    RESERVE,
    SECOND,
    STARTER,
    build_depth_book,
    depth_for,
    merge_depth,
)
from cfb_engine.data.injuries import InjuryRow
from cfb_engine.data.watch import (
    build_watch_book,
    load_awards,
    parse_big_board,
    parse_heisman,
    watch_for,
)
from cfb_engine.output.brief import GameBrief, TeamBrief, _apply_out, _apply_watch
from cfb_engine.output.card import (
    _spoken_context,
    _unit_edge,
    _unit_sentences,
    _units_line,
    _watch_line,
)

# --- depth --------------------------------------------------------------------


def _usage(team: str, pos: str, name: str, share: float) -> dict[str, object]:
    return {"team": team, "position": pos, "name": name, "usage": {"overall": share}}


def _tackles(team: str, pos: str, name: str, tot: int) -> dict[str, object]:
    return {"team": team, "position": pos, "player": name, "statType": "TOT", "stat": str(tot)}


def test_depth_roles_follow_usage_rank_and_starter_count() -> None:
    book = build_depth_book(
        [
            _usage("Texas", "WR", "A One", 0.30),
            _usage("Texas", "WR", "B Two", 0.25),
            _usage("Texas", "WR", "C Three", 0.20),
            _usage("Texas", "WR", "D Four", 0.10),
            _usage("Texas", "QB", "Arch Manning", 0.9),
            _usage("Texas", "QB", "Backup Guy Jr.", 0.1),
        ],
        [
            _tackles("Texas", "LB", "Tackler A", 30),
            _tackles("Texas", "LB", "Tackler B", 20),
            _tackles("Texas", "LB", "Tackler C", 15),
            _tackles("Texas", "LB", "Tackler D", 5),
            {
                "team": "Texas",
                "position": "LB",
                "player": "Tackler A",
                "statType": "SACKS",
                "stat": "3",
            },
        ],
    )
    roles = {n: depth_for(book, "Texas Longhorns", n).role for n in ("A One", "C Three", "D Four")}  # type: ignore[union-attr]
    assert roles == {"A One": STARTER, "C Three": STARTER, "D Four": SECOND}
    assert depth_for(book, "Texas", "Backup Guy").role == SECOND  # suffix ignored
    assert depth_for(book, "Texas", "Tackler D").role == SECOND
    assert depth_for(book, "Texas", "Nobody") is None
    three = build_depth_book(
        [], [_tackles("Texas", "QB", n, load) for n, load in (("Ay", 10), ("Bee", 9), ("Cee", 8))]
    )
    assert depth_for(three, "Texas", "Cee").role == RESERVE


def test_merge_depth_prefers_this_season_but_fills_from_last() -> None:
    now = build_depth_book([_usage("Miami", "TE", "New Guy", 0.5)], [])
    last = build_depth_book(
        [_usage("Miami", "TE", "New Guy", 0.1), _usage("Miami", "TE", "Luka Gilbert", 0.4)], []
    )
    merged = merge_depth(now, last)
    assert depth_for(merged, "Miami", "New Guy").rank == 1
    assert depth_for(merged, "Miami", "Luka Gilbert").rank == 1  # last season's own rank
    assert depth_for(now, "Miami", "Luka Gilbert") is None


def test_apply_out_tags_roles_and_leaves_unknowns_bare() -> None:
    book = build_depth_book([_usage("Texas", "WR", "Ny Carr", 0.3)], [])
    rows = [
        InjuryRow("Ny Carr", "1", "Texas", "WR", "Out", "Knee", "", ""),
        InjuryRow("Some Lineman", "2", "Texas", "OL", "Out for season", "Knee", "", ""),
        InjuryRow("Maybe Guy", "3", "Texas", "RB", "Questionable", "Ankle", "", ""),
    ]
    tb = TeamBrief(name="Texas")
    _apply_out(tb, {"texas": rows}, book)
    assert tb.out == ["WR Ny Carr (starter)", "OL Some Lineman"]


# --- advanced units -------------------------------------------------------------


def _adv(
    team: str, orush: float, opass: float, drush: float, dpass: float, front: float
) -> dict[str, object]:
    return {
        "team": team,
        "offense": {
            "ppa": 0.1,
            "rushingPlays": {"ppa": orush},
            "passingPlays": {"ppa": opass},
        },
        "defense": {
            "ppa": 0.0,
            "rushingPlays": {"ppa": drush},
            "passingPlays": {"ppa": dpass},
            "havoc": {"total": 0.15, "frontSeven": front},
        },
    }


def test_unit_ranks_one_is_best_with_defense_lower_better() -> None:
    book = parse_advanced(
        [
            _adv("A", 0.3, 0.1, 0.10, 0.30, 0.10),
            _adv("B", 0.1, 0.3, 0.30, 0.10, 0.05),
            _adv("C", 0.2, 0.2, 0.20, 0.20, 0.08),
        ],
        {},
    )
    assert set(UNITS) == {"rush_off", "pass_off", "run_def", "pass_def", "pass_rush"}
    assert book.unit_ranks("A") == {
        "rush_off": 1,
        "pass_off": 3,
        "run_def": 1,
        "pass_def": 3,
        "pass_rush": 1,
    }
    assert book.unit_ranks("B") == {
        "rush_off": 3,
        "pass_off": 1,
        "run_def": 3,
        "pass_def": 1,
        "pass_rush": 3,
    }
    assert book.unit_ranks("Nowhere") == {}


def _brief() -> GameBrief:
    away = TeamBrief(
        name="Texas",
        units={"pass_off": 12, "rush_off": 60, "pass_def": 30, "run_def": 20, "pass_rush": 5},
    )
    home = TeamBrief(
        name="Kansas",
        units={"pass_off": 70, "rush_off": 65, "pass_def": 98, "run_def": 9, "pass_rush": 80},
    )
    return GameBrief(home=home, away=away)


def test_unit_sentences_grade_both_sides_and_call_the_edge() -> None:
    s = _unit_sentences(_brief())
    assert (
        s[0]
        == "Texas' elite pass offense (#12) meets a poor Kansas pass defense (#98) — edge Texas"
    )
    assert (
        s[1]
        == "Texas' middling run offense (#60) meets an elite Kansas run defense (#9) — edge Kansas"
    )
    assert (
        s[2]
        == "Kansas' middling pass offense (#70) meets a good Texas pass defense (#30) — edge Texas"
    )
    assert s[3].endswith("(#20) — edge Texas")
    assert s[4] == "Pass rush (front-seven havoc + sacks): Texas #5, Kansas #80"
    assert (
        _unit_edge(_brief()) == "Texas' passing game (#12) against a Kansas pass defense ranked #98"
    )
    assert "Unit matchups" in _units_line(_brief())
    assert _units_line(GameBrief(home=TeamBrief("A"), away=TeamBrief("B"))) == ""
    assert _unit_edge(GameBrief(home=TeamBrief("A"), away=TeamBrief("B"))) is None


def test_unit_edge_can_favour_the_defense() -> None:
    b = _brief()
    b.away.units = {"pass_off": 40, "pass_def": 30}
    b.home.units = {"pass_def": 45, "rush_off": 100, "run_def": 50}
    b.away.units["run_def"] = 5
    assert _unit_edge(b) == "Texas' #5 run defense against Kansas' #100 running game"


# --- player watch ------------------------------------------------------------------


def test_watch_sources_parse_and_merge(tmp_path: Path) -> None:
    heisman = parse_heisman(
        [
            {
                "name": "Arch Manning",
                "team": "Texas",
                "draftkings_odds": "+1000",
                "fanduel_odds": "+900",
            },
            {"name": "Long Shot", "team": "Texas", "draftkings_odds": "+25000"},
            {"name": "No Price", "team": "Texas", "draftkings_odds": ""},
            "junk",
        ]
    )
    assert heisman == [("Arch Manning", "Texas", 1000), ("Long Shot", "Texas", 25000)]
    page = (
        '<div class="mock-row-pick-number">2</div><div class="mock-row-name">Arch Manning</div>'
        '<div class="mock-row-school-position">QB|Texas</div>'
        '<div class="mock-row-pick-number">70</div><div class="mock-row-name">Late Guy</div>'
        '<div class="mock-row-school-position">OT|Kansas</div>'
    )
    assert parse_big_board(page) == [
        (2, "Arch Manning", "QB", "Texas"),
        (70, "Late Guy", "OT", "Kansas"),
    ]
    awards = tmp_path / "awards_watch.json"
    awards.write_text('{"Butkus": ["Anthony Hill, Texas"], "Bad": "x"}')
    book = build_watch_book(heisman, parse_big_board(page), load_awards(awards))
    names = [p.name for p in watch_for(book, "Texas Longhorns")]
    assert names == ["Arch Manning", "Anthony Hill"]  # +25000 and #70 dropped
    arch = watch_for(book, "Texas")[0]
    assert arch.position == "QB"
    assert arch.tags() == ["Heisman +1000", "1st-round draft stock (#2 big board)"]
    assert watch_for(book, "Texas")[1].tags() == ["Butkus watch list"]
    assert load_awards(None) == {} and load_awards(tmp_path / "missing.json") == {}


def test_watch_line_and_narration_use_the_brief() -> None:
    book = build_watch_book([("Arch Manning", "Texas", 1000)], [], {})
    depth = build_depth_book([_usage("Texas", "QB", "Arch Manning", 0.9)], [])
    b = _brief()
    _apply_watch(b.away, book, depth)
    assert b.away.watch == ["QB Arch Manning — Heisman +1000"]
    b.away.out = ["WR Ny Carr (starter)", "OL Big Man"]
    html = _watch_line(b)
    assert "<b>Player watch</b>" in html and "Arch Manning" in html
    spoken = _spoken_context(b)
    assert "Matchup to watch: Texas' passing game (#12)" in spoken
    assert "Texas without starter Ny Carr, Big Man" in spoken
    assert "Names to know: Arch Manning." in spoken
