import json
from datetime import date

from cbb_engine.output import pdf, teamstats
from engine_runtime import ledger

ZERO = [0] * 15


def _box(
    fgm: int, fga: int, tpm: int, fta: int, orb: int, drb: int, to: int, pts: int
) -> list[int]:
    return [fgm, fga, tpm, 0, 0, fta, orb, drb, orb + drb, 0, 0, 0, to, 0, pts]


def _rows(
    day: str, home: str, away: str, hb: list[int], ab: list[int], poss: float
) -> list[list[str]]:
    box = json.dumps([day, 200, home, away, *hb, *ab, poss])
    pad = [""] * 22
    return [
        [day, "", home, "", away, "H", "", *pad, box],
        [day, "", away, "", home, "A", "", *pad, box],
    ]


def test_season_pools_counts_by_venue_and_ranks_with_ties() -> None:
    logs = _rows(
        "11/4/25",
        "Duke",
        "Army",
        _box(30, 60, 6, 20, 10, 25, 10, 80),
        _box(25, 60, 4, 10, 8, 20, 15, 60),
        70,
    )
    logs += _rows(
        "11/8/25",
        "Army",
        "Duke",
        _box(20, 50, 5, 10, 5, 20, 14, 50),
        _box(28, 55, 8, 15, 12, 30, 10, 75),
        65,
    )
    logs += _rows(
        "11/9/25", "Duke", "Navy", _box(1, 1, 0, 0, 0, 0, 0, 1), ZERO, 1
    )  # on the card day: excluded
    s = teamstats.season(
        logs, date(2025, 11, 9), [{"team": "Duke", "adjoe": "120.5", "adjde": "95.0"}]
    )
    duke = s["Duke"]
    assert (duke.games, duke.home.games, duke.away.games) == (2, 1, 1)
    assert duke.home.off.value == 100 * 80 / 70 and duke.away.dfn.value == 100 * 50 / 65
    assert duke.offense.efg.value == (58 + 0.5 * 14) / 115
    assert duke.offense.orb.value == 22 / (22 + 40)
    assert duke.defense.tov.value == 29 / 135 and duke.offense.ftr.value == 35 / 115
    assert duke.adj_off.value == 120.5 and s["Army"].adj_off.rank is None
    assert duke.last_game == date(2025, 11, 8) and "Navy" not in s
    assert (
        teamstats._ranks({"a": 1.0, "b": 1.0, "c": 0.5, "d": None}, high_is_good=True)["c"].rank
        == 3
    )


def test_match_team_prefers_the_longest_school_and_aliases() -> None:
    torvik = ["Michigan", "Michigan St.", "Kansas", "UMKC", "Connecticut", "Miami FL", "Miami OH"]
    assert teamstats.match_team("Michigan St Spartans", torvik) == "Michigan St."
    assert teamstats.match_team("Michigan Wolverines", torvik) == "Michigan"
    assert teamstats.match_team("Kansas City Roos", torvik) == "UMKC"
    assert teamstats.match_team("UConn Huskies", torvik) == "Connecticut"
    assert teamstats.match_team("Miami Hurricanes", torvik) == "Miami FL"
    assert teamstats.match_team("Gonzaga Bulldogs", torvik) is None


def test_card_html_has_board_and_panel_for_each_game() -> None:
    logs = _rows(
        "11/4/25",
        "Duke",
        "Army",
        _box(30, 60, 6, 20, 10, 25, 10, 80),
        _box(25, 60, 4, 10, 8, 20, 15, 60),
        70,
    )
    stats = teamstats.season(logs, date(2025, 11, 10))
    row = ledger.Row(
        slate_date="2025-11-10",
        event_id="e1",
        tip_utc="2025-11-11T00:00:00Z",
        market="game_total",
        side="over",
        line=141.5,
        book="draftkings",
        pass_id="p",
        pass_kind="featured",
        bet_of_record=True,
        status="live",
        american=-110,
        p_fair=0.55,
        edge=0.05,
        tier="Lean",
    )
    body = pdf.build_html(
        date(2025, 11, 10), [row], {"e1": ("Duke Blue Devils", "Army Black Knights")}, stats
    )
    assert "Army Black Knights at Duke Blue Devils" in body and "7:00 PM ET" in body
    assert "class='buy'" in body and "Four factors" in body and "Off rtg" in body
    assert "Duke 5d rest" in body
    bare = pdf.build_html(date(2025, 11, 10), [], {"e2": ("Gonzaga Bulldogs", "Utah Utes")}, {})
    assert "no Torvik games logged yet" in bare


def _q(at: str, market: str, side: str, line: float | None, american: float, book: str) -> object:
    from engine_runtime.quotes import Quote

    return Quote(
        captured_at=at,
        game_date="2025-11-10",
        event_id="e1",
        tip_utc="2025-11-11T00:00:00Z",
        market=market,
        side=side,
        entity="",
        line=line,
        book=book,
        american=american,
        opposite_american=None,
        pass_id="p",
        pass_kind="board",
    )


def test_line_movement_is_median_first_to_last_pass_before_the_card() -> None:
    from cbb_engine.output import movement

    t0, t1, t2 = "2025-11-10T14:00:00Z", "2025-11-10T20:00:00Z", "2025-11-10T23:00:00Z"
    qs = [
        _q(t0, "game_total", "over", 140.5, -110, "dk"),
        _q(t0, "game_total", "over", 141.5, -110, "mgm"),
        _q(t0, "game_total", "under", 141.5, -110, "mgm"),
        _q(t1, "game_total", "over", 139.5, -110, "dk"),
        _q(t1, "game_total", "over", 139.5, -110, "mgm"),
        _q(t0, "game_ats", "home", -3.5, -110, "dk"),
        _q(t1, "game_ats", "home", -5.0, -110, "dk"),
        _q(t2, "game_ats", "home", -7.0, -110, "dk"),  # after the card
        _q(t0, "game_ml", "home", -160, 0, "dk"),
    ]
    m = movement.moves(qs, until=t1)["e1"]  # type: ignore[arg-type]
    assert [(x.market, x.opened, x.now) for x in m] == [
        ("game_total", 141.0, 139.5),
        ("game_ats", -3.5, -5.0),
    ]


def test_unit_edge_rest_and_injuries_lines() -> None:
    from cbb_engine.data.injuries import Injury

    logs = _rows(
        "11/4/25",
        "Duke",
        "Army",
        _box(30, 60, 6, 20, 10, 25, 10, 80),
        _box(25, 60, 4, 10, 8, 20, 15, 60),
        70,
    )
    logs += _rows(
        "11/8/25",
        "Navy",
        "Army",
        _box(20, 60, 2, 5, 5, 20, 15, 50),
        _box(20, 60, 2, 5, 5, 20, 15, 52),
        70,
    )
    s = teamstats.season(logs, date(2025, 11, 9))
    assert pdf.unit_edge(s["Duke"], s["Army"], "home", "away", "Duke", "Army").startswith(
        "Duke O vs Army D: even (off rtg #1 vs def rtg #1); offense better in 4 of 4 factors"
    )
    assert pdf.unit_edge(s["Army"], s["Duke"], "away", "home", "Army", "Duke").startswith(
        "Army O vs Duke D: edge Army offense (off rtg #1 vs def rtg #2)"
    )
    assert pdf.rest_edge(s["Army"], s["Duke"], date(2025, 11, 9), "Army", "Duke") == (
        "Duke +4 days rest (Army on a back-to-back)"
    )
    inj = [
        Injury("A Guard", "Duke", "G", "Ankle", "Out"),
        Injury("B Big", "Oklahoma State", "F", "Knee", "Out"),
    ]
    body = pdf.build_html(
        date(2025, 11, 9), [], {"e1": ("Duke Blue Devils", "Army Black Knights")}, s, injuries=inj
    )
    assert "Duke: A Guard (G, Ankle) Out" in body and "Army: none listed" in body
    assert "not matched to a Torvik team (not shown): Oklahoma State" in body


def test_factor_winner_and_bold_marks_the_edge() -> None:
    assert pdf.factor_winner("efg", 0.569, 0.459) == "O"
    assert pdf.factor_winner("tov", 0.122, 0.156) == "O"
    assert pdf.factor_winner("tov", 0.20, 0.15) == "D"
    assert pdf.factor_winner("orb", 0.30, 0.30) is None
    assert pdf.factor_winner("ftr", None, 0.3) is None
