import json
from datetime import date

from cbb_engine.output import pdf, teamstats
from cbb_engine.output.teamstats import Rated
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


def test_factor_winner_compares_margins_from_the_d1_average() -> None:
    lg = pdf.League(
        {
            ("offense", "efg"): (0.51, 0.03),
            ("defense", "efg"): (0.51, 0.03),
            ("offense", "tov"): (0.17, 0.02),
            ("defense", "tov"): (0.17, 0.02),
        }
    )
    # +2.0 SD offense vs a defense holding teams 2.7 SD under: the defense, though 56.9% > 45.9%
    assert pdf.factor_winner(lg, "efg", 0.569, 0.429) == "D"
    assert pdf.factor_winner(lg, "efg", 0.60, 0.48) == "O"
    assert pdf.factor_winner(lg, "efg", 0.57, 0.45) is None  # 2.0 vs 2.0 SD
    # TOV%: low is good for an offense, high is good for a defense
    assert pdf.factor_winner(lg, "tov", 0.13, 0.18) == "O"
    assert pdf.factor_winner(lg, "tov", 0.18, 0.21) == "D"
    assert pdf.factor_winner(lg, "orb", 0.3, 0.3) is None  # no league spread known


def test_unit_edge_uses_adjusted_margins() -> None:
    from cbb_engine.output.teamstats import Factors, Rated, Split, TeamStats

    def team(name: str, ao: float, ad: float) -> TeamStats:
        return TeamStats(name, 5, Split(), Split(), Factors(), Factors(), Rated(ao), Rated(ad))

    lg = pdf.League({}, adj_off=105.0, adj_def=105.0)
    ark, fla = team("Arkansas", 128.5, 102.7), team("Florida", 126.9, 93.2)
    assert pdf.unit_edge(lg, ark, fla, "Arkansas", "Florida").startswith(
        "Arkansas O vs Florida D: edge Arkansas offense by 11.7 pts/100 (Adj O 128.5, +23.5 vs D-I avg;"
        " Adj D 93.2, +11.8 better than avg); factors: offense 0, defense 0, even 4"
    )
    assert pdf.unit_edge(lg, team("A", 105.5, 105), team("B", 105, 105), "A", "B").startswith(
        "A O vs B D: even ("
    )
    assert "adjusted ratings unavailable" in pdf.unit_edge(
        pdf.League({}), ark, fla, "Arkansas", "Florida"
    )


def test_factor_cells_show_signed_z_with_raw_rate() -> None:
    lg = pdf.League({("offense", "tov"): (0.17, 0.02), ("defense", "tov"): (0.17, 0.02)})
    assert pdf._zcell(lg, "offense", "tov", Rated(0.13)).startswith("+2.0 ")
    assert pdf._zcell(lg, "defense", "tov", Rated(0.13)).startswith("-2.0 ")
    assert "13.0%" in pdf._zcell(lg, "defense", "tov", Rated(0.13))
    assert pdf._zcell(lg, "offense", "efg", Rated(0.5)).startswith("—")


def test_manifest_teams_reads_the_top_level_teams_written_by_capture(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from cbb_engine.cli import _manifest_teams
    from cbb_engine.config import SPORT
    from engine_runtime import quotes

    q = _q("2025-11-10T14:00:00Z", "game_total", "over", 140.5, -110, "dk")
    quotes.write_shard(
        tmp_path,
        SPORT,
        "2025-11-10",
        [q],
        pass_id="p1",
        pass_kind="board",  # type: ignore[list-item]
        extra={"teams": {"e1": ["Duke Blue Devils", "Army Black Knights"]}},
    )
    assert _manifest_teams(tmp_path, date(2025, 11, 10)) == {
        "e1": ("Duke Blue Devils", "Army Black Knights")
    }


def test_board_shows_the_live_buy_over_a_capped_higher_edge_book() -> None:
    from dataclasses import replace

    base = ledger.Row(
        slate_date="2025-11-10",
        event_id="e1",
        tip_utc="2025-11-11T00:00:00Z",
        market="game_total",
        side="over",
        line=141.5,
        book="betmgm",
        pass_id="p",
        pass_kind="featured",
        bet_of_record=True,
        status="live",
        american=-105,
        p_fair=0.56,
        edge=0.06,
        tier="Pass",
        gates=["exposure_cap"],
    )
    buy = replace(base, book="draftkings", american=-110, edge=0.04, tier="Lean", gates=[])
    body = pdf._board([base, buy])
    assert "draftkings" in body and "betmgm" not in body and "class='buy'" in body


def test_injury_parse_reads_either_rotowire_field_names() -> None:
    from cbb_engine.data import injuries

    a, b = injuries.parse(
        [
            {"player": "A", "team": "Duke", "position": "G", "injury": "Knee", "status": "Out"},
            {
                "player": "B",
                "team": "Duke",
                "position": "F",
                "injury_type": "Ankle",
                "IR": "Game Time Decision",
            },
        ]
    )
    assert (a.injury, a.status) == ("Knee", "Out")
    assert (b.injury, b.status) == ("Ankle", "Game Time Decision")
