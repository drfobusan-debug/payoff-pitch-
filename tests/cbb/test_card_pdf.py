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
