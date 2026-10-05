import json
from datetime import date
from pathlib import Path

from nba_engine.data.espn import parse_box, parse_scoreboard

FIX = Path(__file__).parent / "fixtures"


def test_scoreboard_settles_full_game_and_first_half():
    games = parse_scoreboard(
        json.loads((FIX / "espn_scoreboard_20251210.json").read_text()), date(2025, 12, 10)
    )
    by_id = {g.espn_id: g for g in games}
    okc = by_id["401809835"]
    assert (okc.away, okc.home, okc.final_away, okc.final_home) == ("PHX", "OKC", 89, 138)
    assert (okc.h1_away, okc.h1_home, okc.overtimes) == (48, 74, 0)
    assert okc.is_final
    assert by_id["401809836"].away == "SAS"  # ESPN's "SA"


def test_overtime_counts_extra_periods():
    from nba_engine.schemas import GameResult

    g = GameResult(
        "1", date(2025, 1, 1), "A", "B", "STATUS_FINAL", (30, 25, 20, 25, 10), (25, 25, 25, 25, 8)
    )
    assert (g.overtimes, g.final_away, g.final_home, g.h1_away) == (1, 110, 108, 55)


def test_box_lines_and_dnp():
    players = parse_box(json.loads((FIX / "espn_summary_401809835.json").read_text()))
    sga = next(p for p in players if p.name == "Shai Gilgeous-Alexander")
    assert (sga.team, sga.minutes, sga.points) == ("OKC", 27, 28)
    assert sga.pra == sga.points + sga.rebounds + sga.assists
    dnp = parse_box(
        {
            "boxscore": {
                "players": [
                    {
                        "team": {"abbreviation": "OKC"},
                        "statistics": [
                            {
                                "keys": ["minutes"],
                                "athletes": [
                                    {"athlete": {"id": "9"}, "didNotPlay": True, "stats": []}
                                ],
                            }
                        ],
                    }
                ]
            }
        }
    )
    assert dnp[0].dnp
