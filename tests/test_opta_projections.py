"""Reading an outside model off a page that owes us no contract.

VSIN's Opta table is scraped, not consumed from an API, so the failure mode
that matters is not "it broke" -- a missing benchmark costs nothing -- but "it
kept working and the numbers moved". Each test here pins a place where the
page's conventions differ from the engine's and a plausible parse would be
quietly wrong: which side a probability refers to, which team is at home, what
"2+" means, and which club "SDG" is.
"""

from __future__ import annotations

from mlb_engine.data import opta


def _row(
    player: str = "Alan Roden",
    matchup: str = "MIN @ MIL",
    line: str = "0.5",
    odds: str = "O -229 U +170",
    model: str = "U +130",
    edge: str = "+6.4%",
    bet: str = "UNDER",
    conf: str = "&#9733;&#9733;",
    result: str = "HIT 0",
    player_id: int = 31763,
) -> str:
    return (
        "<tr>"
        f'<td><a onclick="loadPlayer({player_id})">{player}</a>'
        f"<div>{matchup} &middot; 2:10PM</div></td>"
        "<td>0.83 &#9660;</td>"
        f"<td>{line} &#9660;</td>"
        f"<td>{odds}</td>"
        f'<td><span style="color:#4b8119">{model}</span></td>'
        f'<td><span class="rank-badge">{edge}</span></td>'
        f"<td><span>{bet}</span></td>"
        f"<td><span>{conf}</span></td>"
        f"<td><span>{result}</span></td>"
        "<td>40% 0.40</td><td>30% 0.30</td><td>47% 0.60</td><td>56% 0.87</td>"
        "</tr>"
    )


def test_it_reads_a_projection_row_off_the_board() -> None:
    row = opta._parse_row(_row(), "Hits", "2026-08-08")
    assert row is not None
    assert row.player == "Alan Roden"
    assert row.player_id == 31763
    assert row.matchup == "MIN @ MIL"
    assert row.market == "batter_h"
    assert row.selection == "Alan Roden H o0.5"
    assert row.line == 0.5
    assert row.projection == 0.83
    assert row.over_odds == -229.0
    assert row.under_odds == 170.0
    assert row.edge == 0.064
    assert row.bet == "under"
    assert row.confidence == 2
    assert row.result == "hit"
    assert row.actual == 0.0


def test_an_under_price_is_flipped_to_the_probability_of_the_over() -> None:
    """Opta quotes whichever side it likes; the engine only ever means the over.

    Taken at face value, ``U +130`` would be stored as a 43% chance of a bet the
    engine reads as the over, so the benchmark would look badly calibrated and
    the disagreement would be an artefact of notation.
    """
    under = opta._parse_row(_row(model="U +130"), "Hits", "2026-08-08")
    over = opta._parse_row(_row(model="O -113"), "Hits", "2026-08-08")
    assert under is not None and over is not None
    assert under.over_prob is not None and over.over_prob is not None
    assert round(under.over_prob, 4) == round(1 - 100 / 230, 4)
    assert round(over.over_prob, 4) == round(113 / 213, 4)


def test_a_milestone_target_is_the_over_on_the_half_point_below_it() -> None:
    """"2+ hits" and "o1.5 hits" are one bet and must share one key."""
    ms = opta._parse_row(_row(line="2+", odds="+141"), "Hits", "2026-08-08")
    ou = opta._parse_row(_row(line="1.5"), "Hits", "2026-08-08")
    assert ms is not None and ou is not None
    assert ms.line == ou.line == 1.5
    assert ms.selection == ou.selection
    # The milestone tab prints one price, so only the over is known there.
    assert ms.over_odds == 141.0
    assert ms.under_odds is None


def test_vs_puts_the_row_s_own_team_at_home() -> None:
    """VSIN writes the player's club first, so "vs" reverses the engine's order."""
    away = opta._parse_row(_row(matchup="MIN @ MIL"), "Hits", "2026-08-08")
    home = opta._parse_row(_row(matchup="MIL vs MIN"), "Hits", "2026-08-08")
    assert away is not None and home is not None
    assert away.matchup == home.matchup == "MIN @ MIL"


def test_vsin_s_team_codes_are_translated_to_the_engine_s() -> None:
    """Seven clubs are spelled differently; unmapped, they never join a slate."""
    row = opta._parse_row(_row(matchup="SDG @ TAM"), "Hits", "2026-08-08")
    assert row is not None
    assert row.matchup == "SD @ TB"
    for vsin, engine in (("KAN", "KC"), ("WAS", "WSH"), ("CHW", "CWS"), ("ARI", "AZ")):
        got = opta._parse_row(_row(matchup=f"{vsin} @ BOS"), "Hits", "2026-08-08")
        assert got is not None
        assert got.matchup == f"{engine} @ BOS"


def test_hits_allowed_is_not_the_batter_s_hits() -> None:
    """Both tabs call a column "Hits"; only the stat code separates them."""
    batter = opta._parse_row(_row(), "Hits", "2026-08-08")
    pitcher = opta._parse_row(_row(player="Shane Bieber", line="5.5"), "HA", "2026-08-08")
    assert batter is not None and pitcher is not None
    assert batter.market == "batter_h"
    assert pitcher.market == "pitcher_h"
    assert pitcher.selection == "Shane Bieber Hits o5.5"


def test_an_ungraded_row_has_no_result_rather_than_a_loss() -> None:
    """Before first pitch every prop is open, not lost."""
    row = opta._parse_row(_row(result="&mdash;"), "Hits", "2026-08-08")
    assert row is not None
    assert row.result is None
    assert row.actual is None


def test_a_row_without_a_line_is_dropped_rather_than_keyed_on_nothing() -> None:
    """Every lineless row shares one key, so keeping them evicts real rows."""
    assert opta._parse_row(_row(line="&mdash;"), "Hits", "2026-08-08") is None


def test_a_header_row_is_not_mistaken_for_a_player() -> None:
    header = "<tr><th>Player</th><th>Proj</th><th>Line</th><th>Odds</th>" \
        "<th>Model</th><th>Edge</th><th>Bet</th><th>Conf</th><th>Result</th></tr>"
    assert opta._parse_row(header, "Hits", "2026-08-08") is None


def test_a_later_capture_grades_the_morning_s_rows_without_losing_them() -> None:
    """The point of running it twice: projections first, outcomes after."""
    morning = opta._parse_row(_row(result="&mdash;"), "Hits", "2026-08-08")
    evening = opta._parse_row(_row(result="HIT 2"), "Hits", "2026-08-08")
    other = opta._parse_row(_row(player="Max Muncy"), "Hits", "2026-08-08")
    assert morning is not None and evening is not None and other is not None
    merged = opta.merge_rows([morning, other], [evening])
    assert len(merged) == 2
    graded = {r.player: r.result for r in merged}
    assert graded["Alan Roden"] == "hit"
    assert graded["Max Muncy"] == "hit"


def test_a_dead_page_yields_no_projections_instead_of_raising() -> None:
    """A benchmark going dark must never take the slate down with it."""

    class _Down(opta.OptaClient):
        def _get(self, url: str, **params: object) -> str:
            return ""

    assert _Down().fetch(day=0, date="2026-08-08") == []
    assert _Down().slate_dates() == {}


def test_the_day_offset_is_read_off_the_page_not_assumed_to_be_today() -> None:
    """``day=0`` is the slate VSIN is showing, which is not always today's date.

    At 3:34am ET on Aug 10 the page still called Aug 9 ``day=0``. Assuming
    otherwise files a capture under a date whose games have not been played.
    """

    class _Page(opta.OptaClient):
        def _get(self, url: str, **params: object) -> str:
            return (
                '<button data-day="-1" onclick="switchDay(-1)">Sat Aug 8</button>'
                '<button data-day="0" onclick="switchDay(0)">Sun Aug 9</button>'
                '<button data-day="1" onclick="switchDay(1)">Mon Aug 10</button>'
            )

    dates = _Page().slate_dates()
    assert dates[-1].endswith("-08-08")
    assert dates[0].endswith("-08-09")
    assert dates[1].endswith("-08-10")


def test_a_reopened_row_cannot_ungrade_a_graded_one() -> None:
    """Captures cross between machines out of order; only grading moves forward.

    A morning projection pushed from one box after the evening's results were
    pushed from another would, under "later wins", blank the outcome.
    """
    open_row = opta._parse_row(_row(result="&mdash;"), "Hits", "2026-08-08")
    graded = opta._parse_row(_row(result="HIT 2"), "Hits", "2026-08-08")
    assert open_row is not None and graded is not None
    assert opta.merge_rows([graded], [open_row])[0].result == "hit"
    assert opta.merge_rows([open_row], [graded])[0].result == "hit"
