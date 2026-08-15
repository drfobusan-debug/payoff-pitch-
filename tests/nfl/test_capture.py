"""The price archive: round trip, idempotence, and the two sides of a spread."""

from __future__ import annotations

import numpy as np

from nfl_engine.data import capture
from nfl_engine.market.board import GameOdds, MarketQuote
from nfl_engine.models.distribution import ScoreDistribution

HOME, AWAY = "KC", "BUF"
MATCHUP = f"{AWAY} @ {HOME}"
TAKEN = "2026-09-10T18:00:00Z"


def board() -> dict[str, GameOdds]:
    odds = GameOdds(matchup=MATCHUP)
    odds.add_ml(HOME, MarketQuote("dk", -150, 130))
    odds.add_ml(AWAY, MarketQuote("dk", 130, -150))
    odds.add_spread(-3.0, HOME, MarketQuote("dk", -110, -110))
    odds.add_spread(-3.0, AWAY, MarketQuote("dk", -110, -110))
    odds.add_spread(-2.5, HOME, MarketQuote("fd", -120, 100))
    odds.add_total(47.5, True, MarketQuote("dk", -105, -115))
    odds.add_total(47.5, False, MarketQuote("dk", -115, -105))
    return {MATCHUP: odds}


def rows() -> list[capture.QuoteRow]:
    return capture.rows_from_board(
        board(), season=2026, week=1, captured_at=TAKEN, dates={MATCHUP: "2026-09-13"}
    )


def test_rows_carry_the_pair_and_the_rung():
    archived = rows()
    assert len(archived) == 7
    spread = {(row.side, row.line) for row in archived if row.market == "spread"}
    # The away side is stored on its own handicap, not the home axis.
    assert spread == {(HOME, -3.0), (AWAY, 3.0), (HOME, -2.5)}
    assert all(row.captured_at == TAKEN for row in archived)
    assert all(row.game_date == "2026-09-13" for row in archived)


def test_board_round_trips_through_the_archive():
    rebuilt = capture.board_from_rows(rows())[MATCHUP]
    original = board()[MATCHUP]
    assert rebuilt.ml.keys() == original.ml.keys()
    assert sorted(rebuilt.spreads) == sorted(original.spreads)
    assert rebuilt.consensus_home_spread() == original.consensus_home_spread()
    assert rebuilt.consensus_total() == original.consensus_total()
    assert rebuilt.spreads[-3.0][AWAY][0].american == -110


def test_repeated_capture_of_an_unmoved_board_writes_nothing(tmp_path):
    first = capture.write_snapshot(rows(), season=2026, week=1, root=tmp_path)
    assert first is not None
    again = [row for row in rows()]
    assert capture.write_snapshot(again, season=2026, week=1, root=tmp_path) is None
    assert len(capture.snapshot_paths(2026, 1, root=tmp_path)) == 1


def test_a_moved_price_is_a_new_snapshot(tmp_path):
    capture.write_snapshot(rows(), season=2026, week=1, root=tmp_path)
    moved = [
        row if row.market != "moneyline" else capture.QuoteRow(**{**row.__dict__, "american": -160})
        for row in rows()
    ]
    second = capture.write_snapshot(moved, season=2026, week=1, root=tmp_path)
    assert second is not None
    assert len(capture.snapshot_paths(2026, 1, root=tmp_path)) == 2


def test_snapshot_survives_the_csv(tmp_path):
    path = capture.write_snapshot(rows(), season=2026, week=1, root=tmp_path)
    assert path is not None
    assert capture.fingerprint(capture.read_snapshot(path)) == capture.fingerprint(rows())


def test_captured_at_does_not_change_the_fingerprint():
    later = [capture.QuoteRow(**{**row.__dict__, "captured_at": "2026-09-11T00:00:00Z"}) for row in rows()]
    assert capture.fingerprint(later) == capture.fingerprint(rows())


def test_the_two_sides_of_a_spread_are_complementary():
    """The invariant that catches an away side priced off the home axis.

    Negating the home handicap gives the home team's probability at the mirrored
    number, not the away team's at its own -- so P(home covers) + P(away covers)
    would come to 1.20 instead of 1, and a fair-looking away dog would read as a
    23-point edge.
    """
    margins = np.array([-10, -7, -3, -3, 1, 3, 3, 6, 7, 14])
    dist = ScoreDistribution(home=np.full(len(margins), 24), away=24 - margins)
    for point in (-7.5, -3.5, -3.0, 0.0, 2.5, 7.0):
        home = dist.spread(point)
        away = dist.spread(-point, home=False)
        assert home.push == away.push
        assert abs(home.win + away.win + home.push - 1.0) < 1e-9
