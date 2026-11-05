from __future__ import annotations

import random

import numpy as np

from scripts import live_total_study as lts


def _item(name: str, ou: float | None, open_: str = "", close: str = "") -> dict:
    return {
        "provider": {"name": name},
        "overUnder": ou,
        "open": {"total": {"american": open_}},
        "close": {"total": {"american": close}},
    }


def test_odds_drop_live_and_model_books_and_signed_prices() -> None:
    doc = {
        "items": [
            _item("ESPN BET", 148.5, "149.5", "-135"),
            _item("ESPN Bet - Live Odds", 155.5),
            _item("teamrankings", 140.0),
            _item("DraftKings", 148.0),
        ]
    }
    got = lts.parse_odds(doc)
    assert got["books"] == {"ESPN BET": 148.5, "DraftKings": 148.0}
    assert got["open"] == 149.5
    assert got["close"] is None  # "-135" is a price, not a line


def test_consensus_drops_a_bad_book() -> None:
    assert lts.consensus({"a": 218.0, "b": 217.5, "c": 57.0, "d": 218.5}) == 218.0
    assert lts.consensus({}) is None


def _ev(home: list[int], away: list[int], *, done: bool = True, kind: int = 2) -> dict:
    def team(side: str, pts: list[int]) -> dict:
        return {"homeAway": side, "score": str(sum(pts)), "linescores": [{"value": p} for p in pts]}

    return {
        "id": "1",
        "season": {"type": kind},
        "competitions": [
            {
                "status": {"type": {"completed": done}},
                "competitors": [team("away", away), team("home", home)],
            }
        ],
    }


def test_game_needs_a_final_regular_season_line_score_that_adds_up() -> None:
    g = lts.parse_game(_ev([1, 0, 1, 0, 1], [0, 1, 1, 0, 0]), "2025-01-01", "2024-25")
    assert g is not None and g["home"] == [1, 0, 1, 0, 1] and g["away"] == [0, 1, 1, 0, 0]
    assert lts.parse_game(_ev([1], [0], done=False), "d", "s") is None
    assert lts.parse_game(_ev([1], [0], kind=3), "d", "s") is None
    bad = _ev([10, 10], [10, 10])
    bad["competitions"][0]["competitors"][0]["score"] = "25"
    assert lts.parse_game(bad, "d", "s") is None


def test_ceiling_is_a_coin_when_the_start_simply_stays_in() -> None:
    rnd = random.Random(7)
    rows = []
    for i in range(4000):
        t = 220.0
        q = [round(rnd.gauss(t / 4, 7)) for _ in range(4)]
        home = [x // 2 for x in q]
        away = [x - h for x, h in zip(q, home, strict=True)]
        rows.append(
            lts.Game(
                str(i),
                f"2025-{1 + i // 400:02d}-{1 + i % 28:02d}",
                str(i // 1000),
                t,
                home,
                away,
                None,
                None,
            )
        )
    w = lts.walk(rows, 2)
    a, b, se, _ = lts._fit(
        np.array([x[2] for x in w]),
        np.array([x[0].final - x[0].pregame for x in w]),
    )
    assert abs(b - 1) < 3 * se
    slope, _ = lts.ceiling(w, lts.SPORTS["nba"], 2)
    assert float(slope.split("%")[0]) < 54
