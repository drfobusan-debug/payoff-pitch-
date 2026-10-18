"""Close test: fixed decision rule, resumable observation CSV, official-starter overrides."""

from __future__ import annotations

from pathlib import Path

from nhl_engine import closetest
from nhl_engine.closetest import Obs


def _obs(
    market: str, p_sim: float, p_close: float, result: float, line: float | None = None
) -> Obs:
    return Obs(
        "2024-01-11",
        "BOS @ BUF",
        market,
        line,
        p_sim,
        p_close,
        result,
        6.0,
        7,
        "confirmed/confirmed",
    )


def test_fit_recovers_slope_and_verdicts() -> None:
    # result miss = 0.5 * disagreement exactly -> slope 0.5, se 0
    rows = [_obs("game_ml", 0.5 + d, 0.5, 0.5 + 0.5 * d) for d in (-0.2, -0.1, 0.1, 0.2)]
    f = closetest.fit(rows, "game_ml")
    assert f.n == 4 and abs(f.slope - 0.5) < 1e-9 and f.se < 1e-9
    assert f.verdict.startswith("anchor ~ 0.50")
    noisy = [
        _obs("game_total", 0.5 + d, 0.5, r, 6.5)
        for d, r in ((0.1, 0), (-0.1, 1), (0.2, 1), (-0.2, 0))
    ]
    g = closetest.fit(noisy, "game_total")
    assert abs(g.slope) <= g.se and "graded-only" in g.verdict
    pushes = [_obs("game_total", 0.6, 0.5, 0.5, 6.0)]
    assert closetest.fit(pushes, "game_total").n == 0


def test_csv_roundtrip_and_report(tmp_path: Path) -> None:
    rows = [_obs("game_ml", 0.55, 0.5, 1.0), _obs("game_total", 0.45, 0.5, 1.0, 6.5)]
    p = tmp_path / "o.csv"
    assert closetest.write_csv(rows, p) == 2
    assert closetest.read_csv(p) == rows
    text = closetest.report(rows)
    assert "game_ml: n=1" in text and "goal level: n=1" in text


def test_goalie_overrides_use_official_starter_ids() -> None:
    from datetime import date as Date

    from nhl_engine.schemas import Game, GameResult, PeriodScore, PlayerLine

    g = Game(game_date=Date(2024, 1, 11), home="BUF", away="OTT", start_utc="2024-01-12T00:00:00Z")

    def _g(pid: int, team: str, name: str, starter: bool) -> PlayerLine:
        return PlayerLine(pid, team, name, "G", 60.0, 0, 0, 0, 0, 0, 30, 32, starter)

    res = GameResult(
        nhl_game_id=1,
        game_date=g.game_date,
        away="OTT",
        home="BUF",
        state="OFF",
        periods=(PeriodScore(1, "REG", 3, 5),),
        decided="REG",
        players=(
            _g(8476341, "OTT", "A. Forsberg", True),
            _g(8476914, "OTT", "J. Korpisalo", False),
            _g(8480045, "BUF", "U. Luukkonen", True),
        ),
    )
    ov = closetest.goalie_overrides([g], {g.matchup: res})
    assert ov["BUF"]["player_id"] == 8480045 and ov["OTT"]["player_id"] == 8476341
    assert ov["OTT"]["status"] == "confirmed"
    assert closetest.goalie_overrides([g], {}) == {}
