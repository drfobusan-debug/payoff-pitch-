"""Replay a played week at its closing prices, through the live code path.

The dry run has an availability problem: the operational layer -- capture, price,
stamp the close, grade, report -- can only be exercised on games that have both
prices and a final score, and in the off-season there are none. Replay solves it
by building a board out of the closing numbers nflverse ships for every game back
to 1999 and feeding it to exactly the functions the live commands call. Nothing is
special-cased: :func:`nfl_engine.pipeline.price_slate` cannot tell a replayed
board from a live one.

**What a replay can and cannot prove.** It proves the plumbing: that rows are
written once, graded on the side taken, settled against the right final score, and
that CLV, ROI and the PPV/NPV report come out of real outcomes. It cannot prove an
execution edge, and it is important to be clear why: the replayed board has one
book, so the de-vigged consensus *is* the price taken and the execution edge is
zero by construction. Every replayed row is therefore a ``Pass``. That is the
correct answer, not a defect -- an engine that manufactured a buy against the only
price on the board would be inventing the edge it claims to find. Live dispersion
across books is what creates a buy, and it exists only on a live board.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date as Date

import pandas as pd

from nfl_engine.data import nflverse
from nfl_engine.data.capture import QuoteRow, rows_from_board
from nfl_engine.market.board import GameOdds, MarketQuote
from nfl_engine.schemas import Game, GameEnvironment, TeamGameInfo

log = logging.getLogger(__name__)

CLOSE_BOOK = "close"
NFLVERSE = "nflverse_close"
# nflverse quotes the closing spread and total with no price attached in some
# older seasons. A missing price is left missing rather than assumed to be -110:
# the whole point of the archive is that the price is the thing being measured.


@dataclass(frozen=True)
class ReplayWeek:
    season: int
    week: int
    first_day: Date
    games: list[Game]
    board: dict[str, GameOdds] = field(default_factory=dict)
    finals: dict[str, tuple[int, int]] = field(default_factory=dict)  # (home, away)

    def quote_rows(self, captured_at: str) -> list[QuoteRow]:
        dates = {game.matchup(): game.game_date.isoformat() for game in self.games}
        return rows_from_board(
            self.board,
            season=self.season,
            week=self.week,
            captured_at=captured_at,
            dates=dates,
            source=NFLVERSE,
        )


def played_weeks(season: int, weeks: list[int] | None = None) -> list[ReplayWeek]:
    """Every week of ``season`` with final scores, oldest first."""
    games = nflverse.games()
    frame = games[games.season == season]
    frame = frame[frame.home_score.notna() & frame.away_score.notna()]
    if weeks:
        frame = frame[frame.week.isin(weeks)]
    out: list[ReplayWeek] = []
    for week in sorted({int(value) for value in frame.week}):
        replay = _week(frame[frame.week == week], season, week)
        if replay.games:
            out.append(replay)
    return out


def _week(frame: pd.DataFrame, season: int, week: int) -> ReplayWeek:
    games: list[Game] = []
    board: dict[str, GameOdds] = {}
    finals: dict[str, tuple[int, int]] = {}
    for row in frame.to_dict("records"):
        home, away = str(row["home_team"]), str(row["away_team"])
        matchup = f"{away} @ {home}"
        games.append(
            Game(
                game_id=str(row["game_id"]),
                season=season,
                week=week,
                game_date=Date.fromisoformat(str(row["gameday"])),
                home=TeamGameInfo(name=home, abbrev=home, is_home=True),
                away=TeamGameInfo(name=away, abbrev=away, is_home=False),
                env=GameEnvironment(
                    roof=_text(row.get("roof")),
                    surface=_text(row.get("surface")),
                    temp_f=_number(row.get("temp")),
                    wind_mph=_number(row.get("wind")),
                ),
                home_rest=_int(row.get("home_rest")),
                away_rest=_int(row.get("away_rest")),
                # Named on the schedule, and announced before kickoff in life, so
                # a replay may read them without seeing the future.
                home_qb_id=_text(row.get("home_qb_id")),
                away_qb_id=_text(row.get("away_qb_id")),
            )
        )
        board[matchup] = _odds(row, matchup, home, away)
        finals[matchup] = (int(row["home_score"]), int(row["away_score"]))
    first = min((game.game_date for game in games), default=Date.today())
    return ReplayWeek(season, week, first, games, board, finals)


def _odds(row: dict[str, object], matchup: str, home: str, away: str) -> GameOdds:
    odds = GameOdds(matchup=matchup)
    home_ml = _number(row.get("home_moneyline"))
    away_ml = _number(row.get("away_moneyline"))
    if home_ml is not None and away_ml is not None:
        odds.add_ml(home, MarketQuote(CLOSE_BOOK, home_ml, away_ml))
        odds.add_ml(away, MarketQuote(CLOSE_BOOK, away_ml, home_ml))

    spread = _number(row.get("spread_line"))
    if spread is not None:
        # nflverse states the closing spread as the home team's expected margin;
        # the board is stored on the handicap axis, so the sign flips.
        home_point = -spread
        home_odds = _number(row.get("home_spread_odds"))
        away_odds = _number(row.get("away_spread_odds"))
        if home_odds is not None:
            odds.add_spread(home_point, home, MarketQuote(CLOSE_BOOK, home_odds, away_odds))
        if away_odds is not None:
            odds.add_spread(home_point, away, MarketQuote(CLOSE_BOOK, away_odds, home_odds))

    total = _number(row.get("total_line"))
    if total is not None:
        over = _number(row.get("over_odds"))
        under = _number(row.get("under_odds"))
        if over is not None:
            odds.add_total(total, True, MarketQuote(CLOSE_BOOK, over, under))
        if under is not None:
            odds.add_total(total, False, MarketQuote(CLOSE_BOOK, under, over))
    return odds


def _number(value: object) -> float | None:
    if value is None or isinstance(value, str):
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _int(value: object) -> int | None:
    out = _number(value)
    return None if out is None else int(out)


def _text(value: object) -> str | None:
    if value is None or not isinstance(value, str) or value == "":
        return None
    return value


__all__ = ["CLOSE_BOOK", "NFLVERSE", "ReplayWeek", "played_weeks"]
