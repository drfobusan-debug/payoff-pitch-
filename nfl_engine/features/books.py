"""The rating and starter books a live slate actually gets, as of one week.

Everything under ``features/`` was fitted and graded through
``scripts/nfl/ratings_study.py``, which builds its own panel and walks it forward.
Nothing built the same books for a *card*, so the CLI priced every game with
``RatingBook()`` -- an empty book whose every team rates exactly average -- and
the ratings layer was measured but not connected.

This is that connection, and it is deliberately the same construction the study
uses, because the alternative is a production path whose numbers no study has
ever graded:

    history = every played game strictly before (season, week)
    asof    = one week index past the last of them
    book    = ratings.fit(history, asof=asof)

**Strictly before is the whole point.** The panel on disk holds the season's
later weeks too, and a card built from ``season == 2025`` rows would be reading
results that have not happened; the mask is on ``(season, week)`` rather than on
the cached ``week_index``, so it is right in a replay of a finished season as
well as on a Sunday morning.

Two ways this returns nothing rather than something wrong. ``MIN_HISTORY_GAMES``
of play-by-play may not be there -- an empty cache, or a fresh install in
September -- and then the book is unusable and ``notes`` says so; with
``MARKET_WEIGHT`` at 1.0 the card is unchanged either way, because the market is
the mean wherever one has posted. What an unusable book costs is the games with
no two-way price, which are reported and never bet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from nfl_engine.data import nflverse
from nfl_engine.features import panel as panel_mod
from nfl_engine.features import quarterback as qb_mod
from nfl_engine.features import ratings as ratings_mod
from nfl_engine.schemas import Game

log = logging.getLogger(__name__)

# Seasons of play-by-play loaded behind a card. The 8-week half-life weights
# anything older than about a season at under 0.02, so this is well past where
# the decay stops caring; it exists to keep the parquet read bounded.
SEASONS_BACK = 3


@dataclass(frozen=True)
class Books:
    """What a slate is priced with, and what to say when it is thin."""

    ratings: ratings_mod.RatingBook = field(default_factory=ratings_mod.RatingBook)
    starters: qb_mod.StarterBook = field(default_factory=qb_mod.StarterBook)
    notes: tuple[str, ...] = ()

    def summary(self) -> str:
        rated = len(self.ratings.teams)
        detail = f"{self.ratings.games_used} games, {rated} teams"
        if not self.ratings.is_usable():
            return f"ratings unusable ({detail}); the market is the mean"
        return f"ratings from {detail}"


def as_of(
    season: int,
    week: int,
    *,
    seasons_back: int = SEASONS_BACK,
    refresh: bool = False,
) -> Books:
    """Books for a slate in ``season`` week ``week``, from earlier games only."""
    games = nflverse.games()
    starters = qb_mod.build(games)
    frame = panel_mod.panel(list(range(season - seasons_back, season + 1)), refresh=refresh)
    if frame.empty:
        log.warning("no play-by-play panel for %s; pricing on the market alone", season)
        return Books(starters=starters, notes=("no_panel",))
    history = _before(panel_mod.with_results(frame, games), season, week)
    if history.empty:
        return Books(starters=starters, notes=("no_rating_history",))
    book = ratings_mod.fit(history, asof=float(history.week_index.max()) + 1.0)
    notes = () if book.is_usable() else ("ratings_thin",)
    return Books(ratings=book, starters=starters, notes=notes)


def attach_qbs(games: list[Game], season: int, week: int) -> int:
    """Name each live game's starters from the schedule; returns how many it named.

    The odds board carries no quarterback, so without this a live card is always
    ``unknown`` on both sides and the fill-in charge can never fire -- the whole
    correction would be live in replay and dead on a Sunday. nflverse fills the
    schedule's ``*_qb_id`` during the week, so this is empty in August and named
    by kickoff, and an unnamed starter is left ``None`` rather than assumed to be
    the incumbent: unknown charges nothing.
    """
    schedule = nflverse.games()
    rows = schedule[(schedule.season == season) & (schedule.week == week)]
    named: dict[tuple[str, str], tuple[str | None, str | None]] = {
        (str(row.home_team), str(row.away_team)): (
            _text(row.home_qb_id),
            _text(row.away_qb_id),
        )
        for row in rows.itertuples()
    }
    filled = 0
    for game in games:
        pair = named.get((game.home.abbrev, game.away.abbrev))
        if pair is None:
            continue
        home_qb, away_qb = pair
        game.home_qb_id = game.home_qb_id or home_qb
        game.away_qb_id = game.away_qb_id or away_qb
        filled += bool(game.home_qb_id) + bool(game.away_qb_id)
    return filled


def _text(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _before(joined: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Played team-games strictly before ``(season, week)``, week-indexed."""
    if joined.empty:
        return joined
    frame = ratings_mod.week_index(joined)
    earlier = (frame.season < season) | ((frame.season == season) & (frame.week < week))
    return frame[earlier].reset_index(drop=True)


@lru_cache(maxsize=8)
def cached(season: int, week: int) -> Books:
    """``as_of`` memoised, for a command that prices and then replays a week."""
    return as_of(season, week)
