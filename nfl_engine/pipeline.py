"""Board in, screened bets out.

The order is fixed by what phase 3 established. The market's own consensus number
sets the mean; the possession simulator turns that mean into a joint score
distribution; every rung on every book's ladder is priced off that one
distribution; the de-vigged consensus on each rung says what the price should be;
and the screens remove. Ratings enter only where the board has no number to
anchor to, and are otherwise carried for reporting.

Anchoring to the consensus rather than to the best price is deliberate: if the
mean came from the outlier we are trying to beat, the outlier would price itself
fair and the execution edge would vanish by construction.

**The one forecasting claim this layer makes does hold out of sample.** Phase 3
killed the rating; what survives is the distribution's ability to price a rung the
market did not quote. ``scripts/nfl/rung_study.py`` anchors to the closing spread
and total for 3,028 games (2015-2025) and compares the model's cover probability
at each neighbouring rung to the realised rate:

    offset      n     model   realised    diff      t
     -1.5     2,972   0.5519   0.5552   -0.0033  -0.36
     -0.5     2,950   0.5149   0.5163   -0.0014  -0.15
     +0.0     2,953   0.4920   0.4903   +0.0017  +0.18
     +0.5     2,961   0.4692   0.4664   +0.0028  +0.31
     +1.5     2,973   0.4341   0.4279   +0.0062  +0.68

Every rung is within 0.6pp of realised, |t| never above 0.75, and the same holds
on totals (largest miss 0.7pp). The half-point *steps* -- the thing ladder
shopping actually buys -- also match: crossing the closing number is worth 0.0228
to the model against 0.0259 realised on spreads, and 0.0144 against 0.0128 on
totals. The model slightly **understates** the value of a half point around the
key numbers, which is the phase-2 shape gap (13.95% at a 3-point margin against
14.83% realised) showing up exactly where expected. It errs toward finding fewer
edges than exist rather than inventing them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date as Date

from nfl_engine.calibration import Calibrator
from nfl_engine.features.adjustments import Situation
from nfl_engine.features.quarterback import StarterBook, margin_delta
from nfl_engine.features.ratings import RatingBook
from nfl_engine.market.board import GameOdds
from nfl_engine.market.ev import PricedBet, best_by_line, price_game
from nfl_engine.market.fair import DEFAULT_METHOD
from nfl_engine.market.screens import Thresholds, apply_screens
from nfl_engine.models.distribution import ScoreDistribution
from nfl_engine.models.drives import DriveSim, ExpectedGame
from nfl_engine.models.expectation import Forecast, forecast
from nfl_engine.schemas import Game


@dataclass(frozen=True)
class GamePricing:
    game: Game
    forecast: Forecast | None
    distribution: ScoreDistribution | None
    bets: list[PricedBet]
    notes: tuple[str, ...] = ()

    def buys(self) -> list[PricedBet]:
        return sorted(
            (bet for bet in self.bets if bet.is_bet()),
            key=lambda bet: -(bet.ev_fair or 0.0),
        )


def anchor(odds: GameOdds) -> tuple[float | None, float | None]:
    """Consensus market margin (home positive) and total.

    The board stores spreads on the home handicap axis, so a home favourite is a
    negative point and the margin the simulator wants is its negation.
    """
    home_point = odds.consensus_home_spread()
    total = odds.consensus_total()
    return (None if home_point is None else -home_point, total)


def price_slate(
    games: list[Game],
    board: dict[str, GameOdds],
    *,
    book: RatingBook | None = None,
    starters: StarterBook | None = None,
    sim: DriveSim | None = None,
    thresholds: Thresholds | None = None,
    method: str = DEFAULT_METHOD,
    best_price_only: bool = True,
    calibrator: Calibrator | None = None,
) -> list[GamePricing]:
    simulator = sim or DriveSim()
    maps = calibrator or Calibrator()
    ratings = book or RatingBook()
    out: list[GamePricing] = []
    for game in games:
        odds = board.get(game.matchup())
        if odds is None:
            out.append(GamePricing(game, None, None, [], ("no_board",)))
            continue
        market_margin, market_total = anchor(odds)
        if market_margin is None or market_total is None:
            # No two-way anchor: the rating is the only mean available, and phase 3
            # says a bet cannot rest on it. Priced for reporting, screened to Pass.
            notes: tuple[str, ...] = ("no_market_anchor",)
        else:
            notes = ()
        qb_points: float = 0.0
        qb_notes: tuple[str, ...] = ()
        if starters is not None:
            qb_points, qb_notes = margin_delta(
                starters,
                season=game.season,
                week=game.week,
                home=game.home.abbrev,
                away=game.away.abbrev,
                home_qb=game.home_qb_id,
                away_qb=game.away_qb_id,
            )
        shot = forecast(
            ratings,
            game.home.abbrev,
            game.away.abbrev,
            situation=situation_of(game),
            market_margin=market_margin,
            market_total=market_total,
            qb_margin_points=qb_points,
            qb_notes=qb_notes,
        )
        distribution = simulator.simulate(_expected(shot))
        bets = price_game(
            odds,
            distribution,
            home=game.home.abbrev,
            away=game.away.abbrev,
            method=method,
        )
        # Before the screens, so a corrected probability is what the disagreement
        # veto and the ledger both see. A market with no accepted map is untouched
        # (see :mod:`nfl_engine.calibration`), so this is a no-op by default.
        bets = [
            replace(bet, model_prob=maps.apply(bet.market, bet.model_prob)) for bet in bets
        ]
        if best_price_only:
            bets = best_by_line(bets)
        bets = apply_screens(bets, thresholds)
        if notes:
            bets = [replace(bet, screens=(*bet.screens, *notes)) for bet in bets]
        out.append(GamePricing(game, shot, distribution, bets, notes))
    return out


def situation_of(game: Game) -> Situation:
    """The game's own context, in the shape the adjustment layer reads.

    Every field is optional and unknown means no adjustment, so a game the
    schedule join missed is priced exactly as it was before there was one.
    """
    return Situation(
        roof=game.env.roof,
        wind_mph=game.env.wind_mph,
        temp_f=game.env.temp_f,
        home_rest=game.home_rest,
        away_rest=game.away_rest,
        neutral_site=game.env.neutral_site,
        div_game=game.div_game,
    )


def _expected(shot: Forecast) -> ExpectedGame:
    return shot.expected_game()


def slate_buys(pricings: list[GamePricing]) -> list[PricedBet]:
    """Every surviving bet on the slate, best execution edge first."""
    bets = [bet for pricing in pricings for bet in pricing.buys()]
    return sorted(bets, key=lambda bet: -(bet.ev_fair or 0.0))


def slate_date(games: list[Game]) -> Date | None:
    return min((game.game_date for game in games), default=None)
