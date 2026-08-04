"""Daily orchestration: slate + market -> score projections -> priced,
tiered recommendations for moneyline, ATS, and totals.

The flow mirrors the sibling MLB engine: pull the board (which doubles as the
schedule), build each game's expected margin/total from team ratings blended
toward the market, Monte Carlo the score, price every side against the book,
calibrate, and classify into Strong / Moderate / Pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as Date

from cfb_engine.calibration import Calibrator, ConfidenceShrink
from cfb_engine.config import Config
from cfb_engine.data.cfbd import CFBDClient, RatingBook
from cfb_engine.data.ensemble import EnsembleProvider, blend_ensemble
from cfb_engine.data.oddsapi import Board, OddsAPIClient
from cfb_engine.data.ratings import build_rating_book
from cfb_engine.data.vsin import hfa_for
from cfb_engine.features.adjustments import Adjustment, compute_adjustment
from cfb_engine.features.context import ContextBook, build_context_book, context_for
from cfb_engine.market import keys
from cfb_engine.market.board import GameOdds
from cfb_engine.market.ev import EVResult, MarketQuote, anchor_to_market, evaluate
from cfb_engine.market.tiers import Tier, classify
from cfb_engine.models.montecarlo import ExpectedGame, GameSimResult, MonteCarlo
from cfb_engine.recommendations import Recommendation
from cfb_engine.schemas import Game, Slate

logger = logging.getLogger(__name__)


@dataclass
class _Means:
    exp_margin: float
    exp_total: float
    source: str  # "ratings+market" | "market" | "ratings"


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        *,
        odds: OddsAPIClient | None = None,
        cfbd: CFBDClient | None = None,
        calibrator: Calibrator | None = None,
    ) -> None:
        self.cfg = cfg
        self.odds = odds or OddsAPIClient(
            cfg.creds.odds_api_key,
            regions=cfg.odds_regions,
            cache_dir=cfg.odds_cache_dir,
            cache_ttl=cfg.odds_cache_ttl,
        )
        self.cfbd = cfbd or CFBDClient(cfg.creds.cfbd_api_key)
        self.ensemble = EnsembleProvider(cfg.cache_dir, cfg.models_dir)
        self.calibrator = calibrator or self._load_calibrator()
        self.shrink = ConfidenceShrink(cfg.shrink_pivot, cfg.shrink_slope) if cfg.shrink_tails else None

    def _load_calibrator(self) -> Calibrator:
        if self.cfg.calibrate and self.cfg.calibration_file.exists():
            try:
                return Calibrator.from_json(self.cfg.calibration_file)
            except (ValueError, OSError) as exc:
                logger.warning("could not load calibration (%s); using identity", exc)
        return Calibrator.identity()

    def _season(self, slate_date: Date) -> int:
        if self.cfg.season:
            return self.cfg.season
        # A January bowl/playoff slate belongs to the prior calendar year's season.
        return slate_date.year - 1 if slate_date.month <= 2 else slate_date.year

    def run(
        self, slate_date: Date, *, slate: Slate | None = None, board: Board | None = None
    ) -> list[Recommendation]:
        if slate is None or board is None:
            slate, board = self.odds.fetch_board(slate_date)
        if not slate.games:
            logger.warning("no NCAAF games found for %s", slate_date)
            return []

        season = self._season(slate_date)
        ratings = build_rating_book(
            self.cfbd.fetch_ratings(season),
            self.cfg.pff_dir,
            self.cfg.ratings_file,
        )
        if self.cfg.ensemble:
            models = self.ensemble.collect(season)
            if models:
                logger.info(
                    "ensemble: %s", ", ".join(f"{m.source}({len(m.net)})" for m in models)
                )
                ratings = blend_ensemble(
                    ratings,
                    models,
                    blend=self.cfg.ensemble_blend,
                    weights=self.ensemble.weights(),
                    target_sd=self.cfg.ensemble_target_sd,
                )
        ctx_book = build_context_book(self.cfbd, season, slate)
        mc = MonteCarlo(self.cfg.model)

        recs: list[Recommendation] = []
        for game in slate.games:
            odds = board.get(game.matchup())
            if odds is None:
                continue
            recs.extend(self._price_game(game, odds, ratings, ctx_book, mc))
        recs.sort(key=lambda r: (_tier_rank(r.tier), -(r.edge or -1.0)))
        return recs

    # -- per game ---------------------------------------------------------
    def _price_game(
        self,
        game: Game,
        odds: GameOdds,
        ratings: RatingBook | None,
        ctx_book: ContextBook,
        mc: MonteCarlo,
    ) -> list[Recommendation]:
        home_hfa = hfa_for(
            game.home.name, self.cfg.model.home_field_pts, enabled=self.cfg.vsin_hfa
        )
        means = self._means(game, odds, ratings, home_hfa)
        if means is None:
            return []
        adj = compute_adjustment(
            context_for(ctx_book, game.home.name, game.away.name),
            self.cfg.features,
            home_hfa,
            game.home.abbrev,
            game.away.abbrev,
        )
        exp = ExpectedGame(
            exp_margin=means.exp_margin + adj.margin_delta,
            exp_total=max(0.0, means.exp_total + adj.total_delta),
            margin_sd=self.cfg.model.margin_sd,
            total_sd=self.cfg.model.total_sd,
        )
        sim = mc.simulate(exp)
        ctx = _GameCtx(game, sim, adj)
        out: list[Recommendation] = []
        out.extend(self._price_ml(ctx, odds))
        out.extend(self._price_ats(ctx, odds))
        out.extend(self._price_total(ctx, odds))
        return out

    def _means(
        self, game: Game, odds: GameOdds, ratings: RatingBook | None, home_hfa: float
    ) -> _Means | None:
        mkt_spread = odds.consensus_home_spread()
        mkt_total = odds.consensus_total()
        market_margin = -mkt_spread if mkt_spread is not None else None
        market_total = mkt_total

        rating_margin = rating_total = None
        if ratings is not None:
            home = ratings.get(game.home.name)
            away = ratings.get(game.away.name)
            if home is not None and away is not None:
                la = ratings.league_avg or self.cfg.model.avg_team_points
                # Log5-style scoring: a team's points scale with its own offense
                # and the opponent's defense relative to the league average.
                home_pts = home.offense * away.defense / la
                away_pts = away.offense * home.defense / la
                # Mean-reversion regression: shrink the rating gap toward zero
                # (guide-style) before adding HFA, since SP+ separation overstates
                # true edge; the total is left on its own scale.
                raw_gap = (home_pts - away_pts) * self.cfg.features.regression_factor
                rating_margin = raw_gap + self.cfg.model.home_field_pts
                rating_total = home_pts + away_pts

        if rating_margin is not None and market_margin is not None:
            w = self.cfg.model.market_blend
            return _Means(
                exp_margin=(1 - w) * rating_margin + w * market_margin,
                exp_total=(1 - w) * (rating_total or market_total or 0.0)
                + w * (market_total or rating_total or 0.0),
                source="ratings+market",
            )
        if rating_margin is not None:
            return _Means(rating_margin, rating_total or 0.0, "ratings")
        if market_margin is not None:
            return _Means(market_margin, market_total or 0.0, "market")
        return None

    # -- markets ----------------------------------------------------------
    def _price_ml(self, ctx: _GameCtx, odds: GameOdds) -> list[Recommendation]:
        home_p = ctx.sim.home_win_prob()
        out = []
        for side, ab, prob in (
            ("home", ctx.home_ab, home_p),
            ("away", ctx.away_ab, 1.0 - home_p),
        ):
            quotes = odds.ml.get(ab)
            if not quotes:
                continue
            out.append(
                self._make_rec(
                    ctx, "game_ml", keys.game_ml(ab), prob, quotes,
                    team_side=side, side="win",
                )
            )
        return out

    def _price_ats(self, ctx: _GameCtx, odds: GameOdds) -> list[Recommendation]:
        point = odds.main_spread()
        if point is None or point not in odds.spreads:
            return []
        home_cover = ctx.sim.cover_prob(point)
        sides = odds.spreads[point]
        out = []
        for team_side, ab, prob, pt in (
            ("home", ctx.home_ab, home_cover, point),
            ("away", ctx.away_ab, 1.0 - home_cover, -point),
        ):
            quotes = sides.get(ab)
            if not quotes:
                continue
            out.append(
                self._make_rec(
                    ctx, "game_ats", keys.game_ats(ab, pt), prob, quotes,
                    line=pt, team_side=team_side, side="cover",
                )
            )
        return out

    def _price_total(self, ctx: _GameCtx, odds: GameOdds) -> list[Recommendation]:
        line = odds.main_total()
        if line is None or line not in odds.totals:
            return []
        over_p = ctx.sim.over_prob(line)
        sides = odds.totals[line]
        out = []
        for is_over, prob, key in (
            (True, over_p, "over"),
            (False, 1.0 - over_p, "under"),
        ):
            quotes = sides.get(key)
            if not quotes:
                continue
            out.append(
                self._make_rec(
                    ctx, "game_total", keys.game_total(is_over, line), prob, quotes,
                    line=line, side=key,
                )
            )
        return out

    # -- recommendation assembly -----------------------------------------
    def _make_rec(
        self,
        ctx: _GameCtx,
        market: str,
        selection: str,
        raw_prob: float,
        quotes: list[MarketQuote],
        *,
        line: float | None = None,
        team_side: str | None = None,
        side: str | None = None,
    ) -> Recommendation:
        model_prob = self.calibrator.apply(market, raw_prob)
        if self.shrink is not None:
            model_prob = self.shrink.apply(model_prob)

        result: EVResult = evaluate(model_prob, quotes)
        bet_prob = anchor_to_market(model_prob, result.fair_prob, self.cfg.market_anchor)
        if bet_prob != model_prob:
            result = evaluate(bet_prob, quotes)

        thr = self.cfg.ev.for_market(market)
        tier, reasons = classify(result, thr)
        # Surface the situational nudges alongside the EV reasoning.
        reasons = [*reasons, *ctx.adj.reasons]
        return Recommendation(
            game_date=ctx.game.game_date,
            game_id=ctx.game.game_id,
            matchup=ctx.matchup,
            market=market,
            selection=selection,
            model_prob=model_prob,
            raw_prob=raw_prob,
            line=line,
            book=result.best_quote.book,
            market_american=result.best_quote.american,
            opposite_american=result.best_quote.opposite_american,
            ev=result.ev,
            edge=result.edge,
            fair_prob=result.fair_prob,
            bet_prob=bet_prob,
            tier=tier,
            reasons=reasons,
            team_side=team_side,
            side=side,
            home_abbrev=ctx.home_ab,
            away_abbrev=ctx.away_ab,
            exp_margin=ctx.sim.exp_margin,
            exp_margin_sd=ctx.sim.exp_margin_sd,
            exp_total=ctx.sim.exp_total,
            exp_total_sd=ctx.sim.exp_total_sd,
        )


class _GameCtx:
    __slots__ = ("game", "sim", "adj", "home_ab", "away_ab", "matchup")

    def __init__(self, game: Game, sim: GameSimResult, adj: Adjustment) -> None:
        self.game = game
        self.sim = sim
        self.adj = adj
        self.home_ab = game.home.abbrev
        self.away_ab = game.away.abbrev
        self.matchup = game.matchup()


def _tier_rank(tier: Tier) -> int:
    return {Tier.STRONG: 0, Tier.MODERATE: 1, Tier.PASS: 2}[tier]
