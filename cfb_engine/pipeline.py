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

from cfb_engine.audit import snapshot
from cfb_engine.calibration import Calibrator, ConfidenceShrink
from cfb_engine.config import Config
from cfb_engine.data.advanced import AdvancedBook, parse_advanced
from cfb_engine.data.cfbd import CFBDClient, RatingBook
from cfb_engine.data.efficiency import EfficiencyProvider, blend_efficiency
from cfb_engine.data.ensemble import EnsembleProvider, blend_ensemble
from cfb_engine.data.injuries import (
    InjuryBook,
    NewsItem,
    fetch_injury_report,
    fetch_news,
    injury_note,
    log_availability,
    unavailable_for,
)
from cfb_engine.data.oddsapi import Board, OddsAPIClient
from cfb_engine.data.portal import PortalBook, portal_note
from cfb_engine.data.preseason import stability_factor
from cfb_engine.data.ratings import build_rating_book
from cfb_engine.data.returning import ReturningBook, build_returning_book
from cfb_engine.data.roster import RosterBook
from cfb_engine.data.starters import StarterBook, starter_absent
from cfb_engine.data.teamnames import school_key
from cfb_engine.data.vsin import hfa_for, hfa_note
from cfb_engine.features.adjustments import Adjustment, compute_adjustment
from cfb_engine.features.context import ContextBook, build_context_book, context_for
from cfb_engine.market import keys
from cfb_engine.market.board import GameOdds
from cfb_engine.market.confidence import (
    MatchupSignal,
    build_signal,
    confidence_adjustment,
    market_veto,
)
from cfb_engine.market.drift import DriftGate
from cfb_engine.market.ev import EVResult, MarketQuote, anchor_to_market, evaluate
from cfb_engine.market.linevalue import drift_probability
from cfb_engine.market.ordering import order_recs
from cfb_engine.market.priceband import PriceBand
from cfb_engine.market.tiers import Tier, bump_tier, classify
from cfb_engine.models.markov import DriveShape, MarkovSim
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
        self.efficiency = EfficiencyProvider(self.cfbd)
        self.calibrator = calibrator or self._load_calibrator()
        self.shrink = ConfidenceShrink(cfg.shrink_pivot, cfg.shrink_slope) if cfg.shrink_tails else None
        self.advanced: AdvancedBook = parse_advanced([], {})
        self.news: dict[str, NewsItem] = {}
        self._roster: dict[int, RosterBook | None] = {}
        self.drift_gate = DriftGate.from_env()
        self.price_band = PriceBand.from_env()
        self._first_board: dict[str, snapshot.SideQuote] = {}

    def _load_calibrator(self) -> Calibrator:
        if self.cfg.calibrate and self.cfg.calibration_file.exists():
            try:
                return Calibrator.from_json(self.cfg.calibration_file)
            except (ValueError, OSError) as exc:
                logger.warning("could not load calibration (%s); using identity", exc)
        return Calibrator.identity()

    def _slate_week(self, season: int, slate_date: Date) -> int:
        """The slate's week number, so efficiency is fit on earlier weeks only.

        Anything the schedule cannot place (bowls, a missing key) becomes week 99,
        which simply means "fit on the whole season so far" -- still leak-free,
        since a game cannot appear in the fit before it has been played.
        """
        target = slate_date.isoformat()
        weeks = [
            meta.week
            for meta in self.cfbd.fetch_schedule(season)
            if meta.week is not None
            and meta.season_type == "regular"
            and meta.start_date[:10] >= target
        ]
        return min(weeks) if weeks else 99

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
        if self.cfg.efficiency:
            book = self.efficiency.book(season, self._slate_week(season, slate_date))
            if book is not None:
                logger.info(
                    "efficiency: %d teams (blend %.2f%s)",
                    len(book.ratings),
                    self.cfg.efficiency_blend,
                    ", fallback ratings" if ratings is None else "",
                )
                ratings = blend_efficiency(
                    ratings,
                    book,
                    blend=self.cfg.efficiency_blend,
                    league_avg=self.cfg.model.avg_team_points,
                )
        returning = (
            build_returning_book(self.cfbd, season) if self.cfg.returning_pts > 0 else None
        )
        ctx_book = build_context_book(self.cfbd, season, slate)
        portal = self.cfbd.fetch_portal(season)
        if portal:
            logger.info("portal book: %d teams", len(portal))
        injuries: InjuryBook = {}
        starters: StarterBook = {}
        if self.cfg.injury_feed:
            injuries = fetch_injury_report()
            if injuries:
                # Usage decides what an absence is worth, so the feed is only read
                # against who had been taking the snaps before this week.
                starters = self.cfbd.fetch_starters(season, self._slate_week(season, slate_date))
                self.news = fetch_news(cache=self.cfg.cache_dir / "injury_news.json")
                logger.info(
                    "injury feed: %d teams, usage book %d teams", len(injuries), len(starters)
                )
        if self.cfg.marking.enabled or self.cfg.sim_engine == "markov":
            self.advanced = self.cfbd.fetch_advanced(season)
            if self.advanced.teams:
                logger.info("advanced stats: %d teams", len(self.advanced.teams))
        self._baseline_board(slate_date, slate, board)
        mc = MonteCarlo(self.cfg.model)
        markov = MarkovSim(self.cfg.model) if self.cfg.sim_engine == "markov" else None

        recs: list[Recommendation] = []
        for game in slate.games:
            odds = board.get(game.matchup())
            if odds is None:
                continue
            recs.extend(
                self._price_game(
                    game, odds, ratings, ctx_book, mc, markov, returning, portal,
                    injuries, starters, season=season,
                )
            )
        return order_recs(recs)

    # -- per game ---------------------------------------------------------
    def _price_game(
        self,
        game: Game,
        odds: GameOdds,
        ratings: RatingBook | None,
        ctx_book: ContextBook,
        mc: MonteCarlo,
        markov: MarkovSim | None = None,
        returning: ReturningBook | None = None,
        portal: PortalBook | None = None,
        injuries: InjuryBook | None = None,
        starters: StarterBook | None = None,
        season: int = 0,
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
        if returning is not None:
            delta = returning.margin_delta(
                game.home.name,
                game.away.name,
                self.cfg.returning_pts,
                self.cfg.returning_max_pts,
            )
            if abs(delta) >= 0.05:
                adj.margin_delta += delta
                side = game.home.abbrev if delta > 0 else game.away.abbrev
                adj.reasons.append(f"{side} returns more production ({delta:+.1f})")
        if means.source == "ratings":
            self._apply_roster(adj, game, season)
        if portal:
            note = portal_note(portal, game.home.name, game.away.name)
            if note is not None:
                adj.reasons.append(note)
        vsin = hfa_note(
            game.home.name, self.cfg.model.home_field_pts, enabled=self.cfg.vsin_hfa
        )
        if vsin is not None:
            adj.reasons.append(vsin)
        if injuries:
            note = injury_note(injuries, game.home.name, game.away.name)
            if note is not None:
                adj.reasons.append(note)
            self._record_absences(adj, game, odds, injuries, starters or {})
        exp = ExpectedGame(
            exp_margin=means.exp_margin + adj.margin_delta,
            exp_total=max(0.0, means.exp_total + adj.total_delta),
            margin_sd=self.cfg.model.margin_sd,
            total_sd=self.cfg.model.total_sd,
        )
        sim = self._simulate(game, exp, mc, markov)
        signal = (
            build_signal(self.advanced, game.home.name, game.away.name)
            if self.cfg.marking.enabled
            else MatchupSignal()
        )
        ctx = _GameCtx(game, sim, adj, signal)
        out: list[Recommendation] = []
        out.extend(self._price_ml(ctx, odds))
        out.extend(self._price_ats(ctx, odds))
        out.extend(self._price_total(ctx, odds))
        return out

    def _simulate(
        self, game: Game, exp: ExpectedGame, mc: MonteCarlo, markov: MarkovSim | None
    ) -> GameSimResult:
        """Markov engine when selected and both teams have pace stats; else normal."""
        if markov is not None:
            home = self.advanced.get(game.home.name)
            away = self.advanced.get(game.away.name)
            if home is not None and away is not None:
                shape = DriveShape(
                    home_drives=home.drives_per_game,
                    away_drives=away.drives_per_game,
                )
                return markov.simulate(exp, shape)
        return mc.simulate(exp)

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
                # true edge; the total is left on its own scale. Volatile rosters
                # (low VSiN stability) shrink the gap harder toward a pick'em.
                regression = self.cfg.features.regression_factor * stability_factor(
                    game.home.name, game.away.name, enabled=self.cfg.vsin_stability
                )
                raw_gap = (home_pts - away_pts) * regression
                rating_margin = raw_gap + home_hfa
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

    def _roster_book(self, season: int) -> RosterBook | None:
        """Production kept plus bought, built on first use.

        Two extra CFBD calls that all but the rare line-less game never need, so
        the book is only fetched when one turns up.
        """
        if season not in self._roster:
            self._roster[season] = self.cfbd.fetch_roster_book(season)
        return self._roster[season]

    def _apply_roster(self, adj: Adjustment, game: Game, season: int) -> None:
        """Charge roster continuity, but only where no market number exists.

        Against a closing spread the same term goes 51.11% ATS, so it is confined
        to the ratings-only margin, where it is worth 0.33 points of held-out RMSE
        and has no market price to be redundant with (see data/roster.py).
        """
        if self.cfg.roster_pts <= 0 or season <= 0:
            return
        book = self._roster_book(season)
        if book is None:
            return
        delta = book.margin_delta(
            game.home.name, game.away.name, self.cfg.roster_pts, self.cfg.roster_max_pts
        )
        if abs(delta) < 0.05:
            return
        adj.margin_delta += delta
        side = game.home.abbrev if delta > 0 else game.away.abbrev
        adj.reasons.append(f"{side} returns and buys more production ({delta:+.1f}, no line)")

    def _record_absences(
        self,
        adj: Adjustment,
        game: Game,
        odds: GameOdds,
        injuries: InjuryBook,
        starters: StarterBook,
    ) -> None:
        """Log every absence with the line at this moment; charge points only if asked.

        The log is the measurement: an absence is worth -2.2 points against the
        close in the *first* game (holdout 59.9% fading) and nothing once it is
        common knowledge (holdout 47.1%), so what matters is whether we hear it
        before the number moves. ``CFBE_INJURY_QB_PTS`` is 0.0 until the log says
        we do.
        """
        spread = odds.consensus_home_spread()
        for name, abbrev, sign in (
            (game.home.name, game.home.abbrev, -1.0),
            (game.away.name, game.away.abbrev, 1.0),
        ):
            rows = unavailable_for(injuries, name)
            if not rows:
                continue
            log_availability(
                self.cfg.availability_file,
                home=game.home.name,
                away=game.away.name,
                rows=rows,
                spread=spread,
                news=self.news,
            )
            starter = starters.get(school_key(name))
            if starter is None or not starter_absent(starter, [row.player for row in rows]):
                continue
            stale = starter.missed_last_week
            detail = f"{abbrev} without QB {starter.name} ({starter.share:.0%} of attempts)"
            if self.cfg.injury_qb_pts <= 0 or stale:
                why = "already priced in" if stale else "reported, not scored"
                adj.reasons.append(f"{detail} [{why}]")
                continue
            adj.margin_delta += sign * self.cfg.injury_qb_pts
            adj.reasons.append(f"{detail} -{self.cfg.injury_qb_pts:.1f}")

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
        tier, mark_reasons = self._mark(ctx.signal, market, team_side, side, line, tier)
        reasons = [*reasons, *mark_reasons]

        drift = self._drift(ctx.matchup, market, selection, side, line, result.fair_prob)
        pass_gate: str | None = None
        if tier != Tier.PASS:
            keep, drift_reason, gate = self.drift_gate.verdict(drift)
            if drift_reason:
                reasons = [*reasons, drift_reason]
            if not keep:
                tier, pass_gate = Tier.PASS, gate
        if tier != Tier.PASS:
            band = self.price_band.for_market(market)
            keep, band_reason, band_gate = band.verdict(result.best_quote.american)
            if band_reason:
                reasons = [*reasons, band_reason]
            if not keep:
                tier, pass_gate = Tier.PASS, band_gate
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
            drift=drift,
            pass_gate=pass_gate,
            team_side=team_side,
            side=side,
            home_abbrev=ctx.home_ab,
            away_abbrev=ctx.away_ab,
            exp_margin=ctx.sim.exp_margin,
            exp_margin_sd=ctx.sim.exp_margin_sd,
            exp_total=ctx.sim.exp_total,
            exp_total_sd=ctx.sim.exp_total_sd,
        )

    # -- market movement since the first board ----------------------------
    def _baseline_board(
        self, slate_date: Date, slate: Slate, board: Board
    ) -> dict[str, snapshot.SideQuote]:
        """Load the slate's first-seen board, writing it on the first run.

        Write-once, because the point of the file is to be the *earliest* board:
        a second run must not quietly redefine its own board as the baseline and
        report every side as unmoved. Sides that only appear later are added --
        the market posts a mid-major weeknight game days after the marquee ones --
        so a late arrival gets a baseline rather than nothing. A failed write is
        logged and swallowed: a snapshot is bookkeeping, and losing it should not
        cost the slate its card.
        """
        path = self.cfg.board_file(slate_date)
        existing = snapshot.load(path)
        fresh = snapshot.board_quotes(slate, board)
        merged = snapshot.merge_first_wins(existing, fresh)
        if merged != existing:
            try:
                snapshot.save(merged, path)
            except OSError as exc:
                logger.warning("could not write first-seen board (%s)", exc)
        self._first_board = merged
        return merged

    def _drift(
        self,
        matchup: str,
        market: str,
        selection: str,
        side: str | None,
        line: float | None,
        fair_prob: float,
    ) -> float | None:
        """No-vig probability points the market has moved toward this side.

        On a spread or total most of the movement is in the number rather than
        the price, so the handicap difference is converted to probability at the
        distribution's local slope and the price difference added on top.
        """
        base = self._first_board.get(snapshot.key(matchup, market, selection))
        if base is None:
            return None
        pts = drift_probability(
            market,
            side,
            from_prob=base.no_vig_prob,
            to_prob=fair_prob,
            from_line=base.line,
            to_line=line,
            margin_sd=self.cfg.model.margin_sd,
            total_sd=self.cfg.model.total_sd,
        )
        return None if pts is None else round(pts, 4)

    def _mark(
        self,
        signal: MatchupSignal,
        market: str,
        team_side: str | None,
        side: str | None,
        line: float | None,
        tier: Tier,
    ) -> tuple[Tier, list[str]]:
        """Apply the metric marking layer: NPV veto first, then confidence bump.

        With bumps off (the default) the support score comes back with zero steps
        and is kept as a reason, so the ledger records what the layer would have
        done and a graded season can settle whether it should.
        """
        if not self.cfg.marking.enabled or not signal.has_efficiency or tier == Tier.PASS:
            return tier, []
        params = self.cfg.marking
        veto = market_veto(market, team_side, side, line, signal, params)
        if veto.dropped:
            return Tier.PASS, [f"veto: {veto.gate}"]
        steps, reasons = confidence_adjustment(market, team_side, side, signal, params)
        if steps == 0:
            return tier, reasons
        return bump_tier(tier, steps), reasons


class _GameCtx:
    __slots__ = ("game", "sim", "adj", "signal", "home_ab", "away_ab", "matchup")

    def __init__(
        self, game: Game, sim: GameSimResult, adj: Adjustment, signal: MatchupSignal
    ) -> None:
        self.game = game
        self.sim = sim
        self.adj = adj
        self.signal = signal
        self.home_ab = game.home.abbrev
        self.away_ab = game.away.abbrev
        self.matchup = game.matchup()

