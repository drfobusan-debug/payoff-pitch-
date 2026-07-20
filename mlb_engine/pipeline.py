"""Daily pipeline orchestration: slate -> features -> model -> filters -> market -> output."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

from mlb_engine.config import Config
from mlb_engine.data.mlb_statsapi import MLBStatsClient
from mlb_engine.data.parks import get_park
from mlb_engine.data.statcast import StatcastRepository
from mlb_engine.data.vsin import VSINClient
from mlb_engine.features.regression import (
    build_batter_regression,
    build_pitcher_regression,
)
from mlb_engine.features.rolling import (
    LEAGUE_RATES,
    OutcomeRates,
    build_batter_profile,
    build_pitcher_profile,
)
from mlb_engine.filters import travel_rest
from mlb_engine.filters.weather import WeatherProvider
from mlb_engine.market.ev import evaluate
from mlb_engine.market.tiers import Tier, classify
from mlb_engine.models.markov_f5 import f5_from_lineups
from mlb_engine.models.matchup import apply_multipliers, combine
from mlb_engine.models.montecarlo import MonteCarlo, TeamSimConfig
from mlb_engine.models.props import p_over
from mlb_engine.models.rbi_rule import evaluate_lineup, rbi_multiplier
from mlb_engine.recommendations import Recommendation
from mlb_engine.schemas import Game, TeamGameInfo

log = logging.getLogger(__name__)


def league_pitcher_rates() -> OutcomeRates:
    r = LEAGUE_RATES
    return OutcomeRates(
        pa=999,
        p_1b=r["1B"],
        p_2b=r["2B"],
        p_3b=r["3B"],
        p_hr=r["HR"],
        p_bb=r["BB"],
        p_k=r["K"],
        p_out=r["OUT"],
    )


@dataclass
class PipelineDeps:
    stats: MLBStatsClient
    statcast: StatcastRepository
    weather: WeatherProvider
    vsin: VSINClient


def load_sprint_speeds(year: int) -> dict[int, float]:
    try:
        from pybaseball import statcast_sprint_speed

        df = statcast_sprint_speed(year, 10)
        return {int(r["player_id"]): float(r["sprint_speed"]) for _, r in df.iterrows()}
    except Exception as exc:  # optional enrichment
        log.warning("sprint speed unavailable: %s", exc)
        return {}


class Pipeline:
    def __init__(self, cfg: Config, deps: PipelineDeps) -> None:
        self.cfg = cfg
        self.deps = deps

    def run(
        self,
        slate_date: Date,
        vsin_csv: Path | None = None,
        seed: int | None = 7,
    ) -> list[Recommendation]:
        w = self.cfg.windows
        slate = self.deps.stats.get_slate(slate_date)
        log.info("Slate %s: %d games", slate_date, len(slate.games))

        statcast = self.deps.statcast.max_window(
            slate_date,
            [w.pitcher_form_days, w.batter_home_away_days, w.batter_vs_rhp_days, w.batter_vs_lhp_days],
        )
        sprint = load_sprint_speeds(slate_date.year)

        quotes = {}
        if vsin_csv and vsin_csv.exists():
            quotes = self.deps.vsin.load_csv(vsin_csv)
            log.info("Loaded %d VSIN market entries", len(quotes))

        recs: list[Recommendation] = []
        mc = MonteCarlo(self.cfg.mc_sims, seed=seed)
        for game in slate.games:
            if not (game.home.lineup_confirmed() and game.away.lineup_confirmed()):
                log.info("skip %s: lineups not posted", game.matchup())
                continue
            if not (game.home.probable_pitcher and game.away.probable_pitcher):
                log.info("skip %s: probable pitcher missing", game.matchup())
                continue
            recs.extend(self._price_game(game, statcast, slate_date, sprint, mc, quotes))
        return recs

    # ------------------------------------------------------------------
    def _team_offense(
        self,
        team: TeamGameInfo,
        opp: TeamGameInfo,
        statcast,
        slate_date: Date,
        sprint: dict[int, float],
    ):
        """Return (bat_vs_starter, bat_vs_pen, rbi_flags) for a lineup."""
        w = self.cfg.windows
        assert opp.probable_pitcher is not None  # guarded in run()
        opp_throws = opp.probable_pitcher.throws.value if opp.probable_pitcher.throws else None

        pit_prof = build_pitcher_profile(
            statcast, opp.probable_pitcher.mlbam_id, slate_date, w.pitcher_form_days
        )
        pit_reg = build_pitcher_regression(
            statcast[statcast["pitcher"] == opp.probable_pitcher.mlbam_id]
        )
        pit_allowed_mult = pit_reg.allowed_multipliers()
        k_mult = pit_reg.k_multiplier()
        league_pit = league_pitcher_rates()

        # travel/rest for this offense (applied at game level via prev game)
        prev = self.deps.stats.last_game_venue(team.team_id, slate_date)

        profiles = []
        regs = []
        bat_vs_starter = []
        bat_vs_pen = []
        for slot in team.lineup:
            pid = slot.player.mlbam_id
            bprof = build_batter_profile(
                statcast, pid, slate_date, w.batter_home_away_days, w.batter_vs_rhp_days, w.batter_vs_lhp_days
            )
            profiles.append(bprof)
            ctx = bprof.for_context(team.is_home, opp_throws)

            breg = build_batter_regression(
                statcast[statcast["batter"] == pid], sprint.get(pid, 27.0)
            )
            regs.append(breg)
            bmult = breg.multipliers()

            vs_start = combine(ctx, pit_prof.allowed)
            vs_start = apply_multipliers(vs_start, bmult)
            vs_start = apply_multipliers(vs_start, pit_allowed_mult)
            vs_start = apply_multipliers(vs_start, {"K": k_mult})

            vs_pen = combine(ctx, league_pit)
            vs_pen = apply_multipliers(vs_pen, bmult)

            bat_vs_starter.append(vs_start)
            bat_vs_pen.append(vs_pen)

        rbi_flags = evaluate_lineup(profiles, self.cfg.rbi_obp_threshold, regs)
        return bat_vs_starter, bat_vs_pen, rbi_flags, prev

    def _apply_env(self, rates_list, mult: dict[str, float]):
        if not mult:
            return rates_list
        return [apply_multipliers(r, mult) for r in rates_list]

    def _price_game(self, game: Game, statcast, slate_date, sprint, mc, quotes):
        assert game.home.probable_pitcher is not None  # guarded in run()
        assert game.away.probable_pitcher is not None
        recs: list[Recommendation] = []
        park = get_park(game.venue.venue_id)

        # weather effect (park-level)
        weather_mult = {}
        if park:
            eff = self.deps.weather.fetch(park, game.game_datetime_utc)
            weather_mult = eff.multipliers()

        home_start, home_pen, home_rbi, home_prev = self._team_offense(
            game.home, game.away, statcast, slate_date, sprint
        )
        away_start, away_pen, away_rbi, away_prev = self._team_offense(
            game.away, game.home, statcast, slate_date, sprint
        )

        # travel/rest per offense
        if park:
            home_tr = travel_rest.compute(
                _prev_to_pg(home_prev), park, slate_date
            ).multipliers()
            away_tr = travel_rest.compute(
                _prev_to_pg(away_prev), park, slate_date
            ).multipliers()
        else:
            home_tr = away_tr = {}

        # apply env filters (weather + travel) to both offenses
        home_start = self._apply_env(self._apply_env(home_start, weather_mult), home_tr)
        home_pen = self._apply_env(self._apply_env(home_pen, weather_mult), home_tr)
        away_start = self._apply_env(self._apply_env(away_start, weather_mult), away_tr)
        away_pen = self._apply_env(self._apply_env(away_pen, weather_mult), away_tr)

        home_cfg = TeamSimConfig(bat_vs_starter=home_start, bat_vs_pen=home_pen)
        away_cfg = TeamSimConfig(bat_vs_starter=away_start, bat_vs_pen=away_pen)
        res = mc.simulate(home_cfg, away_cfg)

        # F5: non-stationary per-lineup-slot Markov (TTO-aware).
        f5 = f5_from_lineups(home_start, away_start)

        ha, aa = game.home.abbrev, game.away.abbrev
        m = game.matchup()

        # ---- game markets ----
        h, a = res.home_runs_full.astype(float), res.away_runs_full.astype(float)
        total = h + a
        margin = h - a
        recs.append(self._mk(game, m, "game", "game_ml", f"{ha} ML", float((margin > 0).mean()),
                             team_side="home", side="win", quotes=quotes))
        recs.append(self._mk(game, m, "game", "game_ml", f"{aa} ML", float((margin < 0).mean()),
                             team_side="away", side="win", quotes=quotes))
        recs.append(self._mk(game, m, "game", "game_rl", f"{ha} -1.5", float((margin > 1.5).mean()),
                             line=-1.5, team_side="home", side="cover", quotes=quotes))
        recs.append(self._mk(game, m, "game", "game_rl", f"{aa} +1.5", float((margin > -1.5).mean()),
                             line=1.5, team_side="away", side="cover", quotes=quotes))
        recs.append(self._mk(game, m, "game", "game_rl", f"{aa} -1.5", float((-margin > 1.5).mean()),
                             line=-1.5, team_side="away", side="cover", quotes=quotes))
        recs.append(self._mk(game, m, "game", "game_rl", f"{ha} +1.5", float((-margin > -1.5).mean()),
                             line=1.5, team_side="home", side="cover", quotes=quotes))
        for line in (7.5, 8.5, 9.5, 10.5):
            recs.append(self._mk(game, m, "game", "game_total", f"Over {line}", p_over(total, line),
                                 line=line, side="over", quotes=quotes))
            recs.append(self._mk(game, m, "game", "game_total", f"Under {line}", 1 - p_over(total, line),
                                 line=line, side="under", quotes=quotes))

        # ---- F5 markets ----
        recs.append(self._mk(game, m, "f5", "f5_ml", f"{ha} F5 ML", f5.p_home_ml,
                             team_side="home", side="win", quotes=quotes))
        recs.append(self._mk(game, m, "f5", "f5_ml", f"{aa} F5 ML", f5.p_away_ml,
                             team_side="away", side="win", quotes=quotes))
        recs.append(self._mk(game, m, "f5", "f5_ml", "F5 Tie", f5.p_tie, side="tie", quotes=quotes))
        for line in (4.5, 5.5):
            po = f5.p_total_over(line)
            recs.append(self._mk(game, m, "f5", "f5_total", f"F5 Over {line}", po,
                                 line=line, side="over", quotes=quotes))
            recs.append(self._mk(game, m, "f5", "f5_total", f"F5 Under {line}", 1 - po,
                                 line=line, side="under", quotes=quotes))
        recs.append(self._mk(game, m, "f5", "f5_rl", f"{ha} F5 -0.5", f5.p_home_cover(0.5),
                             line=-0.5, team_side="home", side="cover", quotes=quotes))
        recs.append(self._mk(game, m, "f5", "f5_rl", f"{aa} F5 +0.5", 1 - f5.p_home_cover(0.5),
                             line=0.5, team_side="away", side="cover", quotes=quotes))

        # ---- batter props ----
        for team_key, tinfo, flags in (("home", game.home, home_rbi), ("away", game.away, away_rbi)):
            recs.extend(self._batter_props(game, m, res, team_key, tinfo, flags, quotes))

        # ---- pitcher props (starters) ----
        # home team's starter faces away hitters -> stats tracked under pit["home"]
        recs.extend(self._pitcher_props(game, m, res, "home", game.home.probable_pitcher, quotes))
        recs.extend(self._pitcher_props(game, m, res, "away", game.away.probable_pitcher, quotes))
        return recs

    def _batter_props(self, game, m, res, team_key, tinfo, flags, quotes):
        out = []
        bat = res.bat[team_key]
        lines = {"H": [0.5, 1.5], "1B": [0.5], "2B": [0.5], "HR": [0.5], "R": [0.5], "RBI": [0.5]}
        for i, slot in enumerate(tinfo.lineup):
            name = slot.player.name
            pid = slot.player.mlbam_id
            flag = flags[i] if i < len(flags) else None
            for stat, sl in lines.items():
                arr = bat[stat][:, i].astype(float)
                if stat == "RBI" and flag is not None:
                    arr = arr * rbi_multiplier(flag)
                for line in sl:
                    out.append(self._mk(
                        game, m, "batter", f"batter_{stat.lower()}",
                        f"{name} {stat} o{line}", p_over(arr, line),
                        line=line, player_id=pid, stat=stat, side="over", quotes=quotes,
                    ))
            hrr = (bat["H"][:, i] + bat["R"][:, i] + bat["RBI"][:, i]).astype(float)
            for line in (1.5, 2.5):
                out.append(self._mk(
                    game, m, "batter", "batter_hrr", f"{name} H+R+RBI o{line}", p_over(hrr, line),
                    line=line, player_id=pid, stat="HRR", side="over", quotes=quotes,
                ))
        return out

    def _pitcher_props(self, game, m, res, team_key, pitcher, quotes):
        out = []
        pit = res.pit[team_key]
        lines = {"K": [4.5, 5.5, 6.5], "outs": [15.5, 17.5], "H": [4.5, 5.5], "BB": [1.5, 2.5], "ER": [2.5, 3.5]}
        label = {"K": "Ks", "outs": "Outs", "H": "Hits", "BB": "Walks", "ER": "ER"}
        for stat, sl in lines.items():
            arr = pit[stat].astype(float)
            for line in sl:
                out.append(self._mk(
                    game, m, "pitcher", f"pitcher_{stat.lower()}",
                    f"{pitcher.name} {label[stat]} o{line}", p_over(arr, line),
                    line=line, player_id=pitcher.mlbam_id, stat=stat, side="over", quotes=quotes,
                ))
        return out

    def _mk(self, game, matchup, category, market, selection, prob, *, line=None,
            team_side=None, player_id=None, stat=None, side=None, quotes=None) -> Recommendation:
        rec = Recommendation(
            game_date=game.game_date,
            game_pk=game.game_pk,
            matchup=matchup,
            category=category,
            market=market,
            selection=selection,
            model_prob=float(min(max(prob, 1e-6), 1 - 1e-6)),
            line=line,
            team_side=team_side,
            player_id=player_id,
            stat=stat,
            side=side,
        )
        key = (matchup, market, selection)
        q = (quotes or {}).get(key)
        if q:
            evres = evaluate(rec.model_prob, q)
            rec.book = evres.best_quote.book
            rec.market_american = evres.best_quote.american
            rec.ev = evres.ev
            rec.edge = evres.edge
            rec.handle_pct = evres.best_quote.handle_pct
            rec.bets_pct = evres.best_quote.bets_pct
            tier, reasons = classify(evres, self.cfg.ev)
            rec.tier = tier
            rec.reasons = reasons
        else:
            rec.tier = Tier.PASS
            rec.reasons = ["no market price"]
        return rec


def _prev_to_pg(prev):
    if prev is None:
        return None
    d, venue_id = prev
    p = get_park(venue_id)
    if not p:
        return None
    return travel_rest.PrevGame(game_date=d, lat=p.lat, lon=p.lon)
