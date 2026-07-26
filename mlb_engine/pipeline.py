"""Daily pipeline orchestration: slate -> features -> model -> filters -> market -> output."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import TypeVar

from mlb_engine.calibration import Calibrator
from mlb_engine.config import Config
from mlb_engine.data import catcher_framing
from mlb_engine.data.divisions import same_division
from mlb_engine.data.fangraphs import (
    FanGraphsClient,
    FanGraphsTail,
    MetricValues,
    load_fangraphs_tail_csv,
)
from mlb_engine.data.managers import get_manager
from mlb_engine.data.mlb_statsapi import MLBStatsClient
from mlb_engine.data.oddsapi import OddsAPIClient
from mlb_engine.data.parks import get_park
from mlb_engine.data.rotowire import RotoGame, RotoLineup, RotowireClient, norm_person
from mlb_engine.data.savant_expected import load_batter_xslg
from mlb_engine.data.statcast import StatcastRepository
from mlb_engine.data.vsin import Split, VSINClient
from mlb_engine.features.efficiency import (
    build_pitcher_efficiency,
    opponent_discipline_factor,
)
from mlb_engine.features.pitch_mix import (
    arsenal_matchup_multiplier,
    build_arsenal,
    build_batter_pitch_profile,
)
from mlb_engine.features.regression import (
    MIN_BBE,
    build_batter_regression,
    build_pitcher_regression,
)
from mlb_engine.features.rolling import (
    LEAGUE_RATES,
    OutcomeRates,
    blend_bb_rate,
    blend_k_rate,
    build_batter_late_rates,
    build_batter_profile,
    build_bullpen_profile,
    build_pitcher_profile,
)
from mlb_engine.features.siera import (
    Siera,
    faces_ace,
    faces_scrub,
    pitcher_siera,
)
from mlb_engine.features.singles_under import (
    SinglesUnderResult,
    evaluate_singles_under,
)
from mlb_engine.features.tails import TailAdjuster
from mlb_engine.features.workload import expected_bf_cap
from mlb_engine.filters import travel_rest
from mlb_engine.filters.defense import TeamDefense, load_team_defense
from mlb_engine.filters.human import HumanFactors
from mlb_engine.filters.schedule import dgang_multipliers, local_hour, parse_utc_hour
from mlb_engine.filters.weather import WeatherProvider
from mlb_engine.market import keys
from mlb_engine.market.ev import MarketQuote, evaluate
from mlb_engine.market.runline import RunLineSignal, runline_adjustment
from mlb_engine.market.tiers import Tier, bump_tier, classify
from mlb_engine.models.comeback import ComebackSignal
from mlb_engine.models.comeback import evaluate as evaluate_comeback
from mlb_engine.models.markov_f5 import f5_from_lineups
from mlb_engine.models.matchup import apply_multipliers, combine
from mlb_engine.models.montecarlo import MonteCarlo, TeamSimConfig
from mlb_engine.models.props import p_over
from mlb_engine.models.rbi_rule import evaluate_lineup
from mlb_engine.models.selectors import (
    RBISelector,
    Selection,
    TBSelector,
    XBHSelector,
    power_floor_reason,
)
from mlb_engine.recommendations import Recommendation
from mlb_engine.schemas import BatterSlot, Game, Hand, Pitcher, Player, Slate, TeamGameInfo

log = logging.getLogger(__name__)

FATIGUE_DEPLETED = 60.0  # bullpen-fatigue score at/above which a pen is "depleted"
SHARP_SPREAD_DIV = 15.0  # VSIN run-line handle%-bets% gap treated as sharp money


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
    oddsapi: OddsAPIClient | None = None
    rotowire: RotowireClient | None = None
    fangraphs: FanGraphsClient | None = None


def _merge_quotes(
    primary: dict[tuple[str, str, str], list[MarketQuote]],
    secondary: dict[tuple[str, str, str], list[MarketQuote]],
) -> dict[tuple[str, str, str], list[MarketQuote]]:
    """Union two quote maps, de-duplicating by book per key (primary wins).

    ``primary`` is the Odds API (multi-book prices); ``secondary`` is VSIN, which
    contributes its Circa line plus handle/bets that the Odds API lacks. A book
    present in both is kept once from the primary source.
    """
    out: dict[tuple[str, str, str], list[MarketQuote]] = {k: list(v) for k, v in primary.items()}
    for key, qs in secondary.items():
        seen = {q.book for q in out.get(key, [])}
        for q in qs:
            if q.book not in seen:
                out.setdefault(key, []).append(q)
                seen.add(q.book)
    return out


def _merge_metric(a: MetricValues, b: MetricValues) -> MetricValues:
    return MetricValues(
        by_id={**a.by_id, **b.by_id},
        by_name={**a.by_name, **b.by_name},
    )


def _load_fg_tail(path: Path) -> FanGraphsTail:
    """Load a FanGraphs tail export, or merge every CSV/XLSX in a directory."""
    if path.is_dir():
        files = sorted(
            f for f in path.iterdir() if f.suffix.lower() in (".csv", ".xlsx", ".xls")
        )
    else:
        files = [path]
    merged = FanGraphsTail()
    for f in files:
        if not f.exists():
            continue
        t = load_fangraphs_tail_csv(f)
        merged = FanGraphsTail(
            siera=_merge_metric(merged.siera, t.siera),
            stuff_plus=_merge_metric(merged.stuff_plus, t.stuff_plus),
            wrc_plus=_merge_metric(merged.wrc_plus, t.wrc_plus),
            xslg=_merge_metric(merged.xslg, t.xslg),
        )
    return merged


def _slate_name_ids(slate: Slate) -> tuple[dict[str, int], dict[str, int]]:
    """Return ``(batter_name->id, pitcher_name->id)`` for the slate's players."""
    bat: dict[str, int] = {}
    pit: dict[str, int] = {}
    for g in slate.games:
        for team in (g.home, g.away):
            for slot in team.lineup:
                if slot.player.mlbam_id:
                    bat[norm_person(slot.player.name)] = slot.player.mlbam_id
            if team.probable_pitcher and team.probable_pitcher.mlbam_id:
                pit[norm_person(team.probable_pitcher.name)] = team.probable_pitcher.mlbam_id
    return bat, pit


_ZKey = TypeVar("_ZKey")


def _zscores(values: dict[_ZKey, float]) -> dict[_ZKey, float]:
    """Population z-score of keyed metric values (``{}`` if too few/flat)."""
    if len(values) < 2:
        return {}
    vals = list(values.values())
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    if var <= 0:
        return {}
    std = var ** 0.5
    return {k: (v - mean) / std for k, v in values.items()}


def _metric_to_id_z(
    mv: MetricValues, name_to_id: dict[str, int], allowed_ids: set[int], invert: bool
) -> dict[int, float]:
    """Directional z per MLBAM id for slate players (id match preferred)."""
    out: dict[int, float] = {}
    if mv.by_id:
        for pid, z in _zscores(mv.by_id).items():
            if pid in allowed_ids:
                out[pid] = -z if invert else z
    elif mv.by_name:
        for name, z in _zscores(mv.by_name).items():
            mid = name_to_id.get(name)
            if mid is not None:
                out[mid] = -z if invert else z
    return out


def _match_roto_game(game: Game, roto_games: list[RotoGame]) -> RotoGame | None:
    """Match a slate game to a Rotowire game by team nickname (city-agnostic)."""
    hn, an = norm_person(game.home.name), norm_person(game.away.name)
    for rg in roto_games:
        rh, ra = norm_person(rg.home.nickname), norm_person(rg.away.nickname)
        if rh and ra and hn.endswith(rh) and an.endswith(ra):
            return rg
    return None


def load_sprint_speeds(year: int) -> dict[int, float]:
    try:
        from pybaseball import statcast_sprint_speed

        df = statcast_sprint_speed(year, 10)
        return {int(r["player_id"]): float(r["sprint_speed"]) for _, r in df.iterrows()}
    except Exception as exc:  # optional enrichment
        log.warning("sprint speed unavailable: %s", exc)
        return {}


_CALIBRATION_FILE = Path(__file__).parent / "data" / "calibration_2024.json"


def _load_calibrator() -> Calibrator:
    """Load the packaged 2024 isotonic calibration map (identity if missing)."""
    if _CALIBRATION_FILE.exists():
        return Calibrator.from_json(_CALIBRATION_FILE)
    log.warning("calibration map %s missing; probabilities left uncalibrated", _CALIBRATION_FILE)
    return Calibrator.identity()


class Pipeline:
    def __init__(self, cfg: Config, deps: PipelineDeps) -> None:
        self.cfg = cfg
        self.deps = deps
        self._team_defense: dict[str, TeamDefense] = {}
        self._framing: dict[int, float] = {}
        self._fatigue: dict[int, float | None] = {}
        self._splits: dict[tuple[str, str, str], Split] = {}
        self._tails = TailAdjuster()
        self._calibrator = _load_calibrator() if cfg.calibrate else Calibrator.identity()
        self._rbi_selector = RBISelector(cfg.rbi_obp_threshold)
        self._xbh_selector = XBHSelector()
        self._tb_selector = TBSelector()

    def run(
        self,
        slate_date: Date,
        vsin_csv: Path | None = None,
        fangraphs_csv: Path | None = None,
        seed: int | None = 7,
        enrich_leaderboards: bool = True,
    ) -> list[Recommendation]:
        """Price a slate.

        ``enrich_leaderboards`` gates the season-to-date Savant/pybaseball
        leaderboards (sprint speed, team defense, xSLG tails). It is left on for
        live runs and turned off by the historical backtester, where those
        full-season leaderboards would leak future information (look-ahead bias).
        """
        w = self.cfg.windows
        slate = self.deps.stats.get_slate(slate_date)
        log.info("Slate %s: %d games", slate_date, len(slate.games))
        if self.deps.rotowire is not None:
            self._enrich_expected_lineups(slate, slate_date)

        statcast = self.deps.statcast.max_window(
            slate_date,
            [w.pitcher_form_days, w.batter_home_away_days, w.batter_vs_rhp_days, w.batter_vs_lhp_days],
        )
        sprint = load_sprint_speeds(slate_date.year) if enrich_leaderboards else {}
        self._team_defense = load_team_defense(slate_date.year) if enrich_leaderboards else {}
        self._framing = catcher_framing.load_framing(slate_date.year) if enrich_leaderboards else {}
        batter_xslg = load_batter_xslg(slate_date.year) if enrich_leaderboards else {}
        fg_bz, fg_pz = self._fangraphs_tail_z(fangraphs_csv, slate)
        self._tails = TailAdjuster.build(statcast, batter_xslg, fg_bz, fg_pz)

        quotes: dict[tuple[str, str, str], list[MarketQuote]] = {}
        self._splits = {}
        if vsin_csv and vsin_csv.exists():
            quotes = self.deps.vsin.load_csv(vsin_csv)
            log.info("Loaded %d VSIN market entries from CSV", len(quotes))
        else:
            quotes, self._splits = self.deps.vsin.fetch(slate)
            log.info(
                "Fetched VSIN public splits: %d moneyline prices, %d handle/bets entries",
                len(quotes), len(self._splits),
            )
        if self.deps.oddsapi is not None and self.deps.oddsapi.available():
            odds = self.deps.oddsapi.fetch(slate)
            quotes = _merge_quotes(odds, quotes)
            log.info("Merged Odds API prices: %d market keys now priced", len(quotes))

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
    def _enrich_expected_lineups(self, slate: Slate, slate_date: Date) -> None:
        """Fill unposted lineups/probables from Rotowire's public expected list.

        MLB Stats API only carries lineups once posted; Rotowire publishes
        expected lineups earlier. For any team without a confirmed lineup we
        resolve Rotowire's names to MLBAM ids via the team roster and populate
        the batting order (and probable pitcher if missing) so the game clears
        the ``lineup_confirmed`` gate. Anything unresolved is left untouched.
        """
        rotowire = self.deps.rotowire
        if rotowire is None:
            return
        need = [
            g for g in slate.games
            if not (g.home.lineup_confirmed() and g.away.lineup_confirmed())
        ]
        if not need:
            return
        roto_games = rotowire.fetch_expected_lineups(slate_date)
        if not roto_games:
            return
        filled = 0
        for game in need:
            rg = _match_roto_game(game, roto_games)
            if rg is None:
                continue
            for team, side in ((game.home, rg.home), (game.away, rg.away)):
                if team.lineup_confirmed() or not side.batters:
                    continue
                if self._apply_expected_lineup(team, side, slate_date.year):
                    filled += 1
        if filled:
            log.info("Filled %d expected lineups from Rotowire", filled)

    def _fangraphs_tail_z(
        self, fangraphs_csv: Path | None, slate: Slate
    ) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
        """Build id-keyed tail z-contributions from a FanGraphs custom-report CSV.

        Accepts a single CSV or a directory of CSVs (e.g. one hitter + one
        pitcher export). Metrics are z-scored across the export population, then
        mapped to this slate's players by MLBAM id when the export carries one,
        else by name (unmatched rows stay neutral). SIERA is inverted so a lower
        SIERA reads as "better" (positive z).
        """
        if fangraphs_csv is None:
            return {}, {}
        fg = _load_fg_tail(fangraphs_csv)
        if fg.is_empty():
            return {}, {}
        bat_ids, pit_ids = _slate_name_ids(slate)
        bat_id_set = set(bat_ids.values())
        pit_id_set = set(pit_ids.values())
        batter_z: dict[int, dict[str, float]] = {}
        pitcher_z: dict[int, dict[str, float]] = {}
        for metric, mv in (("wrc_plus", fg.wrc_plus), ("xslg", fg.xslg)):
            for pid, z in _metric_to_id_z(mv, bat_ids, bat_id_set, invert=False).items():
                batter_z.setdefault(pid, {})[metric] = z
        for metric, mv, invert in (
            ("siera", fg.siera, True),
            ("stuff_plus", fg.stuff_plus, False),
        ):
            for pid, z in _metric_to_id_z(mv, pit_ids, pit_id_set, invert=invert).items():
                pitcher_z.setdefault(pid, {})[metric] = z
        if batter_z or pitcher_z:
            log.info(
                "FanGraphs tails: %d batters, %d pitchers matched", len(batter_z), len(pitcher_z)
            )
        return batter_z, pitcher_z

    def _apply_expected_lineup(self, team: TeamGameInfo, side: RotoLineup, season: int) -> bool:
        """Resolve a Rotowire lineup to MLBAM ids and set it on the team."""
        roster = self.deps.stats.team_roster(team.team_id, season)
        if not roster:
            return False
        by_name: dict[str, tuple[int, Hand | None, Hand | None]] = {
            norm_person(full): (pid, bats, throws) for pid, full, bats, throws in roster
        }
        lineup: list[BatterSlot] = []
        for order, rb in enumerate(side.batters, start=1):
            hit = by_name.get(norm_person(rb.name))
            if hit is None:
                return False  # incomplete resolution -> leave MLB data in place
            pid, bats, _ = hit
            lineup.append(
                BatterSlot(order=order, player=Player(mlbam_id=pid, name=rb.name, bats=bats))
            )
        if len(lineup) < 9:
            return False
        team.lineup = lineup
        if team.probable_pitcher is None and side.pitcher:
            phit = by_name.get(norm_person(side.pitcher))
            if phit is not None:
                pid, _, throws = phit
                team.probable_pitcher = Pitcher(mlbam_id=pid, name=side.pitcher, throws=throws)
        return True

    # ------------------------------------------------------------------
    def _team_offense(
        self,
        team: TeamGameInfo,
        opp: TeamGameInfo,
        statcast,
        slate_date: Date,
        sprint: dict[int, float],
        park,
        weather_mult: dict[str, float] | None,
    ):
        """Return (bat_vs_starter, bat_vs_pen, rbi_flags, prev, selections, regs, sunders)."""
        w = self.cfg.windows
        assert opp.probable_pitcher is not None  # guarded in run()
        opp_throws = opp.probable_pitcher.throws.value if opp.probable_pitcher.throws else None

        pit_prof = build_pitcher_profile(
            statcast, opp.probable_pitcher.mlbam_id, slate_date, w.pitcher_form_days
        )
        pit_rows = statcast[statcast["pitcher"] == opp.probable_pitcher.mlbam_id]
        pit_reg = build_pitcher_regression(pit_rows)
        pit_allowed_mult = pit_reg.allowed_multipliers()
        k_mult = pit_reg.k_multiplier()

        # Stuff/command priors: pull the starter's allowed K and BB rates toward
        # xK% (CSW%/SwStr%) and xBB% (Zone%/chase/F-strike) so thin PA samples
        # regress to his skills, not the flat league mean.
        pit_allowed = blend_k_rate(pit_prof.allowed, pit_reg.expected_k_pct())
        pit_allowed = blend_bb_rate(pit_allowed, pit_reg.expected_bb_pct())

        # Arsenal matching: starter's pitch-mix usage/SwStr% vs. each batter's
        # per-pitch-class whiff/xwOBA (replaces noisy BvP head-to-heads).
        arsenal = build_arsenal(pit_rows)

        # Distribution-tail kicker for a >=2 SD elite/poor starter (suppresses or
        # boosts every batter it faces); neutral for typical arms.
        pit_tail = self._tails.pitcher_multiplier(opp.probable_pitcher.mlbam_id)

        # Opponent bullpen: relievers' late-inning (>=6th) rates over ~3 weeks,
        # plus PPV (K-BB%/CSW/barrel/IVB via pitcher regression on the relief set)
        # and NPV (zone% walk-trap + 3-in-4 fatigue) tripwires.
        bpen = build_bullpen_profile(
            statcast, opp.abbrev, slate_date, w.bullpen_days, w.bullpen_min_inning
        )
        bpen_reg = build_pitcher_regression(bpen.relief)
        bpen_allowed = bpen_reg.allowed_multipliers()
        bpen_k = bpen_reg.k_multiplier()
        avail = (
            self.deps.rotowire.bullpen_availability(opp.abbrev)
            if self.deps.rotowire and self.deps.rotowire.available()
            else None
        )
        bpen_npv = bpen.npv_multipliers(avail)

        # travel/rest for this offense (applied at game level via prev game)
        prev = self.deps.stats.last_game_venue(team.team_id, slate_date)

        profiles = []
        regs = []
        sunders: list[SinglesUnderResult] = []
        bat_vs_starter = []
        bat_vs_pen = []
        selections: list[dict[str, Selection | None]] = []
        for slot_idx, slot in enumerate(team.lineup):
            pid = slot.player.mlbam_id
            bprof = build_batter_profile(
                statcast, pid, slate_date, w.batter_home_away_days, w.batter_vs_rhp_days, w.batter_vs_lhp_days
            )
            profiles.append(bprof)
            ctx = bprof.for_context(team.is_home, opp_throws)

            bslice = statcast[statcast["batter"] == pid]
            breg = build_batter_regression(bslice, sprint.get(pid, 27.0))
            regs.append(breg)
            bmult = breg.multipliers()

            bats = slot.player.bats.value if slot.player.bats else None
            # Switch hitters bat from the side opposite the starter's hand.
            stand = bats if bats in ("L", "R") else ("L" if opp_throws == "R" else "R")
            sunders.append(evaluate_singles_under(bslice, stand))
            xbh_sel = self._xbh_selector.select(
                breg, park=park, weather=weather_mult, slot=slot_idx, bats=bats, opp_hand=opp_throws
            )
            tb_sel = self._tb_selector.select(
                breg, park=park, weather=weather_mult, slot=slot_idx, bats=bats, opp_hand=opp_throws
            )

            bpp = build_batter_pitch_profile(bslice)
            arsenal_mult = arsenal_matchup_multiplier(arsenal, bpp)
            bat_tail = self._tails.batter_multiplier(pid)

            platoon_k = pit_reg.platoon_k_multiplier(bats)

            vs_start = combine(ctx, pit_allowed)
            vs_start = apply_multipliers(vs_start, bmult)
            # V1-style XBH selector feeds the existing 2B/3B multiplier block.
            vs_start = apply_multipliers(vs_start, xbh_sel.outcome_multipliers)
            vs_start = apply_multipliers(vs_start, pit_allowed_mult)
            vs_start = apply_multipliers(vs_start, {"K": k_mult})
            vs_start = apply_multipliers(vs_start, {"K": platoon_k})
            vs_start = apply_multipliers(vs_start, arsenal_mult)
            vs_start = apply_multipliers(vs_start, pit_tail)
            vs_start = apply_multipliers(vs_start, bat_tail)

            # Bullpen matchup: batter's late-inning (>=6th) 3-week rates vs the
            # pen (FanGraphs split when available, else Statcast), then bullpen
            # regression PPV + NPV tripwires.
            late_ctx = None
            if self.deps.fangraphs and self.deps.fangraphs.available():
                late_ctx = self.deps.fangraphs.late_inning_batter_rates(
                    pid, w.bullpen_days, w.bullpen_min_inning
                )
            if late_ctx is None:
                late_ctx = build_batter_late_rates(
                    statcast, pid, slate_date, w.bullpen_days, w.bullpen_min_inning
                )
            vs_pen = combine(late_ctx, bpen.allowed)
            vs_pen = apply_multipliers(vs_pen, bmult)
            # Apply XBH selection to bullpen matchup too.
            vs_pen = apply_multipliers(vs_pen, xbh_sel.outcome_multipliers)
            vs_pen = apply_multipliers(vs_pen, bpen_allowed)
            vs_pen = apply_multipliers(vs_pen, {"K": bpen_k})
            vs_pen = apply_multipliers(vs_pen, bpen_npv)
            vs_pen = apply_multipliers(vs_pen, bat_tail)

            bat_vs_starter.append(vs_start)
            bat_vs_pen.append(vs_pen)
            selections.append({"RBI": None, "XBH": xbh_sel, "TB": tb_sel})

        rbi_flags = evaluate_lineup(profiles, self.cfg.rbi_obp_threshold, regs)
        # Bind RBI selections now that lineup flags are available.
        for i, flag in enumerate(rbi_flags):
            breg = regs[i]
            slot = team.lineup[i]
            bats = slot.player.bats.value if slot.player.bats else None
            selections[i]["RBI"] = self._rbi_selector.select(
                flag,
                breg=breg,
                park=park,
                weather=weather_mult,
                slot=i,
                bats=bats,
                opp_hand=opp_throws,
            )

        return bat_vs_starter, bat_vs_pen, rbi_flags, prev, selections, regs, sunders

    def _apply_env(self, rates_list, mult: dict[str, float]):
        if not mult:
            return rates_list
        return [apply_multipliers(r, mult) for r in rates_list]

    def _apply_all(self, rates_list, mults: list[dict[str, float]]):
        for m in mults:
            rates_list = self._apply_env(rates_list, m)
        return rates_list

    def _bullpen_fatigue(self, team_id: int, slate_date: Date) -> float | None:
        """Cached 0-100 bullpen-fatigue score for a team on the slate date."""
        if team_id not in self._fatigue:
            self._fatigue[team_id] = self.deps.stats.bullpen_fatigue(team_id, slate_date)
        return self._fatigue[team_id]

    def _spread_divergence(self, matchup: str, abbrev: str) -> float | None:
        """VSIN run-line handle% - bets% for a team (sharp side when positive)."""
        for suffix in ("-1.5", "+1.5"):
            sp = self._splits.get((matchup, "game_rl", f"{abbrev} {suffix}"))
            if sp is not None:
                return sp.divergence
        return None

    def _sharp_spread_side(self, matchup: str, home_ab: str, away_ab: str) -> str | None:
        """Return 'home'/'away' when VSIN spread money notably outweighs tickets."""
        hd = self._spread_divergence(matchup, home_ab)
        if hd is not None and hd >= SHARP_SPREAD_DIV:
            return "home"
        ad = self._spread_divergence(matchup, away_ab)
        if ad is not None and ad >= SHARP_SPREAD_DIV:
            return "away"
        return None

    def _defense_multiplier(self, fielding_abbrev: str) -> dict[str, float]:
        """BIP-hit suppression on the offense that faces this fielding team."""
        defense = self._team_defense.get(fielding_abbrev)
        if defense is None:
            return {}
        return defense.bip_multipliers()

    def _team_xwoba(self, statcast, tinfo) -> float | None:
        """Mean batted-ball xwOBA across a lineup (skips thin-sample hitters)."""
        vals = []
        for slot in tinfo.lineup:
            rows = statcast[statcast["batter"] == slot.player.mlbam_id]
            x = rows.loc[rows["launch_speed"].notna(), "estimated_woba_using_speedangle"].dropna()
            if len(x) >= 15:
                vals.append(float(x.mean()))
        if not vals:
            return None
        return sum(vals) / len(vals)

    def _dgang_multiplier(self, prev, today_park, today_iso: str | None, slate_date: Date):
        """Day-game-after-night-game offense tax for a team, from schedule times."""
        if prev is None or today_park is None:
            return {}
        d, venue_id, prev_hr = prev
        prev_park = get_park(venue_id)
        today_hr = parse_utc_hour(today_iso)
        if prev_park is None or prev_hr is None or today_hr is None:
            return {}
        rest_days = (slate_date - d).days
        prev_local = local_hour(prev_hr, prev_park.lon)
        today_local = local_hour(today_hr, today_park.lon)
        return dgang_multipliers(prev_local, today_local, rest_days)

    def _catcher_framing_runs(self, team: TeamGameInfo) -> float:
        """Framing runs of a team's starting catcher.

        Prefers the live Savant feed (keyed by MLBAM id), then the curated
        name table; 0.0 (neutral) when the catcher is unknown or no feed loaded.
        """
        for slot in team.lineup:
            if slot.player.position != "C":
                continue
            live = self._framing.get(slot.player.mlbam_id)
            if live is not None:
                return live
            curated = catcher_framing.framing_runs_for_name(slot.player.name)
            if curated is not None:
                return curated
        return 0.0

    def _umpire_zone_runs(self, game: Game) -> float:
        if self.deps.rotowire and self.deps.rotowire.available():
            z = self.deps.rotowire.umpire_zone_runs(game.game_pk)
            if z is not None:
                return z
        return 0.0

    def _price_game(self, game: Game, statcast, slate_date, sprint, mc, quotes):
        assert game.home.probable_pitcher is not None  # guarded in run()
        assert game.away.probable_pitcher is not None
        recs: list[Recommendation] = []
        park = get_park(game.venue.venue_id)

        # weather effect (park-level)
        weather_mult = {}
        eff = None
        if park:
            eff = self.deps.weather.fetch(park, game.game_datetime_utc)
            weather_mult = eff.multipliers()

        home_start, home_pen, home_rbi, home_prev, home_sels, home_regs, home_su = (
            self._team_offense(
                game.home, game.away, statcast, slate_date, sprint, park, weather_mult
            )
        )
        away_start, away_pen, away_rbi, away_prev, away_sels, away_regs, away_su = (
            self._team_offense(
                game.away, game.home, statcast, slate_date, sprint, park, weather_mult
            )
        )

        # travel/rest (circadian) per team
        if park:
            home_te = travel_rest.compute(_prev_to_pg(home_prev), park, slate_date)
            away_te = travel_rest.compute(_prev_to_pg(away_prev), park, slate_date)
            home_tr, away_tr = home_te.multipliers(), away_te.multipliers()
            # jet-lagged staff allows more HR -> boost the OPPOSING offense
            home_hr_boost = away_te.pitching_multipliers()
            away_hr_boost = home_te.pitching_multipliers()
        else:
            home_tr = away_tr = home_hr_boost = away_hr_boost = {}

        # human element: division familiarity + plate-umpire zone (both offenses),
        # plus opponent-specific hooks (framing/battery, neutral until fed).
        divisional = same_division(game.home.team_id, game.away.team_id)
        ump_zone = self._umpire_zone_runs(game)
        # Each offense faces the OPPOSING starting catcher's framing.
        home_frame = self._catcher_framing_runs(game.away)
        away_frame = self._catcher_framing_runs(game.home)
        home_hf = HumanFactors(
            divisional=divisional, umpire_zone_runs=ump_zone, catcher_framing_runs=home_frame
        )
        away_hf = HumanFactors(
            divisional=divisional, umpire_zone_runs=ump_zone, catcher_framing_runs=away_frame
        )
        home_human = home_hf.offense_multipliers()
        away_human = away_hf.offense_multipliers()

        # fielding defense: each offense's BIP hits suppressed by the OPP defense.
        home_def = self._defense_multiplier(game.away.abbrev)
        away_def = self._defense_multiplier(game.home.abbrev)

        # manager tendencies: TTO hook -> starter BF cap; speed engine -> full
        # offense tilt; platoon aggression -> late-inning (pen) tilt only.
        home_mgr = get_manager(game.home.team_id)
        away_mgr = get_manager(game.away.team_id)

        # schedule pacing: day-game-after-night-game offense tax (per team).
        home_dgang = self._dgang_multiplier(home_prev, park, game.game_datetime_utc, slate_date)
        away_dgang = self._dgang_multiplier(away_prev, park, game.game_datetime_utc, slate_date)

        # apply env filters: weather + own travel + opponent-staff HR boost +
        # human element + opponent fielding defense + manager speed engine + DGANG.
        home_env = [weather_mult, home_tr, home_hr_boost, home_human, home_def,
                    home_mgr.offense_multipliers(), home_dgang]
        away_env = [weather_mult, away_tr, away_hr_boost, away_human, away_def,
                    away_mgr.offense_multipliers(), away_dgang]
        home_start = self._apply_all(home_start, home_env)
        home_pen = self._apply_all(home_pen, [*home_env, home_mgr.pen_multipliers()])
        away_start = self._apply_all(away_start, away_env)
        away_pen = self._apply_all(away_pen, [*away_env, away_mgr.pen_multipliers()])

        # Starter exit model: manager hooks (batters-faced + pitch-count caps)
        # tightened by each starter's own recent workload, plus a pitch-efficiency
        # profile (P/PA, F-Strike%, GB%) so out volume tracks pitch economy, not
        # just batters faced. Drives realistic outs (innings) and strikeout unders.
        w = self.cfg.windows
        home_pit_rows = statcast[statcast["pitcher"] == game.home.probable_pitcher.mlbam_id]
        away_pit_rows = statcast[statcast["pitcher"] == game.away.probable_pitcher.mlbam_id]
        # SIERA of each starter (from Statcast); away batters face the home
        # starter and vice-versa -> map by the batter's own team_key.
        opp_siera = {
            "away": pitcher_siera(home_pit_rows),
            "home": pitcher_siera(away_pit_rows),
        }
        home_cap = expected_bf_cap(
            home_pit_rows, slate_date, w.pitcher_form_days, home_mgr.starter_bf_cap,
        )
        away_cap = expected_bf_cap(
            away_pit_rows, slate_date, w.pitcher_form_days, away_mgr.starter_bf_cap,
        )
        home_eff = build_pitcher_efficiency(
            home_pit_rows, slate_date, w.pitcher_form_days, home_mgr.starter_pitch_cap,
        )
        away_eff = build_pitcher_efficiency(
            away_pit_rows, slate_date, w.pitcher_form_days, away_mgr.starter_pitch_cap,
        )
        # Opponent lineup discipline (pitches-seen-per-PA): each starter's pitch
        # budget is burned faster by the patient lineup he actually faces.
        home_ids = [s.player.mlbam_id for s in game.home.lineup]
        away_ids = [s.player.mlbam_id for s in game.away.lineup]
        home_disc = opponent_discipline_factor(
            statcast, away_ids, slate_date, w.batter_home_away_days
        )
        away_disc = opponent_discipline_factor(
            statcast, home_ids, slate_date, w.batter_home_away_days
        )
        home_cfg = TeamSimConfig(
            bat_vs_starter=home_start,
            bat_vs_pen=home_pen,
            starter_bf_cap=home_cap,
            starter_pitch_cap=home_eff.pitch_cap,
            pitch_eff=min(1.35, home_eff.efficiency_scaler() * home_disc),
            gb_dp_rate=home_eff.gb_dp_rate(),
        )
        away_cfg = TeamSimConfig(
            bat_vs_starter=away_start,
            bat_vs_pen=away_pen,
            starter_bf_cap=away_cap,
            starter_pitch_cap=away_eff.pitch_cap,
            pitch_eff=min(1.35, away_eff.efficiency_scaler() * away_disc),
            gb_dp_rate=away_eff.gb_dp_rate(),
        )
        res = mc.simulate(home_cfg, away_cfg)

        # F5: non-stationary per-lineup-slot Markov (TTO-aware).
        f5 = f5_from_lineups(home_start, away_start)

        ha, aa = game.home.abbrev, game.away.abbrev
        m = game.matchup()

        # ---- game markets ----
        h, a = res.home_runs_full.astype(float), res.away_runs_full.astype(float)
        total = h + a
        margin = h - a

        # Run-line PPV confidence: team xwOBA differential + depleted-favorite
        # bullpen (live from the public StatsAPI workload proxy).
        home_x = self._team_xwoba(statcast, game.home)
        away_x = self._team_xwoba(statcast, game.away)
        home_fat = self._bullpen_fatigue(game.home.team_id, slate_date)
        away_fat = self._bullpen_fatigue(game.away.team_id, slate_date)
        fav_side = "home" if float((margin > 0).mean()) >= 0.5 else "away"
        fav_fat = home_fat if fav_side == "home" else away_fat
        rl_signal = RunLineSignal(
            xwoba_diff=(home_x - away_x) if home_x is not None and away_x is not None else None,
            fav_pen_depleted_side=(
                fav_side if fav_fat is not None and fav_fat >= FATIGUE_DEPLETED else None
            ),
            sharp_money_side=self._sharp_spread_side(m, ha, aa),
        )

        recs.append(self._mk(game, m, "game", "game_ml", keys.game_ml(ha), float((margin > 0).mean()),
                             team_side="home", side="win", quotes=quotes))
        recs.append(self._mk(game, m, "game", "game_ml", keys.game_ml(aa), float((margin < 0).mean()),
                             team_side="away", side="win", quotes=quotes))
        recs.append(self._mk(game, m, "game", "game_rl", keys.game_rl(ha, -1.5), float((margin > 1.5).mean()),
                             line=-1.5, team_side="home", side="cover", quotes=quotes, rl_signal=rl_signal))
        recs.append(self._mk(game, m, "game", "game_rl", keys.game_rl(aa, 1.5), float((margin > -1.5).mean()),
                             line=1.5, team_side="away", side="cover", quotes=quotes, rl_signal=rl_signal))
        recs.append(self._mk(game, m, "game", "game_rl", keys.game_rl(aa, -1.5), float((-margin > 1.5).mean()),
                             line=-1.5, team_side="away", side="cover", quotes=quotes, rl_signal=rl_signal))
        recs.append(self._mk(game, m, "game", "game_rl", keys.game_rl(ha, 1.5), float((-margin > -1.5).mean()),
                             line=1.5, team_side="home", side="cover", quotes=quotes, rl_signal=rl_signal))

        # ---- comeback-resilience flags ----
        recs.extend(self._comeback_recs(
            game, m, home_x, away_x, home_rbi, away_rbi, home_mgr, away_mgr,
            home_fat, away_fat,
        ))

        for line in (7.5, 8.5, 9.5, 10.5):
            recs.append(self._mk(game, m, "game", "game_total", keys.game_total(True, line), p_over(total, line),
                                 line=line, side="over", quotes=quotes))
            recs.append(self._mk(game, m, "game", "game_total", keys.game_total(False, line), 1 - p_over(total, line),
                                 line=line, side="under", quotes=quotes))

        # ---- F5 markets ----
        recs.append(self._mk(game, m, "f5", "f5_ml", keys.f5_ml(ha), f5.p_home_ml,
                             team_side="home", side="win", quotes=quotes))
        recs.append(self._mk(game, m, "f5", "f5_ml", keys.f5_ml(aa), f5.p_away_ml,
                             team_side="away", side="win", quotes=quotes))
        recs.append(self._mk(game, m, "f5", "f5_ml", "F5 Tie", f5.p_tie, side="tie", quotes=quotes))
        for line in (4.5, 5.5):
            po = f5.p_total_over(line)
            recs.append(self._mk(game, m, "f5", "f5_total", keys.f5_total(True, line), po,
                                 line=line, side="over", quotes=quotes))
            recs.append(self._mk(game, m, "f5", "f5_total", keys.f5_total(False, line), 1 - po,
                                 line=line, side="under", quotes=quotes))
        recs.append(self._mk(game, m, "f5", "f5_rl", keys.f5_rl(ha, -0.5), f5.p_home_cover(0.5),
                             line=-0.5, team_side="home", side="cover", quotes=quotes))
        recs.append(self._mk(game, m, "f5", "f5_rl", keys.f5_rl(aa, 0.5), 1 - f5.p_home_cover(0.5),
                             line=0.5, team_side="away", side="cover", quotes=quotes))

        # ---- batter props ----
        for team_key, tinfo, flags, sels, regs, sunders in (
            ("home", game.home, home_rbi, home_sels, home_regs, home_su),
            ("away", game.away, away_rbi, away_sels, away_regs, away_su),
        ):
            recs.extend(
                self._batter_props(
                    game, m, res, team_key, tinfo, flags, sels, regs, sunders,
                    opp_siera[team_key], quotes,
                )
            )

        # ---- pitcher props (starters) ----
        # home team's starter faces away hitters -> stats tracked under pit["home"]
        recs.extend(self._pitcher_props(game, m, res, "home", game.home.probable_pitcher, quotes))
        recs.extend(self._pitcher_props(game, m, res, "away", game.away.probable_pitcher, quotes))

        self._attach_context(recs, park, eff)
        return recs

    @staticmethod
    def _attach_context(recs, park, eff) -> None:
        """Stamp shared park + live-weather context onto every rec in a game."""
        wx_summary = wx_note = None
        wx_hr = None
        if eff is not None:
            wx_hr = eff.hr_mult
            wx_note = eff.note or None
            if eff.conditions is not None:
                wx_summary = eff.conditions.summary()
        for r in recs:
            if park is not None:
                r.park_name = park.name
                r.park_factor = park.park_factor
                r.carry_factor = park.carry_factor
                r.roof = park.roof
            r.wx_summary = wx_summary
            r.wx_hr_mult = wx_hr
            r.wx_note = wx_note

    def _comeback_recs(self, game, m, home_x, away_x, home_rbi, away_rbi,
                       home_mgr, away_mgr, home_fat, away_fat):
        """Emit an informational comeback-resilience flag per team."""
        out = []
        diff = (home_x - away_x) if home_x is not None and away_x is not None else None
        specs = (
            ("home", game.home.abbrev, diff, home_rbi, away_mgr, away_fat),
            ("away", game.away.abbrev, (-diff if diff is not None else None), away_rbi, home_mgr, home_fat),
        )
        for team_side, abbrev, xdiff, flags, opp_mgr, opp_fat in specs:
            obp = None
            if flags:
                obp = sum(f.preceding_obp for f in flags) / len(flags)
            sig = ComebackSignal(
                xwoba_diff=xdiff,
                team_obp=obp,
                opp_starter_bf_cap=opp_mgr.starter_bf_cap,
                opp_bullpen_fatigue=opp_fat,
            )
            a = evaluate_comeback(sig)
            if a.score >= 0.60:
                tier = Tier.STRONG
            elif a.score >= 0.50:
                tier = Tier.MODERATE
            else:
                tier = Tier.PASS
            rec = self._mk(
                game, m, "comeback", "comeback", f"{abbrev} comeback",
                a.score, team_side=team_side, side="resilient",
            )
            rec.tier = tier
            rec.reasons = a.reasons or ["baseline resilience"]
            out.append(rec)
        return out

    @staticmethod
    def _selection_for_stat(stat: str, sels: dict[str, Selection | None]) -> Selection | None:
        """Map a prop stat to the relevant V1-style selector."""
        if stat == "RBI":
            return sels.get("RBI")
        if stat == "TB":
            return sels.get("TB")
        if stat in ("2B", "3B"):
            return sels.get("XBH")
        if stat in ("H", "1B", "HR"):
            return sels.get("TB")
        return None

    def _power_floor_reason(self, breg, stat: str) -> str | None:
        """Pipeline wrapper: apply the contact-quality floor when enabled."""
        if not self.cfg.power_floor:
            return None
        return power_floor_reason(
            breg,
            stat,
            xslg_floor=self.cfg.power_xslg_floor,
            k_ceiling=self.cfg.contact_k_ceiling,
        )

    def _singles_under_reason(
        self, su: SinglesUnderResult | None, opp: Siera | None
    ) -> str | None:
        """Exclude the singles/H/H+R+RBI over for a strong singles-Under profile.

        Vetoed when the batter faces a weak arm (SIERA above the ceiling): a
        scrub inflates cheap singles even for a power bat, so don't fade it.
        """
        if (
            not self.cfg.singles_under
            or su is None
            or su.score < self.cfg.singles_under_min
        ):
            return None
        if self.cfg.singles_siera and faces_scrub(opp, self.cfg.singles_siera_bad):
            return None
        return f"singles under (score {su.score:.1f}): {'; '.join(su.reasons)}"

    def _singles_ace_reason(self, opp: Siera | None) -> str | None:
        """Exclude the singles/hit over when the batter faces an ace starter."""
        if opp is None or not self.cfg.singles_siera:
            return None
        if not faces_ace(opp, self.cfg.singles_siera_ace):
            return None
        return f"vs ace: opp SIERA {opp.siera:.2f} < {self.cfg.singles_siera_ace:.2f}"

    def _batter_gate(
        self, breg, su: SinglesUnderResult | None, opp: Siera | None, stat: str
    ) -> str | None:
        """Combined batter-prop floor: contact-quality, then SIERA/singles-Under."""
        reason = self._power_floor_reason(breg, stat)
        if reason is not None:
            return reason
        if stat in ("H", "1B", "HRR"):
            ace = self._singles_ace_reason(opp)
            if ace is not None:
                return ace
            return self._singles_under_reason(su, opp)
        return None

    def _batter_props(
        self, game, m, res, team_key, tinfo, flags, sels, regs, sunders, opp_siera, quotes
    ):
        out = []
        bat = res.bat[team_key]
        lines = {"H": [0.5, 1.5], "1B": [0.5], "2B": [0.5], "HR": [0.5], "R": [0.5], "RBI": [0.5]}
        for i, slot in enumerate(tinfo.lineup):
            name = slot.player.name
            pid = slot.player.mlbam_id
            rbi_sel = sels[i].get("RBI") if i < len(sels) else None
            tb_sel = sels[i].get("TB") if i < len(sels) else None
            breg = regs[i] if i < len(regs) else None
            su = sunders[i] if i < len(sunders) else None
            feat = (
                {"bat_xslg": breg.xslg, "bat_k_pct": breg.k_pct, "bat_bb_pct": breg.bb_pct}
                if breg is not None and breg.bbe >= MIN_BBE
                else {}
            )
            if su is not None and su.profile.has_data:
                feat["bat_singles_under"] = su.score
            if opp_siera is not None and opp_siera.has_data:
                feat["opp_starter_siera"] = opp_siera.siera
            for stat, sl in lines.items():
                arr = bat[stat][:, i].astype(float)
                if stat == "RBI" and rbi_sel is not None:
                    arr = arr * rbi_sel.factor
                sel = self._selection_for_stat(stat, sels[i]) if i < len(sels) else None
                gate = self._batter_gate(breg, su, opp_siera, stat)
                for line in sl:
                    out.append(self._mk(
                        game, m, "batter", f"batter_{stat.lower()}",
                        keys.batter_prop(name, stat, line), p_over(arr, line),
                        line=line, player_id=pid, stat=stat, side="over", quotes=quotes,
                        selector=sel, gate_reason=gate, **feat,
                    ))
            hrr = (bat["H"][:, i] + bat["R"][:, i] + bat["RBI"][:, i]).astype(float)
            hrr_gate = self._batter_gate(breg, su, opp_siera, "HRR")
            for line in (1.5, 2.5):
                out.append(self._mk(
                    game, m, "batter", "batter_hrr", f"{name} H+R+RBI o{line}", p_over(hrr, line),
                    line=line, player_id=pid, stat="HRR", side="over", quotes=quotes,
                    gate_reason=hrr_gate, **feat,
                ))
            tb = (
                bat["1B"][:, i] + 2 * bat["2B"][:, i] + 3 * bat["3B"][:, i] + 4 * bat["HR"][:, i]
            ).astype(float)
            if tb_sel is not None:
                tb = tb * tb_sel.factor
            tb_sel_out = self._selection_for_stat("TB", sels[i]) if i < len(sels) else None
            tb_gate = self._power_floor_reason(breg, "TB")
            for line in (1.5, 2.5, 3.5):
                out.append(self._mk(
                    game, m, "batter", "batter_tb", f"{name} TB o{line}", p_over(tb, line),
                    line=line, player_id=pid, stat="TB", side="over", quotes=quotes,
                    selector=tb_sel_out, gate_reason=tb_gate, **feat,
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
                    keys.pitcher_prop(pitcher.name, label[stat], line), p_over(arr, line),
                    line=line, player_id=pitcher.mlbam_id, stat=stat, side="over", quotes=quotes,
                ))
        return out

    def _mk(self, game, matchup, category, market, selection, prob, *, line=None,
            team_side=None, player_id=None, stat=None, side=None, quotes=None,
            rl_signal: RunLineSignal | None = None,
            selector: Selection | None = None,
            gate_reason: str | None = None,
            bat_xslg: float | None = None,
            bat_k_pct: float | None = None,
            bat_bb_pct: float | None = None,
            bat_singles_under: float | None = None,
            opp_starter_siera: float | None = None) -> Recommendation:
        raw = float(min(max(prob, 1e-6), 1 - 1e-6))
        calibrated = self._calibrator.apply(market, raw)
        rec = Recommendation(
            game_date=game.game_date,
            game_pk=game.game_pk,
            matchup=matchup,
            category=category,
            market=market,
            selection=selection,
            model_prob=calibrated,
            raw_prob=raw,
            line=line,
            team_side=team_side,
            player_id=player_id,
            stat=stat,
            side=side,
        )
        if selector is not None:
            rec.signal = selector.signal
            rec.factor = selector.factor
            rec.score = selector.score
            rec.profile = selector.profile
        rec.bat_xslg = bat_xslg
        rec.bat_k_pct = bat_k_pct
        rec.bat_bb_pct = bat_bb_pct
        rec.bat_singles_under = bat_singles_under
        rec.opp_starter_siera = opp_starter_siera
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
            if rec.handle_pct is None:
                sp = self._splits.get(key)
                if sp is not None:
                    rec.handle_pct = sp.handle_pct
                    rec.bets_pct = sp.bets_pct
            tier, reasons = classify(evres, self.cfg.ev.for_market(market))
            if rl_signal is not None and tier != Tier.PASS:
                steps, rl_reasons = runline_adjustment(team_side, line, rl_signal)
                if steps:
                    tier = bump_tier(tier, steps)
                reasons.extend(rl_reasons)
            rec.tier = tier
            rec.reasons = reasons
        else:
            rec.tier = Tier.PASS
            rec.reasons = ["no market price"]
            sp = self._splits.get(key)
            if sp is not None:
                rec.handle_pct = sp.handle_pct
                rec.bets_pct = sp.bets_pct
                if sp.handle_pct is not None and sp.bets_pct is not None:
                    rec.reasons.append(
                        f"VSIN handle {sp.handle_pct:.0f}% / bets {sp.bets_pct:.0f}%"
                    )
        # Contact-quality floor: hard-exclude a failing batter prop from betting
        # regardless of price (attacks the low-power/whiff-prone false positives).
        if gate_reason is not None and rec.tier != Tier.PASS:
            rec.tier = Tier.PASS
            rec.reasons = [gate_reason, *rec.reasons]
        return rec


def _prev_to_pg(prev):
    if prev is None:
        return None
    d, venue_id, _ = prev
    p = get_park(venue_id)
    if not p:
        return None
    return travel_rest.PrevGame(game_date=d, lat=p.lat, lon=p.lon)
