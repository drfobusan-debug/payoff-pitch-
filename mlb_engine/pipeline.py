"""Daily pipeline orchestration: slate -> features -> model -> filters -> market -> output."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from datetime import date as Date
from pathlib import Path
from typing import TypeVar

from mlb_engine.calibration import Calibrator, ConfidenceShrink
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
from mlb_engine.data.oddsapi import PRICE_ONLY_MARKETS, OddsAPIClient
from mlb_engine.data.parks import get_park
from mlb_engine.data.rotowire import RotoGame, RotoLineup, RotowireClient, norm_person
from mlb_engine.data.savant_expected import load_batter_xslg
from mlb_engine.data.statcast import StatcastRepository, batted_balls
from mlb_engine.data.vsin import Split, VSINClient, lookup_split
from mlb_engine.features.efficiency import (
    PitcherEfficiency,
    build_pitcher_efficiency,
    opponent_discipline_factor,
    recent_start_form,
)
from mlb_engine.features.hits_gate import HitsContactGate
from mlb_engine.features.hr_gate import HRPowerGate
from mlb_engine.features.hrr_adjust import HRRAdjuster
from mlb_engine.features.lineup_lock import (
    LineupLock,
    LineupLockGate,
    hours_to_first_pitch,
)
from mlb_engine.features.ml_gate import MLPenGate, MLSharpGate
from mlb_engine.features.pitch_mix import (
    ArsenalProfile,
    BatterPitchProfile,
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
    LEVERAGE_INNING,
    OutcomeRates,
    blend_bb_rate,
    blend_hr_rate,
    blend_k_rate,
    build_batter_late_rates,
    build_batter_profile,
    build_bullpen_profile,
    build_pitcher_profile,
    lineup_iso,
    pen_arm_spread,
    scale_hr_rate,
    woba_from_rates,
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
from mlb_engine.features.tb_gate import TBGate
from mlb_engine.features.team_form import compute_luck_gaps, load_team_forms, luck_gap_for
from mlb_engine.features.team_splits import (
    LeagueContact,
    TeamSplits,
    build_team_splits,
    league_contact,
)
from mlb_engine.features.trend import PitcherTrends, pitcher_trends
from mlb_engine.features.workload import expected_bf_cap
from mlb_engine.features.xhr import batter_xhr, park_hr_multiplier
from mlb_engine.features.xtb import LeagueXTB
from mlb_engine.filters import travel_rest
from mlb_engine.filters.defense import TeamDefense, load_team_defense
from mlb_engine.filters.human import HumanFactors
from mlb_engine.filters.schedule import dgang_multipliers, local_hour, parse_utc_hour
from mlb_engine.filters.weather import WeatherProvider
from mlb_engine.market import keys
from mlb_engine.market.ev import MarketQuote, anchor_to_market, evaluate
from mlb_engine.market.odds import american_to_prob
from mlb_engine.market.runline import (
    RunLineSignal,
    RunLineVeto,
    runline_adjustment,
    runline_veto,
)
from mlb_engine.market.tiers import Tier, bump_tier, classify, price_screen
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
from mlb_engine.preview import (
    BestBet,
    BullpenLine,
    GamePreview,
    LineupLine,
    RegFlag,
    StarterLine,
)
from mlb_engine.recommendations import Recommendation
from mlb_engine.schemas import BatterSlot, Game, Hand, Pitcher, Player, Slate, TeamGameInfo

log = logging.getLogger(__name__)

FATIGUE_DEPLETED = 60.0  # bullpen-fatigue score at/above which a pen is "depleted"
SHARP_SPREAD_DIV = 15.0  # VSIN run-line handle%-bets% gap treated as sharp money


def apply_outs_bias(prob: float, bias: float, max_prob: float) -> float:
    """Lift a calibrated pitcher_outs over-probability by a capped bias.

    Applied only at/below ``max_prob`` -- the band where the graded audit showed
    the model under-projected outs -- so the correction targets the passed-but-
    profitable overs without inflating high-confidence tails. A ``bias`` of 0
    leaves the probability untouched.
    """
    if bias == 0.0 or prob > max_prob:
        return prob
    return min(1 - 1e-6, prob + bias)


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


def _pen_arsenal_mult(
    arsenal: ArsenalProfile | None, batter: BatterPitchProfile
) -> dict[str, float]:
    """Arsenal matchup for a pen subset; neutral when its mix is unreadable."""
    if arsenal is None:
        return {}
    return arsenal_matchup_multiplier(arsenal, batter)


def _pen_matchup(
    late_ctx: OutcomeRates,
    pen_allowed: OutcomeRates,
    bmult: dict[str, float],
    xbh_mult: dict[str, float],
    bpen_allowed: dict[str, float],
    bpen_k: float,
    bpen_npv: dict[str, float],
    bat_tail: dict[str, float],
    arsenal_mult: dict[str, float] | None = None,
) -> dict[str, float]:
    """A batter's outcome probs vs a given pen ``pen_allowed`` profile.

    Same multiplier stack for the aggregate, bridge and high-leverage pens; only
    the base ``pen_allowed`` rates and the pen's own arsenal differ.
    """
    vp = combine(late_ctx, pen_allowed)
    vp = apply_multipliers(vp, bmult)
    # Apply XBH selection to the bullpen matchup too.
    vp = apply_multipliers(vp, xbh_mult)
    vp = apply_multipliers(vp, bpen_allowed)
    vp = apply_multipliers(vp, {"K": bpen_k})
    vp = apply_multipliers(vp, bpen_npv)
    if arsenal_mult:
        vp = apply_multipliers(vp, arsenal_mult)
    return apply_multipliers(vp, bat_tail)


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


def load_calibrator(live: Path | None = None) -> Calibrator:
    """Load the isotonic calibration map.

    A map refit locally by ``mlb-engine calibrate`` wins over the packaged 2024
    fit: it is trained on this engine's own graded results, so it also covers
    markets the packaged file never saw (``batter_tb`` among them, which is why
    total bases was pricing off the flatter pooled curve).
    """
    if live is not None and live.exists():
        log.info("using locally refit calibration map %s", live)
        return Calibrator.from_json(live)
    if _CALIBRATION_FILE.exists():
        return Calibrator.from_json(_CALIBRATION_FILE)
    log.warning("calibration map %s missing; probabilities left uncalibrated", _CALIBRATION_FILE)
    return Calibrator.identity()


class Pipeline:
    # Rebound per slate in ``run``. The class-level empty default lets a caller
    # that prices a single recommendation stand up a Pipeline without a board.
    _quote_aliases: dict[tuple[str, str, str], list[MarketQuote]] = {}

    def __init__(self, cfg: Config, deps: PipelineDeps) -> None:
        self.cfg = cfg
        self.deps = deps
        self._team_defense: dict[str, TeamDefense] = {}
        self._framing: dict[int, float] = {}
        self._fatigue: dict[int, float | None] = {}
        self._pen_avail: dict[str, float | None] = {}
        self._splits: dict[tuple[str, str, str], Split] = {}
        self._tails = TailAdjuster()
        self._calibrator = (
            load_calibrator(cfg.calibration_file) if cfg.calibrate else Calibrator.identity()
        )
        self._shrink = (
            ConfidenceShrink(pivot=cfg.shrink_pivot, slope=cfg.shrink_slope)
            if cfg.shrink_tails
            else None
        )
        self._rbi_selector = RBISelector(cfg.rbi_obp_threshold)
        self._xbh_selector = XBHSelector()
        self._tb_selector = TBSelector()
        self._luck_gaps: dict[str, float] = {}
        self._hr_gate = HRPowerGate.from_env()
        self._hits_gate = HitsContactGate.from_env()
        self._tb_gate = TBGate.from_env()
        self._ml_gate = MLSharpGate.from_env()
        self._pen_gate = MLPenGate.from_env()
        self._lineup_gate = LineupLockGate.from_env()
        # Games whose lineup came from Rotowire's projection rather than MLB's
        # posted card, and the per-game lineup/timing read built from them.
        self._projected_lineups: set[int] = set()
        self._lineup_lock: LineupLock | None = None
        self._hrr_adjust = HRRAdjuster.from_env()
        self._previews: list[GamePreview] = []
        self._team_splits: dict[str, TeamSplits] = {}
        self._league_contact = LeagueContact(batter=None, pitcher=None)
        self._league_xtb: LeagueXTB | None = None
        self.slate: Slate | None = None

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
        self.slate = slate
        log.info("Slate %s: %d games", slate_date, len(slate.games))
        self._projected_lineups = set()
        self._pen_avail = {}
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
        self._tails = TailAdjuster.build(
            statcast, batter_xslg, fg_bz, fg_pz, power_split=self.cfg.tail_power_split
        )
        # Expected bases per ball, fitted on the season the slate sits in, so
        # every hitter's xSLG is read off the league's own contact.
        self._league_xtb = LeagueXTB.from_statcast(statcast)
        if self._league_xtb is not None:
            log.info(
                "Fitted league expected total bases: %d cells, %.3f TB per ball in play",
                len(self._league_xtb.cells), self._league_xtb.league,
            )

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
        self._quote_aliases = keys.canonical_index(quotes)

        self._luck_gaps = (
            compute_luck_gaps(load_team_forms(self.cfg.team_form_path))
            if self.cfg.runline_luck_gap
            else {}
        )

        recs: list[Recommendation] = []
        self._previews = []
        # League-wide platoon and home/road offense, for the article's ranked
        # matchup verdict. Read once per slate off the frame already loaded.
        self._team_splits = build_team_splits(
            statcast, slate_date, self.cfg.windows.batter_vs_lhp_days
        )
        self._league_contact = league_contact(
            statcast, slate_date, self.cfg.windows.batter_vs_lhp_days
        )
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

    @property
    def previews(self) -> list[GamePreview]:
        """Per-game slate previews assembled during the last :meth:`run`."""
        return self._previews

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
                    if not side.confirmed:
                        self._projected_lineups.add(game.game_pk)
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
        """Return (bat_vs_starter, bat_vs_pen, bat_vs_pen_close, bat_vs_pen_bridge,
        rbi_flags, prev, selections, regs, sunders, half, opp_starter_regression)
        for a lineup."""
        w = self.cfg.windows
        assert opp.probable_pitcher is not None  # guarded in run()
        opp_throws = opp.probable_pitcher.throws.value if opp.probable_pitcher.throws else None

        pit_prof = build_pitcher_profile(
            statcast, opp.probable_pitcher.mlbam_id, slate_date, w.pitcher_form_days
        )
        pit_rows = statcast[statcast["pitcher"] == opp.probable_pitcher.mlbam_id]
        pit_reg = build_pitcher_regression(pit_rows, shrink=w.starter_contact_shrink)
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
            statcast,
            opp.abbrev,
            slate_date,
            w.bullpen_days,
            w.bullpen_min_inning,
            skill_days=w.bullpen_skill_days,
            xwoba_shrink=w.bullpen_xwoba_shrink,
        )
        # Rates come off the recent window, stuff and command off the longer one.
        bpen_reg = build_pitcher_regression(
            bpen.skill_frame, bullpen=self.cfg.pen_contact_level
        )
        bpen_allowed = bpen_reg.allowed_multipliers()
        bpen_k = bpen_reg.k_multiplier()
        avail = (
            self.deps.rotowire.bullpen_availability(opp.abbrev)
            if self.deps.rotowire and self.deps.rotowire.available()
            else None
        )
        bpen_npv = bpen.npv_multipliers(avail)

        # Arsenal matching for the pen as well as the starter: the corps' mix and
        # per-class SwStr% against the same hitter pitch-class profile. A pen's
        # leverage arms throw a different mix from its bridge arms, so each
        # subset is read on its own and falls back to the whole corps when thin.
        pen_arsenal = pen_bridge_arsenal = pen_lev_arsenal = None
        if self.cfg.pen_arsenal:
            pen_rows = bpen.skill_frame
            pen_arsenal = build_arsenal(pen_rows)
            pen_bridge_arsenal = pen_lev_arsenal = pen_arsenal
            if len(pen_rows) and "inning" in pen_rows:
                lev = build_arsenal(pen_rows[pen_rows["inning"] >= LEVERAGE_INNING])
                bridge = build_arsenal(pen_rows[pen_rows["inning"] < LEVERAGE_INNING])
                pen_lev_arsenal = lev if lev.usage else pen_arsenal
                pen_bridge_arsenal = bridge if bridge.usage else pen_arsenal

        # travel/rest for this offense (applied at game level via prev game)
        prev = self.deps.stats.last_game_venue(team.team_id, slate_date)

        profiles = []
        regs = []
        sunders: list[SinglesUnderResult] = []
        bat_vs_starter = []
        bat_vs_league = []  # same hitters vs a league-average arm, for the preview
        bat_vs_pen = []
        bat_vs_pen_close = []
        bat_vs_pen_bridge = []
        selections: list[dict[str, Selection | None]] = []
        for slot_idx, slot in enumerate(team.lineup):
            pid = slot.player.mlbam_id
            bprof = build_batter_profile(
                statcast, pid, slate_date, w.batter_home_away_days, w.batter_vs_rhp_days,
                w.batter_vs_lhp_days, self.cfg.batter_split_prior,
            )
            profiles.append(bprof)
            ctx = bprof.for_context(team.is_home, opp_throws)

            bslice = statcast[statcast["batter"] == pid]
            # Observed HR/PA counts home runs; expected HR measures the contact
            # that produced them, against the actual walls it was hit toward.
            if self.cfg.xhr_blend:
                ctx = blend_hr_rate(
                    ctx, batter_xhr(bslice).xhr_per_pa, self.cfg.xhr_prior_weight
                )
            # ...which leaves the rate park-neutral, so put tonight's park back:
            # what his own batted balls would be worth against these fences.
            if self.cfg.xhr_park and park is not None:
                ctx = scale_hr_rate(
                    ctx, park_hr_multiplier(bslice, park.venue_id)
                )
            breg = build_batter_regression(
                bslice, sprint.get(pid, 27.0), league_xtb=self._league_xtb
            )
            regs.append(breg)
            bmult = breg.multipliers(
                self.cfg.singles_barrel_slope if self.cfg.singles_barrel else 0.0,
                self.cfg.singles_gb_slope if self.cfg.singles_gb else 0.0,
                self.cfg.singles_ld_slope if self.cfg.singles_gb else 0.0,
                self.cfg.singles_shape,
            )

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
            # Missing bats and suppressing contact quality are different skills,
            # so the K split says nothing about the starter's home-run risk to
            # this side of the plate.
            platoon_hr = pit_reg.platoon_power_multiplier(bats)

            vs_start = combine(ctx, pit_allowed)
            vs_start = apply_multipliers(vs_start, bmult)
            # V1-style XBH selector feeds the existing 2B/3B multiplier block.
            vs_start = apply_multipliers(vs_start, xbh_sel.outcome_multipliers)
            vs_start = apply_multipliers(vs_start, pit_allowed_mult)
            vs_start = apply_multipliers(vs_start, {"K": k_mult})
            vs_start = apply_multipliers(vs_start, {"K": platoon_k})
            vs_start = apply_multipliers(vs_start, {"HR": platoon_hr})
            vs_start = apply_multipliers(vs_start, arsenal_mult)
            vs_start = apply_multipliers(vs_start, pit_tail)
            vs_start = apply_multipliers(vs_start, bat_tail)

            # The same batter, same platoon/venue context, against a
            # league-average arm: the article's reference point for how much of
            # this matchup is the lineup and how much is the man on the mound.
            vs_league = combine(ctx, league_pitcher_rates())
            vs_league = apply_multipliers(vs_league, bmult)
            vs_league = apply_multipliers(vs_league, xbh_sel.outcome_multipliers)
            vs_league = apply_multipliers(vs_league, bat_tail)
            bat_vs_league.append(vs_league)

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
            # Aggregate pen (used once the game is out of hand) vs the team's
            # high-leverage arms (used late in a still-close game).
            pen_args = (bmult, xbh_sel.outcome_multipliers, bpen_allowed, bpen_k, bpen_npv, bat_tail)
            vs_pen = _pen_matchup(
                late_ctx, bpen.allowed, *pen_args,
                arsenal_mult=_pen_arsenal_mult(pen_arsenal, bpp),
            )
            vs_pen_close = _pen_matchup(
                late_ctx, bpen.allowed_leverage, *pen_args,
                arsenal_mult=_pen_arsenal_mult(pen_lev_arsenal, bpp),
            )
            # The middle men who cover the hand-off to the 8th, priced apart from
            # the setup/closer pair they precede.
            vs_pen_bridge = _pen_matchup(
                late_ctx, bpen.bridge, *pen_args,
                arsenal_mult=_pen_arsenal_mult(pen_bridge_arsenal, bpp),
            )

            bat_vs_starter.append(vs_start)
            bat_vs_pen.append(vs_pen)
            bat_vs_pen_close.append(vs_pen_close)
            bat_vs_pen_bridge.append(vs_pen_bridge)
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

        # Reader-facing matchup story: this offense's lineup line, plus the
        # opposing starter + bullpen it will face (built from the same objects
        # the simulator just consumed, so nothing is recomputed).
        half = _preview_half(
            team,
            opp,
            regs,
            pit_reg,
            bpen,
            pitcher_trends(pit_rows, slate_date, w.pitcher_form_days),
            self._team_splits.get(team.abbrev),
            opp_throws,
            pitcher_siera(pit_rows),
            self._league_contact,
            _mean_woba(bat_vs_starter),
            _mean_woba(bat_vs_league),
            _mean_woba(bat_vs_pen),
            _mean_woba(bat_vs_pen_close),
        )

        return (
            bat_vs_starter,
            bat_vs_pen,
            bat_vs_pen_close,
            bat_vs_pen_bridge,
            rbi_flags,
            prev,
            selections,
            regs,
            sunders,
            half,
            pit_reg,
        )

    def _apply_env(self, rates_list, mult: dict[str, float]):
        if not mult:
            return rates_list
        return [apply_multipliers(r, mult) for r in rates_list]

    def _apply_all(self, rates_list, mults: list[dict[str, float]]):
        for m in mults:
            rates_list = self._apply_env(rates_list, m)
        return rates_list

    def _pen_availability(self, abbrev: str) -> float | None:
        """Cached Rotowire bullpen availability (0..1 rested), None when no feed."""
        if abbrev not in self._pen_avail:
            roto = self.deps.rotowire
            self._pen_avail[abbrev] = (
                roto.bullpen_availability(abbrev)
                if roto is not None and roto.available()
                else None
            )
        return self._pen_avail[abbrev]

    def _bullpen_fatigue(self, team_id: int, slate_date: Date) -> float | None:
        """Cached 0-100 bullpen-fatigue score for a team on the slate date."""
        if team_id not in self._fatigue:
            self._fatigue[team_id] = self.deps.stats.bullpen_fatigue(team_id, slate_date)
        return self._fatigue[team_id]

    def _spread_divergence(self, matchup: str, abbrev: str) -> float | None:
        """VSIN run-line handle% - bets% for a team (sharp side when positive)."""
        sp = lookup_split(self._splits, matchup, "game_rl", f"{abbrev} -1.5")
        return sp.divergence if sp is not None else None

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
            x = batted_balls(rows)["estimated_woba_using_speedangle"].dropna()
            if len(x) >= 15:
                vals.append(float(x.mean()))
        if not vals:
            return None
        return sum(vals) / len(vals)

    def _runline_gate_inputs(
        self,
        base: RunLineSignal,
        statcast,
        slate_date: Date,
        *,
        fav: TeamGameInfo,
        dog: TeamGameInfo,
        dog_pit_rows,
        fav_opp_eff: PitcherEfficiency,
    ) -> RunLineSignal:
        """Inputs for the run-line NPV gates, computed only for enabled gates.

        Each is ``None`` when its sample is too thin, which leaves the gate
        keyed on it inert rather than vetoing on noise.
        """
        gates = self.cfg.runline_gates
        w = self.cfg.windows

        fav_iso = opp_gb = None
        if gates.iso_gb:
            fav_iso = lineup_iso(
                statcast,
                [s.player.mlbam_id for s in fav.lineup],
                slate_date,
                w.batter_home_away_days,
            )
            opp_gb = fav_opp_eff.gb_pct

        form = recent_start_form(dog_pit_rows, slate_date) if gates.dog_sp else None

        pen_xwoba = pen_k = None
        if gates.dog_pen:
            pen = build_bullpen_profile(
                statcast,
                dog.abbrev,
                slate_date,
                w.bullpen_days,
                w.bullpen_min_inning,
                skill_days=w.bullpen_skill_days,
                xwoba_shrink=w.bullpen_xwoba_shrink,
            )
            pen_xwoba = pen.xwoba_allowed
            pen_k = pen.k_pct if pen.xwoba_allowed is not None else None

        return replace(
            base,
            fav_iso=fav_iso,
            fav_opp_sp_gb_pct=opp_gb,
            dog_sp_whip_l3=form.whip if form else None,
            dog_sp_hard_hit_l3=form.hard_hit_pct if form else None,
            dog_pen_xwoba=pen_xwoba,
            dog_pen_k_pct=pen_k,
        )

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
        # Late-information read for this game: did we price the posted lineup,
        # and how long before first pitch? Consumed by the game_ml gate below and
        # stamped on every rec in _attach_context.
        self._lineup_lock = self._lineup_gate.read(
            projected=game.game_pk in self._projected_lineups,
            hours=hours_to_first_pitch(game.game_datetime_utc),
        )

        # weather effect (park-level)
        weather_mult = {}
        eff = None
        if park:
            eff = self.deps.weather.fetch(park, game.game_datetime_utc)
            weather_mult = eff.multipliers()

        # The ballpark's own effect on the hit types the HR park multiplier and
        # the weather term do not reach. Both model carry, and carry is what
        # turns a fly ball into a home run -- not what drops a single in front
        # of a deep outfielder or rolls a double into the alley behind him.
        park_mult: dict[str, float] = {}
        if park is not None:
            if self.cfg.park_singles:
                park_mult["1B"] = park.singles_factor
            if self.cfg.park_xbh:
                park_mult["2B"] = park.xbh_factor
                park_mult["3B"] = park.xbh_factor

        (home_start, home_pen, home_pen_close, home_pen_bridge, home_rbi, home_prev,
         home_sels, home_regs, home_su, home_half, home_opp_reg) = self._team_offense(
            game.home, game.away, statcast, slate_date, sprint, park, weather_mult
        )
        (away_start, away_pen, away_pen_close, away_pen_bridge, away_rbi, away_prev,
         away_sels, away_regs, away_su, away_half, away_opp_reg) = self._team_offense(
            game.away, game.home, statcast, slate_date, sprint, park, weather_mult
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
        home_env = [weather_mult, park_mult, home_tr, home_hr_boost, home_human,
                    home_def, home_mgr.offense_multipliers(), home_dgang]
        away_env = [weather_mult, park_mult, away_tr, away_hr_boost, away_human,
                    away_def, away_mgr.offense_multipliers(), away_dgang]
        home_start = self._apply_all(home_start, home_env)
        home_pen_env = [*home_env, home_mgr.pen_multipliers()]
        home_pen = self._apply_all(home_pen, home_pen_env)
        home_pen_close = self._apply_all(home_pen_close, home_pen_env)
        home_pen_bridge = self._apply_all(home_pen_bridge, home_pen_env)
        away_start = self._apply_all(away_start, away_env)
        away_pen_env = [*away_env, away_mgr.pen_multipliers()]
        away_pen = self._apply_all(away_pen, away_pen_env)
        away_pen_close = self._apply_all(away_pen_close, away_pen_env)
        away_pen_bridge = self._apply_all(away_pen_bridge, away_pen_env)

        # Starter exit model: manager hooks (batters-faced + pitch-count caps)
        # tightened by each starter's own recent workload, plus a pitch-efficiency
        # profile (P/PA, F-Strike%, GB%) so out volume tracks pitch economy, not
        # just batters faced. Drives realistic outs (innings) and strikeout unders.
        w = self.cfg.windows
        home_pit_rows = statcast[statcast["pitcher"] == game.home.probable_pitcher.mlbam_id]
        away_pit_rows = statcast[statcast["pitcher"] == game.away.probable_pitcher.mlbam_id]
        # Thin-Statcast starter gate: a starter with too few tracked pitches is
        # priced off an optimistic prior, so veto that game's starter-driven
        # markets rather than bet a matchup the model can't actually read.
        home_sp_thin = self._thin_starter_reason(
            game.home.probable_pitcher.name, len(home_pit_rows)
        )
        away_sp_thin = self._thin_starter_reason(
            game.away.probable_pitcher.name, len(away_pit_rows)
        )
        game_sp_thin = home_sp_thin or away_sp_thin
        # SIERA of each starter (from Statcast); away batters face the home
        # starter and vice-versa -> map by the batter's own team_key.
        opp_siera = {
            "away": pitcher_siera(home_pit_rows),
            "home": pitcher_siera(away_pit_rows),
        }
        # Contact quality the starter each lineup faces allows (reusing the
        # regression the simulator already consumed), so the total-bases gate can
        # drop overs against a barrel/hard-hit suppressor.
        opp_contact = {"home": home_opp_reg, "away": away_opp_reg}
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
            bat_vs_pen_close=home_pen_close,
            bat_vs_pen_bridge=home_pen_bridge if self.cfg.pen_bridge else None,
            starter_bf_cap=home_cap,
            starter_pitch_cap=home_eff.pitch_cap,
            pitch_eff=min(1.35, home_eff.efficiency_scaler() * home_disc),
            gb_dp_rate=home_eff.gb_dp_rate(),
        )
        away_cfg = TeamSimConfig(
            bat_vs_starter=away_start,
            bat_vs_pen=away_pen,
            bat_vs_pen_close=away_pen_close,
            bat_vs_pen_bridge=away_pen_bridge if self.cfg.pen_bridge else None,
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
        # Expected run differential (home perspective): sequencing-luck-free mean
        # of the simulated margin, plus its spread. Surfaced on the card.
        xrd = float(margin.mean())
        xrd_sd = float(margin.std())

        # Run-line PPV confidence: team xwOBA differential + depleted-favorite
        # bullpen (live from the public StatsAPI workload proxy).
        home_x = self._team_xwoba(statcast, game.home)
        away_x = self._team_xwoba(statcast, game.away)
        home_fat = self._bullpen_fatigue(game.home.team_id, slate_date)
        away_fat = self._bullpen_fatigue(game.away.team_id, slate_date)
        fav_side = "home" if float((margin > 0).mean()) >= 0.5 else "away"
        fav_fat = home_fat if fav_side == "home" else away_fat
        rl_signal = self._runline_gate_inputs(
            RunLineSignal(
                xwoba_diff=(
                    (home_x - away_x) if home_x is not None and away_x is not None else None
                ),
                fav_pen_depleted_side=(
                    fav_side if fav_fat is not None and fav_fat >= FATIGUE_DEPLETED else None
                ),
                sharp_money_side=self._sharp_spread_side(m, ha, aa),
                fav_side=fav_side,
                model_total=float(total.mean()),
                luck_gap_home=luck_gap_for(ha, self._luck_gaps),
                luck_gap_away=luck_gap_for(aa, self._luck_gaps),
            ),
            statcast,
            slate_date,
            fav=game.home if fav_side == "home" else game.away,
            dog=game.away if fav_side == "home" else game.home,
            dog_pit_rows=away_pit_rows if fav_side == "home" else home_pit_rows,
            fav_opp_eff=home_eff if fav_side == "away" else away_eff,
        )

        recs.append(self._mk(game, m, "game", "game_ml", keys.game_ml(ha), float((margin > 0).mean()),
                             team_side="home", side="win", quotes=quotes, gate_reason=game_sp_thin,
                             pen_fatigue=home_fat, opp_pen_fatigue=away_fat,
                             pen_availability=self._pen_availability(ha)))
        recs.append(self._mk(game, m, "game", "game_ml", keys.game_ml(aa), float((margin < 0).mean()),
                             team_side="away", side="win", quotes=quotes, gate_reason=game_sp_thin,
                             pen_fatigue=away_fat, opp_pen_fatigue=home_fat,
                             pen_availability=self._pen_availability(aa)))
        recs.append(self._mk(game, m, "game", "game_rl", keys.game_rl(ha, -1.5), float((margin > 1.5).mean()),
                             line=-1.5, team_side="home", side="cover", quotes=quotes, rl_signal=rl_signal, gate_reason=game_sp_thin))
        recs.append(self._mk(game, m, "game", "game_rl", keys.game_rl(aa, 1.5), float((margin > -1.5).mean()),
                             line=1.5, team_side="away", side="cover", quotes=quotes, rl_signal=rl_signal, gate_reason=game_sp_thin))
        recs.append(self._mk(game, m, "game", "game_rl", keys.game_rl(aa, -1.5), float((-margin > 1.5).mean()),
                             line=-1.5, team_side="away", side="cover", quotes=quotes, rl_signal=rl_signal, gate_reason=game_sp_thin))
        recs.append(self._mk(game, m, "game", "game_rl", keys.game_rl(ha, 1.5), float((-margin > -1.5).mean()),
                             line=1.5, team_side="home", side="cover", quotes=quotes, rl_signal=rl_signal, gate_reason=game_sp_thin))

        # ---- comeback-resilience flags ----
        recs.extend(self._comeback_recs(
            game, m, home_x, away_x, home_rbi, away_rbi, home_mgr, away_mgr,
            home_fat, away_fat,
        ))

        for line in (7.5, 8.5, 9.5, 10.5):
            recs.append(self._mk(game, m, "game", "game_total", keys.game_total(True, line), p_over(total, line),
                                 line=line, side="over", quotes=quotes, gate_reason=game_sp_thin))
            recs.append(self._mk(game, m, "game", "game_total", keys.game_total(False, line), 1 - p_over(total, line),
                                 line=line, side="under", quotes=quotes, gate_reason=game_sp_thin))

        # ---- F5 markets ----
        recs.append(self._mk(game, m, "f5", "f5_ml", keys.f5_ml(ha), f5.p_home_ml,
                             team_side="home", side="win", quotes=quotes, gate_reason=game_sp_thin))
        recs.append(self._mk(game, m, "f5", "f5_ml", keys.f5_ml(aa), f5.p_away_ml,
                             team_side="away", side="win", quotes=quotes, gate_reason=game_sp_thin))
        recs.append(self._mk(game, m, "f5", "f5_ml", "F5 Tie", f5.p_tie, side="tie", quotes=quotes, gate_reason=game_sp_thin))
        for line in (4.5, 5.5):
            po = f5.p_total_over(line)
            recs.append(self._mk(game, m, "f5", "f5_total", keys.f5_total(True, line), po,
                                 line=line, side="over", quotes=quotes, gate_reason=game_sp_thin))
            recs.append(self._mk(game, m, "f5", "f5_total", keys.f5_total(False, line), 1 - po,
                                 line=line, side="under", quotes=quotes, gate_reason=game_sp_thin))
        recs.append(self._mk(game, m, "f5", "f5_rl", keys.f5_rl(ha, -0.5), f5.p_home_cover(0.5),
                             line=-0.5, team_side="home", side="cover", quotes=quotes, gate_reason=game_sp_thin))
        recs.append(self._mk(game, m, "f5", "f5_rl", keys.f5_rl(aa, 0.5), 1 - f5.p_home_cover(0.5),
                             line=0.5, team_side="away", side="cover", quotes=quotes, gate_reason=game_sp_thin))

        # ---- batter props ----
        for team_key, tinfo, flags, sels, regs, sunders, opp_sp in (
            ("home", game.home, home_rbi, home_sels, home_regs, home_su,
             game.away.probable_pitcher),
            ("away", game.away, away_rbi, away_sels, away_regs, away_su,
             game.home.probable_pitcher),
        ):
            recs.extend(
                self._batter_props(
                    game, m, res, team_key, tinfo, flags, sels, regs, sunders,
                    opp_siera[team_key], opp_contact[team_key], quotes,
                    park=park, weather_mult=weather_mult,
                    opp_throws=(
                        opp_sp.throws.value
                        if opp_sp is not None and opp_sp.throws
                        else None
                    ),
                )
            )

        # ---- pitcher props (starters) ----
        # home team's starter faces away hitters -> stats tracked under pit["home"]
        recs.extend(self._pitcher_props(game, m, res, "home", game.home.probable_pitcher, quotes, home_sp_thin))
        recs.extend(self._pitcher_props(game, m, res, "away", game.away.probable_pitcher, quotes, away_sp_thin))

        self._attach_context(recs, park, eff, xrd, xrd_sd, self._lineup_lock)

        # ---- reader-facing slate preview ----
        # home starter is the arm the AWAY lineup faced (away_half.opp_*), and
        # the home offense faces the away pitching (home_half.opp_*).
        away_half.opp_pen.fatigue = _fnum(away_fat)  # away pen faced by home bats
        home_half.opp_pen.fatigue = _fnum(home_fat)
        self._previews.append(
            self._build_preview(
                game, m, recs, total, margin, xrd, xrd_sd,
                park, eff, home_half, away_half,
            )
        )
        return recs

    def _build_preview(
        self, game, matchup, recs, total, margin, xrd, xrd_sd,
        park, eff, home_half, away_half,
    ) -> GamePreview:
        """Assemble the per-game preview record from the priced slate."""
        p_home_win = float((margin > 0).mean())
        p_blowout = float((abs(margin) >= 4).mean())
        p_close = float((abs(margin) <= 1).mean())

        # Moneyline market: pull the two game_ml recs for implied prob + edge.
        ha, aa = game.home.abbrev, game.away.abbrev
        ml = {r.team_side: r for r in recs if r.market == "game_ml" and r.team_side in ("home", "away")}
        home_ml = ml.get("home")
        away_ml = ml.get("away")
        home_prob = float(home_ml.model_prob) if home_ml else p_home_win
        away_prob = float(away_ml.model_prob) if away_ml else 1.0 - p_home_win
        fav_side = "home" if home_prob >= away_prob else "away"
        fav_rec = home_ml if fav_side == "home" else away_ml
        fav_team = ha if fav_side == "home" else aa
        fav_odds = _fnum(fav_rec.market_american) if fav_rec else None

        # Best bets in this game: the engine's own buy tiers, best EV first.
        buys = [
            r for r in recs
            if r.tier in (Tier.STRONG, Tier.MODERATE) and r.ev is not None
        ]
        buys.sort(key=lambda r: (r.tier != Tier.STRONG, -(r.ev or 0.0)))
        best_bets = [
            BestBet(
                selection=r.selection,
                market=r.market,
                odds=_fnum(r.market_american),
                model_prob=float(r.model_prob),
                edge=_fnum(r.edge),
                ev=_fnum(r.ev),
                tier=r.tier.value,
            )
            for r in buys[:4]
        ]

        wx_summary = None
        wx_hr = None
        if eff is not None:
            wx_hr = _fnum(eff.hr_mult)
            if eff.conditions is not None:
                wx_summary = eff.conditions.summary()

        game_date = recs[0].game_date.isoformat() if recs else ""
        return GamePreview(
            game_date=game_date,
            game_pk=int(game.game_pk),
            matchup=matchup,
            home=ha,
            away=aa,
            home_starter=away_half.opp_starter,
            away_starter=home_half.opp_starter,
            home_lineup=home_half.lineup,
            away_lineup=away_half.lineup,
            home_pen=away_half.opp_pen,
            away_pen=home_half.opp_pen,
            xrd=round(float(xrd), 2),
            xrd_sd=round(float(xrd_sd), 2),
            total_mean=round(float(total.mean()), 2),
            p_home_win=round(p_home_win, 3),
            p_blowout=round(p_blowout, 3),
            p_close=round(p_close, 3),
            park_name=park.name if park else None,
            park_factor=_fnum(park.park_factor) if park else None,
            roof=park.roof if park else None,
            wx_summary=wx_summary,
            wx_hr_mult=wx_hr,
            home_ml_prob=round(home_prob, 3),
            away_ml_prob=round(away_prob, 3),
            fav_side=fav_side,
            fav_team=fav_team,
            fav_odds=fav_odds,
            fav_implied=(
                round(american_to_prob(fav_odds), 3) if fav_odds is not None else None
            ),
            fav_edge=_fnum(fav_rec.edge) if fav_rec else None,
            best_bets=best_bets,
        )

    @staticmethod
    def _attach_context(recs, park, eff, xrd=None, xrd_sd=None, lock=None) -> None:
        """Stamp shared park + weather + lineup-lock context onto a game's recs."""
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
            r.xrd = xrd
            r.xrd_sd = xrd_sd
            if lock is not None:
                r.lineup_status = lock.status
                r.hours_to_first_pitch = lock.hours_to_first_pitch

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

    @staticmethod
    def _hits_context(park) -> float | None:
        """Tonight's hit environment: the park's singles factor, 1.0 = neutral.

        ``None`` when the park is unknown, which leaves the contact gate neutral
        rather than guessing.

        This reads ``singles_factor`` rather than the runs park factor it used
        to. The runs factor is mostly home runs and carries no singles signal
        (+0.09 across the 30 parks), so an average bat was being bought at
        Yankee Stadium -- one of the worst singles parks -- and blocked at
        Busch, the best. The weather no longer contributes: fitted against
        realised singles it measures nothing (see ``filters/weather.py``), so
        the ballpark is the whole of tonight's hit environment.
        """
        if park is None:
            return None
        return float(park.singles_factor)

    def _batter_gate(
        self,
        breg,
        su: SinglesUnderResult | None,
        opp: Siera | None,
        stat: str,
        opp_contact=None,
        slot: int | None = None,
        context: float | None = None,
        platoon_disadvantage: bool = False,
    ) -> str | None:
        """Combined batter-prop floor: contact-quality, then SIERA/singles-Under."""
        reason = self._power_floor_reason(breg, stat)
        if reason is not None:
            return reason
        if stat in ("H", "1B", "HRR"):
            ace = self._singles_ace_reason(opp)
            if ace is not None:
                return ace
            pa_risk = self._hits_gate.platoon_pa_reason(slot, platoon_disadvantage)
            if pa_risk is not None:
                return pa_risk
            keep, hits_reason = self._hits_gate.allows(breg, context)
            if not keep:
                # A poor bat whose night is also against him is a fade, not just
                # a no-buy; say so, so the under price is graded as a bet.
                return (
                    self._hits_gate.under_reason(
                        breg, context, slot, platoon_disadvantage
                    )
                    or hits_reason
                )
            return self._singles_under_reason(su, opp)
        if stat == "HR":
            # The hitter's own power is gated at tier time (it needs the barrel
            # trend windows); the matchup and the lineup spot are known here.
            if opp_contact is not None:
                matchup = self._hr_gate.opponent_reason(
                    opp_contact.barrel_allowed,
                    opp_contact.hard_hit_allowed,
                    opp_contact.bbe,
                )
                if matchup is not None:
                    return matchup
            return self._hr_gate.slot_reason(slot)
        return None

    def _batter_props(
        self, game, m, res, team_key, tinfo, flags, sels, regs, sunders, opp_siera,
        opp_contact, quotes, park=None, weather_mult=None, opp_throws=None
    ):
        out = []
        bat = res.bat[team_key]
        context = self._hits_context(park)
        lines = {"H": [0.5, 1.5], "1B": [0.5], "2B": [0.5], "HR": [0.5], "R": [0.5], "RBI": [0.5]}
        for i, slot in enumerate(tinfo.lineup):
            name = slot.player.name
            pid = slot.player.mlbam_id
            bats = slot.player.bats.value if slot.player.bats else None
            # Same-handed and not a switch hitter: the side of the platoon a
            # bench bat gets lifted from.
            platoon_bad = bats is not None and opp_throws is not None and bats == opp_throws
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
                if stat == "RBI" and rbi_sel is not None and self.cfg.legacy_prop_post_mult:
                    arr = arr * rbi_sel.factor
                sel = self._selection_for_stat(stat, sels[i]) if i < len(sels) else None
                gate = self._batter_gate(
                    breg, su, opp_siera, stat, opp_contact=opp_contact, slot=i + 1,
                    context=context, platoon_disadvantage=platoon_bad,
                )
                for line in sl:
                    out.append(self._mk(
                        game, m, "batter", f"batter_{stat.lower()}",
                        keys.batter_prop(name, stat, line), p_over(arr, line),
                        line=line, player_id=pid, stat=stat, side="over", quotes=quotes,
                        selector=sel, gate_reason=gate, **feat,
                    ))
            hrr = (bat["H"][:, i] + bat["R"][:, i] + bat["RBI"][:, i]).astype(float)
            hrr_gate = self._batter_gate(
                breg, su, opp_siera, "HRR", slot=i + 1,
                context=context, platoon_disadvantage=platoon_bad,
            )
            hrr_sweet = tb_sel.bat_sweet_spot if tb_sel is not None else None
            hrr_xslg = tb_sel.bat_xslg if tb_sel is not None else None
            for line in (1.5, 2.5):
                out.append(self._mk(
                    game, m, "batter", "batter_hrr", f"{name} H+R+RBI o{line}", p_over(hrr, line),
                    line=line, player_id=pid, stat="HRR", side="over", quotes=quotes,
                    gate_reason=hrr_gate, hrr_sweet=hrr_sweet, hrr_xslg=hrr_xslg, **feat,
                ))
            tb = (
                bat["1B"][:, i] + 2 * bat["2B"][:, i] + 3 * bat["3B"][:, i] + 4 * bat["HR"][:, i]
            ).astype(float)
            if tb_sel is not None and self.cfg.legacy_prop_post_mult:
                tb = tb * tb_sel.factor
            tb_sel_out = self._selection_for_stat("TB", sels[i]) if i < len(sels) else None
            tb_gate = self._tb_gate_reason(breg, tb_sel, opp_contact)
            for line in (1.5, 2.5, 3.5):
                out.append(self._mk(
                    game, m, "batter", "batter_tb", f"{name} TB o{line}", p_over(tb, line),
                    line=line, player_id=pid, stat="TB", side="over", quotes=quotes,
                    selector=tb_sel_out, gate_reason=tb_gate, **feat,
                ))
        return out

    def _pitcher_props(self, game, m, res, team_key, pitcher, quotes, gate_reason=None):
        out = []
        pit = res.pit[team_key]
        lines = {"K": [4.5, 5.5, 6.5], "outs": [15.5, 17.5], "H": [4.5, 5.5], "BB": [1.5, 2.5], "ER": [2.5, 3.5]}
        label = {"K": "Ks", "outs": "Outs", "H": "Hits", "BB": "Walks", "ER": "ER"}
        for stat, sl in lines.items():
            arr = pit[stat].astype(float)
            for line in sl:
                gate = None
                if stat == "K" and line > self.cfg.pitcher_k_max_buy_line:
                    gate = (
                        f"pitcher_k o{line} above buy cap "
                        f"{self.cfg.pitcher_k_max_buy_line}"
                    )
                out.append(self._mk(
                    game, m, "pitcher", f"pitcher_{stat.lower()}",
                    keys.pitcher_prop(pitcher.name, label[stat], line), p_over(arr, line),
                    line=line, player_id=pitcher.mlbam_id, stat=stat, side="over", quotes=quotes,
                    gate_reason=gate or gate_reason,
                ))
        return out

    def _thin_starter_reason(self, name: str, pitches: int) -> str | None:
        """Gate reason when a starter has too little Statcast to be priced.

        Returns ``None`` (no gate) when the gate is disabled or the sample clears
        the ``thin_starter_min_pitches`` floor.
        """
        if not self.cfg.thin_starter_gate:
            return None
        floor = self.cfg.thin_starter_min_pitches
        if pitches >= floor:
            return None
        return f"thin Statcast: {name} {pitches}p < {floor}"

    def _tb_gate_reason(self, breg, tb_sel: Selection | None, opp) -> str | None:
        """Gate reason excluding a total-bases over from betting.

        Total bases is a power market, so it is gated on the hitter's barrel rate
        and max exit velocity -- the metrics the graded window showed separate the
        over winners -- rather than on xSLG alone, and on the contact quality the
        opposing starter allows. Falls back to the shared contact floor when the
        TB gate is disabled so the market is never left ungated.
        """
        if not self._tb_gate.enabled:
            return self._power_floor_reason(breg, "TB")
        barrel = tb_sel.hr_barrel if tb_sel is not None else None
        max_ev = tb_sel.hr_max_ev if tb_sel is not None else None
        bbe = tb_sel.hr_bbe if tb_sel is not None else None
        reason = self._tb_gate.power_reason(barrel, max_ev, bbe)
        if reason is not None:
            return reason
        return self._tb_gate.opponent_reason(
            opp.barrel_allowed, opp.hard_hit_allowed, opp.bbe
        )

    def _apply_outs_bias(self, prob: float) -> float:
        return apply_outs_bias(
            prob,
            self.cfg.pitcher_outs_prob_bias,
            self.cfg.pitcher_outs_bias_max_prob,
        )

    def _mk(self, game, matchup, category, market, selection, prob, *, line=None,
            team_side=None, player_id=None, stat=None, side=None, quotes=None,
            rl_signal: RunLineSignal | None = None,
            selector: Selection | None = None,
            gate_reason: str | None = None,
            bat_xslg: float | None = None,
            bat_k_pct: float | None = None,
            bat_bb_pct: float | None = None,
            bat_singles_under: float | None = None,
            opp_starter_siera: float | None = None,
            hrr_sweet: float | None = None,
            hrr_xslg: float | None = None,
            pen_fatigue: float | None = None,
            opp_pen_fatigue: float | None = None,
            pen_availability: float | None = None) -> Recommendation:
        raw = float(min(max(prob, 1e-6), 1 - 1e-6))
        calibrated = self._calibrator.apply(market, raw)
        if self._shrink is not None:
            calibrated = self._shrink.apply(calibrated)
        if market == "pitcher_outs":
            calibrated = self._apply_outs_bias(calibrated)
        if market == "batter_hrr":
            calibrated = self._hrr_adjust.apply(calibrated, line, hrr_sweet, hrr_xslg)
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
        rec.pen_fatigue = pen_fatigue
        rec.opp_pen_fatigue = opp_pen_fatigue
        # NPV gates run whether or not the market is priced: an unpriced run
        # line is already a Pass, but the audit still needs to know which gate
        # removed it to grade the counterfactual.
        veto = (
            runline_veto(team_side, line, rl_signal, self.cfg.runline_gates)
            if rl_signal is not None
            else RunLineVeto()
        )
        if veto.triggered:
            rec.veto_gate = veto.gate

        key = (matchup, market, selection)
        q = (quotes or {}).get(key)
        if q is None:
            q = self._quote_aliases.get((matchup, market, keys.canonical(selection)))
        if q:
            evres = evaluate(rec.model_prob, q)
            if self.cfg.market_anchor > 0:
                # Re-price against a probability pulled toward the market. Only
                # the bet probability moves; rec.model_prob stays the model's.
                bet_prob = anchor_to_market(
                    rec.model_prob, evres.fair_prob, self.cfg.market_anchor
                )
                evres = evaluate(bet_prob, q)
            rec.book = evres.best_quote.book
            rec.market_american = evres.best_quote.american
            rec.opposite_american = evres.best_quote.opposite_american
            rec.ev = evres.ev
            rec.edge = evres.edge
            rec.fair_prob = evres.fair_prob
            rec.bet_prob = evres.model_prob
            rec.handle_pct = evres.best_quote.handle_pct
            rec.bets_pct = evres.best_quote.bets_pct
            if rec.handle_pct is None:
                sp = lookup_split(self._splits, *key)
                if sp is not None:
                    rec.handle_pct = sp.handle_pct
                    rec.bets_pct = sp.bets_pct
            thr = self.cfg.ev.for_market(market)
            tier, reasons = classify(evres, thr)
            screened = price_screen(evres, thr)
            gate = screened[0] if screened is not None else None
            if veto.triggered:
                # A gate vetoes outright; the xwOBA/sharp-money signals only
                # nudge the tier of a selection that survived the gates.
                tier = Tier.PASS
                gate = veto.gate
                reasons.append(veto.reason())
            elif rl_signal is not None and tier != Tier.PASS:
                steps, rl_reasons = runline_adjustment(team_side, line, rl_signal)
                if steps:
                    tier = bump_tier(tier, steps)
                reasons.extend(rl_reasons)
            if (
                market == "batter_hr"
                and tier != Tier.PASS
                and selector is not None
            ):
                keep, hr_reason = self._hr_gate.allows(
                    selector.hr_max_ev, selector.hr_barrel, selector.hr_bbe,
                    selector.hr_barrel_pa, selector.hr_fb_ld_ev,
                )
                if not keep:
                    tier = Tier.PASS
                    gate = "hr_barrel_gate"
                if hr_reason:
                    reasons.append(hr_reason)
            if market == "game_ml" and tier != Tier.PASS:
                keep, ml_reason = self._ml_gate.allows(
                    rec.handle_pct, rec.bets_pct
                )
                if not keep:
                    tier = Tier.PASS
                    gate = "ml_handle_gate"
                if ml_reason:
                    reasons.append(ml_reason)
            # Sharp money can promote a side the model passed on, but not one the
            # price cannot pay: the upgrade is evidence about the number, not a
            # licence to bet a negative expectation at it.
            if market == "game_ml" and tier == Tier.PASS and evres.ev > thr.min_ev:
                up, up_reason = self._ml_gate.upgrades(
                    rec.handle_pct, rec.bets_pct, evres.fair_prob
                )
                if up:
                    tier = Tier.MODERATE
                    gate = None
                if up_reason:
                    reasons.append(up_reason)
            # Availability gates run last so they also veto a sharp-money
            # upgrade: sharp money on a team whose high-leverage arms are gone,
            # or on a lineup that may not bat, is money bet on stale inputs.
            if market == "game_ml" and tier != Tier.PASS:
                keep, pen_reason = self._pen_gate.allows(
                    pen_fatigue, opp_pen_fatigue, pen_availability
                )
                if not keep:
                    tier = Tier.PASS
                    gate = "pen_availability"
                if pen_reason:
                    reasons.append(pen_reason)
                keep, lock_reason = self._lineup_gate.allows(self._lineup_lock)
                if not keep:
                    tier = Tier.PASS
                    gate = "lineup_lock"
                if lock_reason:
                    reasons.append(lock_reason)
            rec.tier = tier
            rec.reasons = reasons
            # A Pass with no named screen was demoted by a tier adjustment
            # (sharp money against it, or strong-only mode) rather than rejected
            # outright; naming it keeps every Pass row attributable.
            rec.pass_gate = (gate or "tier_downgrade") if tier == Tier.PASS else None
        else:
            rec.tier = Tier.PASS
            rec.pass_gate = veto.gate if veto.triggered else "unpriced"
            rec.reasons = [veto.reason()] if veto.triggered else ["no market price"]
            sp = lookup_split(self._splits, *key)
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
            rec.pass_gate = "contact_floor"
            rec.reasons = [gate_reason, *rec.reasons]
        # Price-only markets (e.g. singles) are fetched to persist the under
        # quote, never to bet the side we price. Hard-pass the over after every
        # tier decision so pricing the market cannot re-enable buying it.
        if market in PRICE_ONLY_MARKETS and rec.tier != Tier.PASS:
            rec.tier = Tier.PASS
            rec.pass_gate = "price_only"
            rec.reasons = ["price captured for audit only", *rec.reasons]
        return rec


def _prev_to_pg(prev):
    if prev is None:
        return None
    d, venue_id, _ = prev
    p = get_park(venue_id)
    if not p:
        return None
    return travel_rest.PrevGame(game_date=d, lat=p.lat, lon=p.lon)


def _fnum(x) -> float | None:
    """Coerce to a finite float, or None (keeps NaN out of the JSON/report)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


@dataclass
class PreviewHalf:
    """One offense's slate-preview story: its lineup + the pitching it faces."""

    lineup: LineupLine
    opp_starter: StarterLine
    opp_pen: BullpenLine


def _mean_woba(matchups: list[dict[str, float]]) -> float | None:
    """Lineup-average wOBA implied by the simulator's per-hitter matchup rates."""
    if not matchups:
        return None
    return round(sum(woba_from_rates(m) for m in matchups) / len(matchups), 3)


def _preview_half(
    team,
    opp,
    regs,
    pit_reg,
    bpen,
    trends: PitcherTrends,
    splits: TeamSplits | None,
    opp_throws: str | None,
    opp_siera: Siera,
    league: LeagueContact,
    proj_woba: float | None,
    proj_woba_vs_league: float | None,
    pen_proj_woba: float | None,
    pen_proj_woba_close: float | None,
) -> PreviewHalf:
    """Assemble the preview half from objects the simulator already computed."""
    assert opp.probable_pitcher is not None
    starter = StarterLine(
        name=opp.probable_pitcher.name,
        pitches=int(pit_reg.pitches),
        k_pct=_fnum(pit_reg.k_pct) or 0.0,
        xk_pct=_fnum(pit_reg.expected_k_pct()) or 0.0,
        bb_pct=_fnum(pit_reg.bb_pct) or 0.0,
        xbb_pct=_fnum(pit_reg.expected_bb_pct()) or 0.0,
        csw=_fnum(pit_reg.csw) or 0.0,
        whiff=_fnum(pit_reg.whiff) or 0.0,
        swstr=_fnum(pit_reg.swstr) or 0.0,
        zone_pct=_fnum(pit_reg.zone_pct) or 0.0,
        xwoba_allowed=_fnum(pit_reg.xwoba_allowed) or 0.0,
        barrel_allowed=_fnum(pit_reg.barrel_allowed) or 0.0,
        dxwoba=_fnum(pit_reg.dxwoba) or 0.0,
        spin=_fnum(pit_reg.spin),
        hard_hit_allowed=_fnum(pit_reg.hard_hit_allowed),
        babip_allowed=_fnum(pit_reg.babip_allowed),
        siera=opp_siera.siera if opp_siera.has_data else None,
        siera_trend=trends.siera.delta,
        stuff_trend=trends.stuff.delta,
        vfa_trend=trends.vfa.delta,
        league_xwoba_allowed=league.pitcher,
    )
    split = splits.vs_hand(opp_throws) if splits is not None else None
    overall = splits.overall if splits is not None else None
    venue = splits.at_venue(bool(team.is_home)) if splits is not None else None

    named = [
        (slot.player.name, r)
        for slot, r in zip(team.lineup, regs, strict=False)
        if r.bbe >= MIN_BBE
    ]

    def _mean(vals: list[float | None]) -> float:
        clean = [v for v in vals if v is not None]
        return sum(clean) / len(clean) if clean else 0.0

    woba = _mean([_fnum(r.woba) for _, r in named])
    xwoba = _mean([_fnum(r.xwoba) for _, r in named])
    xslg = _mean([_fnum(r.xslg) for _, r in named])
    barrel = _mean([_fnum(r.barrel_rate) for _, r in named])
    # regression: dxwoba = xwoba - woba. Negative => overperforming (hot, due to
    # cool off); positive => underperforming (cold, buy-low / due to heat up).
    by_gap = sorted(named, key=lambda nr: nr[1].dxwoba)
    hot = [
        RegFlag(name=n, points=round(-r.dxwoba * 1000, 1))
        for n, r in by_gap
        if r.dxwoba <= -0.030
    ][:3]
    cold = [
        RegFlag(name=n, points=round(r.dxwoba * 1000, 1))
        for n, r in reversed(by_gap)
        if r.dxwoba >= 0.030
    ][:3]
    lineup = LineupLine(
        n=len(named),
        woba=round(woba, 3),
        xwoba=round(xwoba, 3),
        dxwoba=round(xwoba - woba, 3),
        xslg=round(xslg, 3),
        barrel=round(barrel, 3),
        hot=hot,
        cold=cold,
        vs_hand=opp_throws,
        split_woba=None if split is None else split.woba,
        split_rank=None if split is None else split.rank,
        split_of=None if split is None else split.of,
        split_bucket=None if split is None else split.bucket,
        home_woba=None if splits is None else splits.home_woba,
        away_woba=None if splits is None else splits.away_woba,
        is_home=bool(team.is_home),
        team_woba=None if overall is None else overall.woba,
        team_rank=None if overall is None else overall.rank,
        team_of=None if overall is None else overall.of,
        venue_rank=None if venue is None else venue.rank,
        venue_of=None if venue is None else venue.of,
        league_xwoba=league.batter,
        proj_woba=proj_woba,
        proj_woba_vs_league=proj_woba_vs_league,
    )

    spread, arms = pen_arm_spread(bpen.relief)
    pen = BullpenLine(
        xwoba_allowed=_fnum(bpen.xwoba_allowed),
        k_pct=_fnum(bpen.k_pct),
        zone_pct=_fnum(bpen.zone_pct),
        recent_load=_fnum(bpen.recent_load),
        fatigue=None,  # filled in at game level from the StatsAPI proxy
        proj_woba=pen_proj_woba,
        proj_woba_close=pen_proj_woba_close,
        arm_spread=spread,
        arms=arms,
    )
    return PreviewHalf(lineup=lineup, opp_starter=starter, opp_pen=pen)
