"""Central configuration for the college-football prediction engine.

Credentials come from environment variables (nothing sensitive is committed).
Everything else has sensible defaults overridable via ``CFBE_``-prefixed
environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date as Date
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw not in ("0", "false", "False")


def _env_csv(name: str) -> tuple[str, ...] | None:
    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


@dataclass(frozen=True)
class ModelParams:
    """Score-simulation parameters.

    A college-football final margin has a large irreducible spread: the
    standard deviation of ``actual margin - closing spread`` is ~15-16 points,
    far wider than the NFL. The total's residual is a touch tighter. Both are
    the empirical dispersion the Monte Carlo draws around the ratings-implied
    means, and both are overridable so the audit can retune them against the
    graded ledger.
    """

    # Points a rating gap translates into on the scoreboard. Ratings are already
    # expressed in points, so this is 1.0 unless the rating scale needs shrinking.
    rating_to_points: float = field(
        default_factory=lambda: _env_float("CFBE_RATING_TO_POINTS", 1.0)
    )
    # Home-field advantage in points, added to the home margin. Measured on 7,345
    # home-site games (2014-2025) by regressing on the SP+ gap, which carries no
    # venue: home field delivered +2.33 +/- 0.16 pts, so 2.4 stands. The market
    # prices +2.84 -- it charged 1.29 pts too much in 2014-2016 and +0.07 since
    # 2022 -- and neutral sites came in at +0.13 in the price, hence a 0.0 default
    # for FeatureParams.neutral_site_hfa.
    home_field_pts: float = field(default_factory=lambda: _env_float("CFBE_HFA_PTS", 2.4))
    # League-average points scored per team per game (sets the total baseline).
    avg_team_points: float = field(default_factory=lambda: _env_float("CFBE_AVG_TEAM_PTS", 27.5))
    # Standard deviation of the game margin around its mean.
    margin_sd: float = field(default_factory=lambda: _env_float("CFBE_MARGIN_SD", 16.0))
    # Standard deviation of the total around its mean.
    total_sd: float = field(default_factory=lambda: _env_float("CFBE_TOTAL_SD", 13.0))
    # Correlation between margin and total draws (mild: blowouts trend slightly
    # higher-scoring). Kept small; the two are close to independent.
    margin_total_corr: float = field(
        default_factory=lambda: _env_float("CFBE_MARGIN_TOTAL_CORR", 0.06)
    )
    # Monte Carlo draws per game.
    n_sims: int = field(default_factory=lambda: _env_int("CFBE_MC_SIMS", 40000))
    # Share of the ratings-implied margin/total pulled toward the market's
    # implied margin/total (0 = pure ratings, 1 = pure market). Blends the two
    # priors so a thin/absent rating never manufactures a phantom edge; the
    # model still departs from the market by ``1 - blend`` of its own view.
    market_blend: float = field(default_factory=lambda: _env_float("CFBE_MARKET_BLEND", 0.35))


@dataclass(frozen=True)
class FeatureParams:
    """Situational adjustments layered onto the ratings-implied means before the
    Monte Carlo, echoing the classic handicapping angles (VSiN-style): home
    field, rest/fatigue, travel, weather, and mean-reversion regression.

    Each is a point delta (margin from the home team's perspective, or total),
    is individually toggle-able, and is a no-op whenever its input data is
    missing -- so the engine degrades gracefully to a pure ratings+market model.
    """

    enabled: bool = field(default_factory=lambda: _env_bool("CFBE_FEATURES", True))

    # -- home field -------------------------------------------------------
    # HFA to apply at a neutral site (bowls, kickoff games). The model's base
    # HFA (ModelParams.home_field_pts) is removed and this substituted.
    neutral_site_hfa: float = field(
        default_factory=lambda: _env_float("CFBE_NEUTRAL_HFA", 0.0)
    )

    # -- rest / fatigue ---------------------------------------------------
    # Points per net rest-day edge (home rest days - away rest days), capped.
    rest_pts_per_day: float = field(default_factory=lambda: _env_float("CFBE_REST_PTS_DAY", 0.18))
    rest_max_pts: float = field(default_factory=lambda: _env_float("CFBE_REST_MAX_PTS", 3.0))
    # A full bye (>= this many rest days) adds a fixed prep bonus for that team.
    bye_days: int = field(default_factory=lambda: _env_int("CFBE_BYE_DAYS", 13))
    bye_bonus_pts: float = field(default_factory=lambda: _env_float("CFBE_BYE_BONUS_PTS", 1.0))

    # -- travel -----------------------------------------------------------
    # Away-team penalty per 1000 miles travelled to the venue, ignored under
    # ``travel_min_miles`` and capped at ``travel_max_pts``.
    travel_pts_per_1000mi: float = field(
        default_factory=lambda: _env_float("CFBE_TRAVEL_PTS_1000MI", 0.7)
    )
    travel_min_miles: float = field(default_factory=lambda: _env_float("CFBE_TRAVEL_MIN_MI", 300.0))
    travel_max_pts: float = field(default_factory=lambda: _env_float("CFBE_TRAVEL_MAX_PTS", 2.5))

    # -- weather (totals) -------------------------------------------------
    # Wind above the threshold knocks points off the total, per mph, capped.
    #
    # These default to zero, i.e. observed and printed but not priced, because
    # the closing total already contains them. Measured on 7,651 outdoor games
    # (2014-2025) against the *closing total's* residual, not the raw score:
    #
    #   wind                     r=-0.0125 (t=-1.09), -0.058 pts per mph
    #   wind >= 15mph as a flag  r=-0.0077 (t=-0.68)
    #   wind x pass rate         r=-0.0021 (t=-0.19)   the named mechanism
    #   temperature              r=+0.0152 (t=+1.33)
    #   R2 over the closing total: +0.00018 for wind, +0.00000 for the interaction
    #
    # Precipitation is the one term with a pulse (r=-0.0233, t=-2.04; unders hit
    # 55.6% in 0.5-1.5mm/hr and 57.6% above that, and 59.4% when rain meets 8mph+
    # wind) -- but it dies out of time: 56.6% unders in 2014-2019, 47.8% in
    # 2023-2025. Same shape as returning production, so it is not scored either.
    # Set the env vars to price any of them; the old defaults were 0.45/mph over
    # 12mph capped at 7, which is ~8x the measured slope in the same direction.
    wind_threshold_mph: float = field(default_factory=lambda: _env_float("CFBE_WIND_MPH", 12.0))
    wind_total_per_mph: float = field(
        default_factory=lambda: _env_float("CFBE_WIND_TOTAL_PER_MPH", 0.0)
    )
    wind_total_max: float = field(default_factory=lambda: _env_float("CFBE_WIND_TOTAL_MAX", 7.0))
    precip_total_pts: float = field(default_factory=lambda: _env_float("CFBE_PRECIP_TOTAL_PTS", 0.0))
    cold_threshold_f: float = field(default_factory=lambda: _env_float("CFBE_COLD_F", 32.0))
    cold_total_pts: float = field(default_factory=lambda: _env_float("CFBE_COLD_TOTAL_PTS", 0.0))

    # -- regression -------------------------------------------------------
    # Shrink the ratings-implied margin toward zero (mean reversion): early and
    # mid-season SP+ gaps overstate true separation, so scale the rating margin
    # by this factor before blending with the market. 1.0 disables it.
    regression_factor: float = field(
        default_factory=lambda: _env_float("CFBE_REGRESSION_FACTOR", 0.90)
    )


@dataclass(frozen=True)
class EVThresholds:
    """Cutoffs for buy tiers: an EV floor to clear, then edge to rank on."""

    # The price has to pay at all at the best number we can bet. Deliberately 0
    # rather than a margin: ``EV = decimal_odds x edge``, so an EV *margin* is a
    # cheaper bar the longer the price. The old 0.06 Strong cutoff asked a -400
    # favourite for 4.8 points of edge and a +300 dog for 1.5, which is how the
    # MLB engine's Strong tier filled with plus-money dogs and inverted (39.9%
    # against Moderate's 46.9%). The bar is sized in edge below, not here.
    min_ev: float = field(default_factory=lambda: _env_float("CFBE_MIN_EV", 0.0))
    # Minimum model edge over the no-vig market price required to buy.
    min_edge: float = field(default_factory=lambda: _env_float("CFBE_MIN_EDGE", 0.02))
    # Extra edge, in probability points over ``min_edge``, that promotes a buy to
    # Strong -- price-independent, unlike an EV cutoff.
    strong_edge_gap: float = field(
        default_factory=lambda: _env_float("CFBE_EDGE_STRONG_GAP", 0.02)
    )
    # Disagreement with the devigged market beyond which the edge reads as model
    # error rather than a bet. The market is the better forecaster here by a wide
    # margin -- the closing spread carries r=+.647 against the final margin while
    # the engine's efficiency gap adds nothing to it -- so a wide departure is
    # evidence against the sim. 1.0 disables the cap.
    max_edge: float = field(default_factory=lambda: _env_float("CFBE_MAX_EDGE", 0.08))
    strong_only: bool = field(default_factory=lambda: _env_bool("CFBE_STRONG_ONLY", False))

    def for_market(self, market: str) -> EVThresholds:
        """Per-market thresholds, overridable via ``CFBE_MIN_EDGE_<MARKET>`` etc."""
        suffix = market.upper()
        return EVThresholds(
            min_ev=_env_float(f"CFBE_MIN_EV_{suffix}", self.min_ev),
            min_edge=_env_float(f"CFBE_MIN_EDGE_{suffix}", self.min_edge),
            strong_edge_gap=_env_float(f"CFBE_EDGE_STRONG_GAP_{suffix}", self.strong_edge_gap),
            max_edge=_env_float(f"CFBE_MAX_EDGE_{suffix}", self.max_edge),
            strong_only=_env_bool(f"CFBE_STRONG_ONLY_{suffix}", self.strong_only),
        )


@dataclass(frozen=True)
class Credentials:
    """Credentials for data sources and SMTP (never logged)."""

    # CollegeFootballData.com (https://collegefootballdata.com) API key.
    cfbd_api_key: str | None = field(default_factory=lambda: os.getenv("CFBD_API_KEY"))
    # The Odds API (https://the-odds-api.com) key: multi-book prices. Prefer the
    # vendor-canonical THE_ODDS_API_KEY; fall back to the legacy ODDS_API_KEY.
    odds_api_key: str | None = field(
        default_factory=lambda: os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
    )
    # Gmail credentials used by daily-card / audit emails.
    gmail_user: str | None = field(
        default_factory=lambda: os.getenv("GMAIL_USER") or os.getenv("EMAIL_ADDRESS")
    )
    gmail_app_password: str | None = field(
        default_factory=lambda: os.getenv("GMAIL_APP_PASSWORD")
    )
    # Generic SMTP relay (optional alternative to Gmail).
    smtp_server: str | None = field(default_factory=lambda: os.getenv("SMTP_SERVER"))
    smtp_user: str | None = field(default_factory=lambda: os.getenv("SMTP_USER"))
    smtp_pass: str | None = field(default_factory=lambda: os.getenv("SMTP_PASS"))

    def has_cfbd(self) -> bool:
        return bool(self.cfbd_api_key)

    def has_odds_api(self) -> bool:
        return bool(self.odds_api_key)

    def has_email(self) -> bool:
        return bool(self.gmail_app_password)


@dataclass(frozen=True)
class MarkingParams:
    """Metric-driven 'marking' layer: PPV confidence bumps + NPV veto gates.

    Fed by CFBD advanced stats, this layer only re-tiers an already-priced bet:
    confidence bumps nudge a selection up/down when the efficiency metrics that
    carry documented PPV (net PPA, success rate, havoc, finishing drives) agree
    or disagree with it, and NPV gates drop a bet outright when the matchup is
    structurally hostile to it (a turnover-luck regression candidate, a
    low/high-scoring totals environment, an efficiency blowout laying points).
    Every threshold degrades to a no-op when its input stat is missing.
    """

    enabled: bool = field(default_factory=lambda: _env_bool("CFBE_MARKING", True))
    # Whether the support score may actually move a tier. Default off: measured
    # against what the closing spread missed over 8,009 games (2014-2025), every
    # metric the score is built from is indistinguishable from noise -- net PPA
    # r=+.0035 (t=+0.32), success rate -.0056, explosiveness -.0047, havoc
    # +.0078, points-per-opportunity +.0051, and the gap adds +0.00000 R2 on top
    # of the spread. The published cover rates the weights came from are real but
    # descriptive; they are not value over the price. The score is still computed
    # and printed as a reason so it can be graded before it is trusted.
    confidence_bumps: bool = field(default_factory=lambda: _env_bool("CFBE_MARK_BUMPS", False))
    # Weighted support score (sum of PPV-weighted agreeing metrics minus
    # disagreeing) needed to move a bet one tier up / down.
    bump_up: float = field(default_factory=lambda: _env_float("CFBE_MARK_BUMP_UP", 0.20))
    bump_down: float = field(default_factory=lambda: _env_float("CFBE_MARK_BUMP_DOWN", 0.20))
    # Deadband (in each metric's own units) below which a metric is "neutral".
    ppa_deadband: float = field(default_factory=lambda: _env_float("CFBE_MARK_PPA_DEAD", 0.03))
    # NPV gates (each individually switchable).
    veto_turnover: bool = field(default_factory=lambda: _env_bool("CFBE_VETO_TURNOVER", True))
    veto_totals_env: bool = field(default_factory=lambda: _env_bool("CFBE_VETO_TOTALS_ENV", True))
    veto_ats_blowout: bool = field(default_factory=lambda: _env_bool("CFBE_VETO_ATS_BLOWOUT", True))
    # A per-game turnover margin at/above which a team is a regression (fade)
    # candidate (~ +9 turnovers over a 12-game season).
    turnover_extreme: float = field(default_factory=lambda: _env_float("CFBE_TO_EXTREME", 0.75))
    # Net-PPA gap (home minus away, per play) at which the losing side is too
    # outclassed to trust its cover when it is also laying points.
    ppa_blowout: float = field(default_factory=lambda: _env_float("CFBE_PPA_BLOWOUT", 0.25))


@dataclass(frozen=True)
class Config:
    model: ModelParams = field(default_factory=ModelParams)
    ev: EVThresholds = field(default_factory=EVThresholds)
    features: FeatureParams = field(default_factory=FeatureParams)
    marking: MarkingParams = field(default_factory=MarkingParams)
    creds: Credentials = field(default_factory=Credentials)

    # Score simulator: "normal" (correlated-bivariate-normal Monte Carlo) or
    # "markov" (drive-based possession model). Markov re-shapes the score
    # distribution around the *same* ratings-implied means using pace and
    # per-drive scoring structure; the two share a means so a backtest can A/B
    # the distribution shape cleanly. Falls back to normal when a game lacks
    # advanced stats for both teams.
    sim_engine: str = field(default_factory=lambda: os.getenv("CFBE_SIM_ENGINE", "normal"))

    # Season year for CFBD ratings lookups. Defaults to the current year; the
    # pipeline overrides it from the slate date so an early-January bowl slate
    # still reads the finishing season's ratings.
    season: int | None = field(default_factory=lambda: _env_int("CFBE_SEASON", 0) or None)

    # Apply the historical isotonic probability calibration before EV/tiers.
    calibrate: bool = field(
        default_factory=lambda: os.getenv("CFBE_CALIBRATE", "1") not in ("0", "false", "")
    )
    # Compress the over-confident tails after calibration.
    shrink_tails: bool = field(default_factory=lambda: _env_bool("CFBE_SHRINK_TAILS", True))
    shrink_pivot: float = field(default_factory=lambda: _env_float("CFBE_SHRINK_PIVOT", 0.62))
    shrink_slope: float = field(default_factory=lambda: _env_float("CFBE_SHRINK_SLOPE", 0.55))

    # Ensemble of public power models (Sagarin/FPI/FEI + TSI/CFB-Graphs drop-ins)
    # blended into the CFBD SP+ net rating. ``ensemble_blend`` is the pull toward
    # the consensus (0 = SP+ only); ``ensemble_target_sd`` fixes the common points
    # spread the standardized models are rescaled to (0 = auto from SP+).
    ensemble: bool = field(default_factory=lambda: _env_bool("CFBE_ENSEMBLE", True))
    ensemble_blend: float = field(
        default_factory=lambda: _env_float("CFBE_ENSEMBLE_BLEND", 0.35)
    )
    ensemble_target_sd: float = field(default_factory=lambda: _env_float("CFBE_ENSEMBLE_SD", 0.0))

    # Opponent-adjusted per-play efficiency (CFBD game PPA, ridge-fit on games
    # played strictly before the slate week -- see data/efficiency.py).
    #
    # ``efficiency`` only controls whether the book is built; with it on and the
    # blend at 0 the ratings are unchanged and efficiency serves purely as the
    # *fallback* when SP+ is unavailable, which beats echoing the market back at
    # itself (standalone r 0.56 / MAE 13.1 on held-out margins, 2014-2025).
    #
    # ``efficiency_blend`` pulls the SP+ net rating toward efficiency. It defaults
    # to 0 because after the closing spread is in the model efficiency adds
    # nothing: partial r -0.001 (season-clustered 95% CI [-0.022, +0.020]),
    # held-out MAE 12.218 -> 12.220, 50.1% ATS on the disagreements (-4.4% ROI).
    efficiency: bool = field(default_factory=lambda: _env_bool("CFBE_EFFICIENCY", True))
    efficiency_blend: float = field(
        default_factory=lambda: _env_float("CFBE_EFFICIENCY_BLEND", 0.0)
    )

    # Returning-production experience edge, in points of margin per unit of gap
    # (see data/returning.py). The only candidate with residual signal after the
    # closing spread -- partial r +0.039, CI [+0.020, +0.058], fitted at +2.5
    # pts/unit -- but betting it goes 51.96% ATS against a 52.38% break-even, so
    # it ships off. Set CFBE_RETURNING_PTS=2.5 to enable.
    returning_pts: float = field(default_factory=lambda: _env_float("CFBE_RETURNING_PTS", 0.0))
    returning_max_pts: float = field(
        default_factory=lambda: _env_float("CFBE_RETURNING_MAX_PTS", 3.0)
    )

    # Roster continuity -- production kept *plus* production bought in the portal
    # (see data/roster.py) -- on the ratings-only margin, the fallback used when a
    # game has no consensus spread. Fitted at +6.5 pts per unit of gap, walk-forward
    # RMSE 18.231 -> 17.901 in all four held-out seasons, with the 8-point cap chosen
    # by sweep. It never contests a market number: against the closing spread the
    # same term goes 51.11% ATS, so it is deliberately confined to the fallback.
    roster_pts: float = field(default_factory=lambda: _env_float("CFBE_ROSTER_PTS", 6.5))
    roster_max_pts: float = field(
        default_factory=lambda: _env_float("CFBE_ROSTER_MAX_PTS", 8.0)
    )

    # Use the VSiN guide's per-team home-field-advantage table (overrides the flat
    # ``model.home_field_pts`` for listed home teams). Off by default: the guide
    # buckets teams by their own three-year home ATS record, so it grades 64% /
    # 30% inside that window and r=+0.042 outside it, and a program's home edge
    # over the market does not persist year to year at all (r=+0.017, p=.68).
    # :mod:`cfb_engine.data.vsin` records the measurement. The table is still
    # printed on the card; set the env var to price it again.
    vsin_hfa: bool = field(default_factory=lambda: _env_bool("CFBE_VSIN_HFA", False))

    # Read the injury feed and the box-score usage book. On by default because it
    # only reports and logs: an absence is printed on the card and appended to the
    # availability log with the line at that moment, which is what measures whether
    # we hear the news before the market moves.
    injury_feed: bool = field(default_factory=lambda: _env_bool("CFBE_INJURY_FEED", True))

    # Points to charge a team missing an established starting quarterback. Measured
    # at -2.2 against the closing spread (604 team-games, 55.3% fading, t=+2.60,
    # holdout 56.7%) -- but the whole effect sits in the *first* game of an absence
    # (holdout 59.9% against 47.1% once it is common knowledge), so it is a bet on
    # hearing the news early, not on knowing the backup is playing. Default 0.0
    # until the availability log shows we get there before the number does;
    # :mod:`cfb_engine.data.injuries` records the measurement.
    injury_qb_pts: float = field(default_factory=lambda: _env_float("CFBE_INJURY_QB_PTS", 0.0))

    # Use the VSiN guide's 0-19 roster-stability score to shrink a rating gap
    # toward a pick'em when both teams have volatile rosters. Measured off by
    # default: continuity does move how much of a gap the data supports, but by
    # ~0.05 of it (low-continuity b 0.916 vs high 0.968 over 6,818 games), against
    # a haircut up to 0.25 wide, and walk-forward every dose of it is worse than
    # none (rating MAE 12.549 flat -> 12.654 at the shipped floor). The effect is
    # also smallest in September, which is the window the term was written for.
    # ``scripts/cfb/stability_study.py`` reproduces it.
    vsin_stability: bool = field(default_factory=lambda: _env_bool("CFBE_VSIN_STABILITY", False))

    # Weight given to the devigged market price when forming the probability the
    # EV screen bets on (see market.ev.anchor_to_market). Default 0 (off).
    market_anchor: float = field(default_factory=lambda: _env_float("CFBE_MARKET_ANCHOR", 0.0))

    # Odds API budget/behaviour.
    odds_regions: str = field(default_factory=lambda: os.getenv("CFBE_ODDS_REGIONS", "us"))
    odds_cache_ttl: int = field(default_factory=lambda: _env_int("CFBE_ODDS_CACHE_TTL", 1800))

    # Default recipient for the daily card / nightly audit email.
    email_to: str | None = field(
        default_factory=lambda: os.getenv("CFBE_EMAIL_TO") or os.getenv("CFB_EMAIL_TO")
    )
    audit_email: str = field(
        default_factory=lambda: os.getenv("CFBE_AUDIT_EMAIL", "drfobusan@gmail.com")
    )
    smtp_host: str = field(
        default_factory=lambda: os.getenv("CFBE_SMTP_HOST")
        or os.getenv("SMTP_HOST")
        or "smtp.gmail.com"
    )
    smtp_port: int = field(
        default_factory=lambda: _env_int("CFBE_SMTP_PORT", _env_int("SMTP_PORT", 465))
    )

    # Directories.
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("CFBE_DATA_DIR", str(Path.home() / ".cfb_engine")))
    )

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def odds_cache_dir(self) -> Path:
        return self.cache_dir / "oddsapi"

    @property
    def calibration_file(self) -> Path:
        override = os.getenv("CFBE_CALIBRATION_FILE")
        if override:
            return Path(override)
        return self.data_dir / "calibration_live.json"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"

    @property
    def audit_dir(self) -> Path:
        return self.data_dir / "audit"

    @property
    def pff_dir(self) -> Path:
        """Drop-in folder for PFF team-grade CSV exports."""
        return self.data_dir / "pff"

    @property
    def models_dir(self) -> Path:
        """Drop-in folder for external model exports (TSI, CFB Graphs, ...)."""
        return self.data_dir / "models"

    @property
    def ratings_file(self) -> Path:
        """Optional local team-ratings CSV override (team,off,def[,pace])."""
        override = os.getenv("CFBE_RATINGS_FILE")
        if override:
            return Path(override)
        return self.data_dir / "ratings.csv"

    @property
    def ledger_file(self) -> Path:
        """Persistent per-bet audit ledger (CSV), appended across all slates."""
        override = os.getenv("CFBE_LEDGER_FILE")
        if override:
            return Path(override)
        return self.audit_dir / "ledger.csv"

    def predictions_file(self, day: Date) -> Path:
        """Persisted recommendations for one slate (grading input)."""
        return self.audit_dir / f"predictions_{day.isoformat()}.json"

    def closing_file(self, day: Date) -> Path:
        """Closing-line snapshot captured near kickoff for one slate."""
        return self.audit_dir / f"closing_{day.isoformat()}.json"

    @property
    def availability_file(self) -> Path:
        """Append-only log of absences, each stamped with the line at capture."""
        override = os.getenv("CFBE_AVAILABILITY_FILE")
        if override:
            return Path(override)
        return self.audit_dir / "availability.jsonl"

    @property
    def scorecard_file(self) -> Path:
        """Rolling PPV/NPV-by-tier-and-market scorecard (CSV), appended nightly."""
        return self.audit_dir / "scorecard.csv"

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.cache_dir,
            self.output_dir,
            self.audit_dir,
            self.pff_dir,
            self.models_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    cfg = Config()
    cfg.ensure_dirs()
    return cfg
