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
    # Home-field advantage in points, added to the home margin.
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
    wind_threshold_mph: float = field(default_factory=lambda: _env_float("CFBE_WIND_MPH", 12.0))
    wind_total_per_mph: float = field(
        default_factory=lambda: _env_float("CFBE_WIND_TOTAL_PER_MPH", 0.45)
    )
    wind_total_max: float = field(default_factory=lambda: _env_float("CFBE_WIND_TOTAL_MAX", 7.0))
    precip_total_pts: float = field(default_factory=lambda: _env_float("CFBE_PRECIP_TOTAL_PTS", 2.5))
    cold_threshold_f: float = field(default_factory=lambda: _env_float("CFBE_COLD_F", 32.0))
    cold_total_pts: float = field(default_factory=lambda: _env_float("CFBE_COLD_TOTAL_PTS", 1.5))

    # -- regression -------------------------------------------------------
    # Shrink the ratings-implied margin toward zero (mean reversion): early and
    # mid-season SP+ gaps overstate true separation, so scale the rating margin
    # by this factor before blending with the market. 1.0 disables it.
    regression_factor: float = field(
        default_factory=lambda: _env_float("CFBE_REGRESSION_FACTOR", 0.90)
    )


@dataclass(frozen=True)
class EVThresholds:
    """Expected-value cutoffs (EV per $1 staked) for buy tiers."""

    strong_buy: float = field(default_factory=lambda: _env_float("CFBE_EV_STRONG", 0.06))
    moderate_buy: float = field(default_factory=lambda: _env_float("CFBE_EV_MODERATE", 0.025))
    # Minimum model edge over the no-vig market price required to buy.
    min_edge: float = field(default_factory=lambda: _env_float("CFBE_MIN_EDGE", 0.02))
    strong_only: bool = field(default_factory=lambda: _env_bool("CFBE_STRONG_ONLY", False))

    def for_market(self, market: str) -> EVThresholds:
        """Per-market thresholds, overridable via ``CFBE_EV_STRONG_<MARKET>`` etc."""
        suffix = market.upper()
        return EVThresholds(
            strong_buy=_env_float(f"CFBE_EV_STRONG_{suffix}", self.strong_buy),
            moderate_buy=_env_float(f"CFBE_EV_MODERATE_{suffix}", self.moderate_buy),
            min_edge=_env_float(f"CFBE_MIN_EDGE_{suffix}", self.min_edge),
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
class Config:
    model: ModelParams = field(default_factory=ModelParams)
    ev: EVThresholds = field(default_factory=EVThresholds)
    features: FeatureParams = field(default_factory=FeatureParams)
    creds: Credentials = field(default_factory=Credentials)

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

    # Use the VSiN guide's per-team home-field-advantage table (overrides the flat
    # ``model.home_field_pts`` for listed home teams). Unlisted teams keep the default.
    vsin_hfa: bool = field(default_factory=lambda: _env_bool("CFBE_VSIN_HFA", True))

    # Use the VSiN guide's 0-19 roster-stability score to shrink a preseason
    # rating gap toward a pick'em when both teams have volatile rosters.
    vsin_stability: bool = field(default_factory=lambda: _env_bool("CFBE_VSIN_STABILITY", True))

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
