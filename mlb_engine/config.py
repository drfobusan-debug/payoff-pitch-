"""Central configuration for the MLB prediction engine.

Values are sourced from environment variables where relevant (credentials in
particular) so that nothing sensitive is committed. Everything else has sensible
defaults that can be overridden via environment variables prefixed ``MLBE_``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from mlb_engine.features.regression import SINGLES_BARREL_SLOPE, SINGLES_GB_SLOPE


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _env_csv(name: str) -> tuple[str, ...] | None:
    """Comma-separated override, or ``None`` to keep the caller's default."""
    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


@dataclass(frozen=True)
class RollingWindows:
    """Rolling look-back windows (in days) for stat aggregation."""

    pitcher_form_days: int = field(default_factory=lambda: _env_int("MLBE_PITCHER_FORM_DAYS", 28))
    batter_home_away_days: int = field(
        default_factory=lambda: _env_int("MLBE_BATTER_HOME_AWAY_DAYS", 21)
    )
    batter_vs_rhp_days: int = field(default_factory=lambda: _env_int("MLBE_BATTER_VS_RHP_DAYS", 21))
    batter_vs_lhp_days: int = field(default_factory=lambda: _env_int("MLBE_BATTER_VS_LHP_DAYS", 42))
    biomech_days: int = field(default_factory=lambda: _env_int("MLBE_BIOMECH_DAYS", 28))
    # Bullpen: relievers' last ~3 weeks and batters' late-inning last ~3 weeks.
    bullpen_days: int = field(default_factory=lambda: _env_int("MLBE_BULLPEN_DAYS", 21))
    bullpen_min_inning: int = field(default_factory=lambda: _env_int("MLBE_BULLPEN_MIN_INNING", 6))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw not in ("0", "false", "False")


@dataclass(frozen=True)
class EVThresholds:
    """Expected-value cutoffs (in EV per $1 staked) for buy tiers."""

    strong_buy: float = field(default_factory=lambda: _env_float("MLBE_EV_STRONG", 0.08))
    moderate_buy: float = field(default_factory=lambda: _env_float("MLBE_EV_MODERATE", 0.03))
    # Below moderate_buy -> "pass"
    # Minimum model edge over the no-vig market price required to buy (thin-edge
    # guard). Raise it to trade volume for realized PPV/NPV.
    min_edge: float = field(default_factory=lambda: _env_float("MLBE_MIN_EDGE", 0.02))
    # Strict selection: when set, downgrade every Moderate buy to Pass so only
    # Strong buys fire.
    strong_only: bool = field(default_factory=lambda: _env_bool("MLBE_STRONG_ONLY", False))

    def for_market(self, market: str) -> EVThresholds:
        """Per-market thresholds, overridable via ``MLBE_EV_STRONG_<MARKET>`` etc.

        Lets a market the audit flags as a false-positive pocket (e.g.
        ``pitcher_outs``) be tightened independently -- raising its buy bar lifts
        realized PPV -- without touching the rest of the sheet. Falls back to the
        global cutoffs when no override is set.
        """
        suffix = market.upper()
        return EVThresholds(
            strong_buy=_env_float(f"MLBE_EV_STRONG_{suffix}", self.strong_buy),
            moderate_buy=_env_float(f"MLBE_EV_MODERATE_{suffix}", self.moderate_buy),
            min_edge=_env_float(f"MLBE_MIN_EDGE_{suffix}", self.min_edge),
            strong_only=_env_bool(f"MLBE_STRONG_ONLY_{suffix}", self.strong_only),
        )


@dataclass(frozen=True)
class RunLineGates:
    """NPV gates that veto a run-line selection outright.

    Each gate removes selections whose *failure* to cover is highly predictable,
    lifting realized NPV at the cost of bet volume. They ship disabled so each
    can be A/B'd against the ledger one at a time (see ``runline_metrics``).
    """

    # Favorite -1.5: a low-power lineup facing a ground-ball starter has almost
    # no blowout path (needs multi-run homers it cannot hit).
    # On by default: over eight graded slates this gate removed six favorite
    # -1.5s that went 1-5, against a 59.6% baseline for the run lines it kept.
    # Thresholds are the engine's own tracked-batted-ball scale, which reads a
    # few points below the public leaderboards the .140/.50 figures come from.
    iso_gb: bool = field(default_factory=lambda: _env_bool("MLBE_RL_GATE_ISO_GB", True))
    iso_max: float = field(default_factory=lambda: _env_float("MLBE_RL_ISO_MAX", 0.170))
    gb_min: float = field(default_factory=lambda: _env_float("MLBE_RL_GB_MIN", 0.40))

    # Underdog +1.5: a starter putting runners on and giving up hard contact is
    # a blowout waiting to happen.
    # Both underdog gates stay off: over the same eight slates they removed 35
    # +1.5s that won at 66-75%, i.e. they deleted winners. The sim already
    # prices weak-starter and weak-bullpen matchups, so the gates double-count.
    dog_sp: bool = field(default_factory=lambda: _env_bool("MLBE_RL_GATE_DOG_SP", False))
    dog_sp_whip_max: float = field(default_factory=lambda: _env_float("MLBE_RL_DOG_WHIP_MAX", 1.45))
    dog_sp_hard_hit_max: float = field(
        default_factory=lambda: _env_float("MLBE_RL_DOG_HARD_HIT_MAX", 0.45)
    )

    # Underdog +1.5: a bullpen that cannot strand inherited runners hands the
    # favorite the late-innings cushion that breaks the spread.
    dog_pen: bool = field(default_factory=lambda: _env_bool("MLBE_RL_GATE_DOG_PEN", False))
    dog_pen_xwoba_max: float = field(
        default_factory=lambda: _env_float("MLBE_RL_DOG_PEN_XWOBA_MAX", 0.330)
    )
    dog_pen_k_min: float = field(default_factory=lambda: _env_float("MLBE_RL_DOG_PEN_K_MIN", 0.18))

    # Favorite -1.5: low-total games trend to 1-run margins. Redundant with the
    # simulated margin distribution, so it is off by default.
    low_total: bool = field(default_factory=lambda: _env_bool("MLBE_RL_GATE_TOTAL", False))
    low_total_max: float = field(default_factory=lambda: _env_float("MLBE_RL_TOTAL_MAX", 7.0))


@dataclass(frozen=True)
class Credentials:
    """Credentials for subscription data sources and SMTP (never logged)."""

    fangraphs_user: str | None = field(default_factory=lambda: os.getenv("FANGRAPHS_USER"))
    fangraphs_pass: str | None = field(default_factory=lambda: os.getenv("FANGRAPHS_PASS"))
    rotowire_user: str | None = field(default_factory=lambda: os.getenv("ROTOWIRE_USER"))
    rotowire_pass: str | None = field(default_factory=lambda: os.getenv("ROTOWIRE_PASS"))
    vsin_user: str | None = field(default_factory=lambda: os.getenv("VSIN_USER"))
    vsin_pass: str | None = field(default_factory=lambda: os.getenv("VSIN_PASS"))
    # The Odds API (https://the-odds-api.com) key: multi-book ML/run-line/total prices.
    # Prefer the vendor-canonical THE_ODDS_API_KEY; fall back to the legacy ODDS_API_KEY.
    odds_api_key: str | None = field(
        default_factory=lambda: os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
    )
    # Generic SMTP credentials used by nightly audit emails.
    smtp_server: str | None = field(default_factory=lambda: os.getenv("SMTP_SERVER"))
    smtp_port: int = field(default_factory=lambda: _env_int("SMTP_PORT", 587))
    smtp_user: str | None = field(default_factory=lambda: os.getenv("SMTP_USER"))
    smtp_pass: str | None = field(default_factory=lambda: os.getenv("SMTP_PASS"))
    # Gmail credentials used by daily-card emails.
    gmail_user: str | None = field(
        default_factory=lambda: os.getenv("GMAIL_USER") or os.getenv("EMAIL_ADDRESS")
    )
    gmail_app_password: str | None = field(
        default_factory=lambda: os.getenv("GMAIL_APP_PASSWORD")
    )

    def has_fangraphs(self) -> bool:
        return bool(self.fangraphs_user and self.fangraphs_pass)

    def has_rotowire(self) -> bool:
        return bool(self.rotowire_user and self.rotowire_pass)

    def has_vsin(self) -> bool:
        return bool(self.vsin_user and self.vsin_pass)

    def has_odds_api(self) -> bool:
        return bool(self.odds_api_key)

    def has_email(self) -> bool:
        return bool(self.gmail_app_password)


@dataclass(frozen=True)
class Config:
    windows: RollingWindows = field(default_factory=RollingWindows)
    ev: EVThresholds = field(default_factory=EVThresholds)
    runline_gates: RunLineGates = field(default_factory=RunLineGates)
    creds: Credentials = field(default_factory=Credentials)

    # RBI hard-rule threshold: on-base pct of preceding 3 batters over 3wk window.
    rbi_obp_threshold: float = field(default_factory=lambda: _env_float("MLBE_RBI_OBP", 0.345))

    # Default recipient for the nightly audit email.
    audit_email: str = field(
        default_factory=lambda: os.getenv("MLBE_AUDIT_EMAIL", "drfobusan@gmail.com")
    )
    # Daily-card email delivery. Env names mirror scripts/email_results.py
    # (MLB_EMAIL_TO / SMTP_HOST / SMTP_PORT), with MLBE_-prefixed overrides.
    email_to: str | None = field(
        default_factory=lambda: os.getenv("MLBE_EMAIL_TO") or os.getenv("MLB_EMAIL_TO")
    )
    smtp_host: str = field(
        default_factory=lambda: os.getenv("MLBE_SMTP_HOST")
        or os.getenv("SMTP_HOST")
        or "smtp.gmail.com"
    )
    smtp_port: int = field(
        default_factory=lambda: _env_int("MLBE_SMTP_PORT", _env_int("SMTP_PORT", 465))
    )

    # Monte Carlo simulation count per game.
    mc_sims: int = field(default_factory=lambda: _env_int("MLBE_MC_SIMS", 20000))

    # Apply the historical isotonic probability calibration before EV/tiers.
    calibrate: bool = field(
        default_factory=lambda: os.getenv("MLBE_CALIBRATE", "1") not in ("0", "false", "")
    )
    # Compress the over-confident tails after calibration (see ConfidenceShrink).
    shrink_tails: bool = field(default_factory=lambda: _env_bool("MLBE_SHRINK_TAILS", True))
    shrink_pivot: float = field(default_factory=lambda: _env_float("MLBE_SHRINK_PIVOT", 0.60))
    shrink_slope: float = field(default_factory=lambda: _env_float("MLBE_SHRINK_SLOPE", 0.55))

    # Post-simulation TB/RBI selector scaling. The selector's park/weather and
    # batted-ball terms are already applied inside the simulation, so scaling
    # the simulated count arrays again double-counts them; kept only as an
    # escape hatch for reproducing pre-fix runs.
    legacy_prop_post_mult: bool = field(
        default_factory=lambda: _env_bool("MLBE_LEGACY_PROP_POST_MULT", False)
    )

    # Barrel rate is a negative for singles: power hitters take the same number
    # of hits but convert them to extra bases. The slope prices the half of that
    # effect the simulated K rate does not already carry; ``power_split`` stops
    # the distribution-tail bonus lifting 1B by the same factor it lifts HR.
    singles_barrel: bool = field(default_factory=lambda: _env_bool("MLBE_SINGLES_BARREL", True))
    singles_barrel_slope: float = field(
        default_factory=lambda: _env_float("MLBE_SINGLES_BARREL_SLOPE", SINGLES_BARREL_SLOPE)
    )
    tail_power_split: bool = field(
        default_factory=lambda: _env_bool("MLBE_TAIL_POWER_SPLIT", True)
    )

    # Ground-ball rate is the batted-ball half of the same story and the largest
    # remaining contact term, but on eight slates it is not separable from zero.
    # Off by default: enabling it grades a counterfactual without moving picks.
    singles_gb: bool = field(default_factory=lambda: _env_bool("MLBE_SINGLES_GB", False))
    singles_gb_slope: float = field(
        default_factory=lambda: _env_float("MLBE_SINGLES_GB_SLOPE", SINGLES_GB_SLOPE)
    )

    # Odds API credit budget. The vendor bills markets x regions per request, so
    # a 16-game slate at every market it can name costs ~230 credits. Props are
    # restricted to the markets with a positive graded edge (see
    # data.oddsapi.DEFAULT_PROP_MARKETS); MLBE_ODDS_PROPS takes a comma list.
    odds_props: tuple[str, ...] | None = field(
        default_factory=lambda: _env_csv("MLBE_ODDS_PROPS")
    )
    odds_f5: bool = field(default_factory=lambda: _env_bool("MLBE_ODDS_F5", True))
    # Re-running the same slate inside the TTL costs nothing.
    odds_cache_ttl: int = field(default_factory=lambda: _env_int("MLBE_ODDS_CACHE_TTL", 1800))
    # Credits held in reserve so one runaway slate cannot drain the plan.
    odds_min_credits: int = field(default_factory=lambda: _env_int("MLBE_ODDS_MIN_CREDITS", 200))

    # Weight given to the devigged market price when forming the probability the
    # EV screen bets on (see market.ev.anchor_to_market). The model's own
    # probability is untouched, so PPV/NPV and the calibration refit still
    # measure the model. Because the screen is affine in the probability, a weight
    # w is equivalent to demanding edge >= threshold / (1 - w): it raises the toll
    # on disagreeing with the market rather than making the engine defer to it.
    # Default 0 (off). Nine retro-priced slates: ROI -5.4% at 0, -4.1% at 0.4,
    # -3.5% at 0.6 on a third as many bets, -12.9% at 0.8 -- every interval still
    # spans zero, so this shrinks a loss rather than earning a profit. Judge a
    # weight on closing line value, which resolves in far fewer bets than ROI.
    market_anchor: float = field(default_factory=lambda: _env_float("MLBE_MARKET_ANCHOR", 0.0))

    # Directories.
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MLBE_DATA_DIR", str(Path.home() / ".mlb_engine")))
    )

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def odds_cache_dir(self) -> Path:
        return self.cache_dir / "oddsapi"

    @property
    def calibration_file(self) -> Path:
        """Isotonic map to price with: a locally refit one wins if it exists.

        ``mlb-engine calibrate`` writes ``calibration_live.json`` into the data
        directory from the audit ledger, so an operator who has graded history
        prices off their own results instead of the packaged 2024 fit.
        """
        override = os.getenv("MLBE_CALIBRATION_FILE")
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
    def fangraphs_dir(self) -> Path:
        """Default drop-in folder for FanGraphs custom-report exports."""
        return self.data_dir / "fangraphs"

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.cache_dir,
            self.output_dir,
            self.audit_dir,
            self.fangraphs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    cfg = Config()
    cfg.ensure_dirs()
    return cfg
