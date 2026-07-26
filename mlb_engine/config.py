"""Central configuration for the MLB prediction engine.

Values are sourced from environment variables where relevant (credentials in
particular) so that nothing sensitive is committed. Everything else has sensible
defaults that can be overridden via environment variables prefixed ``MLBE_``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


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
    iso_gb: bool = field(default_factory=lambda: _env_bool("MLBE_RL_GATE_ISO_GB", False))
    iso_max: float = field(default_factory=lambda: _env_float("MLBE_RL_ISO_MAX", 0.140))
    gb_min: float = field(default_factory=lambda: _env_float("MLBE_RL_GB_MIN", 0.50))

    # Underdog +1.5: a starter putting runners on and giving up hard contact is
    # a blowout waiting to happen.
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

    # Directories.
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MLBE_DATA_DIR", str(Path.home() / ".mlb_engine")))
    )

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

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
