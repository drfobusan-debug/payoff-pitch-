"""Central configuration for the NFL prediction engine.

Credentials come from environment variables (nothing sensitive is committed).
Everything else has a default overridable via an ``NFLE_``-prefixed variable.
Defaults that are measurements rather than choices carry their provenance in the
comment, so a future refit can see what a change would be arguing with.
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw not in ("0", "false", "False")


def data_dir() -> Path:
    """Where cached source data and the ledger live."""
    raw = os.getenv("NFLE_DATA_DIR")
    root = Path(raw).expanduser() if raw else Path.home() / ".nfl_engine"
    return root


def cache_dir() -> Path:
    return data_dir() / "cache"


def output_dir() -> Path:
    """Where the reader-facing package is written (card, workbook, PDF)."""
    raw = os.getenv("NFLE_OUTPUT_DIR")
    return Path(raw).expanduser() if raw else data_dir() / "output"


@dataclass(frozen=True)
class ModelParams:
    """Score-simulation parameters.

    The NFL's scoreboard is coarse: points arrive in 3s and 7s, and the final
    margin piles up on those sums. Over 2015-2025 (n=3,028) 14.8% of games ended
    with a 3-point margin and 8.7% with 7, against 2.7% and 2.6% for a normal
    with the same mean and sd. So the primary simulator is possession-based and
    discrete (:mod:`nfl_engine.models.drives`); the normal is kept only as a
    control that shares the same means, so a backtest can attribute a result to
    the distribution's *shape* rather than to the forecast.
    """

    # Monte Carlo draws per game.
    n_sims: int = field(default_factory=lambda: _env_int("NFLE_SIMS", 40000))
    # League-average points per team per game, the baseline the ratings move.
    # 2015-2025: home 23.78, away 21.88.
    avg_team_points: float = field(default_factory=lambda: _env_float("NFLE_AVG_TEAM_PTS", 22.8))
    # Home-field advantage in points. Deliberately *not* a constant in the
    # pipeline: measured means were 2.75 (1999-2007), 2.44 (2008-2015), 1.94
    # (2016-2019), 1.08 (2020-2021, no crowds) and 2.33 (2022-2025). This is the
    # fallback for when the rolling estimate has no history to fit on.
    home_field_pts: float = field(default_factory=lambda: _env_float("NFLE_HFA_PTS", 1.9))
    # Normal-control dispersion. 2015-2025 margin sd 14.16, total sd 13.88, and
    # the residual sd of margin minus closing spread is 13.2.
    margin_sd: float = field(default_factory=lambda: _env_float("NFLE_MARGIN_SD", 13.2))
    total_sd: float = field(default_factory=lambda: _env_float("NFLE_TOTAL_SD", 13.4))
    # Correlation of margin and total draws. Measured corr(home score, away
    # score) is -0.02, i.e. the two teams' scores are close to independent.
    margin_total_corr: float = field(default_factory=lambda: _env_float("NFLE_MT_CORR", 0.05))
    # Pull of the ratings-implied margin/total toward the market's implied pair
    # (0 = pure ratings, 1 = bet the market). Higher than the CFB engine's 0.35
    # because the NFL market is the harder opponent: the closing spread's
    # residual sd is 13.2 points and it correlates only r=0.43 with the final
    # margin, so most of the variance is irreducible and a rating that departs
    # far from the line is more likely wrong than early.
    market_blend: float = field(default_factory=lambda: _env_float("NFLE_MARKET_BLEND", 0.55))


@dataclass(frozen=True)
class Credentials:
    """Credentials for data sources and SMTP (never logged)."""

    # The Odds API (https://the-odds-api.com): multi-book prices.
    odds_api_key: str | None = field(
        default_factory=lambda: os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
    )
    gmail_user: str | None = field(
        default_factory=lambda: os.getenv("GMAIL_USER") or os.getenv("EMAIL_ADDRESS")
    )
    gmail_app_password: str | None = field(
        default_factory=lambda: os.getenv("GMAIL_APP_PASSWORD")
    )

    def has_odds_api(self) -> bool:
        return bool(self.odds_api_key)

    def has_email(self) -> bool:
        return bool(self.gmail_app_password)


@dataclass(frozen=True)
class Delivery:
    """Where the weekly package goes. Shares the MLB engine's SMTP variables so
    one ``engine.env`` serves both, with ``NFLE_``-prefixed overrides.
    """

    email_to: str | None = field(
        default_factory=lambda: os.getenv("NFLE_EMAIL_TO") or os.getenv("MLBE_EMAIL_TO")
    )
    smtp_host: str = field(
        default_factory=lambda: os.getenv("NFLE_SMTP_HOST")
        or os.getenv("SMTP_HOST")
        or "smtp.gmail.com"
    )
    smtp_port: int = field(
        default_factory=lambda: _env_int("NFLE_SMTP_PORT", _env_int("SMTP_PORT", 465))
    )


@dataclass(frozen=True)
class Config:
    model: ModelParams = field(default_factory=ModelParams)
    creds: Credentials = field(default_factory=Credentials)
    delivery: Delivery = field(default_factory=Delivery)

    # Score simulator: "drives" (possession-based, discrete) or "normal"
    # (bivariate-normal control). Both consume the same means.
    sim_engine: str = field(default_factory=lambda: os.getenv("NFLE_SIM_ENGINE", "drives"))


def load_config() -> Config:
    return Config()
