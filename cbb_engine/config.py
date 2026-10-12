"""CBB configuration: ``CBBE_*`` env overrides, books, cadence, paths.

No thresholds live here that the runtime needs; the runtime is sport-blind.
The Odds API key and SMTP variables are the ones every engine already reads,
so one ``engine.env`` serves all six.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

SPORT = "cbb"
SPORT_KEY = "basketball_ncaab"
EXECUTION_BOOKS: tuple[str, ...] = ("draftkings", "betmgm")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw not in ("0", "false", "False")


def data_dir() -> Path:
    raw = os.getenv("CBBE_DATA_DIR")
    return Path(raw).expanduser() if raw else Path.home() / ".cbb_engine"


def runtime_root() -> Path:
    """Where the runtime keeps quote shards, pass logs and the ledger for this sport."""
    raw = os.getenv("ENGINE_RUNTIME_DIR")
    return Path(raw).expanduser() if raw else Path.home() / ".engine_runtime"


def dry_run() -> bool:
    """``CBBE_DRY_RUN=1``: fake slates through the pipeline, no ledger writes."""
    return _env_bool("CBBE_DRY_RUN", False)


@dataclass(frozen=True)
class Cadence:
    """§5 cadence table. Minutes; windows in hours before tip."""

    board_far: int = 60  # T-6h and beyond
    board_mid: int = 30  # T-6h -> T-3h
    board_near: int = 15  # inside T-3h, to tip
    featured: int = 10  # inside T-3h
    featured_window_h: float = 3.0
    mid_window_h: float = 6.0
    close_first_min: int = 60  # close passes T-60 -> T-5
    close_last_min: int = 5
    watch: int = 5  # ESPN/RotoWire only; no credits
    horizon_hours: int = 30


@dataclass(frozen=True)
class Credentials:
    odds_api_key: str | None = field(
        default_factory=lambda: os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
    )


@dataclass(frozen=True)
class Delivery:
    gmail_user: str | None = field(default_factory=lambda: os.getenv("GMAIL_USER") or os.getenv("EMAIL_ADDRESS"))
    gmail_app_password: str | None = field(default_factory=lambda: os.getenv("GMAIL_APP_PASSWORD"))
    email_to: str | None = field(default_factory=lambda: os.getenv("CBBE_EMAIL_TO") or os.getenv("MLBE_EMAIL_TO"))
    smtp_host: str = field(default_factory=lambda: os.getenv("CBBE_SMTP_HOST") or os.getenv("SMTP_HOST") or "smtp.gmail.com")
    smtp_port: int = field(default_factory=lambda: _env_int("CBBE_SMTP_PORT", _env_int("SMTP_PORT", 465)))


@dataclass(frozen=True)
class Exposure:
    max_live_buys: int = field(default_factory=lambda: _env_int("CBBE_MAX_LIVE_BUYS", 8))
    max_per_book: int = field(default_factory=lambda: _env_int("CBBE_MAX_PER_BOOK", 6))


@dataclass(frozen=True)
class Config:
    creds: Credentials = field(default_factory=Credentials)
    delivery: Delivery = field(default_factory=Delivery)
    exposure: Exposure = field(default_factory=Exposure)
    cadence: Cadence = field(default_factory=Cadence)
    regions: str = field(default_factory=lambda: os.getenv("CBBE_REGIONS", "us"))
    # Per-event markets are the dominant spend; off until the written budget allows them.
    event_markets_on: bool = field(default_factory=lambda: _env_bool("CBBE_EVENT_MARKETS", False))
    max_events: int = field(default_factory=lambda: _env_int("CBBE_MAX_EVENTS", 40))
    cache_ttl: int = field(default_factory=lambda: _env_int("CBBE_CACHE_TTL", 0))


def load_config() -> Config:
    return Config()


__all__ = [
    "EXECUTION_BOOKS",
    "SPORT",
    "SPORT_KEY",
    "Cadence",
    "Config",
    "Credentials",
    "data_dir",
    "dry_run",
    "load_config",
    "runtime_root",
]
