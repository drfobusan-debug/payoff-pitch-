"""Central configuration for the NHL engine.

Credentials come from environment variables (nothing sensitive is committed).
Everything else has a default overridable via an ``NHLE_``-prefixed variable.
The Odds API key is shared with the other engines so one ``engine.env`` serves
all of them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw not in ("0", "false", "False")


def data_dir() -> Path:
    """Where the archive, cache and (later) the ledger live."""
    raw = os.getenv("NHLE_DATA_DIR")
    return Path(raw).expanduser() if raw else Path.home() / ".nhl_engine"


def cache_dir() -> Path:
    return data_dir() / "cache"


def output_dir() -> Path:
    raw = os.getenv("NHLE_OUTPUT_DIR")
    return Path(raw).expanduser() if raw else data_dir() / "output"


@dataclass(frozen=True)
class Credentials:
    """Credentials for data sources (never logged)."""

    odds_api_key: str | None = field(
        default_factory=lambda: os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
    )

    def has_odds_api(self) -> bool:
        return bool(self.odds_api_key)


@dataclass(frozen=True)
class CaptureParams:
    """How much of the board one capture pass is allowed to spend.

    A featured-board pull is one bulk call (3 credits). Every period, team-total
    and prop market is a *per-event* call costing about one credit per market
    that returns data -- measured 25 credits for one event with all 31 keys on
    2026-09-28 -- so a full pass over a 15-game slate is ~400 credits. The cap
    keeps a misconfigured schedule from draining the key in a day.
    """

    max_events: int = field(default_factory=lambda: _env_int("NHLE_CAPTURE_MAX_EVENTS", 20))
    # Only archive events starting within this many hours; keeps the per-event
    # spend on today's games rather than the whole posted week.
    horizon_hours: int = field(default_factory=lambda: _env_int("NHLE_CAPTURE_HORIZON_H", 30))


@dataclass(frozen=True)
class Config:
    creds: Credentials = field(default_factory=Credentials)
    capture: CaptureParams = field(default_factory=CaptureParams)
    # Carry the archive on the shared engine-state branch so a second machine
    # (or a rebuilt one) has every price this one ever saw.
    state_sync: bool = field(default_factory=lambda: _env_bool("NHLE_STATE_SYNC", True))


def load_config() -> Config:
    return Config()
