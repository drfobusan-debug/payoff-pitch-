"""Central configuration for the NBA engine.

Credentials come from environment variables (nothing sensitive is committed).
Everything else has a default overridable via an ``NBAE_``-prefixed variable.
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


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw not in ("0", "false", "False")


def preseason() -> bool:
    """``NBAE_PRESEASON=1`` points every command at the exhibition feed, kept apart."""
    return _env_bool("NBAE_PRESEASON", False)


def data_dir() -> Path:
    """Where the archive, history, cache and (later) the ledger live.

    Preseason games go under ``preseason/`` so nothing fitted on the regular
    season ever reads them.
    """
    raw = os.getenv("NBAE_DATA_DIR")
    root = Path(raw).expanduser() if raw else Path.home() / ".nba_engine"
    return root / "preseason" if preseason() else root


def cache_dir() -> Path:
    return data_dir() / "cache"


@dataclass(frozen=True)
class Credentials:
    """Credentials for data sources (never logged)."""

    odds_api_key: str | None = field(
        default_factory=lambda: os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
    )


@dataclass(frozen=True)
class CaptureParams:
    """How much of the board one live capture pass is allowed to spend.

    The featured board (ML / spread / total for every game) is one bulk call,
    3 credits. The per-event call for the first-half and five prop markets is
    about 10 credits per market returned on the historical endpoint and 1 per
    market live; one 2025-12-11 event returned all 11 keys (110 historical
    credits). ``max_events`` keeps a misconfigured schedule from draining the
    key in a day.
    """

    max_events: int = field(default_factory=lambda: _env_int("NBAE_CAPTURE_MAX_EVENTS", 16))
    horizon_hours: int = field(default_factory=lambda: _env_int("NBAE_CAPTURE_HORIZON_H", 30))
    # The close pass captures events tipping within this many minutes.
    close_window_min: int = field(default_factory=lambda: _env_int("NBAE_CLOSE_WINDOW_MIN", 10))


@dataclass(frozen=True)
class HistoryParams:
    """Guards on the historical pull, which spends from the key every engine shares.

    ``min_credits`` is the balance the pull will not go below: the live MLB,
    CFB, NFL, NHL and NBA captures keep running on whatever is left. A run also
    stops at its own ``--budget``.
    """

    min_credits: int = field(default_factory=lambda: _env_int("NBAE_HISTORY_MIN_CREDITS", 500_000))


@dataclass(frozen=True)
class AlarmParams:
    """What counts as news between two looks at the board.

    A consensus (median across books) move at least this large re-captures and
    marks the game for re-pricing; so does any availability change for a player
    on a team playing that day. These are starting values: the plan fits them on
    the 2023-26 archive by how often a move of that size followed a real
    availability change (docs/nba/master_plan.md §5.4).
    """

    ml_prob: float = field(default_factory=lambda: _env_float("NBAE_ALARM_ML_PROB", 0.03))
    spread_pts: float = field(default_factory=lambda: _env_float("NBAE_ALARM_SPREAD_PTS", 1.0))
    total_pts: float = field(default_factory=lambda: _env_float("NBAE_ALARM_TOTAL_PTS", 1.5))
    # Settle gate: an alerted game stays pending until two consecutive polls of
    # its consensus each move less than this (points for spread/total, no-vig
    # probability for ML), polled every ``settle_interval_s`` up to
    # ``settle_polls`` times per tick.
    settle_pts: float = field(default_factory=lambda: _env_float("NBAE_SETTLE_PTS", 0.5))
    settle_ml: float = field(default_factory=lambda: _env_float("NBAE_SETTLE_ML", 0.01))
    settle_interval_s: float = field(
        default_factory=lambda: _env_float("NBAE_SETTLE_INTERVAL_S", 60.0)
    )
    settle_polls: int = field(default_factory=lambda: int(_env_float("NBAE_SETTLE_POLLS", 4)))


@dataclass(frozen=True)
class Config:
    creds: Credentials = field(default_factory=Credentials)
    capture: CaptureParams = field(default_factory=CaptureParams)
    history: HistoryParams = field(default_factory=HistoryParams)
    alarm: AlarmParams = field(default_factory=AlarmParams)
    state_sync: bool = field(default_factory=lambda: _env_bool("NBAE_STATE_SYNC", True))
    preseason: bool = field(default_factory=preseason)

    @property
    def sport_key(self) -> str:
        return "basketball_nba_preseason" if self.preseason else "basketball_nba"


def load_config() -> Config:
    return Config()


__all__ = [
    "AlarmParams",
    "CaptureParams",
    "Config",
    "Credentials",
    "HistoryParams",
    "cache_dir",
    "data_dir",
    "load_config",
    "preseason",
]
