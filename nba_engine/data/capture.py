"""Durable capture of every NBA quote the engine sees.

The archive format, snapshot dedupe and readers are the NHL engine's
(``nhl_engine.data.capture``): one side per row, content-addressed snapshot
files under ``<data>/prices/<YYYY-MM-DD>/``. Only the market map is NBA's.
"""

from __future__ import annotations

from nhl_engine.data.capture import (
    FIELDS,
    ODDSAPI,
    QuoteRow,
    archive_summary,
    day_dir,
    fingerprint,
    last_quotes,
    latest_snapshot,
    now_utc,
    prices_dir,
    read_day,
    read_snapshot,
    snapshot_paths,
    write_snapshot,
)

# Provider market key -> (engine market key, ot_rule). The plan's eleven
# markets (§2); every key confirmed on a 2025-12-11 historical event.
MARKET_MAP: dict[str, tuple[str, str]] = {
    "h2h": ("game_ml", "incl_ot"),
    "spreads": ("game_ats", "incl_ot"),
    "totals": ("game_total", "incl_ot"),
    "h2h_h1": ("h1_ml", "reg_only"),
    "spreads_h1": ("h1_ats", "reg_only"),
    "totals_h1": ("h1_total", "reg_only"),
    "player_points": ("pl_pts", "incl_ot"),
    "player_threes": ("pl_3pm", "incl_ot"),
    "player_rebounds": ("pl_reb", "incl_ot"),
    "player_assists": ("pl_ast", "incl_ot"),
    "player_points_rebounds_assists": ("pl_pra", "incl_ot"),
}

GAME_MARKETS: tuple[str, ...] = ("h2h", "spreads", "totals")
HALF_MARKETS: tuple[str, ...] = ("h2h_h1", "spreads_h1", "totals_h1")
PROP_MARKETS: tuple[str, ...] = tuple(k for k in MARKET_MAP if k.startswith("player_"))
EVENT_MARKETS: tuple[str, ...] = HALF_MARKETS + PROP_MARKETS
ALL_MARKETS: tuple[str, ...] = tuple(MARKET_MAP)

__all__ = [
    "ALL_MARKETS",
    "EVENT_MARKETS",
    "FIELDS",
    "GAME_MARKETS",
    "HALF_MARKETS",
    "MARKET_MAP",
    "ODDSAPI",
    "PROP_MARKETS",
    "QuoteRow",
    "archive_summary",
    "day_dir",
    "fingerprint",
    "last_quotes",
    "latest_snapshot",
    "now_utc",
    "prices_dir",
    "read_day",
    "read_snapshot",
    "snapshot_paths",
    "write_snapshot",
]
