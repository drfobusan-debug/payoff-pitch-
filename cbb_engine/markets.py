"""Market map (§2): provider key -> engine key, settlement rule, year-1 status.

``status`` is carried on every ledger row and the live promotion test filters to
``live``. Changing a status is a decision recorded in the plan, not a code edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Status = Literal["live", "shadow", "research"]
STATUSES: tuple[Status, ...] = ("live", "shadow", "research")


@dataclass(frozen=True)
class Market:
    provider: str
    key: str
    settle: str  # incl_ot | reg_only | half
    status: Status
    per_event: bool = False  # needs /events/{id}/odds (costed separately)


MARKETS: tuple[Market, ...] = (
    Market("h2h", "game_ml", "incl_ot", "research"),
    Market("spreads", "game_ats", "incl_ot", "research"),
    Market("totals", "game_total", "incl_ot", "live"),
    Market("alternate_spreads", "game_ats_alt", "incl_ot", "shadow", per_event=True),
    Market("alternate_totals", "game_total_alt", "incl_ot", "shadow", per_event=True),
    Market("team_totals", "team_total", "incl_ot", "shadow", per_event=True),
    Market("h2h_h1", "h1_ml", "half", "research", per_event=True),
    Market("spreads_h1", "h1_ats", "half", "shadow", per_event=True),
    Market("totals_h1", "h1_total", "half", "shadow", per_event=True),
    Market("player_points", "pl_pts", "incl_ot", "research", per_event=True),
    Market("player_rebounds", "pl_reb", "incl_ot", "research", per_event=True),
    Market("player_assists", "pl_ast", "incl_ot", "research", per_event=True),
    Market("player_threes", "pl_3pm", "incl_ot", "research", per_event=True),
)

BY_PROVIDER: dict[str, Market] = {m.provider: m for m in MARKETS}
BY_KEY: dict[str, Market] = {m.key: m for m in MARKETS}
FEATURED: tuple[str, ...] = ("h2h", "spreads", "totals")
EVENT_MARKETS: tuple[str, ...] = tuple(m.provider for m in MARKETS if m.per_event)
LIVE_KEYS: tuple[str, ...] = tuple(m.key for m in MARKETS if m.status == "live")


def status_of(engine_key: str) -> Status:
    """Alt lines on a main market inherit the alt row's status."""
    return BY_KEY[engine_key].status if engine_key in BY_KEY else "research"


__all__ = [
    "BY_KEY",
    "BY_PROVIDER",
    "EVENT_MARKETS",
    "FEATURED",
    "LIVE_KEYS",
    "MARKETS",
    "STATUSES",
    "Market",
    "Status",
    "status_of",
]
