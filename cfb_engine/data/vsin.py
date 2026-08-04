"""VSiN 2026 CFB Betting Guide -- per-team home-field advantage.

The guide publishes a home-field-advantage rating per program, bucketed by each
team's three-year straight-up and against-the-spread home win percentage. We use
it to override the model's flat home-field points on a per-home-team basis;
teams the guide does not list keep the model default (a middle-of-the-road HFA).

Only the numeric HFA table is encoded here -- the guide's win totals, futures,
stability scores, and coaching notes are season-long or qualitative and are not
per-game pricing inputs.
"""

from __future__ import annotations

from cfb_engine.data.teamnames import school_key

# Point values by guide tier. Teams outside these tiers fall in the guide's
# unlisted middle band and keep the model's default HFA.
_TIER_35 = (
    "Alabama", "Boise State", "BYU", "Georgia Southern", "Indiana", "Iowa",
    "Jacksonville State", "James Madison", "Marshall", "Miami (OH)", "Mississippi",
    "Missouri", "North Dakota State", "Notre Dame", "Ohio", "Ohio State", "Oregon",
    "SMU", "Texas", "Texas Tech", "Toledo", "UTSA", "Washington", "Western Michigan",
)
_TIER_30 = (
    "Arizona", "Ball State", "Central Michigan", "Connecticut", "Delaware",
    "Georgia Tech", "Hawaii", "Kansas State", "LSU", "Miami (FL)", "Oklahoma",
    "South Florida", "USC", "Utah State", "Washington State", "Western Kentucky",
)
_TIER_15 = (
    "Arkansas", "Baylor", "Florida Atlantic", "Michigan State", "Middle Tennessee",
    "Northern Illinois", "Stanford", "UCLA",
)
_TIER_10 = (
    "Charlotte", "Georgia State", "Kent State", "Massachusetts", "Nevada",
    "Purdue", "Tulsa", "UTEP",
)


def _build() -> dict[str, float]:
    table: dict[str, float] = {}
    for pts, names in ((3.5, _TIER_35), (3.0, _TIER_30), (1.5, _TIER_15), (1.0, _TIER_10)):
        for name in names:
            table[school_key(name)] = pts
    return table


VSIN_HFA: dict[str, float] = _build()


def hfa_for(home_name: str, default: float, *, enabled: bool = True) -> float:
    """Home-field points for ``home_name``: the VSiN value if listed, else ``default``."""
    if not enabled:
        return default
    return VSIN_HFA.get(school_key(home_name), default)
