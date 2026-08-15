"""VSiN 2026 CFB Betting Guide -- per-team home-field advantage.

The guide publishes a home-field-advantage rating per program, bucketed by each
team's three-year straight-up and against-the-spread home win percentage. That
bucketing is the problem: the three-year window *is* 2023-2025, so on those
seasons the table is being read against the results it was built from. Measured
against the closing spread's residual on 2,966 games at listed venues:

    2014-2022 (outside the window)   r=+0.0424  p=.047
    2023-2025 (inside the window)    r=+0.2928  p=.000
        tier 3.5, inside:  n=342  home cover 64.0%
        tier 1.0, inside:  n=108  home cover 29.6%

A 64%-to-30% spread is what a table looks like when it is tested on its own
fitting sample, not what a forecast looks like. And the quantity it claims to
measure does not persist: a program's own home-game residual predicts its next
season's at r=+0.017 (p=.68, 581 team-seasons). Home edge over the market is not
a stable team property, so a per-team HFA table cannot be forecasting one.

So the table is reported and not priced -- ``CFBE_VSIN_HFA=1`` restores the
override. The flat default it falls back to was measured at the same time and
held up: regressing on the SP+ gap over 7,345 home-site games, home field
delivered +2.33 +/- 0.16 points against a model default of 2.40 (the market
prices +2.84, and that gap has closed to +0.07 since 2022).

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


def hfa_note(home_name: str, default: float, *, enabled: bool = True) -> str | None:
    """What the guide would have charged, for the card, when it is switched off.

    Silent while the override is live (the number is already in the price) and
    silent for teams the guide does not list.
    """
    if enabled:
        return None
    listed = VSIN_HFA.get(school_key(home_name))
    if listed is None or abs(listed - default) < 0.05:
        return None
    return f"VSiN home field {listed:.1f} vs {default:.1f} used [reported, not scored]"
