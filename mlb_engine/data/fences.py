"""Outfield fence geometry, for mapping a batted ball against the wall it faced.

A home run is a distance question answered by the park: the same fly ball is a
home run at 318 feet down the right-field line in the Bronx and a loud out to
the same spot in Detroit. Raw home-run totals therefore carry the shape of the
parks a hitter happened to visit; scoring each batted ball against the fence it
was actually hit toward is what strips that noise out (see
:mod:`mlb_engine.features.xhr`).

Each park is five anchors -- left-field line, left-center, center, right-center,
right-field line -- at spray angles -45/-22.5/0/+22.5/+45 degrees, with the wall
height at each. Distances are the published dimensions; the anchors are a
piecewise-linear approximation of a real (curved, occasionally jagged) wall, so
treat a single batted ball's answer as approximate and the aggregate over a
season as meaningful.

Wall height matters as much as distance in a handful of parks -- Fenway's 310ft
left-field line is not a short porch once 37 feet of Green Monster is in the way
-- so :func:`effective_distance` converts height into the extra carry a ball
needs to clear it.
"""

from __future__ import annotations

from dataclasses import dataclass

# Spray angles of the five anchors, left-field line to right-field line.
# Matches the convention in ``mlb_engine.features.regression._pull_air_rate``:
# negative is the third-base side, 0 is straightaway center.
ANCHOR_ANGLES = (-45.0, -22.5, 0.0, 22.5, 45.0)

# A fly ball reaches the wall on a descending arc of roughly 40 degrees, so each
# foot of wall above the standard ~8ft costs about this much extra carry.
HEIGHT_TO_DISTANCE = 1.2
STANDARD_WALL_HEIGHT = 8.0


@dataclass(frozen=True)
class Fence:
    """Distances (feet) and wall heights (feet) at the five spray anchors."""

    distances: tuple[float, float, float, float, float]
    heights: tuple[float, float, float, float, float]

    def effective(self) -> tuple[float, ...]:
        """Distances adjusted for wall height: what a ball must actually carry."""
        return tuple(
            d + max(h - STANDARD_WALL_HEIGHT, 0.0) * HEIGHT_TO_DISTANCE
            for d, h in zip(self.distances, self.heights, strict=True)
        )


# Published outfield dimensions, keyed by MLB venue id (see
# ``mlb_engine.data.parks.PARKS``). Where a park's listed dimensions do not fall
# on an anchor angle, the nearest published marker is used.
FENCES: dict[int, Fence] = {
    1: Fence((330, 387, 396, 370, 330), (8, 8, 8, 8, 18)),  # Angel Stadium
    2: Fence((333, 384, 410, 373, 318), (7, 7, 7, 7, 21)),  # Camden Yards (2022+)
    3: Fence((310, 379, 390, 380, 302), (37, 17, 17, 5, 3)),  # Fenway Park
    4: Fence((330, 375, 400, 375, 335), (8, 8, 8, 8, 8)),  # Rate Field
    5: Fence((325, 370, 405, 375, 325), (19, 9, 9, 9, 9)),  # Progressive Field
    7: Fence((330, 387, 410, 387, 330), (9, 9, 9, 9, 9)),  # Kauffman Stadium
    12: Fence((315, 370, 404, 370, 322), (11, 9, 9, 9, 9)),  # Tropicana Field
    14: Fence((328, 375, 400, 375, 328), (10, 10, 10, 10, 10)),  # Rogers Centre
    15: Fence((330, 374, 407, 374, 335), (8, 8, 25, 8, 8)),  # Chase Field
    17: Fence((355, 368, 400, 368, 353), (11, 11, 11, 11, 11)),  # Wrigley Field
    19: Fence((347, 390, 415, 375, 350), (8, 8, 8, 8, 8)),  # Coors Field
    22: Fence((330, 385, 395, 385, 330), (8, 8, 8, 8, 8)),  # Dodger Stadium
    31: Fence((325, 389, 399, 375, 320), (6, 6, 10, 21, 21)),  # PNC Park
    32: Fence((344, 371, 400, 374, 345), (8, 8, 8, 8, 8)),  # American Family Field
    680: Fence((331, 378, 401, 381, 326), (8, 8, 8, 8, 8)),  # T-Mobile Park
    2392: Fence((315, 362, 409, 373, 326), (19, 19, 10, 10, 7)),  # Daikin Park
    2394: Fence((342, 370, 412, 365, 330), (8, 8, 8, 8, 8)),  # Comerica Park
    2395: Fence((339, 364, 399, 415, 309), (8, 8, 8, 25, 25)),  # Oracle Park
    2529: Fence((330, 375, 403, 375, 325), (8, 8, 8, 8, 8)),  # Sutter Health Park
    2602: Fence((328, 379, 404, 370, 325), (12, 8, 8, 8, 8)),  # Great American
    2680: Fence((336, 390, 396, 391, 322), (8, 8, 8, 8, 8)),  # Petco Park
    2681: Fence((329, 374, 401, 369, 330), (12, 6, 6, 6, 6)),  # Citizens Bank Park
    2889: Fence((336, 375, 400, 375, 335), (8, 8, 8, 8, 8)),  # Busch Stadium
    3289: Fence((335, 358, 408, 398, 330), (8, 8, 8, 8, 8)),  # Citi Field
    3309: Fence((336, 377, 402, 370, 335), (8, 8, 8, 14, 14)),  # Nationals Park
    3312: Fence((339, 377, 404, 367, 328), (8, 8, 8, 23, 23)),  # Target Field
    3313: Fence((318, 399, 408, 385, 314), (8, 8, 8, 8, 8)),  # Yankee Stadium
    4169: Fence((344, 386, 400, 387, 335), (11, 8, 8, 8, 8)),  # loanDepot park
    4705: Fence((335, 385, 400, 375, 325), (16, 8, 8, 8, 8)),  # Truist Park
    5325: Fence((329, 372, 407, 374, 326), (14, 8, 8, 8, 8)),  # Globe Life Field
}

# Used for a venue with no entry (a neutral-site or newly opened park), so an
# unknown park scores as league-average rather than dropping the batted ball.
LEAGUE_FENCE = Fence((331, 378, 403, 376, 328), (9, 9, 9, 9, 9))

# Statcast home-team abbreviation -> venue id, so a historical batted ball can be
# scored against the park it was actually hit in.
TEAM_VENUE: dict[str, int] = {
    "LAA": 1,
    "BAL": 2,
    "BOS": 3,
    "CWS": 4,
    "CHW": 4,
    "CLE": 5,
    "KC": 7,
    "TB": 12,
    "TOR": 14,
    "AZ": 15,
    "ARI": 15,
    "CHC": 17,
    "COL": 19,
    "LAD": 22,
    "PIT": 31,
    "MIL": 32,
    "SEA": 680,
    "HOU": 2392,
    "DET": 2394,
    "SF": 2395,
    "ATH": 2529,
    "OAK": 2529,
    "CIN": 2602,
    "SD": 2680,
    "PHI": 2681,
    "STL": 2889,
    "NYM": 3289,
    "WSH": 3309,
    "WSN": 3309,
    "MIN": 3312,
    "NYY": 3313,
    "MIA": 4169,
    "ATL": 4705,
    "TEX": 5325,
}


def get_fence(venue_id: int | None) -> Fence:
    """Fence geometry for a venue, falling back to a league-average park."""
    if venue_id is None:
        return LEAGUE_FENCE
    return FENCES.get(venue_id, LEAGUE_FENCE)


def fence_for_team(abbrev: str | None) -> Fence:
    """Fence geometry for the park a Statcast home-team abbreviation plays in."""
    if not abbrev:
        return LEAGUE_FENCE
    return get_fence(TEAM_VENUE.get(abbrev))


def wall_distance(fence: Fence, spray_deg: float) -> float:
    """Height-adjusted distance to the wall at a spray angle, interpolated.

    Angles outside the foul lines are clamped to the line anchors.
    """
    eff = fence.effective()
    x = max(ANCHOR_ANGLES[0], min(ANCHOR_ANGLES[-1], spray_deg))
    for i in range(len(ANCHOR_ANGLES) - 1):
        lo, hi = ANCHOR_ANGLES[i], ANCHOR_ANGLES[i + 1]
        if lo <= x <= hi:
            t = (x - lo) / (hi - lo)
            return eff[i] + t * (eff[i + 1] - eff[i])
    return eff[-1]
