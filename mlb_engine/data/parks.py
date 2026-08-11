"""Static ballpark reference data keyed by MLB Stats API ``venue_id``.

Fields
------
lat, lon          : venue coordinates (for weather lookups)
orientation_deg   : compass bearing (degrees, 0=N, 90=E) from home plate toward
                    center field. Used to convert wind direction into an
                    out-to-CF / in-from-CF component.
roof              : "open" | "retractable" | "closed" (dome). Closed/covered
                    parks neutralize weather effects.
park_factor       : runs park factor, 100 = league-neutral (>100 hitter-friendly).
wind_factor       : structural wind-receptivity (WAM). 1.0 = wind reaches the
                    field normally; open low-profile bowls (Wrigley) >1.0;
                    shielded high-grandstand parks (Oracle, Progressive) <1.0.
carry_factor      : fence/outfield profile multiplier on temperature/wind HR
                    payoff. Shallow / high-wall parks (Fenway, GABP, short
                    porches) >1.0; deep outfields (Comerica, Kauffman) <1.0.
singles_factor    : component park factor for singles, 1.0 = neutral. See below.
xbh_factor        : component park factor for doubles+triples, 1.0 = neutral.

Why singles need their own factor
---------------------------------
``park_factor`` is a *runs* factor, and runs are mostly home runs, so it says
almost nothing about singles: across the 30 parks the measured singles index
correlates **+0.09** with the runs factor and **-0.19** with the home-run index.
The two run opposite ways at the extremes -- Yankee Stadium is among the worst
singles parks in baseball (index 86) while scoring hitter-friendly on runs, and
Busch is the best (109) while scoring 98 -- because a park that turns flares
into home runs is subtracting singles, and a huge outfield that forces defenders
deep is adding them.

``singles_factor`` is built the standard way: each park's singles rate per plate
appearance against the rate the same hitters posted everywhere else, over
129,080 plate appearances. Those raw indices span 86..109, but only part of that
is the ballpark -- splitting the season into alternate days and correlating the
two halves across parks gives a reliability of **0.40** (against 0.85 for home
runs, which are far less dependent on where the fielders happen to stand). The
stored numbers keep 40% of each park's deviation from neutral accordingly, which
leaves a 0.945..1.035 spread. They are one season of data and should be widened
as more accumulates: shrinking is the conservative error here, since it prices
the park effect as smaller than measured rather than larger.

Why doubles need their own factor, and a bigger one
---------------------------------------------------
The same argument as singles, except the effect is three times the size and
repeats better. Where the fence *is* decides home runs; how much ground lies in
front of it decides doubles, and outfield geometry varies far more between parks
than fence distance does. Measured the same way -- each park's 2B+3B rate per PA
against the rate the same hitters posted everywhere else, 129,080 plate
appearances -- the raw index spans **0.77 (Wrigley) to 1.42 (Coors)**, against
singles' 0.86..1.09.

It is also the most reliable of the three component factors: splitting the
season into alternate days and correlating across parks gives **0.59**, against
0.40 for singles. Stored values keep 59% of each park's deviation accordingly,
leaving a 0.864..1.246 spread. Fitting on the first 60% of the season and
testing on the rest reproduces the ordering out of time (r = 0.36 across 30
parks) and improves held-out log loss.

It is not a restatement of what was already here: it correlates +0.34 with
``singles_factor`` and +0.42 with the runs factor. Kauffman is the clearest
case -- a 100 runs factor and a neutral singles index, but the deepest alleys in
baseball and a 1.17 doubles index, so the runs number hides it completely.

Coordinates and orientations are approximate but sufficient for wind projection.
Park factors are multi-year approximations and can be recalibrated over time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Park:
    venue_id: int
    name: str
    lat: float
    lon: float
    orientation_deg: float
    roof: str
    park_factor: float
    wind_factor: float = 1.0
    carry_factor: float = 1.0
    singles_factor: float = 1.0
    xbh_factor: float = 1.0


PARKS: dict[int, Park] = {
    1: Park(1, "Angel Stadium", 33.8003, -117.8827, 45.0, "open", 98.0, singles_factor=0.981, xbh_factor=0.898),
    2: Park(2, "Oriole Park at Camden Yards", 39.2839, -76.6217, 30.0, "open", 101.0, singles_factor=1.001, xbh_factor=0.961),
    3: Park(3, "Fenway Park", 42.3467, -71.0972, 45.0, "open", 104.0, carry_factor=1.15, singles_factor=1.012, xbh_factor=1.067),
    4: Park(4, "Rate Field", 41.8299, -87.6338, 130.0, "open", 101.0, singles_factor=0.982, xbh_factor=0.961),
    5: Park(5, "Progressive Field", 41.4962, -81.6852, 0.0, "open", 98.0, wind_factor=0.6, singles_factor=1.016, xbh_factor=1.015),
    7: Park(7, "Kauffman Stadium", 39.0517, -94.4803, 45.0, "open", 100.0, carry_factor=0.82, singles_factor=1.008, xbh_factor=1.172),
    12: Park(12, "Tropicana Field", 27.7683, -82.6534, 60.0, "closed", 96.0, singles_factor=0.995, xbh_factor=1.128),
    14: Park(14, "Rogers Centre", 43.6414, -79.3894, 0.0, "retractable", 101.0, singles_factor=1.005, xbh_factor=0.989),
    15: Park(15, "Chase Field", 33.4455, -112.0667, 0.0, "retractable", 103.0, singles_factor=1.019, xbh_factor=1.005),
    17: Park(17, "Wrigley Field", 41.9484, -87.6553, 30.0, "open", 103.0, wind_factor=1.20, carry_factor=1.05, singles_factor=1.018, xbh_factor=0.864),
    19: Park(19, "Coors Field", 39.7559, -104.9942, 0.0, "open", 112.0, wind_factor=1.05, carry_factor=1.08, singles_factor=1.026, xbh_factor=1.246),
    22: Park(22, "Dodger Stadium", 34.0739, -118.24, 25.0, "open", 98.0, singles_factor=0.974, xbh_factor=0.911),
    31: Park(31, "PNC Park", 40.4469, -80.0057, 120.0, "open", 98.0, singles_factor=1.021, xbh_factor=1.136),
    32: Park(32, "American Family Field", 43.028, -87.9712, 30.0, "retractable", 101.0, singles_factor=0.989, xbh_factor=0.935),
    680: Park(680, "T-Mobile Park", 47.5914, -122.3325, 0.0, "retractable", 95.0, singles_factor=0.989, xbh_factor=0.881),
    2392: Park(2392, "Daikin Park", 29.7573, -95.3555, 20.0, "retractable", 101.0, singles_factor=0.959, xbh_factor=0.963),
    2394: Park(2394, "Comerica Park", 42.339, -83.0485, 150.0, "open", 98.0, wind_factor=0.95, carry_factor=0.85, singles_factor=0.999, xbh_factor=0.933),
    2395: Park(2395, "Oracle Park", 37.7786, -122.3893, 90.0, "open", 96.0, wind_factor=0.35, carry_factor=0.90, singles_factor=1.010, xbh_factor=0.970),
    2529: Park(2529, "Sutter Health Park", 38.5802, -121.5135, 45.0, "open", 100.0, singles_factor=1.027, xbh_factor=1.138),
    2602: Park(2602, "Great American Ball Park", 39.0975, -84.5069, 30.0, "open", 104.0, wind_factor=1.05, carry_factor=1.12, singles_factor=0.954, xbh_factor=0.946),
    2680: Park(2680, "Petco Park", 32.7073, -117.157, 0.0, "open", 95.0, carry_factor=0.90, singles_factor=0.992, xbh_factor=0.871),
    2681: Park(2681, "Citizens Bank Park", 39.9061, -75.1665, 15.0, "open", 103.0, wind_factor=1.10, carry_factor=1.10, singles_factor=1.009, xbh_factor=1.049),
    2889: Park(2889, "Busch Stadium", 38.6226, -90.1928, 60.0, "open", 98.0, singles_factor=1.035, xbh_factor=1.008),
    3289: Park(3289, "Citi Field", 40.7571, -73.8458, 25.0, "open", 97.0, carry_factor=0.95, singles_factor=0.998, xbh_factor=0.979),
    3309: Park(3309, "Nationals Park", 38.873, -77.0074, 30.0, "open", 100.0, singles_factor=1.014, xbh_factor=0.966),
    3312: Park(3312, "Target Field", 44.9817, -93.2776, 90.0, "open", 100.0, singles_factor=0.995, xbh_factor=1.077),
    3313: Park(3313, "Yankee Stadium", 40.8296, -73.9262, 75.0, "open", 102.0, carry_factor=1.10, singles_factor=0.945, xbh_factor=1.078),
    4169: Park(4169, "loanDepot park", 25.7781, -80.2197, 30.0, "retractable", 97.0, singles_factor=0.996, xbh_factor=0.996),
    4705: Park(4705, "Truist Park", 33.8907, -84.4677, 25.0, "open", 101.0, singles_factor=1.012, xbh_factor=0.996),
    5325: Park(5325, "Globe Life Field", 32.7473, -97.0847, 45.0, "retractable", 100.0, singles_factor=1.004, xbh_factor=1.020),
}


def get_park(venue_id: int) -> Park | None:
    return PARKS.get(venue_id)
