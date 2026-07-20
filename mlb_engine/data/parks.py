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


PARKS: dict[int, Park] = {
    1: Park(1, "Angel Stadium", 33.8003, -117.8827, 45.0, "open", 98.0),
    2: Park(2, "Oriole Park at Camden Yards", 39.2839, -76.6217, 30.0, "open", 101.0),
    3: Park(3, "Fenway Park", 42.3467, -71.0972, 45.0, "open", 104.0),
    4: Park(4, "Rate Field", 41.8299, -87.6338, 130.0, "open", 101.0),
    5: Park(5, "Progressive Field", 41.4962, -81.6852, 0.0, "open", 98.0),
    7: Park(7, "Kauffman Stadium", 39.0517, -94.4803, 45.0, "open", 100.0),
    12: Park(12, "Tropicana Field", 27.7683, -82.6534, 60.0, "closed", 96.0),
    14: Park(14, "Rogers Centre", 43.6414, -79.3894, 0.0, "retractable", 101.0),
    15: Park(15, "Chase Field", 33.4455, -112.0667, 0.0, "retractable", 103.0),
    17: Park(17, "Wrigley Field", 41.9484, -87.6553, 30.0, "open", 103.0),
    19: Park(19, "Coors Field", 39.7559, -104.9942, 0.0, "open", 112.0),
    22: Park(22, "Dodger Stadium", 34.0739, -118.24, 25.0, "open", 98.0),
    31: Park(31, "PNC Park", 40.4469, -80.0057, 120.0, "open", 98.0),
    32: Park(32, "American Family Field", 43.028, -87.9712, 30.0, "retractable", 101.0),
    680: Park(680, "T-Mobile Park", 47.5914, -122.3325, 0.0, "retractable", 95.0),
    2392: Park(2392, "Daikin Park", 29.7573, -95.3555, 20.0, "retractable", 101.0),
    2394: Park(2394, "Comerica Park", 42.339, -83.0485, 150.0, "open", 98.0),
    2395: Park(2395, "Oracle Park", 37.7786, -122.3893, 90.0, "open", 96.0),
    2529: Park(2529, "Sutter Health Park", 38.5802, -121.5135, 45.0, "open", 100.0),
    2602: Park(2602, "Great American Ball Park", 39.0975, -84.5069, 30.0, "open", 104.0),
    2680: Park(2680, "Petco Park", 32.7073, -117.157, 0.0, "open", 95.0),
    2681: Park(2681, "Citizens Bank Park", 39.9061, -75.1665, 15.0, "open", 103.0),
    2889: Park(2889, "Busch Stadium", 38.6226, -90.1928, 60.0, "open", 98.0),
    3289: Park(3289, "Citi Field", 40.7571, -73.8458, 25.0, "open", 97.0),
    3309: Park(3309, "Nationals Park", 38.873, -77.0074, 30.0, "open", 100.0),
    3312: Park(3312, "Target Field", 44.9817, -93.2776, 90.0, "open", 100.0),
    3313: Park(3313, "Yankee Stadium", 40.8296, -73.9262, 75.0, "open", 102.0),
    4169: Park(4169, "loanDepot park", 25.7781, -80.2197, 30.0, "retractable", 97.0),
    4705: Park(4705, "Truist Park", 33.8907, -84.4677, 25.0, "open", 101.0),
    5325: Park(5325, "Globe Life Field", 32.7473, -97.0847, 45.0, "retractable", 100.0),
}


def get_park(venue_id: int) -> Park | None:
    return PARKS.get(venue_id)
