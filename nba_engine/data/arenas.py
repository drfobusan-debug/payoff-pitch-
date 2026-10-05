"""Where each team plays: coordinates, elevation and time zone.

Travel miles, time-zone shifts and the altitude flag come from here. These are
facts about buildings, not fitted values. A neutral-site game is placed by its
venue's city when the city is listed, and otherwise carries no travel reading
(``None``), so the game is never charged travel it did not have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Site:
    lat: float
    lon: float
    elevation_ft: int
    tz: str


HOME: dict[str, Site] = {
    "ATL": Site(33.757, -84.396, 1050, "America/New_York"),
    "BOS": Site(42.366, -71.062, 20, "America/New_York"),
    "BKN": Site(40.683, -73.976, 30, "America/New_York"),
    "CHA": Site(35.225, -80.839, 750, "America/New_York"),
    "CHI": Site(41.881, -87.674, 600, "America/Chicago"),
    "CLE": Site(41.496, -81.688, 650, "America/New_York"),
    "DAL": Site(32.790, -96.810, 430, "America/Chicago"),
    "DEN": Site(39.749, -105.008, 5280, "America/Denver"),
    "DET": Site(42.341, -83.055, 600, "America/Detroit"),
    "GSW": Site(37.768, -122.388, 10, "America/Los_Angeles"),
    "HOU": Site(29.751, -95.362, 50, "America/Chicago"),
    "IND": Site(39.764, -86.155, 715, "America/Indiana/Indianapolis"),
    "LAC": Site(33.945, -118.343, 100, "America/Los_Angeles"),
    "LAL": Site(34.043, -118.267, 300, "America/Los_Angeles"),
    "MEM": Site(35.138, -90.051, 300, "America/Chicago"),
    "MIA": Site(25.781, -80.188, 10, "America/New_York"),
    "MIL": Site(43.045, -87.917, 600, "America/Chicago"),
    "MIN": Site(44.979, -93.276, 830, "America/Chicago"),
    "NOP": Site(29.949, -90.082, 10, "America/Chicago"),
    "NYK": Site(40.751, -73.993, 30, "America/New_York"),
    "OKC": Site(35.463, -97.515, 1200, "America/Chicago"),
    "ORL": Site(28.539, -81.384, 100, "America/New_York"),
    "PHI": Site(39.901, -75.172, 30, "America/New_York"),
    "PHX": Site(33.446, -112.071, 1100, "America/Phoenix"),
    "POR": Site(45.532, -122.667, 50, "America/Los_Angeles"),
    "SAC": Site(38.580, -121.500, 30, "America/Los_Angeles"),
    "SAS": Site(29.427, -98.438, 650, "America/Chicago"),
    "TOR": Site(43.643, -79.379, 250, "America/Toronto"),
    "UTA": Site(40.768, -111.901, 4226, "America/Denver"),
    "WAS": Site(38.898, -77.021, 50, "America/New_York"),
}

NEUTRAL_CITY: dict[str, Site] = {
    "las vegas": Site(36.103, -115.178, 2030, "America/Los_Angeles"),
    "mexico city": Site(19.404, -99.097, 7350, "America/Mexico_City"),
    "paris": Site(48.839, 2.379, 120, "Europe/Paris"),
    "indianapolis": Site(39.764, -86.155, 715, "America/Indiana/Indianapolis"),
    "berlin": Site(52.508, 13.443, 110, "Europe/Berlin"),
    "london": Site(51.503, 0.003, 30, "Europe/London"),
    "salt lake city": Site(40.768, -111.901, 4226, "America/Denver"),
}

ALTITUDE_FT = 4000


def game_site(home: str, neutral: bool, city: str) -> Site | None:
    """Where the game is played: the home arena unless ESPN flags a neutral site."""
    if not neutral:
        return HOME.get(home)
    return NEUTRAL_CITY.get(city.strip().lower())


def miles(a: Site, b: Site) -> float:
    """Great-circle distance in statute miles."""
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dp, dl = p2 - p1, math.radians(b.lon - a.lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 3958.8 * math.asin(math.sqrt(h))


__all__ = ["ALTITUDE_FT", "HOME", "NEUTRAL_CITY", "Site", "game_site", "miles"]
