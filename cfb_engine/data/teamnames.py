"""Team-name normalization and short-code generation.

The Odds API and CollegeFootballData both name teams "School + Mascot" (e.g.
"Alabama Crimson Tide"), but not always identically. ``norm`` collapses a name
to a comparison key; ``short_code`` produces a compact, human-readable display
code for cards and Excel by dropping the trailing mascot.
"""

from __future__ import annotations

import re

# Multi-word mascots that must be stripped as a unit so the school name survives
# (e.g. "Notre Dame Fighting Irish" -> "Notre Dame"). Single trailing mascot
# words are dropped generically.
_MULTIWORD_MASCOTS = (
    "crimson tide",
    "fighting irish",
    "fighting illini",
    "golden gophers",
    "golden bears",
    "golden flashes",
    "golden hurricane",
    "green wave",
    "demon deacons",
    "red raiders",
    "red wolves",
    "tar heels",
    "yellow jackets",
    "scarlet knights",
    "ragin cajuns",
    "mean green",
    "horned frogs",
    "sun devils",
    "mountaineers",
    "wolf pack",
    "black knights",
    "blue devils",
    "blue raiders",
    "blue hens",
    "boll weevils",
    "seminoles",
)


def norm(name: str) -> str:
    """Lowercase alphanumeric comparison key."""
    return re.sub(r"[^a-z0-9 ]", "", str(name).lower()).strip()


# External power-rating sources (Sagarin, FPI, FEI, ...) label teams by school
# only ("Ohio State", "Miami (FL)", "Texas A&M"), while CFBD/Odds use
# "School Mascot". ``ALIASES`` maps the many school spellings onto one canonical
# school key so an ensemble rating lands on the right CFBD team.
_SCHOOL_ALIASES: dict[str, str] = {
    "miami fl": "miami",
    "miami florida": "miami",
    "miami oh": "miami ohio",
    "miami ohio": "miami ohio",
    "ole miss": "mississippi",
    "pitt": "pittsburgh",
    "uconn": "connecticut",
    "umass": "massachusetts",
    "ucf": "central florida",
    "usf": "south florida",
    "smu": "southern methodist",
    "tcu": "texas christian",
    "byu": "brigham young",
    "unlv": "nevada las vegas",
    "ul monroe": "louisiana monroe",
    "ull": "louisiana",
    "louisiana lafayette": "louisiana",
    "la lafayette": "louisiana",
    "southern miss": "southern mississippi",
    "san jose st": "san jose state",
    "app state": "appalachian state",
    "fla atlantic": "florida atlantic",
    "fla international": "florida international",
    "fiu": "florida international",
    "hawaii": "hawaii",
    "nc state": "north carolina state",
    "n c state": "north carolina state",
    "texas am": "texas am",
    "sam houston st": "sam houston",
    "st francis pa": "saint francis",
}


# Single-word mascots common to sources that ship "School Mascot" (e.g. ESPN's
# displayName). Stripped only when they are a *known* mascot, so real school
# names whose last token is a plain word ("Ohio State") are never truncated.
_MASCOTS = frozenset(
    {
        "buckeyes", "wolverines", "crimson", "tide", "bulldogs", "tigers", "gators",
        "volunteers", "commodores", "razorbacks", "aggies", "rebels", "wildcats",
        "sooners", "cowboys", "longhorns", "jayhawks", "cyclones", "bears",
        "mountaineers", "cavaliers", "hokies", "hurricanes", "seminoles", "gamecocks",
        "trojans", "bruins", "ducks", "beavers", "huskies", "cougars", "utes",
        "buffaloes", "cardinal", "wolfpack", "eagles", "knights", "bulls", "owls",
        "hurricane", "spartans", "nittany", "lions", "hawkeyes", "badgers", "gophers",
        "cornhuskers", "terrapins", "boilermakers", "hoosiers", "fighting",
        "illini", "panthers", "orange", "ramblin", "jackets", "deacons", "devils",
        "irish", "aztecs", "broncos", "rams", "falcons", "raiders", "mustangs", "frogs", "bearcats", "cajuns", "warhawks", "chanticleers",
        "hilltoppers", "blazers", "miners", "vandals", "redhawks",
        "chippewas", "rockets", "zips", "bobcats", "cardinals",
        "minutemen", "midshipmen",
    }
)


def school_key(name: str) -> str:
    """Canonical school-only key for cross-source team matching.

    Normalizes, strips a recognized mascot suffix (multi-word first, then a
    single known mascot word), expands ``St.`` -> ``State``, and resolves known
    aliases so "Miami (FL)", "Miami Hurricanes" and "Miami" collapse to one key
    while school names ending in a plain word ("Ohio State") stay intact.
    """
    school = norm(name)
    for mascot in _MULTIWORD_MASCOTS:
        if school.endswith(" " + mascot):
            school = school[: -len(mascot) - 1].strip()
            break
    parts = school.split()
    if len(parts) > 1 and parts[-1] in _MASCOTS:
        parts = parts[:-1]
    school = " ".join(parts)
    school = re.sub(r"\bst\b", "state", school)
    school = re.sub(r"\bu\b", "", school).strip()
    school = re.sub(r"\s+", " ", school)
    return _SCHOOL_ALIASES.get(school, school)


def _strip_mascot(name: str) -> str:
    low = norm(name)
    for mascot in _MULTIWORD_MASCOTS:
        if low.endswith(" " + mascot):
            return name[: len(name) - len(mascot) - 1].strip()
    parts = name.split()
    if len(parts) > 1:
        return " ".join(parts[:-1])
    return name


def short_code(name: str, *, maxlen: int = 14) -> str:
    """A compact display label: the school name, mascot removed, length-capped."""
    school = _strip_mascot(name).strip()
    if not school:
        school = name
    return school if len(school) <= maxlen else school[:maxlen].rstrip()
