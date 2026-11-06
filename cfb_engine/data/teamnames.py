"""Team-name normalization and short-code generation.

The Odds API and CollegeFootballData both name teams "School + Mascot" (e.g.
"Alabama Crimson Tide"), but not always identically. ``norm`` collapses a name
to a comparison key; ``short_code`` produces a compact, human-readable display
code for cards and Excel by dropping the trailing mascot.
"""

from __future__ import annotations

import re
import unicodedata

# Multi-word mascots that must be stripped as a unit so the school name survives
# (e.g. "Notre Dame Fighting Irish" -> "Notre Dame"). Single trailing mascot
# words are dropped generically.
_MULTIWORD_MASCOTS = (
    "nittany lions",
    "runnin bulldogs",
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
    "rainbow warriors",
    "golden panthers",
    "golden eagles",
    "thundering herd",
    "black bears",
    "fighting camels",
)


def norm(name: str) -> str:
    """Lowercase alphanumeric comparison key.

    Transliterates accents (``San José`` -> ``san jose``) and treats hyphens and
    slashes as spaces (``Louisiana-Monroe`` -> ``louisiana monroe``) so the same
    school spelled differently across sources collapses to one key.
    """
    decomposed = unicodedata.normalize("NFKD", str(name))
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    spaced = ascii_only.lower().replace("-", " ").replace("/", " ")
    cleaned = re.sub(r"[^a-z0-9 ]", "", spaced)
    return re.sub(r"\s+", " ", cleaned).strip()


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
    "houston baptist": "houston christian",
    "william and mary": "william mary",
    "penn st": "penn state",
    "app state": "appalachian state",
    "fla atlantic": "florida atlantic",
    "fla international": "florida international",
    "fiu": "florida international",
    "hawaii": "hawaii",
    "nc state": "north carolina state",
    "n c state": "north carolina state",
    "texas am": "texas am",
    "sam houston st": "sam houston",
    "sam houston state": "sam houston",
    "st francis pa": "saint francis",
    "middle tennessee state": "middle tennessee",
    "ualbany": "albany",
    "grambling state": "grambling",
    "southern university": "southern",
    "liu": "long island university",
    "long island": "long island university",
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
        "minutemen", "midshipmen", "bison", "hornets",
    }
)


# Every word a card label can be carrying past the school name: a mascot the
# strip missed, or a suffix CFBD's spelling drops ("Grambling State" vs
# "Grambling").
LABEL_SUFFIX_WORDS = frozenset(
    {"state", "university", "college"}
    | _MASCOTS
    | {word for mascot in _MULTIWORD_MASCOTS for word in mascot.split()}
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


def _mascot_prefix(fragment: str) -> bool:
    """Whether ``fragment`` is the start of a known mascot (``black`` -> Black Bears)."""
    return bool(fragment) and any(m.startswith(fragment) for m in (*_MULTIWORD_MASCOTS, *_MASCOTS))


def label_matches(label: str, full: str) -> bool:
    """Whether a length-capped display label names the same school as ``full``.

    ``short_code`` cuts "School Mascot" at 14 characters, so the label may be a
    prefix of the school (``New Mexico Sta``), the school plus the start of its
    mascot (``Penn State Nit``, ``Gardner-Webb R``), or a prefix of an alias
    spelling (``Houston Baptis`` for Houston Christian). Compared against the
    full name, its school key, and every alias that resolves to that key.
    """
    left = norm(label)
    if not left:
        return False
    capped = len(label.strip()) >= 14
    right, key = norm(full), school_key(full)
    candidates = {right, key} | {a for a, c in _SCHOOL_ALIASES.items() if c == key}
    for cand in candidates:
        if left == cand or cand.startswith(left):
            return True
        if left.startswith(cand + " "):
            rest = left[len(cand) + 1 :]
            if capped or _mascot_prefix(rest):
                return True
    return school_key(label) == key


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
