"""Static MLB team -> division map (MLB Stats API team IDs).

Used by the human-element layer for the division-rivalry familiarity signal:
teams that face each other many times a season lose the pitcher's novelty edge.
"""

from __future__ import annotations

DIVISION: dict[int, str] = {
    # AL East
    110: "AL_E",  # BAL
    111: "AL_E",  # BOS
    147: "AL_E",  # NYY
    139: "AL_E",  # TB
    141: "AL_E",  # TOR
    # AL Central
    145: "AL_C",  # CWS
    114: "AL_C",  # CLE
    116: "AL_C",  # DET
    118: "AL_C",  # KC
    142: "AL_C",  # MIN
    # AL West
    117: "AL_W",  # HOU
    108: "AL_W",  # LAA
    133: "AL_W",  # OAK / ATH
    136: "AL_W",  # SEA
    140: "AL_W",  # TEX
    # NL East
    144: "NL_E",  # ATL
    146: "NL_E",  # MIA
    121: "NL_E",  # NYM
    143: "NL_E",  # PHI
    120: "NL_E",  # WSH
    # NL Central
    112: "NL_C",  # CHC
    113: "NL_C",  # CIN
    158: "NL_C",  # MIL
    134: "NL_C",  # PIT
    138: "NL_C",  # STL
    # NL West
    109: "NL_W",  # ARI
    115: "NL_W",  # COL
    119: "NL_W",  # LAD
    135: "NL_W",  # SD
    137: "NL_W",  # SF
}


def same_division(team_a: int, team_b: int) -> bool:
    da, db = DIVISION.get(team_a), DIVISION.get(team_b)
    return da is not None and da == db
