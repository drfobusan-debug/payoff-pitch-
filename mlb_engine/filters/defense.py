"""Team fielding-defense layer (weighted, OAA/FRV-based).

Fielding explains ~12-15% of total game variance and ~7% in this engine's
hierarchy. It is modeled off Statcast Outs Above Average / Fielding Run Value
(never errors or fielding%, which lack predictive power), converted to runs and
split by the positional value hierarchy so it touches only balls in play:

- Infield range (SS+2B+3B+1B) -> suppresses grounder singles (1B).
- Outfield range (CF+LF+RF) -> suppresses gap extra-base hits (2B/3B).
- Strikeouts, walks and home runs (the three true outcomes) are untouched.

Run conversions: OF 1 OAA = 0.90 runs, IF 1 OAA = 0.75 runs, OF arm kill = 1.00.
Positional value weights: SS 17.5%, 2B 17.5%, CF 25%, 3B/LF/RF 10% each, 1B 10%
(SS+2B = 35%, CF = 25%, 3B+corner OF = 30%, 1B = 10%).

Team FRV is the validated fallback (split into IF/OF shares); per-position OAA,
DER and the middle-infield NPV activate the full hierarchy when a positional
feed is supplied. Neutral (1.0) when no fielding data exists, so it never
fabricates an edge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# FanGraphs team abbreviations that differ from MLB Stats API abbreviations.
_FG_TO_STATSAPI = {
    "CHW": "CWS",
    "KCR": "KC",
    "SDP": "SD",
    "SFG": "SF",
    "TBR": "TB",
    "WSN": "WSH",
}

# OAA -> runs prevented (Baseball Savant fielding run value linear weights).
OF_RUN_PER_OAA = 0.90
IF_RUN_PER_OAA = 0.75
ARM_RUN_PER_KILL = 1.00

# Positional value hierarchy -> infield vs. outfield share of team FRV.
IF_SHARE = 0.55  # SS .175 + 2B .175 + 3B .10 + 1B .10
OF_SHARE = 0.45  # CF .25 + LF .10 + RF .10

GAMES = 162.0
# Convert per-game runs prevented into a fractional BIP-hit change.
_BIP_HITS_PER_GAME = 6.0
_RUN_PER_HIT = 0.5
_RUN_DENOM = _BIP_HITS_PER_GAME * _RUN_PER_HIT  # runs per full BIP-hit swing
_MAX_EFFECT = 0.06  # bounded, consistent with fielding's ~7% variance share

# Elite defensive-efficiency anchors.
DER_BASELINE = 0.700
MIDDLE_IF_NPV_THRESHOLD = -8.0  # combined SS+2B OAA below this -> guaranteed leak


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _run_to_mult(runs_prevented_per_game: float) -> float:
    return 1.0 - _clip(runs_prevented_per_game / _RUN_DENOM, -_MAX_EFFECT, _MAX_EFFECT)


@dataclass(frozen=True)
class TeamDefense:
    """A fielding team's defensive profile (season-scale OAA/FRV/DER)."""

    frv: float = 0.0  # team fielding run value (fallback aggregate)
    infield_oaa: float | None = None  # SS+2B+3B+1B OAA
    outfield_oaa: float | None = None  # CF+LF+RF OAA
    middle_if_oaa: float | None = None  # SS+2B OAA (NPV tripwire)
    der: float | None = None  # defensive efficiency rating

    def bip_multipliers(self) -> dict[str, float]:
        """Multipliers on the opposing offense's balls-in-play hits."""
        if self.infield_oaa is not None:
            if_runs = self.infield_oaa * IF_RUN_PER_OAA / GAMES
        else:
            if_runs = self.frv * IF_SHARE / GAMES
        if self.outfield_oaa is not None:
            of_runs = self.outfield_oaa * OF_RUN_PER_OAA / GAMES
        else:
            of_runs = self.frv * OF_SHARE / GAMES

        one_b = _run_to_mult(if_runs)
        xbh = _run_to_mult(of_runs)

        # DER (team-wide out conversion) nudges all BIP hits.
        if self.der is not None:
            d = _clip((self.der - DER_BASELINE) * 2.0, -_MAX_EFFECT, _MAX_EFFECT)
            one_b *= 1.0 - d
            xbh *= 1.0 - d

        # NPV: a broken middle infield leaks grounder singles (1B) regardless.
        if self.middle_if_oaa is not None and self.middle_if_oaa < MIDDLE_IF_NPV_THRESHOLD:
            leak = _clip((MIDDLE_IF_NPV_THRESHOLD - self.middle_if_oaa) * 0.004, 0.0, _MAX_EFFECT)
            one_b *= 1.0 + leak

        return {"1B": one_b, "2B": xbh, "3B": xbh}


def defense_hit_multiplier(fielding_value: float) -> float:
    """Scalar team BIP-hit multiplier from a single fielding run/OAA value."""
    return _run_to_mult(fielding_value / GAMES)


def load_team_defense(year: int) -> dict[str, TeamDefense]:
    """Return ``{statsapi_abbrev: TeamDefense}`` for the season.

    Prefers the public Baseball Savant OAA leaderboard (per-position OAA + FRV,
    which activates the full fielding hierarchy). Falls back to the FanGraphs
    team-fielding aggregate (scalar FRV) if Savant is unavailable, and to an
    empty dict (neutral defense) if both fail.
    """
    from mlb_engine.data.savant_fielding import load_team_oaa

    savant = load_team_oaa(year)
    if savant:
        return {
            ab: TeamDefense(
                frv=v["frv"],
                infield_oaa=v["infield_oaa"],
                outfield_oaa=v["outfield_oaa"],
                middle_if_oaa=v["middle_if_oaa"],
            )
            for ab, v in savant.items()
        }
    return {ab: TeamDefense(frv=val) for ab, val in load_team_fielding(year).items()}


def load_team_fielding(year: int) -> dict[str, float]:
    """Return {statsapi_abbrev: fielding_value} from FanGraphs team fielding.

    Prefers OAA, then DRS, then Def. Returns {} on any failure (neutral).
    """
    try:
        from pybaseball import team_fielding

        df = team_fielding(year)
    except Exception as exc:  # optional enrichment
        log.warning("team fielding unavailable: %s", exc)
        return {}

    col = next((c for c in ("OAA", "DRS", "Def") if c in df.columns), None)
    if col is None or "Team" not in df.columns:
        log.warning("team fielding columns missing (have %s)", list(df.columns)[:12])
        return {}

    out: dict[str, float] = {}
    for _, row in df.iterrows():
        fg = str(row["Team"]).strip()
        abbrev = _FG_TO_STATSAPI.get(fg, fg)
        try:
            out[abbrev] = float(row[col])
        except (TypeError, ValueError):
            continue
    return out
