"""Team fielding-defense layer.

Converts a team's season fielding value (Outs Above Average, with DRS/Def as
fallbacks) into a bounded multiplier on the *opponent's* balls-in-play hits
(1B/2B/3B). Good defense turns more BIP into outs, suppressing BABIP-hits without
touching K/BB/HR (which defense does not influence). Neutral (1.0) when no
fielding feed is available, so it never fabricates an edge.
"""

from __future__ import annotations

import logging

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

# Bounded scaling: ~+/-20 OAA -> ~-/+5% BIP hits.
_OAA_SCALE = 0.0025
_MAX_EFFECT = 0.05


def defense_hit_multiplier(fielding_value: float) -> float:
    """Multiplier on opponent BIP hits given a team's fielding runs/outs value."""
    eff = max(-_MAX_EFFECT, min(_MAX_EFFECT, fielding_value * _OAA_SCALE))
    return 1.0 - eff


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
