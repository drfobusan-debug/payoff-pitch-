"""Public Baseball Savant fielding feed (Outs Above Average leaderboard).

Savant's fielding leaderboard is a free, no-login CSV endpoint (same host as the
Statcast search the engine already uses), so it activates the full per-position
fielding hierarchy in ``filters/defense.py`` without any subscription. Each
fielder's OAA and Fielding Run Value are aggregated per team into infield /
outfield / middle-infield OAA plus a team FRV total.

Returns ``{}`` on any failure so the defense layer stays neutral rather than
fabricating an edge.
"""

from __future__ import annotations

import io
import logging

import pandas as pd

from mlb_engine.data import http

log = logging.getLogger(__name__)

_URL = (
    "https://baseballsavant.mlb.com/leaderboard/outs_above_average"
    "?type=Fielder&year={year}&min=q&csv=true"
)
_HEADERS = {"User-Agent": "Mozilla/5.0 (mlb-prediction-engine)"}

# Savant display nickname -> MLB Stats API team abbreviation.
_SAVANT_TO_STATSAPI = {
    "Angels": "LAA", "Astros": "HOU", "Athletics": "ATH", "Blue Jays": "TOR",
    "Braves": "ATL", "Brewers": "MIL", "Cardinals": "STL", "Cubs": "CHC",
    "D-backs": "AZ", "Dodgers": "LAD", "Giants": "SF", "Guardians": "CLE",
    "Mariners": "SEA", "Marlins": "MIA", "Mets": "NYM", "Nationals": "WSH",
    "Orioles": "BAL", "Padres": "SD", "Phillies": "PHI", "Pirates": "PIT",
    "Rangers": "TEX", "Rays": "TB", "Red Sox": "BOS", "Reds": "CIN",
    "Rockies": "COL", "Royals": "KC", "Tigers": "DET", "Twins": "MIN",
    "White Sox": "CWS", "Yankees": "NYY",
}

_INFIELD = {"1B", "2B", "3B", "SS"}
_OUTFIELD = {"LF", "CF", "RF"}
_MIDDLE_IF = {"2B", "SS"}


def fetch_oaa(year: int, timeout: int = 30) -> pd.DataFrame:
    """Return the raw Savant fielder OAA leaderboard for a season."""
    resp = http.get(_URL.format(year=year), headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def load_team_oaa(year: int) -> dict[str, dict[str, float]]:
    """Return ``{statsapi_abbrev: {frv, infield_oaa, outfield_oaa, middle_if_oaa}}``.

    Empty dict on any failure so the caller falls back to neutral defense.
    """
    try:
        df = fetch_oaa(year)
    except Exception as exc:  # optional enrichment
        log.warning("Savant fielding unavailable: %s", exc)
        return {}

    need = {"display_team_name", "primary_pos_formatted",
            "outs_above_average", "fielding_runs_prevented"}
    if not need.issubset(df.columns):
        log.warning("Savant fielding columns missing (have %s)", list(df.columns)[:8])
        return {}

    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        abbrev = _SAVANT_TO_STATSAPI.get(str(row["display_team_name"]).strip())
        if abbrev is None:
            continue
        pos = str(row["primary_pos_formatted"]).strip()
        try:
            oaa = float(row["outs_above_average"])
            frv = float(row["fielding_runs_prevented"])
        except (TypeError, ValueError):
            continue
        agg = out.setdefault(
            abbrev, {"frv": 0.0, "infield_oaa": 0.0, "outfield_oaa": 0.0, "middle_if_oaa": 0.0}
        )
        agg["frv"] += frv
        if pos in _INFIELD:
            agg["infield_oaa"] += oaa
        if pos in _OUTFIELD:
            agg["outfield_oaa"] += oaa
        if pos in _MIDDLE_IF:
            agg["middle_if_oaa"] += oaa
    return out
