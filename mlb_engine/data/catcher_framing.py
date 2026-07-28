"""Catcher pitch-framing (Baseball Savant catcher-framing leaderboard).

Elite framers steal called strikes (more K, fewer BB for the pitching side);
poor framers give them back. This maps a starting catcher to the
``catcher_framing_runs`` input of ``HumanFactors`` (see ``filters/human.py``),
which already sizes framing at ~1/5 of the umpire effect.

The primary source is Savant's free catcher-framing leaderboard CSV (same host
as the Statcast search and OAA feed the engine already uses); its
``rv_tot`` column is the season framing run value, keyed by MLBAM
player id. The curated table below is the offline fallback for the catchers with
the most documented, stable values. ``framing_runs_for_name`` returns ``None``
for unknown catchers so the layer stays neutral rather than fabricating an edge.
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

log = logging.getLogger(__name__)

_URL = (
    "https://baseballsavant.mlb.com/leaderboard/catcher-framing"
    "?year={year}&min=q&csv=true"
)
_HEADERS = {"User-Agent": "Mozilla/5.0 (mlb-prediction-engine)"}
_RUNS_COLS = ("rv_tot", "runs_extra_strikes", "runs")  # framing run value (first present wins)
_ID_COLS = ("id", "player_id")


def load_framing(year: int, timeout: int = 30) -> dict[int, float]:
    """Return ``{mlbam_player_id: framing_runs}`` from Savant.

    Empty dict on any failure so the caller falls back to the curated table.
    """
    try:
        resp = requests.get(_URL.format(year=year), headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:  # optional enrichment
        log.warning("Savant catcher framing unavailable: %s", exc)
        return {}

    runs_col = next((c for c in _RUNS_COLS if c in df.columns), None)
    id_col = next((c for c in _ID_COLS if c in df.columns), None)
    if runs_col is None or id_col is None:
        log.warning("Savant framing columns missing (have %s)", list(df.columns)[:8])
        return {}

    out: dict[int, float] = {}
    for _, row in df.iterrows():
        try:
            out[int(row[id_col])] = float(row[runs_col])
        except (TypeError, ValueError):
            continue
    return out


# Season catcher framing runs vs. average (positive = steals strikes).
CURATED: dict[str, float] = {
    # Elite framers -> more strikes -> Under lean (K up, BB down).
    "patrick bailey": 16.0,
    "austin hedges": 13.0,
    "cal raleigh": 12.0,
    "jose trevino": 11.0,
    "gabriel moreno": 10.0,
    "sean murphy": 9.0,
    "alejandro kirk": 8.0,
    # Poor framers -> gives strikes back -> Over lean (K down, BB up).
    "keibert ruiz": -10.0,
    "salvador perez": -9.0,
    "martin maldonado": -8.0,
    "elias diaz": -7.0,
}


def _norm(name: str) -> str:
    return " ".join(name.lower().replace(".", "").split())


def framing_runs_for_name(name: str) -> float | None:
    """Curated framing runs for a catcher, or ``None`` if unknown."""
    return CURATED.get(_norm(name))
