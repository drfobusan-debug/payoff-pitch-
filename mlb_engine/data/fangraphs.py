"""FanGraphs client (optional projection blending).

Supports a CSV drop-in of FanGraphs projections/splits today; authenticated live
fetch is scaffolded and validated once credentials are supplied. Projections, if
present, are blended with the Statcast-derived rates in the pipeline.

Also ingests a FanGraphs *custom-report* CSV export (Leaderboards -> Export Data)
for the distribution-tail metrics FanGraphs owns -- SIERA & Stuff+ (pitchers),
wRC+ & xSLG (batters). A date-ranged custom report matches the engine's rolling
windows. FanGraphs CSVs key on Name (not MLBAM id), so rows are matched by name
downstream; unmatched rows stay neutral.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from mlb_engine.config import Credentials
from mlb_engine.data.rotowire import norm_person
from mlb_engine.features.rolling import OutcomeRates

log = logging.getLogger(__name__)


# Metric -> candidate FanGraphs CSV header spellings (lower-cased, punctuation-free).
_TAIL_COLUMNS = {
    "siera": ("siera",),
    "stuff_plus": ("stuff+", "stf+", "stuffplus", "pitchingstuff"),
    "wrc_plus": ("wrc+", "wrcplus", "wrc"),
    "xslg": ("xslg", "estslg", "expectedslg"),
}


@dataclass(frozen=True)
class FanGraphsTail:
    """Name-keyed FanGraphs tail metrics from a custom-report CSV export."""

    siera: dict[str, float] = field(default_factory=dict)
    stuff_plus: dict[str, float] = field(default_factory=dict)
    wrc_plus: dict[str, float] = field(default_factory=dict)
    xslg: dict[str, float] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.siera or self.stuff_plus or self.wrc_plus or self.xslg)


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    norm_cols = {re.sub(r"[^a-z0-9+]", "", str(c).lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9+]", "", cand)
        if key in norm_cols:
            return norm_cols[key]
    return None


def load_fangraphs_tail_csv(path: Path) -> FanGraphsTail:
    """Parse a FanGraphs custom-report CSV into name-keyed tail metrics.

    Tolerant of column spelling/ordering; a "Name"/"Player" column plus any of
    the SIERA/Stuff+/wRC+/xSLG columns are detected automatically. Returns an
    empty ``FanGraphsTail`` (all neutral) on any failure.
    """
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # optional enrichment
        log.warning("FanGraphs tail CSV unreadable (%s): %s", path, exc)
        return FanGraphsTail()

    name_col = _find_column(df, ("name", "player", "playername"))
    if name_col is None:
        log.warning("FanGraphs tail CSV has no Name column (have %s)", list(df.columns)[:8])
        return FanGraphsTail()

    metrics: dict[str, dict[str, float]] = {m: {} for m in _TAIL_COLUMNS}
    for metric, candidates in _TAIL_COLUMNS.items():
        col = _find_column(df, candidates)
        if col is None:
            continue
        for _, row in df.iterrows():
            key = norm_person(row[name_col])
            try:
                val = float(row[col])
            except (TypeError, ValueError):
                continue
            if val == val:  # skip NaN
                metrics[metric][key] = val
    return FanGraphsTail(
        siera=metrics["siera"],
        stuff_plus=metrics["stuff_plus"],
        wrc_plus=metrics["wrc_plus"],
        xslg=metrics["xslg"],
    )


class FanGraphsClient:
    def __init__(self, creds: Credentials, timeout: int = 25) -> None:
        self.creds = creds
        self.timeout = timeout

    def available(self) -> bool:
        return self.creds.has_fangraphs()

    def load_projections_csv(self, path: Path) -> pd.DataFrame:
        """Load a FanGraphs projections CSV (e.g., Depth Charts/ZiPS export)."""
        return pd.read_csv(path)

    def late_inning_batter_rates(
        self, batter_id: int, days: int = 21, min_inning: int = 6
    ) -> OutcomeRates | None:
        """Batter PA-outcome rates from a FanGraphs split (last ``days``, innings
        ``min_inning``-9) for the bullpen matchup.

        This is the FanGraphs equivalent of ``build_batter_late_rates`` and, when
        live access is available, overrides the Statcast-derived late-inning
        split. Returns ``None`` until credentials are validated so the pipeline
        falls back to Statcast.
        """
        if not self.available():
            return None
        log.info("FanGraphs late-inning split pending live access for %s", batter_id)
        return None

    def login(self) -> bool:
        if not self.available():
            log.info("FanGraphs credentials not provided; skipping login")
            return False
        raise NotImplementedError(
            "FanGraphs live login pending credential validation; use load_projections_csv()."
        )
