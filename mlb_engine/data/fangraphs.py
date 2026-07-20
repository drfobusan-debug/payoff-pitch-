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
class MetricValues:
    """One tail metric's values, keyed by MLBAM id when present, else by name."""

    by_id: dict[int, float] = field(default_factory=dict)
    by_name: dict[str, float] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.by_id or self.by_name)


@dataclass(frozen=True)
class FanGraphsTail:
    """FanGraphs tail metrics from a custom-report export (CSV or XLSX).

    Each metric prefers exact MLBAM-id matching (FanGraphs exports include an
    ``MLBAMID`` column) and falls back to name matching when that column is
    absent.
    """

    siera: MetricValues = field(default_factory=MetricValues)
    stuff_plus: MetricValues = field(default_factory=MetricValues)
    wrc_plus: MetricValues = field(default_factory=MetricValues)
    xslg: MetricValues = field(default_factory=MetricValues)

    def is_empty(self) -> bool:
        return not (self.siera or self.stuff_plus or self.wrc_plus or self.xslg)


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    norm_cols = {re.sub(r"[^a-z0-9+]", "", str(c).lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9+]", "", cand)
        if key in norm_cols:
            return norm_cols[key]
    return None


def _read_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(path)
        return pd.read_csv(path)
    except Exception as exc:  # optional enrichment
        log.warning("FanGraphs tail export unreadable (%s): %s", path, exc)
        return None


def load_fangraphs_tail_csv(path: Path) -> FanGraphsTail:
    """Parse a FanGraphs custom-report export (CSV/XLSX) into tail metrics.

    Tolerant of column spelling/ordering; an ``MLBAMID`` column (preferred) or a
    ``Name`` column plus any SIERA/Stuff+/wRC+/xSLG columns are auto-detected.
    Returns an empty ``FanGraphsTail`` (all neutral) on any failure.
    """
    df = _read_table(path)
    if df is None:
        return FanGraphsTail()

    id_col = _find_column(df, ("mlbamid", "mlbam", "mlbid", "mlb_id"))
    name_col = _find_column(df, ("name", "player", "playername"))
    if id_col is None and name_col is None:
        log.warning("FanGraphs tail export has no MLBAMID/Name column (have %s)",
                    list(df.columns)[:8])
        return FanGraphsTail()

    out: dict[str, MetricValues] = {}
    for metric, candidates in _TAIL_COLUMNS.items():
        col = _find_column(df, candidates)
        if col is None:
            out[metric] = MetricValues()
            continue
        by_id: dict[int, float] = {}
        by_name: dict[str, float] = {}
        for _, row in df.iterrows():
            try:
                val = float(row[col])
            except (TypeError, ValueError):
                continue
            if val != val:  # NaN
                continue
            pid = _as_int(row[id_col]) if id_col is not None else None
            if pid is not None:
                by_id[pid] = val
            elif name_col is not None:
                by_name[norm_person(row[name_col])] = val
        out[metric] = MetricValues(by_id=by_id, by_name=by_name)
    return FanGraphsTail(
        siera=out["siera"], stuff_plus=out["stuff_plus"],
        wrc_plus=out["wrc_plus"], xslg=out["xslg"],
    )


def _as_int(v: object) -> int | None:
    try:
        f = float(str(v))
    except (TypeError, ValueError):
        return None
    return int(f) if f == f else None


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
