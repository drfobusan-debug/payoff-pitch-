"""FanGraphs client (optional projection blending).

Supports a CSV drop-in of FanGraphs projections/splits today; authenticated live
fetch is scaffolded and validated once credentials are supplied. Projections, if
present, are blended with the Statcast-derived rates in the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from mlb_engine.config import Credentials

log = logging.getLogger(__name__)


class FanGraphsClient:
    def __init__(self, creds: Credentials, timeout: int = 25) -> None:
        self.creds = creds
        self.timeout = timeout

    def available(self) -> bool:
        return self.creds.has_fangraphs()

    def load_projections_csv(self, path: Path) -> pd.DataFrame:
        """Load a FanGraphs projections CSV (e.g., Depth Charts/ZiPS export)."""
        return pd.read_csv(path)

    def login(self) -> bool:
        if not self.available():
            log.info("FanGraphs credentials not provided; skipping login")
            return False
        raise NotImplementedError(
            "FanGraphs live login pending credential validation; use load_projections_csv()."
        )
