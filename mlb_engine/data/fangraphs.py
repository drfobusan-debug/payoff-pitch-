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
from mlb_engine.features.rolling import OutcomeRates

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
