"""Rotowire client (optional confirmed lineups / weather fallback).

Confirmed lineups and park weather primarily come from the MLB Stats API and
Open-Meteo; Rotowire is a fallback/enrichment source. CSV drop-in supported
today; authenticated live fetch scaffolded for once credentials are supplied.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from mlb_engine.config import Credentials

log = logging.getLogger(__name__)


class RotowireClient:
    def __init__(self, creds: Credentials, timeout: int = 25) -> None:
        self.creds = creds
        self.timeout = timeout

    def available(self) -> bool:
        return self.creds.has_rotowire()

    def load_lineups_csv(self, path: Path) -> pd.DataFrame:
        """Load a Rotowire expected-lineups CSV (fallback for unposted lineups)."""
        return pd.read_csv(path)

    def bullpen_availability(self, team_abbrev: str) -> float | None:
        """Return a team's bullpen availability (0..1, higher = more rested).

        Rotowire publishes reliever usage/availability (recent pitch counts, who
        is unavailable). Wired as the primary source for the bullpen fatigue
        (3-in-4) NPV; returns ``None`` until live access is validated, in which
        case the Statcast workload proxy is used instead.
        """
        if not self.available():
            return None
        log.info("Rotowire bullpen availability pending live access for %s", team_abbrev)
        return None

    def login(self) -> bool:
        if not self.available():
            log.info("Rotowire credentials not provided; skipping login")
            return False
        raise NotImplementedError(
            "Rotowire live login pending credential validation; use load_lineups_csv()."
        )
