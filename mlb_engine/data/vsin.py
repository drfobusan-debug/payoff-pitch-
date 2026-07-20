"""VSIN client: DraftKings/Circa odds and handle/bets betting splits.

Two ingestion paths are supported:

1. CSV drop-in (fully supported today). Export or assemble a CSV and point the
   engine at it; columns:
       matchup, market, selection, book, american, handle_pct, bets_pct
   ``book`` is "draftkings" or "circa"; handle_pct/bets_pct are optional.

2. Authenticated live fetch (requires a VSIN Pro login). The scraping selectors
   depend on VSIN's authenticated pages and must be validated against a live
   account before use; until then the CSV path is the source of truth.

The engine degrades gracefully: with no VSIN data it still emits model
probabilities and fair odds, just without EV/tiering.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from mlb_engine.config import Credentials
from mlb_engine.market.ev import MarketQuote

log = logging.getLogger(__name__)

LOGIN_URL = "https://data.vsin.com/login/"  # validate against live account


class VSINClient:
    def __init__(self, creds: Credentials, timeout: int = 25) -> None:
        self.creds = creds
        self.timeout = timeout
        self._logged_in = False

    def available(self) -> bool:
        return self.creds.has_vsin()

    # --- CSV path (supported today) ---------------------------------------
    def load_csv(self, path: Path) -> dict[tuple[str, str, str], list[MarketQuote]]:
        """Return {(matchup, market, selection): [MarketQuote, ...]}."""
        df = pd.read_csv(path)
        required = {"matchup", "market", "selection", "book", "american"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"VSIN CSV missing columns: {sorted(missing)}")
        out: dict[tuple[str, str, str], list[MarketQuote]] = {}
        for _, row in df.iterrows():
            key = (str(row["matchup"]), str(row["market"]), str(row["selection"]))
            quote = MarketQuote(
                book=str(row["book"]).lower(),
                american=float(row["american"]),
                handle_pct=_opt_float(row.get("handle_pct")),
                bets_pct=_opt_float(row.get("bets_pct")),
            )
            out.setdefault(key, []).append(quote)
        return out

    # --- Live path (needs credential validation) --------------------------
    def login(self) -> bool:
        if not self.available():
            log.info("VSIN credentials not provided; skipping live login")
            return False
        # NOTE: implement against the live authenticated flow once credentials
        # are supplied as secrets. Kept explicit rather than guessed.
        raise NotImplementedError(
            "VSIN live login requires validation against a real account; "
            "use load_csv() until credentials are wired in."
        )

    def fetch_quotes(self, date_iso: str) -> dict[tuple[str, str, str], list[MarketQuote]]:
        if not self.available():
            return {}
        raise NotImplementedError(
            "VSIN live fetch pending credential validation; use load_csv()."
        )


def _opt_float(v: object) -> float | None:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
