"""VSIN client: DraftKings/Circa moneyline odds and handle/bets betting splits.

Two ingestion paths are supported:

1. Live public fetch (default). VSIN's betting-splits page is served publicly at
   ``data.vsin.com/betting-splits/`` and carries, per game, each team's
   moneyline price plus handle%/bets% for the spread, total, and moneyline. No
   login is required for this page; ``fetch_quotes`` scrapes it and returns
   moneyline quotes keyed to the engine's selections. VSIN's public splits do
   **not** expose run-line or total prices, so only moneyline EV is derived from
   it -- run-line/total prices must come from the CSV drop-in.

2. CSV drop-in (for run-line/total prices or manual overrides). Point the engine
   at a CSV with columns:
       matchup, market, selection, book, american, handle_pct, bets_pct
   ``book`` is "draftkings" or "circa"; handle_pct/bets_pct are optional.

The engine degrades gracefully: with no VSIN data it still emits model
probabilities and fair odds, just without EV/tiering.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import pandas as pd
import requests

from mlb_engine.config import Credentials
from mlb_engine.market.ev import MarketQuote
from mlb_engine.schemas import Slate

log = logging.getLogger(__name__)

SPLITS_URL = "https://data.vsin.com/betting-splits/?source={book}&sport=MLB"
_HEADERS = {"User-Agent": "Mozilla/5.0 (mlb-prediction-engine)"}
# VSIN "source" code -> engine book label.
_BOOKS = {"DK": "draftkings", "circa": "circa"}


class VSINClient:
    def __init__(self, creds: Credentials, timeout: int = 25) -> None:
        self.creds = creds
        self.timeout = timeout

    def available(self) -> bool:
        return self.creds.has_vsin()

    # --- CSV path ---------------------------------------------------------
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

    # --- Live public splits path -----------------------------------------
    def fetch_quotes(self, slate: Slate) -> dict[tuple[str, str, str], list[MarketQuote]]:
        """Scrape the public VSIN splits pages -> moneyline quotes for the slate.

        Matches each VSIN team row to a slate team by normalized full name and
        emits a ``game_ml`` quote (american price + moneyline handle/bets) for
        each book. Returns an empty dict if the pages are unavailable.
        """
        name_to_team: dict[str, tuple[str, str]] = {}
        for g in slate.games:
            for tm in (g.home, g.away):
                name_to_team[_norm_name(tm.name)] = (g.matchup(), tm.abbrev)

        out: dict[tuple[str, str, str], list[MarketQuote]] = {}
        for src, book in _BOOKS.items():
            for name, american, hnd, bet in self._fetch_book(src):
                match = name_to_team.get(_norm_name(name))
                if match is None or american is None:
                    continue
                matchup, abbrev = match
                qkey = (matchup, "game_ml", f"{abbrev} ML")
                out.setdefault(qkey, []).append(
                    MarketQuote(book=book, american=american, handle_pct=hnd, bets_pct=bet)
                )
        return out

    def _fetch_book(self, src: str) -> list[tuple[str, float | None, float | None, float | None]]:
        """Return (team_name, ml_american, ml_handle_pct, ml_bets_pct) rows."""
        url = SPLITS_URL.format(book=src)
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=self.timeout)
            resp.raise_for_status()
            tables = pd.read_html(io.StringIO(resp.text))
        except (requests.RequestException, ValueError) as exc:
            log.warning("VSIN splits fetch failed for %s: %s", src, exc)
            return []
        if not tables:
            return []
        df = tables[0]
        rows: list[tuple[str, float | None, float | None, float | None]] = []
        for _, r in df.iterrows():
            name = str(r.iloc[1]).strip()
            if not name or name.lower() == "nan":
                continue
            rows.append((name, _american(r.iloc[8]), _pct(r.iloc[9]), _pct(r.iloc[10])))
        return rows


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def _pct(v: object) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", str(v))
    return float(m.group()) if m else None


def _american(v: object) -> float | None:
    m = re.search(r"[+-]?\d+", str(v).replace(" ", ""))
    return float(m.group()) if m else None


def _opt_float(v: object) -> float | None:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
