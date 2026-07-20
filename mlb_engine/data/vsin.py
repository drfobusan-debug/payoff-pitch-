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
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

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


@dataclass(frozen=True)
class Split:
    """VSIN handle/bets percentages for a selection (no price attached)."""

    handle_pct: float | None = None
    bets_pct: float | None = None

    @property
    def divergence(self) -> float | None:
        if self.handle_pct is None or self.bets_pct is None:
            return None
        return self.handle_pct - self.bets_pct


class _RawRow(NamedTuple):
    name: str
    spread_line: float | None
    spread_handle: float | None
    spread_bets: float | None
    total_line: float | None
    total_handle: float | None
    total_bets: float | None
    ml_american: float | None
    ml_handle: float | None
    ml_bets: float | None


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
    Quotes = dict[tuple[str, str, str], list[MarketQuote]]
    Splits = dict[tuple[str, str, str], Split]

    def fetch(self, slate: Slate) -> tuple[Quotes, Splits]:
        """Scrape the public VSIN splits pages for the slate.

        Returns ``(quotes, splits)``:
        - ``quotes``: priced ``game_ml`` moneyline quotes (american + handle/bets)
          for each book -- the only market VSIN exposes a price for.
        - ``splits``: handle/bets-only entries for moneyline, run line, and total
          selections (VSIN gives no run-line/total price, so these carry no EV;
          they surface the public/sharp split and feed the run-line PPV layer).

        Each VSIN row is matched to a slate team by normalized full name. On the
        total, VSIN lists the visitor row as the Over and the home row as the
        Under. Returns empty mappings if the pages are unavailable.
        """
        name_to_team: dict[str, tuple[str, str, bool]] = {}
        for g in slate.games:
            for tm in (g.home, g.away):
                name_to_team[_norm_name(tm.name)] = (g.matchup(), tm.abbrev, tm.is_home)

        quotes: VSINClient.Quotes = {}
        splits: VSINClient.Splits = {}
        for src, book in _BOOKS.items():
            for row in self._fetch_book(src):
                match = name_to_team.get(_norm_name(row.name))
                if match is None:
                    continue
                matchup, abbrev, is_home = match
                if row.ml_american is not None:
                    quotes.setdefault((matchup, "game_ml", f"{abbrev} ML"), []).append(
                        MarketQuote(
                            book=book, american=row.ml_american,
                            handle_pct=row.ml_handle, bets_pct=row.ml_bets,
                        )
                    )
                    splits[(matchup, "game_ml", f"{abbrev} ML")] = Split(row.ml_handle, row.ml_bets)
                if row.spread_line is not None:
                    sel = f"{abbrev} {row.spread_line:+.1f}"
                    splits[(matchup, "game_rl", sel)] = Split(row.spread_handle, row.spread_bets)
                if row.total_line is not None:
                    side = "Under" if is_home else "Over"
                    sel = f"{side} {row.total_line}"
                    splits[(matchup, "game_total", sel)] = Split(row.total_handle, row.total_bets)
        return quotes, splits

    def fetch_quotes(self, slate: Slate) -> Quotes:
        """Backwards-compatible accessor for just the priced moneyline quotes."""
        return self.fetch(slate)[0]

    def _fetch_book(self, src: str) -> list[_RawRow]:
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
        rows: list[_RawRow] = []
        for _, r in tables[0].iterrows():
            name = str(r.iloc[1]).strip()
            if not name or name.lower() == "nan":
                continue
            rows.append(_RawRow(
                name=name,
                spread_line=_num(r.iloc[2]), spread_handle=_pct(r.iloc[3]), spread_bets=_pct(r.iloc[4]),
                total_line=_num(r.iloc[5]), total_handle=_pct(r.iloc[6]), total_bets=_pct(r.iloc[7]),
                ml_american=_american(r.iloc[8]), ml_handle=_pct(r.iloc[9]), ml_bets=_pct(r.iloc[10]),
            ))
        return rows


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def _pct(v: object) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", str(v))
    return float(m.group()) if m else None


def _num(v: object) -> float | None:
    m = re.search(r"[+-]?\d+(?:\.\d+)?", str(v).replace(" ", ""))
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
