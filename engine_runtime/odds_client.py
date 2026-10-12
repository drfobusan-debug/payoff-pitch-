"""One Odds API client for the runtime: raw JSON in, credits debited, headers kept.

Nothing here maps provider markets to engine keys or forms a probability; the
sport's capture module does that from the payload. Every paid request goes
through ``credits.debit`` first and reports the response headers back so the
ledger carries the key's own remaining count.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests

from engine_runtime import credits

log = logging.getLogger(__name__)
BASE = "https://api.the-odds-api.com/v4/sports"


class OddsClient:
    def __init__(
        self,
        sport: str,
        sport_key: str,
        api_key: str | None,
        *,
        regions: str = "us",
        timeout: int = 25,
        cache_dir: Path | None = None,
        cache_ttl: int = 600,
        runtime_root: Path | None = None,
    ) -> None:
        self.sport = sport
        self.sport_key = sport_key
        self.api_key = api_key
        self.regions = regions
        self.timeout = timeout
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        self.runtime_root = runtime_root
        self.credits_remaining: int | None = None
        self.credits_last: int | None = None
        self.spent = 0

    def available(self) -> bool:
        return bool(self.api_key)

    def events(self, **params: str) -> list[dict[str, object]]:
        """``/events`` is free; use it to size a capture before spending."""
        data = self._get(
            f"{BASE}/{self.sport_key}/events", cost=0, endpoint="events", params=dict(params)
        )
        return data if isinstance(data, list) else []

    def board(
        self, markets: tuple[str, ...], *, pass_id: str = "", **params: str
    ) -> list[dict[str, object]]:
        """Featured markets for every posted game: cost = markets x regions."""
        cost = len(markets) * len(self.regions.split(","))
        data = self._get(
            f"{BASE}/{self.sport_key}/odds",
            cost=cost,
            endpoint="odds",
            pass_id=pass_id,
            params={"markets": ",".join(markets), **params},
        )
        return data if isinstance(data, list) else []

    def event(
        self, event_id: str, markets: tuple[str, ...], *, pass_id: str = "", **params: str
    ) -> dict[str, object] | None:
        """Per-event markets: cost = markets x regions for the one event."""
        cost = len(markets) * len(self.regions.split(","))
        data = self._get(
            f"{BASE}/{self.sport_key}/events/{event_id}/odds",
            cost=cost,
            endpoint="event_odds",
            pass_id=pass_id,
            events=1,
            params={"markets": ",".join(markets), **params},
        )
        return data if isinstance(data, dict) else None

    def _cache_path(self, url: str, query: dict[str, str]) -> Path | None:
        if self.cache_dir is None:
            return None
        import hashlib

        key = hashlib.sha1((url + json.dumps(query, sort_keys=True)).encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _get(
        self,
        url: str,
        *,
        cost: int,
        endpoint: str,
        pass_id: str = "",
        events: int = 0,
        params: dict[str, str] | None = None,
    ) -> object:
        if not self.available():
            return None
        params = params or {}
        query = {"regions": self.regions, "oddsFormat": "american", **params}
        cache = self._cache_path(url, query)
        if (
            cache is not None
            and cache.exists()
            and time.time() - cache.stat().st_mtime < self.cache_ttl
        ):
            try:
                return json.loads(cache.read_text())
            except ValueError:
                pass
        if cost and not credits.debit(
            self.sport,
            cost,
            endpoint=endpoint,
            markets=params.get("markets", ""),
            regions=self.regions,
            events=events,
            remaining_header=self.credits_remaining,
            pass_id=pass_id,
            root=self.runtime_root,
        ):
            log.info("skipped %s: credit budget", endpoint)
            return None
        try:
            resp = requests.get(
                url, params={"apiKey": self.api_key or "", **query}, timeout=self.timeout
            )
            remaining = resp.headers.get("x-requests-remaining")
            if remaining is not None:
                self.credits_remaining = int(float(remaining))
            last = resp.headers.get("x-requests-last")
            if last is not None:
                self.credits_last = int(float(last))
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning(
                "Odds API %s failed: %s", endpoint, str(exc).replace(self.api_key or "\0", "***")
            )
            return None
        self.spent += cost
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(payload))
        return payload


__all__ = ["BASE", "OddsClient"]
