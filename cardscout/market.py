"""TCGplayer market prices through tcgcsv.com.

tcgcsv republishes TCGplayer's catalog once a day: every group (set), every
product in it (singles and sealed), and the low/mid/high/market price per
printing. It asks callers to identify themselves with a real User-Agent and to
fetch once a day, which is what ``sync`` does.

A card is a product with a ``Number`` in its extended data; everything else in
a group (booster boxes, ETBs, collections, bundles) is sealed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from cardscout.config import POKEMON_CATEGORY, TCGCSV_BASE, USER_AGENT, cache_dir

CACHE_TTL = 20 * 3600


@dataclass(frozen=True)
class Group:
    group_id: int
    name: str
    abbreviation: str
    published_on: str  # YYYY-MM-DD


@dataclass
class Product:
    product_id: int
    group_id: int
    name: str
    clean_name: str
    url: str
    number: str | None  # "038/131" style for cards, None for sealed
    rarity: str | None
    is_sealed: bool
    prices: dict[str, float] = field(default_factory=dict)  # printing -> market price
    low: dict[str, float] = field(default_factory=dict)

    @property
    def market(self) -> float | None:
        """Market price of the cheapest normal printing (Normal > Holofoil > others)."""
        for key in ("Normal", "Holofoil", "Reverse Holofoil"):
            if key in self.prices:
                return self.prices[key]
        return min(self.prices.values()) if self.prices else None

    @property
    def short_number(self) -> str | None:
        return self.number.split("/")[0].lstrip("0") or "0" if self.number else None


class Tcgcsv:
    def __init__(self, session: requests.Session | None = None, cache: Path | None = None):
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.cache = cache or cache_dir()

    def _get(self, path: str, ttl: int = CACHE_TTL) -> list[dict]:
        cached = self.cache / (path.replace("/", "_") + ".json")
        if cached.exists() and time.time() - cached.stat().st_mtime < ttl:
            return json.loads(cached.read_text())["results"]
        r = self.session.get(f"{TCGCSV_BASE}/{path}", timeout=60)
        r.raise_for_status()
        payload = r.json()
        cached.write_text(json.dumps(payload))
        return payload["results"]

    def groups(self) -> list[Group]:
        out = []
        for g in self._get(f"{POKEMON_CATEGORY}/groups"):
            out.append(
                Group(
                    group_id=g["groupId"],
                    name=g["name"],
                    abbreviation=g.get("abbreviation") or "",
                    published_on=(g.get("publishedOn") or "")[:10],
                )
            )
        return out

    def products(self, group_id: int) -> list[Product]:
        raw = self._get(f"{POKEMON_CATEGORY}/{group_id}/products")
        prices = self._get(f"{POKEMON_CATEGORY}/{group_id}/prices")
        by_id: dict[int, Product] = {}
        for p in raw:
            ext = {e["name"]: e.get("value") for e in p.get("extendedData", [])}
            number = ext.get("Number")
            by_id[p["productId"]] = Product(
                product_id=p["productId"],
                group_id=group_id,
                name=p["name"],
                clean_name=p.get("cleanName") or p["name"],
                url=p.get("url") or "",
                number=number,
                rarity=ext.get("Rarity"),
                is_sealed=number is None,
            )
        for pr in prices:
            prod = by_id.get(pr["productId"])
            if prod is None:
                continue
            printing = pr.get("subTypeName") or "Normal"
            if pr.get("marketPrice") is not None:
                prod.prices[printing] = float(pr["marketPrice"])
            if pr.get("lowPrice") is not None:
                prod.low[printing] = float(pr["lowPrice"])
        return list(by_id.values())


def recent_groups(groups: list[Group], since: str) -> list[Group]:
    """Groups published on or after ``since`` (YYYY-MM-DD), newest first."""
    picked = [g for g in groups if g.published_on and g.published_on >= since]
    return sorted(picked, key=lambda g: g.published_on, reverse=True)
