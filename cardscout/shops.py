"""Shop inventory scrapers.

Two kinds:

* ``shopify`` -- any Shopify storefront publishes ``/products.json`` (250 per
  page, every variant with price and availability). Most independent card
  shops are Shopify, so a new shop is usually one line in ``data/shops.json``.
* ``jsonld`` -- for a single product page on any site that embeds
  ``schema.org/Product`` markup (Target, Walmart and most big-box sites do,
  when they let a script read the page at all).

Prices come back in USD; shops in another currency carry an ``fx_to_usd``
multiplier in their config.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import requests

from cardscout.config import BROWSER_AGENT
from cardscout.store import Listing

POKEMON_RE = re.compile(r"pok[eé]mon|pokemon|\bptcg\b|\btcg\b", re.I)
# Product types or titles that are not cards or sealed card product.
JUNK_RE = re.compile(
    r"sleeve|deck box|playmat|binder(?! collection)|toploader|storage|figure|plush|"
    r"funko|t-shirt|hoodie|poster(?! collection)|keychain|pin\b|mug|sticker(?! collection)|"
    r"dice|token|event|tournament|entry|ticket|gift card|grading|submission|"
    r"acrylic|magnetic|display case|protector|"
    r"digimon|yu-?gi-?oh|one piece|magic: the gathering|\bmtg\b|lorcana|dragon ball|"
    r"weiss|union arena|flesh and blood|star wars|riftbound",
    re.I,
)


@dataclass
class Shop:
    name: str
    kind: str
    url: str
    currency: str = "USD"
    fx_to_usd: float = 1.0
    pokemon_filter: bool = True
    collection: str | None = None
    enabled: bool = True


def load_shops(path: Path | None = None) -> list[Shop]:
    if path is not None:
        raw = json.loads(path.read_text())
    else:
        raw = json.loads(resources.files("cardscout.data").joinpath("shops.json").read_text())
    shops = []
    for s in raw["shops"]:
        shops.append(
            Shop(
                name=s["name"],
                kind=s.get("kind", "shopify"),
                url=s["url"].rstrip("/"),
                currency=s.get("currency", "USD"),
                fx_to_usd=float(s.get("fx_to_usd", 1.0)),
                pokemon_filter=bool(s.get("pokemon_filter", True)),
                collection=s.get("collection"),
                enabled=bool(s.get("enabled", True)),
            )
        )
    return shops


def _is_pokemon(product: dict) -> bool:
    blob = " ".join(
        [
            product.get("title") or "",
            product.get("product_type") or "",
            product.get("vendor") or "",
            " ".join(product.get("tags") or []),
        ]
    )
    return bool(POKEMON_RE.search(blob))


def shopify_listings(
    shop: Shop,
    session: requests.Session | None = None,
    max_pages: int = 40,
    sleep: float = 0.5,
) -> list[Listing]:
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", BROWSER_AGENT)
    base = shop.url
    if shop.collection:
        base = f"{base}/collections/{shop.collection}"
    out: list[Listing] = []
    for page in range(1, max_pages + 1):
        r = session.get(f"{base}/products.json", params={"limit": 250, "page": page}, timeout=30)
        if r.status_code != 200:
            break
        products = r.json().get("products") or []
        if not products:
            break
        for p in products:
            if shop.pokemon_filter and not _is_pokemon(p):
                continue
            if JUNK_RE.search((p.get("title") or "") + " " + (p.get("product_type") or "")):
                continue
            url = f"{shop.url}/products/{p['handle']}"
            for v in p.get("variants") or []:
                try:
                    price = float(v["price"]) * shop.fx_to_usd
                except (KeyError, TypeError, ValueError):
                    continue
                if price <= 0:
                    continue
                variant = v.get("title") or ""
                out.append(
                    Listing(
                        shop=shop.name,
                        title=p["title"],
                        variant="" if variant == "Default Title" else variant,
                        price=round(price, 2),
                        url=url,
                        available=bool(v.get("available", True)),
                    )
                )
        if len(products) < 250:
            break
        time.sleep(sleep)
    return out


def jsonld_listing(
    shop_name: str, url: str, session: requests.Session | None = None
) -> Listing | None:
    """Read one product page's schema.org Product block. None when blocked or absent."""
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", BROWSER_AGENT)
    r = session.get(url, timeout=30)
    if r.status_code != 200:
        return None
    for m in re.finditer(
        r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>', r.text, re.S
    ):
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        for node in data if isinstance(data, list) else [data]:
            node = node.get("@graph", [node])[0] if isinstance(node, dict) else node
            if not isinstance(node, dict) or node.get("@type") not in ("Product", ["Product"]):
                continue
            offer = node.get("offers") or {}
            if isinstance(offer, list):
                offer = offer[0] if offer else {}
            try:
                price = float(str(offer.get("price") or offer.get("lowPrice")).replace(",", ""))
            except (TypeError, ValueError):
                continue
            avail = str(offer.get("availability") or "").lower()
            return Listing(
                shop=shop_name,
                title=node.get("name") or "",
                variant="",
                price=price,
                url=url,
                available="instock" in avail.replace(" ", ""),
            )
    return None
