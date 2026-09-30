"""Listings priced under market.

Edge is ``(market - price) / market``. A shop price is what you pay; TCGplayer
market is what the card last traded for, so the edge is against the thing you
could otherwise buy the card at, before shipping.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class Deal:
    shop: str
    title: str
    variant: str
    price: float
    market: float
    url: str
    product_name: str
    set_name: str
    printing: str | None
    is_sealed: bool
    confidence: float

    @property
    def edge(self) -> float:
        return (self.market - self.price) / self.market

    @property
    def saving(self) -> float:
        return self.market - self.price


def find_deals(
    listings: Iterable[sqlite3.Row],
    market: dict[tuple[int, str], float],
    min_edge: float = 0.10,
    min_price: float = 5.0,
    min_confidence: float = 0.5,
    sealed: bool | None = None,
) -> list[Deal]:
    deals: list[Deal] = []
    for li in listings:
        if not li["available"] or li["product_id"] is None:
            continue
        if li["confidence"] < min_confidence:
            continue
        if sealed is not None and bool(li["is_sealed"]) != sealed:
            continue
        key = (li["product_id"], li["printing"] or "Normal")
        mkt = market.get(key)
        if mkt is None:
            # sealed products are priced under "Normal"; cards may only have one printing
            alts = [v for (pid, _), v in market.items() if pid == li["product_id"]]
            mkt = min(alts) if alts else None
        if not mkt or mkt < min_price:
            continue
        d = Deal(
            shop=li["shop"],
            title=li["title"],
            variant=li["variant"] or "",
            price=li["price"],
            market=mkt,
            url=li["url"],
            product_name=li["product_name"] or "",
            set_name=li["set_name"] or "",
            printing=li["printing"],
            is_sealed=bool(li["is_sealed"]),
            confidence=li["confidence"],
        )
        if d.edge >= min_edge:
            deals.append(d)
    deals.sort(key=lambda d: d.saving, reverse=True)
    return deals
