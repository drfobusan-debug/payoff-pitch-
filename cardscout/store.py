"""SQLite ledger: the catalog, the market price snapshots, the shop listings seen.

Every ``sync`` appends one row per product to ``market_prices`` and one row per
shop listing to ``listings``; nothing is overwritten, which is what makes the
forecasts possible. ``latest_*`` views read the newest snapshot.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cardscout.config import db_path
from cardscout.market import Group, Product

SCHEMA = """
CREATE TABLE IF NOT EXISTS groups (
    group_id INTEGER PRIMARY KEY, name TEXT, abbreviation TEXT, published_on TEXT);
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY, group_id INTEGER, name TEXT, clean_name TEXT,
    url TEXT, number TEXT, rarity TEXT, is_sealed INTEGER);
CREATE TABLE IF NOT EXISTS market_prices (
    snapshot TEXT, product_id INTEGER, printing TEXT, market REAL, low REAL,
    PRIMARY KEY (snapshot, product_id, printing));
CREATE TABLE IF NOT EXISTS listings (
    snapshot TEXT, shop TEXT, title TEXT, variant TEXT, price REAL, url TEXT,
    available INTEGER, product_id INTEGER, printing TEXT, confidence REAL);
CREATE INDEX IF NOT EXISTS ix_listings_snapshot ON listings (snapshot);
CREATE INDEX IF NOT EXISTS ix_market_product ON market_prices (product_id, snapshot);
CREATE TABLE IF NOT EXISTS drop_observations (
    observed_at TEXT, retailer TEXT, store TEXT, product TEXT, in_stock INTEGER,
    source TEXT, note TEXT);
"""


def now_snapshot() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Listing:
    shop: str
    title: str
    variant: str
    price: float
    url: str
    available: bool
    product_id: int | None = None
    printing: str | None = None
    confidence: float = 0.0


class Ledger:
    def __init__(self, path: Path | None = None):
        self.path = path or db_path()
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # -- catalog ---------------------------------------------------------
    def upsert_groups(self, groups: Iterable[Group]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO groups VALUES (?,?,?,?)",
            [(g.group_id, g.name, g.abbreviation, g.published_on) for g in groups],
        )
        self.conn.commit()

    def upsert_products(self, products: Iterable[Product], snapshot: str) -> int:
        rows, prices = [], []
        for p in products:
            rows.append(
                (
                    p.product_id,
                    p.group_id,
                    p.name,
                    p.clean_name,
                    p.url,
                    p.number,
                    p.rarity,
                    int(p.is_sealed),
                )
            )
            for printing, market in p.prices.items():
                prices.append((snapshot, p.product_id, printing, market, p.low.get(printing)))
        self.conn.executemany("INSERT OR REPLACE INTO products VALUES (?,?,?,?,?,?,?,?)", rows)
        self.conn.executemany("INSERT OR REPLACE INTO market_prices VALUES (?,?,?,?,?)", prices)
        self.conn.commit()
        return len(prices)

    def products(self, group_ids: Iterable[int] | None = None, sealed: bool | None = None):
        sql = "SELECT * FROM products WHERE 1=1"
        args: list[object] = []
        if group_ids is not None:
            ids = list(group_ids)
            sql += f" AND group_id IN ({','.join('?' * len(ids))})"
            args += ids
        if sealed is not None:
            sql += " AND is_sealed = ?"
            args.append(int(sealed))
        return self.conn.execute(sql, args).fetchall()

    def products_with_printings(self):
        """Every product plus a '|'-joined list of the printings it has been priced in."""
        return self.conn.execute(
            "SELECT p.*, (SELECT GROUP_CONCAT(DISTINCT printing) FROM market_prices m "
            "             WHERE m.product_id = p.product_id) AS printings FROM products p"
        ).fetchall()

    def group_names(self) -> dict[int, str]:
        return {r["group_id"]: r["name"] for r in self.conn.execute("SELECT * FROM groups")}

    def search_products(self, text: str, limit: int = 20):
        """Every word must appear in the product or set name; a number word must match
        the collector number. Exact name hits and newer sets rank first."""
        words = [w for w in re.split(r"[\s,/#()]+", text.strip()) if w]
        clauses: list[str] = []
        args: list[object] = []
        for w in words:
            if w.isdigit():
                clauses.append(
                    "(CAST(SUBSTR(p.number, 1, INSTR(p.number || '/', '/') - 1) AS INTEGER) = ?)"
                )
                args.append(int(w))
            else:
                clauses.append("(p.name LIKE ? OR g.name LIKE ?)")
                args += [f"%{w}%", f"%{w}%"]
        where = " AND ".join(clauses) or "1=1"
        name_words = " ".join(w for w in words if not w.isdigit())
        return self.conn.execute(
            f"SELECT p.*, g.name AS set_name FROM products p JOIN groups g USING (group_id) "
            f"WHERE {where} ORDER BY (LOWER(p.name) = LOWER(?)) DESC, g.published_on DESC LIMIT ?",
            (*args, name_words, limit),
        ).fetchall()

    # -- market history ----------------------------------------------------
    def latest_market(self) -> dict[tuple[int, str], float]:
        rows = self.conn.execute(
            "SELECT product_id, printing, market FROM market_prices m "
            "WHERE snapshot = (SELECT MAX(snapshot) FROM market_prices m2 "
            "                  WHERE m2.product_id = m.product_id AND m2.printing = m.printing)"
        )
        return {(r["product_id"], r["printing"]): r["market"] for r in rows}

    def market_history(self, product_id: int, printing: str | None = None):
        sql = "SELECT snapshot, printing, market FROM market_prices WHERE product_id = ?"
        args: list[object] = [product_id]
        if printing:
            sql += " AND printing = ?"
            args.append(printing)
        return self.conn.execute(sql + " ORDER BY snapshot", args).fetchall()

    # -- listings ----------------------------------------------------------
    def add_listings(self, listings: Iterable[Listing], snapshot: str) -> int:
        rows = [
            (
                snapshot,
                li.shop,
                li.title,
                li.variant,
                li.price,
                li.url,
                int(li.available),
                li.product_id,
                li.printing,
                li.confidence,
            )
            for li in listings
        ]
        self.conn.executemany("INSERT INTO listings VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def latest_listings(self, shop: str | None = None):
        sql = (
            "SELECT l.*, p.name AS product_name, p.is_sealed, g.name AS set_name "
            "FROM listings l LEFT JOIN products p USING (product_id) "
            "LEFT JOIN groups g ON g.group_id = p.group_id "
            "WHERE l.snapshot = (SELECT MAX(snapshot) FROM listings l2 WHERE l2.shop = l.shop)"
        )
        args: list[object] = []
        if shop:
            sql += " AND l.shop = ?"
            args.append(shop)
        return self.conn.execute(sql, args).fetchall()

    def listing_history(self, product_id: int):
        return self.conn.execute(
            "SELECT snapshot, shop, price FROM listings WHERE product_id = ? AND available = 1 "
            "ORDER BY snapshot",
            (product_id,),
        ).fetchall()

    # -- drops -------------------------------------------------------------
    def add_drop_observation(
        self,
        retailer: str,
        store: str,
        product: str,
        in_stock: bool,
        source: str,
        note: str = "",
        observed_at: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO drop_observations VALUES (?,?,?,?,?,?,?)",
            (observed_at or now_snapshot(), retailer, store, product, int(in_stock), source, note),
        )
        self.conn.commit()

    def drop_observations(self, retailer: str | None = None):
        sql = "SELECT * FROM drop_observations"
        args: list[object] = []
        if retailer:
            sql += " WHERE retailer = ?"
            args.append(retailer)
        return self.conn.execute(sql + " ORDER BY observed_at", args).fetchall()
