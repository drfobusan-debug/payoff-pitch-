"""Forecast when Pokemon product hits the shelf at big-box stores near home.

Big-box restocks are not published. What is known is the distributor rhythm
(MJ Holding / Excell trucks, Target's early-morning online drops) and what you
see with your own eyes. So the forecast is a prior per retailer -- weekday and
hour-of-day weights encoding the widely reported patterns -- updated with the
sightings logged through ``cardscout drops observe`` and by the URL watcher.
Each logged in-stock sighting adds weight to its weekday/hour bucket, so after
a month of checks the forecast is your stores' schedule, not the internet's.

Times are Eastern (home is Richmond, VA).
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from importlib import resources
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from cardscout.config import BROWSER_AGENT, HOME_TZ, data_dir

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
# Hour buckets: (label, start_hour, end_hour) in local time.
BUCKETS = (
    ("overnight 12-6am", 0, 6),
    ("morning 6-10am", 6, 10),
    ("midday 10am-2pm", 10, 14),
    ("afternoon 2-6pm", 14, 18),
    ("evening 6pm-12am", 18, 24),
)

# Prior weight per (weekday index, bucket index). Anything not listed is 1.
# Sources: Target online drops Tue/Fri ~3-6am ET and shelves stocked before
# open on weekdays; Walmart online Wed overnight, distributor freight tagged
# Wed-Sat so Thu/Fri mornings; Costco/Sam's pallets midweek, unannounced;
# Dollar Tree on the store's weekly truck; Pokemon Center weekday mornings.
PRIORS: dict[str, dict[tuple[int, int], float]] = {
    "target": {
        (1, 0): 8,
        (4, 0): 8,  # Tue / Fri overnight online drops
        (1, 1): 5,
        (2, 1): 3,
        (3, 1): 3,
        (4, 1): 5,
        (0, 1): 2,  # weekday mornings before open
    },
    "walmart": {
        (2, 0): 6,
        (2, 4): 3,  # Wed overnight online
        (3, 1): 6,
        (4, 1): 6,
        (5, 1): 3,  # Thu/Fri/Sat mornings after freight
        (3, 2): 3,
        (4, 2): 3,
    },
    "costco": {(1, 2): 3, (2, 2): 4, (3, 2): 4, (1, 1): 2, (2, 1): 3, (3, 1): 3},
    "samsclub": {(1, 2): 3, (2, 2): 4, (3, 2): 4, (2, 1): 3, (3, 1): 3},
    "bestbuy": {(1, 1): 3, (2, 1): 3, (3, 1): 3, (1, 2): 2, (2, 2): 2, (3, 2): 2},
    "dollartree": {
        (1, 1): 3,
        (2, 1): 3,
        (3, 1): 3,
        (4, 1): 3,
        (1, 2): 2,
        (2, 2): 2,
        (3, 2): 2,
        (4, 2): 2,
    },
    "gamestop": {(1, 2): 3, (4, 2): 3, (1, 3): 2, (4, 3): 2},
    "pokemoncenter": {(0, 2): 3, (1, 2): 3, (2, 2): 3, (3, 2): 4, (4, 2): 3},
}
# How many sightings it takes for the data to weigh as much as the prior.
PRIOR_STRENGTH = 8.0

# Weekend in-store priors are low everywhere: distributors do not run Sunday.
for _r in PRIORS.values():
    for _b in range(len(BUCKETS)):
        _r.setdefault((6, _b), 0.3)


@dataclass
class Store:
    retailer: str
    name: str
    address: str
    miles: float


@dataclass
class Window:
    start: datetime
    bucket: str
    retailer: str
    probability: float  # share of this retailer's weekly stocking weight
    from_data: bool

    @property
    def label(self) -> str:
        return f"{WEEKDAYS[self.start.weekday()]} {self.start:%m/%d} {self.bucket}"


def load_config() -> dict:
    user = data_dir() / "drops.json"
    if user.exists():
        return json.loads(user.read_text())
    return json.loads(resources.files("cardscout.data").joinpath("drops.json").read_text())


def stores(config: dict | None = None, retailer: str | None = None) -> list[Store]:
    cfg = config or load_config()
    out = [
        Store(s["retailer"], s["name"], s.get("address", ""), float(s.get("miles", 0)))
        for s in cfg["stores"]
    ]
    if retailer:
        out = [s for s in out if s.retailer == retailer]
    return sorted(out, key=lambda s: s.miles)


def bucket_index(hour: int) -> int:
    for i, (_, lo, hi) in enumerate(BUCKETS):
        if lo <= hour < hi:
            return i
    return len(BUCKETS) - 1


def _weights(
    retailer: str, observations: Iterable[sqlite3.Row], tz: ZoneInfo
) -> tuple[dict[tuple[int, int], float], int]:
    prior = PRIORS.get(retailer, {})
    total_prior = sum(prior.get((d, b), 1.0) for d in range(7) for b in range(len(BUCKETS)))
    counts: dict[tuple[int, int], float] = {}
    n = 0
    for row in observations:
        if not row["in_stock"]:
            continue
        stamp = datetime.fromisoformat(row["observed_at"].replace("Z", "+00:00")).astimezone(tz)
        key = (stamp.weekday(), bucket_index(stamp.hour))
        counts[key] = counts.get(key, 0.0) + 1.0
        n += 1
    weights: dict[tuple[int, int], float] = {}
    for d in range(7):
        for b in range(len(BUCKETS)):
            p = prior.get((d, b), 1.0) / total_prior
            weights[(d, b)] = PRIOR_STRENGTH * p + counts.get((d, b), 0.0)
    return weights, n


def forecast_windows(
    retailer: str,
    observations: Iterable[sqlite3.Row],
    now: datetime | None = None,
    days: int = 7,
    top: int = 5,
    tz_name: str = HOME_TZ,
) -> list[Window]:
    tz = ZoneInfo(tz_name)
    now = (now or datetime.now(tz)).astimezone(tz)
    weights, n = _weights(retailer, observations, tz)
    total = sum(weights.values())
    out: list[Window] = []
    day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for offset in range(days):
        day = day0 + timedelta(days=offset)
        for b, (label, lo, _hi) in enumerate(BUCKETS):
            start = day.replace(hour=lo)
            if start + timedelta(hours=1) < now:
                continue
            w = weights[(day.weekday(), b)]
            # share of a week's stocking weight that falls in this window
            out.append(Window(start, label, retailer, w / total, n >= PRIOR_STRENGTH))
    out.sort(key=lambda w: w.probability, reverse=True)
    return out[:top]


# -- URL watcher -----------------------------------------------------------

SCHEMA_IN_RE = re.compile(r'"availability"\s*:\s*"[^"]*InStock', re.I)
SCHEMA_OUT_RE = re.compile(
    r'"availability"\s*:\s*"[^"]*(OutOfStock|SoldOut|Discontinued|PreOrder)', re.I
)
# an add-to-cart <button>; the tag attributes tell us whether it is disabled
CART_BUTTON_RE = re.compile(
    r"<button\b([^>]*)>(?:(?!</button>).){0,200}?add to (?:cart|bag)", re.I | re.S
)
IN_STOCK_RE = re.compile(r'add to cart|"instock"|ship it|pick it up', re.I)
OUT_RE = re.compile(
    r"out of stock|sold out|currently unavailable|coming soon",
    re.I,
)
BLOCK_RE = re.compile(r"captcha|perimeterx|px-cdn|access denied|robot", re.I)


def stock_signal(body: str) -> tuple[bool | None, str]:
    """Read a product page. Structured data beats button state beats loose text:
    review text says 'sold out' on pages that are in stock, and JS bundles
    contain 'Add to cart' on pages that are not."""
    if SCHEMA_IN_RE.search(body):
        return True, "in stock (schema.org availability)"
    if SCHEMA_OUT_RE.search(body):
        return False, "out of stock (schema.org availability)"
    buttons = [m.group(1) for m in CART_BUTTON_RE.finditer(body)]
    if buttons:
        if all(re.search(r"\bdisabled\b", attrs, re.I) for attrs in buttons):
            return False, "out of stock (add-to-cart button disabled)"
        return True, "in stock (add-to-cart button enabled)"
    if OUT_RE.search(body) and not IN_STOCK_RE.search(body):
        return False, "out of stock"
    if IN_STOCK_RE.search(body) and not OUT_RE.search(body):
        return True, "in stock"
    return None, "no stock signal on page (rendered by JavaScript?)"


@dataclass
class Check:
    url: str
    status: int
    in_stock: bool | None  # None when the page could not be read
    reason: str


def check_url(url: str, session: requests.Session | None = None) -> Check:
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", BROWSER_AGENT)
    try:
        r = session.get(url, timeout=30)
    except requests.RequestException as exc:
        return Check(url, 0, None, f"request failed: {exc.__class__.__name__}")
    body = r.text
    if r.status_code != 200 or (len(body) < 20000 and BLOCK_RE.search(body)):
        return Check(
            url,
            r.status_code,
            None,
            "blocked (bot wall) -- log sightings by hand or run from a residential IP",
        )
    in_stock, reason = stock_signal(body)
    return Check(url, r.status_code, in_stock, reason)


def retailer_of(url: str) -> str:
    host = re.sub(r"^https?://(www\.)?", "", url).split("/")[0].lower()
    for key in PRIORS:
        if host.startswith(key):
            return key
    return host


def write_user_config(path: Path | None = None) -> Path:
    """Copy the packaged store list to ~/.cardscout/drops.json so it can be edited."""
    dest = path or (data_dir() / "drops.json")
    if not dest.exists():
        dest.write_text(json.dumps(load_config(), indent=2))
    return dest
