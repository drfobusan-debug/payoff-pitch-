"""End-to-end-ish tests for cardscout against a tiny in-memory catalog.

The catalog is fed through the real Ledger (a temp SQLite file) so the matcher,
deal finder and forecasts are exercised on the same row shapes the CLI uses.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from cardscout import deals, drops, forecast
from cardscout.market import Group, Product, Tcgcsv
from cardscout.match import Matcher, attach, pick_printing
from cardscout.shops import JUNK_RE, _is_pokemon
from cardscout.store import Ledger, Listing

ET = ZoneInfo("America/New_York")


def _card(pid, gid, name, number, prices):
    return Product(pid, gid, name, name, "", number, "Rare", False, prices=prices)


def _sealed(pid, gid, name, price):
    return Product(pid, gid, name, name, "", None, None, True, prices={"Normal": price})


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    led = Ledger(tmp_path / "t.sqlite")
    led.upsert_groups(
        [
            Group(1, "ME: Mega Evolution Promo", "MEP", "2025-09-26"),
            Group(2, "ME01: Mega Evolution", "ME01", "2025-09-26"),
            Group(3, "ME02: Pitch Black", "ME02", "2025-11-14"),
            Group(4, "Fossil", "FO", "1999-10-10"),
            Group(5, "SV: Prismatic Evolutions", "PRE", "2025-01-17"),
            Group(6, "SWSH01: Sword & Shield Base Set", "SSH", "2020-02-07"),
            Group(7, "My First Battle", "MFB", "2023-11-03"),
        ]
    )
    led.upsert_products(
        [
            _card(10, 1, "Charmander - 038", "038", {"Normal": 33.38}),
            _card(11, 2, "Charmander - 038", "038/132", {"Normal": 0.2, "Reverse Holofoil": 0.6}),
            _card(
                12,
                4,
                "Lapras - 10",
                "10/62",
                {"Unlimited Holofoil": 40.0, "1st Edition Holofoil": 205.0},
            ),
            _sealed(20, 2, "Mega Evolution Elite Trainer Box", 90.0),
            _sealed(21, 2, "Mega Evolution Elite Trainer Box Case", 1000.0),
            _sealed(22, 3, "Pitch Black Elite Trainer Box", 80.0),
            _sealed(23, 2, "Mega Evolution Booster Box", 250.0),
            _sealed(24, 5, "Prismatic Evolutions Booster Bundle", 70.0),
            _sealed(25, 5, "Prismatic Evolutions Mini Tin [Vaporeon]", 25.0),
            _sealed(26, 6, "Sword & Shield Booster Pack", 10.78),
            _sealed(27, 7, "Pikachu", 17.56),
            _card(13, 5, "Umbreon ex - 161/131", "161/131", {"Holofoil": 1400.0}),
        ],
        snapshot="2026-09-01T00:00:00Z",
    )
    return led


@pytest.fixture
def matcher(ledger: Ledger) -> Matcher:
    return Matcher(ledger.products_with_printings(), ledger.group_names())


# -- market feed ------------------------------------------------------------


class _Resp:
    def __init__(self, payload):
        self._p = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class _Session:
    def __init__(self, payloads):
        self.payloads = payloads
        self.headers: dict[str, str] = {}
        self.calls = 0

    def get(self, url, timeout):
        self.calls += 1
        return _Resp(self.payloads[url.rsplit("/3/", 1)[1]])


def test_tcgcsv_parses_cards_and_sealed_and_caches(tmp_path: Path):
    payloads = {
        "99/products": {
            "results": [
                {
                    "productId": 1,
                    "name": "Charmander - 038",
                    "extendedData": [{"name": "Number", "value": "038"}],
                },
                {"productId": 2, "name": "Elite Trainer Box", "extendedData": []},
            ]
        },
        "99/prices": {
            "results": [
                {"productId": 1, "subTypeName": "Normal", "marketPrice": 33.4, "lowPrice": 30},
                {"productId": 2, "subTypeName": "Normal", "marketPrice": 90.0},
                {"productId": 7, "subTypeName": "Normal", "marketPrice": 1.0},
            ]
        },
    }
    sess = _Session(payloads)
    feed = Tcgcsv(session=sess, cache=tmp_path)  # type: ignore[arg-type]
    prods = {p.product_id: p for p in feed.products(99)}
    assert prods[1].number == "038" and not prods[1].is_sealed and prods[1].market == 33.4
    assert prods[2].is_sealed and prods[2].market == 90.0
    assert sess.headers["User-Agent"].startswith("cardscout/")
    feed.products(99)
    assert sess.calls == 2, "second read is served from the on-disk cache"


# -- matching ---------------------------------------------------------------


def test_card_matches_by_number_name_and_set(matcher: Matcher):
    c, printing, conf = matcher.match("Charmander 038/MEP Pokemon Center Promo")
    assert c is not None and c.product_id == 10 and conf >= 0.5
    c, printing, _ = matcher.match("Charmander 038/132 - Mega Evolution Reverse Holo")
    assert c is not None and c.product_id == 11 and printing == "Reverse Holofoil"


def test_card_wrong_name_same_number_is_rejected(matcher: Matcher):
    c, _, _ = matcher.match("Pikachu 038/132 Mega Evolution")
    assert c is None


def test_printing_falls_back_to_unlimited_not_first_edition(matcher: Matcher):
    c, printing, _ = matcher.match("Pokemon Fossil Holo Card - Lapras 10/62")
    assert c is not None and c.product_id == 12
    assert printing == "Unlimited Holofoil"
    _, printing, _ = matcher.match("Pokemon Fossil 1st Edition Holo - Lapras 10/62")
    assert printing == "1st Edition Holofoil"


def test_pick_printing_prefers_offered_tier():
    assert pick_printing(("Normal", "Holofoil"), "Mewtwo holo") == "Holofoil"
    assert pick_printing(("Holofoil",), "Mewtwo") == "Holofoil"
    assert pick_printing((), "anything") == "Normal"


def test_sealed_box_does_not_match_case(matcher: Matcher):
    c, _, conf = matcher.match("Pokemon Mega Evolution Elite Trainer Box")
    assert c is not None and c.product_id == 20 and conf >= 0.5
    c, _, _ = matcher.match("Pokemon Mega Evolution Elite Trainer Box Case (10 ETBs)")
    assert c is not None and c.product_id == 21


def test_sealed_other_set_and_language_are_rejected(matcher: Matcher):
    c, _, _ = matcher.match("Mega Evolution: Pitch Black Elite Trainer Box")
    assert c is not None and c.product_id == 22
    c, _, _ = matcher.match("Pokemon Japanese Mega Evolution Booster Box")
    assert c is None, "a Japanese box must not be priced against the English one"
    c, _, _ = matcher.match("Pokemon Mega Evolution Booster Bundle")
    assert c is None, "no Mega Evolution bundle exists in this catalog"


def test_sealed_series_variant_and_toy_titles(matcher: Matcher):
    # a Scarlet & Violet pack is not a Sword & Shield one, even with identical tokens
    assert matcher.match("Pokemon Scarlet & Violet Booster Pack")[0] is None
    assert matcher.match("Sword & Shield Booster Pack")[0].product_id == 26
    # the bracket names the variant; an unqualified title cannot be priced as one
    assert matcher.match("Prismatic Evolutions Mini Tin")[0] is None
    assert matcher.match("Prismatic Evolutions Mini Tin - Vaporeon")[0].product_id == 25
    # one shared word ("Pikachu") is not a sealed match
    assert matcher.match("Nanoblock Pokemon Series Pikachu")[0] is None


def test_graded_slabs_are_not_priced_off_raw_market(matcher: Matcher):
    assert matcher.match("Umbreon ex 161/131 Prismatic Evolutions")[0].product_id == 13
    assert matcher.match("Umbreon ex 161/131 Prismatic Evolutions PSA 10")[0] is None
    assert matcher.match("Umbreon ex 161/131 CGC 9.5 Graded")[0] is None


def test_junk_and_non_pokemon_filters():
    assert JUNK_RE.search("Evoretro Acrylic Case for Pokemon ETB")
    assert JUNK_RE.search("Digimon Card Game Next Adventure Booster Pack")
    assert not JUNK_RE.search("Pokemon Prismatic Evolutions Booster Bundle")
    assert _is_pokemon({"title": "Pokémon TCG: Scarlet & Violet ETB"})
    assert not _is_pokemon({"title": "Magic Booster Box"})


# -- deals ------------------------------------------------------------------


def test_find_deals_uses_edge_and_confidence(ledger: Ledger, matcher: Matcher):
    listings = [
        Listing("Shop", "Pokemon Mega Evolution Elite Trainer Box", "", 72.0, "u1", True),
        Listing("Shop", "Pokemon Mega Evolution Elite Trainer Box", "", 88.0, "u2", True),
        Listing("Shop", "Pokemon Mega Evolution Booster Box", "", 100.0, "u3", False),
        Listing("Shop", "Charmander 038/MEP Promo", "", 25.0, "u4", True),
    ]
    assert attach(matcher, listings) == 4
    ledger.add_listings(listings, snapshot="2026-09-02T00:00:00Z")
    found = deals.find_deals(ledger.latest_listings(), ledger.latest_market(), min_edge=0.10)
    urls = {d.url for d in found}
    assert urls == {"u1", "u4"}, "88 vs 90 is under 10%; sold-out box is skipped"
    sealed_only = deals.find_deals(ledger.latest_listings(), ledger.latest_market(), sealed=True)
    assert [d.url for d in sealed_only] == ["u1"]
    assert round(sealed_only[0].edge, 2) == 0.20


# -- price forecast ---------------------------------------------------------


def test_forecast_flat_when_history_is_short():
    rows = [(f"2026-09-0{d}T00:00:00Z", 30.0 + d) for d in range(1, 4)]
    f = forecast.forecast(rows, horizon_days=30)
    assert f is not None and f.flat and f.point == f.last == 33.0 and f.direction == "flat"


def test_forecast_follows_a_trend_with_bounds():
    rows = [(f"2026-08-{d:02d}T12:00:00Z", 100.0 * 1.01**d) for d in range(1, 29)]
    f = forecast.forecast(rows, horizon_days=30)
    assert f is not None and not f.flat and f.n == 28
    assert f.direction == "up" and f.point > f.last
    assert f.low <= f.point <= f.high
    assert f.point < f.last * 1.01**30, "trend is damped, not extrapolated forever"


def test_forecast_collapses_same_day_snapshots():
    rows = [("2026-09-01T01:00:00Z", 10.0), ("2026-09-01T09:00:00Z", 12.0)]
    f = forecast.forecast(rows)
    assert f is not None and f.n == 1 and f.last == 12.0
    assert forecast.forecast([]) is None


# -- drops ------------------------------------------------------------------


def _obs(stamp: str, in_stock: bool = True):
    return {"observed_at": stamp, "in_stock": in_stock}


def test_richmond_config_and_priors():
    cfg = drops.load_config()
    assert cfg["home"]["zip"] == "23229"
    names = {s.retailer for s in drops.stores(cfg)}
    assert {"target", "walmart", "costco", "dollartree"} <= names
    assert drops.stores(cfg)[0].miles <= drops.stores(cfg)[-1].miles


def test_target_prior_favours_tuesday_and_friday_early_morning():
    now = datetime(2026, 9, 14, 8, 0, tzinfo=ET)  # a Monday
    windows = drops.forecast_windows("target", [], now=now, top=3)
    assert all(not w.from_data for w in windows)
    days = {w.start.weekday() for w in windows}
    assert days <= {1, 4}, [w.label for w in windows]


def test_observations_reshape_the_forecast():
    now = datetime(2026, 9, 14, 8, 0, tzinfo=ET)
    # thirty Saturday-afternoon (2:30pm ET) sightings at the local Walmart
    sat = datetime(2026, 6, 6, 18, 30, tzinfo=timezone.utc)
    assert sat.weekday() == 5
    obs = [_obs((sat + timedelta(weeks=w)).strftime("%Y-%m-%dT%H:%M:%SZ")) for w in range(10)] * 3
    windows = drops.forecast_windows("walmart", obs, now=now, top=1)
    assert windows[0].from_data and windows[0].start.weekday() == 5
    assert "12" in windows[0].bucket or "afternoon" in windows[0].bucket.lower()


def test_retailer_of_url():
    assert drops.retailer_of("https://www.target.com/p/x/-/A-1") == "target"
    assert drops.retailer_of("https://www.walmart.com/ip/1") == "walmart"


class _Page:
    def __init__(self, status, text):
        self.status_code, self.text = status, text


class _WatchSession:
    headers: dict[str, str] = {}

    def __init__(self, page):
        self.page = page

    def get(self, url, timeout):
        return self.page


def test_check_url_classifies_blocked_out_and_in_stock():
    blocked = drops.check_url("https://x", _WatchSession(_Page(435, "px-cdn captcha")))  # type: ignore[arg-type]
    assert blocked.in_stock is None and "blocked" in blocked.reason
    out = drops.check_url("https://x", _WatchSession(_Page(200, "<p>Sold out</p>")))  # type: ignore[arg-type]
    assert out.in_stock is False
    body = '{"offers": {"availability": "https://schema.org/InStock"}}'
    live = drops.check_url("https://x", _WatchSession(_Page(200, body)))  # type: ignore[arg-type]
    assert live.in_stock is True


def test_stock_signal_prefers_button_state_over_loose_text():
    # Target: reviews say "sold out", the JS bundle says "Add to cart"; the
    # disabled button is the truth
    page = '<p>usually sold out</p><button type="button" disabled="">Add to cart</button>'
    assert drops.stock_signal(page)[0] is False
    page = '<p>usually sold out</p><button type="button" class="x">Add to cart</button>'
    assert drops.stock_signal(page)[0] is True
    page = '<button disabled>Add to cart</button>{"availability":"http://schema.org/InStock"}'
    assert drops.stock_signal(page)[0] is True
    assert drops.stock_signal("<p>hello</p>")[0] is None
