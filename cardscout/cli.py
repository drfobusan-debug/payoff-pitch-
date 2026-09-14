"""``cardscout`` -- find Pokemon cards and sealed product priced under market.

    cardscout sync                       # pull TCGplayer market + scrape shops (run daily)
    cardscout deals                      # shop listings under market, biggest saving first
    cardscout deals --sealed --min-edge 0.15
    cardscout price "charmander 038"     # market price + forecast for a card or box
    cardscout forecast --sealed --days 90
    cardscout drops forecast target      # next likely restock windows near 23229
    cardscout drops observe target "Target Willow Lawn" "Prismatic ETB"
    cardscout drops watch URL [URL ...]  # poll product pages, log stock changes
    cardscout drops stores

Market is TCGplayer via tcgcsv.com. Shops are the Shopify stores listed in
cardscout/data/shops.json (add your own with --shops FILE). Everything is
recorded in ~/.cardscout/cardscout.sqlite; the forecasts are fitted to that
history, so they get better the longer sync has been running.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from cardscout import deals as deals_mod
from cardscout import drops as drops_mod
from cardscout import forecast as fc
from cardscout import market as market_mod
from cardscout import shops as shops_mod
from cardscout.match import Matcher, attach
from cardscout.store import Ledger, now_snapshot


def _money(v: float | None) -> str:
    return f"${v:,.2f}" if v is not None else "  --  "


# -- sync ---------------------------------------------------------------------


def cmd_sync(args: argparse.Namespace) -> int:
    ledger = Ledger()
    snapshot = now_snapshot()
    client = market_mod.Tcgcsv()
    groups = client.groups()
    ledger.upsert_groups(groups)
    picked = groups
    if args.since_days:
        since = (datetime.now(timezone.utc) - timedelta(days=args.since_days)).strftime("%Y-%m-%d")
        picked = market_mod.recent_groups(groups, since)
    print(f"market: {len(picked)} sets", file=sys.stderr)
    n_prices = 0
    for g in picked:
        try:
            products = client.products(g.group_id)
        except requests.RequestException as exc:
            print(f"  {g.name}: {exc}", file=sys.stderr)
            continue
        n_prices += ledger.upsert_products(products, snapshot)
    print(f"market: {n_prices} prices recorded", file=sys.stderr)

    if args.no_shops:
        return 0
    shops = shops_mod.load_shops(Path(args.shops) if args.shops else None)
    matcher = Matcher(ledger.products_with_printings(), ledger.group_names())
    session = requests.Session()
    for shop in shops:
        if not shop.enabled or (args.shop and shop.name != args.shop):
            continue
        try:
            listings = shops_mod.listings_for(shop, session)
        except (requests.RequestException, ValueError) as exc:
            print(f"  {shop.name}: {exc}", file=sys.stderr)
            continue
        matched = attach(matcher, listings)
        ledger.add_listings(listings, snapshot)
        print(f"  {shop.name}: {len(listings)} listings, {matched} matched", file=sys.stderr)
    return 0


# -- deals --------------------------------------------------------------------


def cmd_deals(args: argparse.Namespace) -> int:
    ledger = Ledger()
    sealed = True if args.sealed else False if args.singles else None
    found = deals_mod.find_deals(
        ledger.latest_listings(args.shop),
        ledger.latest_market(),
        min_edge=args.min_edge,
        min_price=args.min_price,
        min_confidence=args.min_confidence,
        sealed=sealed,
    )
    if not found:
        print("no listings under market (run `cardscout sync` first, or lower --min-edge)")
        return 0
    print(f"{len(found)} listings under TCGplayer market (edge = saving / market)\n")
    for d in found[: args.limit]:
        kind = "sealed" if d.is_sealed else (d.printing or "card")
        conf = "" if d.confidence >= 0.8 else f"  [match {d.confidence:.0%}]"
        print(
            f"{_money(d.price):>10} vs {_money(d.market):>10}  {d.edge:+.0%}  "
            f"{d.product_name} ({d.set_name}) {kind}{conf}"
        )
        print(f"{'':>10}    {d.shop}: {d.title} {d.variant}".rstrip())
        print(f"{'':>10}    {d.url}")
    return 0


# -- price / forecast -----------------------------------------------------------


def _print_forecast(ledger: Ledger, row, days: int) -> None:
    market = ledger.latest_market()
    printings = sorted(k[1] for k in market if k[0] == row["product_id"])
    for printing in printings or ["Normal"]:
        hist = [
            (r["snapshot"], r["market"]) for r in ledger.market_history(row["product_id"], printing)
        ]
        f = fc.forecast(hist, days)
        if f is None:
            continue
        tag = f" [{printing}]" if len(printings) > 1 else ""
        line = f"  market {_money(f.last)}{tag}"
        if f.flat:
            line += f"  ({f.n} snapshot{'s' if f.n != 1 else ''}; need {fc.MIN_POINTS} days for a trend)"
        else:
            line += (
                f"  {days}d forecast {_money(f.point)} ({f.change:+.0%}, {f.direction}) "
                f"range {_money(f.low)}-{_money(f.high)}  trend {f.daily_trend * 100:+.2f}%/day n={f.n}"
            )
        print(line)


def cmd_price(args: argparse.Namespace) -> int:
    ledger = Ledger()
    rows = ledger.search_products(args.query, limit=args.limit)
    if not rows:
        print("nothing in the catalog matches; run `cardscout sync` first")
        return 1
    for row in rows:
        num = f" {row['number']}" if row["number"] else ""
        kind = "sealed" if row["is_sealed"] else row["rarity"] or "card"
        print(f"{row['name']}{num} -- {row['set_name']} ({kind})")
        _print_forecast(ledger, row, args.days)
        if row["url"]:
            print(f"  {row['url']}")
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    """Rank the catalog by forecast move; needs MIN_POINTS days of sync history."""
    ledger = Ledger()
    market = ledger.latest_market()
    sealed = True if args.sealed else False if args.singles else None
    products = {r["product_id"]: r for r in ledger.products(sealed=sealed)}
    names = ledger.group_names()
    ranked = []
    for (pid, printing), price in market.items():
        if pid not in products or price < args.min_price:
            continue
        hist = [(r["snapshot"], r["market"]) for r in ledger.market_history(pid, printing)]
        f = fc.forecast(hist, args.days)
        if f is None or f.flat:
            continue
        ranked.append((f, products[pid], printing))
    if not ranked:
        print(
            f"no product has {fc.MIN_POINTS}+ days of history yet; keep running `cardscout sync` daily"
        )
        return 0
    ranked.sort(key=lambda t: t[0].change, reverse=not args.drops)
    print(f"{'movers down' if args.drops else 'movers up'} over {args.days} days")
    for f, p, printing in ranked[: args.limit]:
        print(
            f"{_money(f.last):>10} -> {_money(f.point):>10} ({f.change:+.0%})  "
            f"{p['name']} ({names.get(p['group_id'], '')}) {printing}"
        )
    return 0


# -- drops ------------------------------------------------------------------------


def cmd_drops_stores(args: argparse.Namespace) -> int:
    cfg = drops_mod.load_config()
    home = cfg["home"]
    print(f"home: {home['city']} {home['zip']} (area code {home['area_code']})")
    for s in drops_mod.stores(cfg, args.retailer):
        print(f"  {s.miles:>4.0f} mi  {s.retailer:<12} {s.name} -- {s.address}")
    print(f"\n{cfg.get('note', '')}")
    print(f"edit: {drops_mod.write_user_config()}")
    return 0


def cmd_drops_forecast(args: argparse.Namespace) -> int:
    ledger = Ledger()
    retailers = [args.retailer] if args.retailer else list(drops_mod.PRIORS)
    for retailer in retailers:
        obs = ledger.drop_observations(retailer)
        windows = drops_mod.forecast_windows(retailer, obs, days=args.days, top=args.top)
        n_hits = sum(1 for o in obs if o["in_stock"])
        basis = (
            f"{n_hits} logged sightings + reported pattern"
            if n_hits
            else "reported pattern only (log sightings to localize)"
        )
        print(f"{retailer}: {basis}")
        for w in windows:
            print(f"  {w.probability:5.1%}  {w.label}")
        near = drops_mod.stores(retailer=retailer)
        if near:
            print("  check: " + "; ".join(s.name for s in near[:3]))
    return 0


def cmd_drops_observe(args: argparse.Namespace) -> int:
    ledger = Ledger()
    ledger.add_drop_observation(
        args.retailer, args.store, args.product, not args.empty, "manual", args.note or ""
    )
    state = "empty shelf" if args.empty else "in stock"
    print(f"logged: {args.retailer} / {args.store} / {args.product} -- {state}")
    return 0


def cmd_drops_watch(args: argparse.Namespace) -> int:
    ledger = Ledger()
    session = requests.Session()
    last: dict[str, bool | None] = {}
    while True:
        for url in args.urls:
            chk = drops_mod.check_url(url, session)
            retailer = drops_mod.retailer_of(url)
            stamp = datetime.now().strftime("%H:%M")
            if chk.in_stock is None:
                print(f"{stamp}  ?  {retailer}: {chk.reason}  {url}")
            else:
                mark = "IN STOCK" if chk.in_stock else "out"
                print(f"{stamp}  {mark:<8} {retailer}  {url}")
                if last.get(url) != chk.in_stock:
                    ledger.add_drop_observation(
                        retailer, "online", url, chk.in_stock, "watch", chk.reason
                    )
            last[url] = chk.in_stock
        if args.once:
            return 0
        time.sleep(args.every)


# -- parser ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cardscout", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sync", help="pull market prices and shop inventories")
    s.add_argument("--since-days", type=int, help="only sets released within N days (default: all)")
    s.add_argument("--shops", help="JSON file of shops (default: packaged list)")
    s.add_argument("--shop", help="only this shop")
    s.add_argument("--no-shops", action="store_true", help="market only")
    s.set_defaults(func=cmd_sync)

    d = sub.add_parser("deals", help="listings priced under market")
    d.add_argument("--min-edge", type=float, default=0.10)
    d.add_argument("--min-price", type=float, default=5.0, help="ignore market prices below this")
    d.add_argument("--min-confidence", type=float, default=0.5)
    d.add_argument("--sealed", action="store_true")
    d.add_argument("--singles", action="store_true")
    d.add_argument("--shop")
    d.add_argument("--limit", type=int, default=40)
    d.set_defaults(func=cmd_deals)

    pr = sub.add_parser("price", help="market price and forecast for a product")
    pr.add_argument("query")
    pr.add_argument("--days", type=int, default=30)
    pr.add_argument("--limit", type=int, default=8)
    pr.set_defaults(func=cmd_price)

    f = sub.add_parser("forecast", help="biggest forecast movers")
    f.add_argument("--days", type=int, default=30)
    f.add_argument("--sealed", action="store_true")
    f.add_argument("--singles", action="store_true")
    f.add_argument("--drops", action="store_true", help="rank by forecast fall instead of rise")
    f.add_argument("--min-price", type=float, default=5.0)
    f.add_argument("--limit", type=int, default=25)
    f.set_defaults(func=cmd_forecast)

    dr = sub.add_parser("drops", help="retail restock forecasting near home")
    dsub = dr.add_subparsers(dest="drops_cmd", required=True)
    st = dsub.add_parser("stores", help="stores near 23229")
    st.add_argument("retailer", nargs="?")
    st.set_defaults(func=cmd_drops_stores)
    fo = dsub.add_parser("forecast", help="next likely restock windows")
    fo.add_argument("retailer", nargs="?", choices=sorted(drops_mod.PRIORS))
    fo.add_argument("--days", type=int, default=7)
    fo.add_argument("--top", type=int, default=5)
    fo.set_defaults(func=cmd_drops_forecast)
    ob = dsub.add_parser("observe", help="log what you saw on the shelf")
    ob.add_argument("retailer", choices=sorted(drops_mod.PRIORS))
    ob.add_argument("store")
    ob.add_argument("product")
    ob.add_argument("--empty", action="store_true", help="shelf was empty")
    ob.add_argument("--note")
    ob.set_defaults(func=cmd_drops_observe)
    wa = dsub.add_parser("watch", help="poll product URLs for stock changes")
    wa.add_argument("urls", nargs="+")
    wa.add_argument("--every", type=int, default=300, help="seconds between polls")
    wa.add_argument("--once", action="store_true")
    wa.set_defaults(func=cmd_drops_watch)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, PermissionError, NotADirectoryError) as exc:
        print(f"cardscout: {exc} (check CARDSCOUT_DATA_DIR)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
