# cardscout -- shop the Pokémon card market

`cardscout` catalogs TCGplayer market prices for every Pokémon single and sealed
product, scrapes independent card shops, flags listings priced under market,
forecasts where a price is heading from the history it has recorded, and
forecasts when Richmond-area big-box stores (ZIP 23229) are most likely to
restock.

Nothing here buys anything. A "deal" is a listing under TCGplayer market, a price
forecast is a damped trend fitted to recorded snapshots, and a drop window is a
probability, not a promise.

## Install

```bash
pip install -e .          # from the repo root; adds the `cardscout` command
```

Data lives in `~/.cardscout/` (override with `CARDSCOUT_DATA_DIR`): a SQLite
ledger of every snapshot plus a cache of the market feed.

## Daily loop

```bash
cardscout sync                      # market prices for all 220 sets + every shop's inventory
cardscout deals                     # listings under market, best edge first
cardscout deals --sealed --min-edge 0.15
cardscout price "charmander 038"    # market price, trend, and any shop listings for a card
cardscout forecast --sealed --days 90
```

Run `sync` daily (cron, Task Scheduler, or by hand). Each run appends a snapshot
rather than overwriting, and the forecasts are only as good as the history: with
fewer than seven daily snapshots a forecast is flat at the last price and says so.

### Where the numbers come from

- **Market price**: TCGplayer's market price via [tcgcsv.com](https://tcgcsv.com),
  a free daily mirror of TCGplayer's catalog and prices. Singles carry a price
  per printing (Normal / Holofoil / Reverse / 1st Edition); sealed products a
  single price.
- **Shops**: any store exposing a Shopify `products.json` feed, listed in
  `cardscout/data/shops.json`. Add a shop by adding its domain; set `"fx_to_usd"`
  for non-USD stores. Shops behind a bot wall are skipped, never worked around.
- **Matching**: shop titles are matched to catalog products by collector number
  plus card name and set, or for sealed goods by product-type agreement (a
  booster box never matches a case, English never matches Japanese) and set
  name. Each match carries a confidence; `deals` shows it when under 100% and
  hides matches under 50%.

### Reading `deals`

```
$49.95 vs $64.01  +22%  Mega Evolution Booster Bundle (ME01: Mega Evolution) sealed
         The Card Vault: Pokémon Mega Evolution Booster Bundle
         https://...
```

`edge = (market - price) / market`. Shop prices exclude shipping and tax, and
for vintage singles a shop's copy may be in worse condition than the NM price
TCGplayer quotes -- check the listing before you buy.

## Drops (retail restock forecasting)

```bash
cardscout drops stores                          # Target/Walmart/Costco/... near 23229
cardscout drops forecast target                 # most likely restock windows, next 7 days
cardscout drops observe target "Target Willow Lawn" "Prismatic ETB"
cardscout drops observe walmart "Walmart Broad St" "Prismatic ETB" --empty
cardscout drops watch https://www.target.com/p/... https://www.walmart.com/ip/...
```

The forecast starts from retailer priors (Target: Tue/Fri overnight-to-morning;
Walmart: Wed overnight and Thu/Fri mornings; Costco/Sam's: bulk arrivals, any
weekday morning; Dollar Tree: the day after the store's freight truck) and
shifts toward what you actually observe. Log every visit, hits and misses,
with `drops observe`; once you have logged more sightings than the prior's
weight, the windows are marked as coming from your data.

`drops watch` reads a product page for an in-stock / out-of-stock signal and logs
it. Target and Walmart usually answer with a bot wall from a datacenter IP; the
watcher reports "blocked" and moves on -- run it from home, or log by hand.

The store list (`cardscout drops stores`) is approximate. Edit
`~/.cardscout/drops.json` (written the first time you run `drops stores`) to add,
remove or correct stores.

## Development

```bash
.venv/bin/ruff check cardscout tests/cardscout
.venv/bin/mypy cardscout
.venv/bin/python -m pytest tests/cardscout
```
