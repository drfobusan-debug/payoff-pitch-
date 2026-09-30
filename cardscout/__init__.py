"""``cardscout`` -- shop the Pokemon card market the way ``lineshop`` shops the board.

The market price is TCGplayer's, read through tcgcsv.com (a free daily mirror of
TCGplayer's catalog and prices for every set, singles and sealed alike). Shop
inventories come from any store that exposes a Shopify ``products.json`` plus a
generic JSON-LD fallback. Every sync writes a snapshot to SQLite, so the price
history the forecasts fit is the history this tool has seen.

Nothing here buys anything: a deal is a listing priced under market, a forecast
is a trend fitted to recorded prices, and a drop window is a probability, not a
promise.
"""

__version__ = "0.1.0"
