"""College Football prediction engine.

A daily NCAAF game-prediction and EV betting engine that mirrors the
architecture of the sibling MLB engine (``mlb_engine``): pull the slate and the
market, build probabilistic score projections from team power ratings, price
every market against the book, and export Strong / Moderate / Pass
recommendations for **moneyline**, **against-the-spread (ATS)**, and **totals**
-- with a reader-facing card, a slate-preview article + audio, a nightly
self-audit, and a running per-bet ledger with closing-line-value scoring.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
