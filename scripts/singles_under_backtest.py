"""Backtest betting the UNDER on batter singles (0.5) on the model's fade branch.

The engine fades most singles overs (model_prob < 0.5). This grades the mirror
strategy -- bet the *under* on those -- to test whether the model's strong
singles NPV converts to profit once the under's real price is paid.

Under price source, in order:
  1. the ledger's persisted ``under_odds`` column (written every run), else
  2. The Odds API *historical* endpoint (``--historical``) for older slates
     whose rows predate under-price persistence.

A pick's under wins when the graded over lost (batter recorded 0 singles).
"""

from __future__ import annotations

import argparse
import json
import os
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

KEY = os.environ.get("THE_ODDS_API_KEY") or os.environ.get("ODDS_API_KEY")
BASE = "https://api.the-odds-api.com/v4"
LEDGER = os.path.expanduser("~/.mlb_engine/audit/ledger.csv")


def norm(name: str) -> str:
    n = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in n if not unicodedata.combining(c)).strip().lower()


def player_from_selection(sel: str) -> str:
    s = str(sel)
    for tok in (" 1B o0.5", " 1B", " o0.5"):
        s = s.replace(tok, "")
    return norm(s)


def american_to_decimal(a: float) -> float:
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)


def _get(url: str) -> dict | list | None:
    try:
        return json.load(urllib.request.urlopen(url, timeout=30))
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print("  HTTP", e.code, str(e.read()[:120]))
        return None
    except Exception as e:  # noqa: BLE001
        print("  ERR", e)
        return None


def historical_under_prices(date: str) -> dict[str, float]:
    """{normalized_player: best under american} from a pre-game snapshot."""
    if not KEY:
        return {}
    ev = _get(f"{BASE}/historical/sports/baseball_mlb/events?apiKey={KEY}&date={date}T12:00:00Z")
    events = ev.get("data", []) if isinstance(ev, dict) else []
    events = [e for e in events if str(e.get("commence_time", "")).startswith(date)]
    out: dict[str, float] = {}
    for e in events:
        ct = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        snap = (ct - timedelta(minutes=90)).strftime("%Y-%m-%dT%H:%M:%SZ")
        d = _get(
            f"{BASE}/historical/sports/baseball_mlb/events/{e['id']}/odds"
            f"?apiKey={KEY}&regions=us&markets=batter_singles&oddsFormat=american&date={snap}"
        )
        data = d.get("data") if isinstance(d, dict) else None
        if not data:
            continue
        for bm in data.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "batter_singles":
                    continue
                for oc in mkt.get("outcomes", []):
                    if str(oc.get("name", "")).lower() != "under":
                        continue
                    p, price = player_from_selection(oc.get("description", "")), oc.get("price")
                    if p and price is not None:
                        out[p] = max(out.get(p, -10_000), float(price))
    return out


def report(x: pd.DataFrame, label: str) -> None:
    if not len(x):
        print(f"\n{label}: no picks")
        return
    x = x.copy()
    x["dec"] = x["under_price"].map(american_to_decimal)
    x["pnl"] = np.where(x["under_won"] == 1, x["dec"] - 1.0, -1.0)
    pnl = x["pnl"].to_numpy()
    rng = np.random.default_rng(0)
    boot = [rng.choice(pnl, len(pnl), replace=True).mean() for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n{label}")
    print(f"  n={len(x)}  under win%={x['under_won'].mean() * 100:.1f}  "
          f"ROI/u={x['pnl'].mean() * 100:+.1f}%  net={x['pnl'].sum():+.1f}u  "
          f"median under price={x['under_price'].median():.0f}")
    print(f"  95% bootstrap ROI/u CI: [{lo * 100:+.1f}%, {hi * 100:+.1f}%]  "
          f"(breakeven 0% {'INSIDE' if lo <= 0 <= hi else 'OUTSIDE'} the interval)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--dates", nargs="*", help="restrict to these YYYY-MM-DD dates")
    ap.add_argument("--under-max-juice", type=float, default=-135.0,
                    help="keep unders priced this or better, i.e. odds >= value")
    ap.add_argument("--historical", action="store_true",
                    help="reconstruct missing under prices from the Odds API historical endpoint")
    args = ap.parse_args()

    g = pd.read_csv(args.ledger)
    s = g[g["market"] == "batter_1b"].copy()
    if args.dates:
        s = s[s["date"].isin(args.dates)]
    s["mp"] = pd.to_numeric(s["model_prob"], errors="coerce")
    s["player"] = s["selection"].map(player_from_selection)
    s = s[s["mp"] < 0.5].copy()  # NPV branch: the model faded the over
    s["under_won"] = (s["result"] == "loss").astype(int)  # over missed => under hit
    under_col = s["under_odds"] if "under_odds" in s.columns else pd.Series(index=s.index, dtype=float)
    s["under_price"] = pd.to_numeric(under_col, errors="coerce")

    from_ledger = int(s["under_price"].notna().sum())
    if args.historical:
        for date in sorted(s.loc[s["under_price"].isna(), "date"].unique()):
            prices = historical_under_prices(date)
            mask = (s["date"] == date) & s["under_price"].isna()
            s.loc[mask, "under_price"] = s.loc[mask, "player"].map(prices)

    matched = s.dropna(subset=["under_price"]).copy()
    print(f"faded-over singles: {len(s)}  from ledger under_odds: {from_ledger}  "
          f"matched total: {len(matched)}")
    print("matched by date:\n" + matched.groupby("date").size().to_string())

    j = args.under_max_juice
    report(matched, "ALL faded-over singles (any under price)")
    report(matched[matched["under_price"] >= j], f"Unders priced {j:.0f} or better (odds >= {j:.0f})")
    report(matched[matched["under_price"] < j], f"Unders more juiced than {j:.0f} (excluded)")


if __name__ == "__main__":
    main()
