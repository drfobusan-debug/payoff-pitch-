"""Backtest a conviction floor on batter props against the graded ledger.

For every batter-prop market it compares the *current* buy set (tier =
Strong/Moderate buy) against the buys that survive a conviction floor
(``model_prob >= FLOOR`` AND ``ev > 0``), and reports the win rate (PPV) of
each so we can see, per market, whether the floor lifts PPV and how many buys
it keeps.

Run on the machine that owns the ledger:

    python scripts/batter_floor_backtest.py            # floor = 0.58
    python scripts/batter_floor_backtest.py --floor 0.55
    python scripts/batter_floor_backtest.py --ledger /path/to/ledger.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mlb_engine.audit.grade import LOSS, WIN
from mlb_engine.audit.ledger import LedgerEntry, load_ledger
from mlb_engine.config import load_config
from mlb_engine.market.tiers import Tier

BUY_TIERS = {Tier.STRONG.value, Tier.MODERATE.value}


def _ppv(rows: list[LedgerEntry]) -> tuple[int, int, float | None]:
    """(n graded, wins, PPV) over win/loss rows (pushes excluded)."""
    graded = [r for r in rows if r.result in (WIN, LOSS)]
    wins = sum(1 for r in graded if r.result == WIN)
    n = len(graded)
    return n, wins, (wins / n if n else None)


def _fmt(pct: float | None) -> str:
    return f"{pct * 100:5.1f}%" if pct is not None else "   -- "


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=0.58, help="default floor for all markets")
    ap.add_argument(
        "--market-floor",
        action="append",
        default=[],
        metavar="MARKET=PROB",
        help="per-market override, e.g. --market-floor batter_1b=0.44 "
        "(repeatable); overrides --floor for that market",
    )
    ap.add_argument("--ledger", type=Path, default=None)
    args = ap.parse_args()

    overrides: dict[str, float] = {}
    for spec in args.market_floor:
        mkt, _, val = spec.partition("=")
        overrides[mkt.strip()] = float(val)

    path = args.ledger or (load_config().audit_dir / "ledger.csv")
    if not path.exists():
        raise SystemExit(f"No ledger at {path}; run `mlb-engine audit` first")

    ledger = [e for e in load_ledger(path) if e.market.startswith("batter_")]
    if not ledger:
        raise SystemExit(f"No batter-prop rows in {path}")

    markets = sorted({e.market for e in ledger})

    def floor_for(market: str) -> float:
        return overrides.get(market, args.floor)

    print(f"Ledger: {path}")
    print(f"Conviction floor: model_prob >= {args.floor:.2f} AND ev > 0 (default)")
    for mkt, val in sorted(overrides.items()):
        print(f"  override: {mkt} >= {val:.2f}")
    print(f"Batter rows: {len(ledger)}  markets: {len(markets)}\n")

    header = (
        f"{'market':<16} {'floor':>5} "
        f"{'buys':>5} {'PPV now':>8}  |  "
        f"{'kept':>5} {'PPV floor':>10} {'dropped':>8} {'ROI/u now':>10} {'ROI/u flr':>10}"
    )
    print(header)
    print("-" * len(header))

    def _kept(r: LedgerEntry) -> bool:
        return r.model_prob >= floor_for(r.market) and (r.ev is not None and r.ev > 0)

    def row_line(name: str, rows: list[LedgerEntry], floor_label: str) -> None:
        buys = [r for r in rows if r.tier in BUY_TIERS]
        kept = [r for r in buys if _kept(r)]
        n0, _, ppv0 = _ppv(buys)
        n1, _, ppv1 = _ppv(kept)
        roi0 = sum(r.pnl for r in buys) / n0 if n0 else None
        roi1 = sum(r.pnl for r in kept) / n1 if n1 else None
        roi0s = f"{roi0 * 100:+8.1f}%" if roi0 is not None else "   --  "
        roi1s = f"{roi1 * 100:+8.1f}%" if roi1 is not None else "   --  "
        print(
            f"{name:<16} {floor_label:>5} "
            f"{n0:>5} {_fmt(ppv0):>8}  |  "
            f"{n1:>5} {_fmt(ppv1):>10} {n0 - n1:>8} {roi0s:>10} {roi1s:>10}"
        )

    for m in markets:
        row_line(m, [e for e in ledger if e.market == m], f"{floor_for(m):.2f}")
    print("-" * len(header))
    row_line("ALL batter", ledger, "mix")
    print(
        "\nPPV now = win% of current Strong/Moderate buys.  "
        "PPV floor = win% of the buys that clear the floor.\n"
        "dropped = buys the floor turns into Pass.  ROI/u = net units per 1u buy."
    )


if __name__ == "__main__":
    main()
