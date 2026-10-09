"""Ask whether the batter-over miss is exposure or rate.

Every batter over in the ledger realizes below the model -- H+R+RBI overs by 8
points, total-bases overs by 23 -- while the unders realize above it. Two very
different causes look the same in the aggregate: the model may be counting plate
appearances the hitter never gets (exposure), or it may be pricing each plate
appearance too richly (rate). They call for opposite repairs, so this joins each
graded H+R+RBI row to the batter's realized box-score line and reads the miss
conditional on the exposure he actually got, by lineup slot.

Nothing here changes a probability. It writes a table and a joined CSV.

    python scripts/hrr_exposure_study.py [--ledger PATH] [--out PATH]

The box scores are cached under ``~/.mlb_engine/cache/boxscores``, the same place
the audit keeps them, so a second run costs no requests.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("hrr_exposure")

BASE = "https://statsapi.mlb.com/api/v1"
# The number of plate appearances a Poisson mean has to buy to reach the model's
# stated probability. HRR is over-dispersed relative to Poisson -- a hit, the run
# it may lead to and the RBI it may drive are one event three times -- so an
# inverted mean is an upper bound on what the model claims, and is read here only
# for its slope across exposure, never as a level.
PA_BUCKETS = ([-1, 2, 3, 4, 5, 9], ["<=2", "3", "4", "5", "6+"])


def _schedule(date: str, cache: Path, session: requests.Session) -> list[int]:
    p = cache / f"schedule_{date}.json"
    if p.exists():
        return [int(x) for x in json.loads(p.read_text())]
    params = {"sportId": "1", "date": date}
    r = session.get(f"{BASE}/schedule", params=params, timeout=30).json()
    pks = [
        int(g["gamePk"])
        for d in r.get("dates", [])
        for g in d.get("games", [])
        if str(g.get("status", {}).get("abstractGameState", "")).lower() == "final"
    ]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(pks))
    return pks


def _batting_lines(
    game_pk: int, cache: Path, session: requests.Session
) -> dict[str, dict[str, int]]:
    """``full name -> line``, including the batting slot and whether he started."""
    p = cache / f"{game_pk}.json"
    if p.exists():
        box = json.loads(p.read_text()).get("boxscore", {})
    else:
        ls = session.get(f"{BASE}/game/{game_pk}/linescore", timeout=30).json()
        box = session.get(f"{BASE}/game/{game_pk}/boxscore", timeout=30).json()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"linescore": ls, "boxscore": box}))
    out: dict[str, dict[str, int]] = {}
    for side in ("home", "away"):
        for player in box.get("teams", {}).get(side, {}).get("players", {}).values():
            bat = (player.get("stats", {}) or {}).get("batting", {}) or {}
            if not bat:
                continue
            order = str(player.get("battingOrder") or "")
            out[(player.get("person", {}) or {}).get("fullName", "")] = {
                "pa": int(bat.get("plateAppearances", 0) or 0),
                "h": int(bat.get("hits", 0) or 0),
                "r": int(bat.get("runs", 0) or 0),
                "rbi": int(bat.get("rbi", 0) or 0),
                # 100, 200 ... is the slot; a trailing non-zero marks a substitute
                # who inherited it, whose plate appearances are not the bet's.
                "slot": int(order[0]) if order[:1].isdigit() else 0,
                "starter": int(order.endswith("00")),
            }
    return out


def join_box_scores(hrr: pd.DataFrame, cache: Path) -> pd.DataFrame:
    """Attach each graded row's realized line, dropping what cannot be matched."""
    session = requests.Session()
    per_date: dict[str, dict[str, dict[str, int]]] = {}
    for date in sorted(hrr["date"].unique()):
        merged: dict[str, dict[str, int]] = {}
        for pk in _schedule(str(date), cache, session):
            try:
                merged.update(_batting_lines(pk, cache, session))
            except requests.RequestException as exc:
                log.warning("no box score for %s: %s", pk, exc)
        per_date[str(date)] = merged
        log.info("%s: %d batting lines", date, len(merged))

    keys = ("pa", "h", "r", "rbi", "slot", "starter")
    found = [
        per_date.get(str(d), {}).get(str(b))
        for d, b in zip(hrr["date"], hrr["batter"], strict=True)
    ]
    for key in keys:
        hrr[key] = [np.nan if rec is None else float(rec[key]) for rec in found]
    hrr["hrr"] = hrr["h"] + hrr["r"] + hrr["rbi"]
    return hrr[hrr["pa"].notna()].copy()


def load_hrr(ledger: Path) -> pd.DataFrame:
    d = pd.read_csv(ledger, low_memory=False)
    if "source" in d.columns:
        d = d[(d["source"].isna()) | (d["source"] == "engine")]
    h = d[(d["market"] == "batter_hrr") & (d["result"].isin(["win", "loss"]))].copy()
    h["batter"] = h["selection"].str.replace(r"\s+H\+R\+RBI.*$", "", regex=True)
    h["side"] = h["selection"].str.extract(r"([ou])\d")[0].map({"o": "over", "u": "under"})
    h["won"] = h["result"] == "win"
    return h


def report(joined: pd.DataFrame) -> None:
    bins, labels = PA_BUCKETS
    fav = joined[joined["model_prob"] >= 0.5].copy()
    fav["pa_bucket"] = pd.cut(fav["pa"], bins, labels=labels)
    fav["miss"] = fav["won"] - fav["model_prob"]
    fav["market_miss"] = fav["won"] - fav["fair_prob"]

    print("\nthe model's miss conditional on the exposure the hitter got")
    print(
        fav.groupby(["side", "pa_bucket"], observed=True)
        .agg(
            n=("won", "size"),
            won=("won", "mean"),
            model=("model_prob", "mean"),
            miss=("miss", "mean"),
            market_miss=("market_miss", "mean"),
            hrr=("hrr", "mean"),
        )
        .round(3)
    )

    over = joined[(joined["side"] == "over") & (joined["line"] == 1.5)].copy()
    slot = over.groupby("slot").agg(
        n=("won", "size"),
        model=("model_prob", "mean"),
        won=("won", "mean"),
        pa=("pa", "mean"),
        hrr=("hrr", "mean"),
    )
    slot["miss"] = slot["won"] - slot["model"]
    print("\nevery graded o1.5, by the slot he actually batted in")
    print(slot.round(3))

    # The one test that separates a failed exposure claim from noise: hold the
    # slot fixed and ask whether the overs the model likes more than the book
    # realize more exposure than the ones it likes less.
    over["above"] = np.where(over["model_prob"] >= over["fair_prob"], "above", "below")
    print("\nwithin slot, does the model's own disagreement buy exposure?")
    print(
        over.groupby(["slot", "above"])
        .agg(n=("pa", "size"), pa=("pa", "mean"), hrr=("hrr", "mean"), won=("won", "mean"))
        .unstack()
        .round(3)
        .to_string()
    )
    pooled = over.groupby("above").agg(
        n=("pa", "size"), pa=("pa", "mean"), hrr=("hrr", "mean"), won=("won", "mean")
    )
    print("\npooled:")
    print(pooled.round(3))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", type=Path, default=Path.home() / ".mlb_engine/audit/ledger.csv")
    ap.add_argument("--cache", type=Path, default=Path.home() / ".mlb_engine/cache/boxscores")
    ap.add_argument("--out", type=Path, help="write the joined rows here")
    args = ap.parse_args()

    hrr = load_hrr(args.ledger)
    joined = join_box_scores(hrr, args.cache)
    print(f"joined {len(joined)} of {len(hrr)} graded H+R+RBI rows")
    if args.out is not None:
        joined.to_csv(args.out, index=False)
        print(f"rows -> {args.out}")
    report(joined)


if __name__ == "__main__":
    main()
