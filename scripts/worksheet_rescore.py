"""Re-score the worksheet ledger's older rows under the current scoring rule.

Every change to how the worksheet scores batting, bullpens or starters bumps
``WEIGHTS_VERSION`` and the Audit counts only rows written under the current
tag, so each change restarts the sample. This rebuilds the older rows as they
would have been written today: the day's Statcast frame (through the day
before) and FanGraphs tables as of that day are rebuilt, the stored starters
are re-scored, ``away_w``/``home_w``/``gap``/``fav`` are recomputed, the
favourite's no-vig probability is re-read from the stored moneylines, and a
graded row is re-graded from its stored runs. Prices, splits and runs are never
touched. A row whose starter no longer scores (below the IP floor under the
current rule) becomes incomplete and drops out of the Audit, as a fresh row
would.

Usage:
    python -m scripts.worksheet_rescore            # dry run: prints what would change
    python -m scripts.worksheet_rescore --write    # rewrites audit/worksheet_ledger.csv (a .bak copy is kept)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import replace
from datetime import date as Date
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from mlb_engine.config import load_config
from mlb_engine.output import daily_worksheet as ws
from mlb_engine.schemas import Hand, Pitcher, TeamGameInfo

log = logging.getLogger(__name__)


def _pitcher(name: str, sps: ws.StarterTable, hands: dict[int, str]) -> Pitcher | None:
    if not name:
        return None
    pid = sps.by_name.get(ws._norm_name(name))
    hand = hands.get(pid) if pid is not None else None
    return Pitcher(mlbam_id=pid or 0, name=name, throws=Hand(hand) if hand in ("L", "R") else None)


def _team(abbrev: str, starter: str, home: bool, sps: ws.StarterTable, hands: dict[int, str]) -> TeamGameInfo:
    return TeamGameInfo(team_id=0, name=abbrev, abbrev=abbrev, is_home=home,
                        probable_pitcher=_pitcher(starter, sps, hands))


def rescore_row(
    r: ws.LedgerRow, bats: dict[str, ws.Ranking], pens: ws.Ranking, sps: ws.StarterTable,
    hands: dict[int, str],
) -> ws.LedgerRow:
    away = _team(r.away, r.away_sp, False, sps, hands)
    home = _team(r.home, r.home_sp, True, sps, hands)
    a = ws.team_line(away, home, bats, pens, sps)
    h = ws.team_line(home, away, bats, pens, sps)
    gap = round(a.weighted() - h.weighted(), 1)
    fav = r.away if gap >= 0 else r.home
    fav_ml, dog_ml = (r.away_ml, r.home_ml) if fav == r.away else (r.home_ml, r.away_ml)
    complete = a.sp is not None and h.sp is not None
    result = r.result
    if result in ("fav", "dog", "tie"):
        if not complete:
            result = ""
        elif r.away_runs is not None and r.home_runs is not None:
            if r.away_runs == r.home_runs:
                result = "tie"
            else:
                winner = r.away if r.away_runs > r.home_runs else r.home
                result = "fav" if winner == fav else "dog"
    return replace(
        r, away_bat=a.bat_vs_hand, away_bp=a.bp, away_sp_pts=a.sp,
        home_bat=h.bat_vs_hand, home_bp=h.bp, home_sp_pts=h.sp,
        away_w=a.weighted(), home_w=h.weighted(), gap=gap, fav=fav, complete=complete,
        fav_implied=ws.implied(fav_ml, dog_ml), weights=ws.WEIGHTS_VERSION, result=result,
    )


def _hands(df: pd.DataFrame) -> dict[int, str]:
    sub = df[["pitcher", "p_throws"]].dropna()
    return {int(p): str(h) for p, h in sub.groupby("pitcher")["p_throws"].agg(lambda s: s.mode().iat[0]).items()}


def rescore(rows: list[ws.LedgerRow], cache_dir: Path) -> tuple[list[ws.LedgerRow], list[str]]:
    """The ledger with every row not on ``WEIGHTS_VERSION`` rebuilt; plus a change log."""
    out, notes = [], []
    stale = sorted({r.date for r in rows if r.weights != ws.WEIGHTS_VERSION})
    by_date: dict[str, tuple] = {}
    for d in stale:
        day = Date.fromisoformat(d)
        as_of = day - timedelta(days=1)
        log.info("rescore %s: loading Statcast through %s", d, as_of)
        df = ws._prepare(ws.season_pitches(cache_dir, day))
        by_date[d] = (ws.bullpen_ranking(as_of), ws.offense_rankings(df, as_of), ws.starter_table(df, as_of),
                      _hands(df))
    for r in rows:
        if r.weights == ws.WEIGHTS_VERSION:
            out.append(r)
            continue
        pens, bats, sps, hands = by_date[r.date]
        new = rescore_row(r, bats, pens, sps, hands)
        flags = []
        if new.fav != r.fav:
            flags.append(f"fav {r.fav}->{new.fav}")
        if new.complete != r.complete:
            flags.append("complete" if new.complete else "incomplete")
        if new.result != r.result:
            flags.append(f"result {r.result or '-'}->{new.result or '-'}")
        notes.append(f"{r.date} {r.game}: gap {r.gap:+.1f}->{new.gap:+.1f} [{r.weights}]"
                     + (" " + ", ".join(flags) if flags else ""))
        out.append(new)
    return out, notes


def _summary(rows: list[ws.LedgerRow]) -> str:
    t = ws.tally(rows)[-1]
    return f"{t.n} graded on {ws.WEIGHTS_VERSION}: fav {t.fav_wins}-{t.n - t.fav_wins}, priced {t.implied_n}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="rewrite the ledger (default: report only)")
    ap.add_argument("--ledger", type=Path, default=None, help="ledger path (default: the audit dir's)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    path = args.ledger or ws.ledger_path(cfg)
    rows = ws.clean_ledger(ws.load_ledger(path))
    print(f"before: {_summary(rows)}")
    new, notes = rescore(rows, cfg.cache_dir)
    for n in notes:
        print(n)
    print(f"after:  {_summary(new)}  ({len(notes)} rows re-scored)")
    if args.write:
        bak = path.with_name(f"{path.name}.bak-{datetime.now():%Y%m%d%H%M%S}")
        shutil.copy(path, bak)
        ws.save_ledger(path, new)
        print(f"written {path} (backup {bak})")
    else:
        print("dry run; re-run with --write to keep")
    return 0


if __name__ == "__main__":
    sys.exit(main())
