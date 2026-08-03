"""Regenerate just the pitcher + batter regression reports (standalone PDFs)
and a single combined MP3, matching the morning house style."""

from __future__ import annotations

import json

import pandas as pd

import scripts.comprehensive_report as cr
import scripts.pitcher_slate_analysis as psa
from mlb_engine.config import load_config
from mlb_engine.output.audit_insight import to_mp3


def main() -> None:
    import sys
    from datetime import date as Date

    if len(sys.argv) > 1:
        cr.DAY = Date.fromisoformat(sys.argv[1])
    if len(sys.argv) > 2:
        cr.STATCAST_PKL = sys.argv[2]
    cfg = load_config()
    day = cr.DAY
    pv_raw = json.load(open(cfg.audit_dir / f"previews_{day.isoformat()}.json"))
    preds_raw = json.load(open(cfg.audit_dir / f"predictions_{day.isoformat()}.json"))
    pv_by_pk = {g["game_pk"]: g for g in pv_raw}
    df = pd.read_pickle(cfg.cache_dir / cr.STATCAST_PKL)

    ppos, pneg, pctxs = psa.build_profiles(pv_raw, preds_raw, df)
    pitcher_html = psa.build_html(day, ppos, pneg, pctxs, preds_raw)
    pitcher_narr = psa.build_narration(day, ppos, pneg, pctxs, preds_raw)

    bpos, bneg = cr.build_batter_profiles(preds_raw, df)
    bctxs = cr._batter_ctx(preds_raw, pv_by_pk)
    batter_html = cr.build_batter_html(day, bpos, bneg, bctxs, preds_raw)
    batter_narr = cr.build_batter_narration(day, bpos, bneg, preds_raw)

    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    mound_pdf = out / f"PayoffPitch_Mound_{day.isoformat()}.pdf"
    batter_pdf = out / f"PayoffPitch_Batter_{day.isoformat()}.pdf"
    mound_pdf.write_bytes(cr.merge_pdf([pitcher_html]))
    batter_pdf.write_bytes(cr.merge_pdf([batter_html]))
    mp3 = out / f"PayoffPitch_Regression_{day.isoformat()}.mp3"
    to_mp3(pitcher_narr + " " + batter_narr, mp3)

    print(f"pitchers: positive={len(ppos)} negative={len(pneg)}")
    print(f"batters:  positive={len(bpos)} negative={len(bneg)}")
    print("MOUND PDF:", mound_pdf)
    print("BATTER PDF:", batter_pdf)
    print("MP3:", mp3)


if __name__ == "__main__":
    main()
