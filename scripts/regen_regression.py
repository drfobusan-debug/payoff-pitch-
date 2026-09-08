"""Regenerate the regression reports: the combined prose article (arms + bats
in one PDF), the two stat-card PDFs it was written from, and one MP3.

Run it from the repository root::

    python -m scripts.regen_regression [--date YYYY-MM-DD] [--statcast FRAME]
"""

from __future__ import annotations

import json

import pandas as pd

import scripts.comprehensive_report as cr
import scripts.pitcher_slate_analysis as psa
import scripts.regression_article as art
from mlb_engine.config import load_config
from mlb_engine.output.audit_insight import to_mp3
from scripts.slate_inputs import predictions_path, resolve_day, statcast_frame


def main(argv: list[str] | None = None) -> None:
    args = cr.parse_args(argv)
    cfg = load_config()
    day = resolve_day(cfg.audit_dir, args.date)
    preds_path = predictions_path(cfg.audit_dir, day)
    frame = statcast_frame(cfg.cache_dir, day, args.statcast)
    pv_raw = json.load(open(cfg.audit_dir / f"previews_{day.isoformat()}.json"))
    preds_raw = json.loads(preds_path.read_text())
    pv_by_pk = {g["game_pk"]: g for g in pv_raw}
    df = pd.read_pickle(frame)
    print(f"slate {day.isoformat()}  predictions {preds_path.name}  frame {frame.name}")

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
    article_pdf = out / f"PayoffPitch_Regression_{day.isoformat()}.pdf"
    article_pdf.write_bytes(
        cr.merge_pdf(
            [art.build_html(day, ppos, pneg, pctxs, bpos, bneg, bctxs, preds_raw)]
        )
    )
    mp3 = out / f"PayoffPitch_Regression_{day.isoformat()}.mp3"
    to_mp3(pitcher_narr + " " + batter_narr, mp3)

    print(f"pitchers: positive={len(ppos)} negative={len(pneg)}")
    print(f"batters:  positive={len(bpos)} negative={len(bneg)}")
    print("ARTICLE PDF:", article_pdf)
    print("MOUND PDF:", mound_pdf)
    print("BATTER PDF:", batter_pdf)
    print("MP3:", mp3)


if __name__ == "__main__":
    main()
