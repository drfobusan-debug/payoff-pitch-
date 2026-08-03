"""Collapse-volatility article for a slate.

Reproduces the collapse-propensity index (leading indicators of multi-run
"crooked" innings) and renders it as a house-style PDF + MP3.

Index (per team, scored separately for starters and bullpens, z-scored across
the 30 teams within each role over the trailing Statcast window):

    collapse = z(BB%) + z(Barrel%) + z(HardHit%) + z(HR%) - z(CSW%)

Higher = more blow-up prone (more traffic + loud contact, less called+swinging
strike command). Starters = each team's inning-1 pitcher on a given date;
relievers = everyone else. A game's "fireworks" score sums both staffs'
starter + bullpen collapse; a volatile game widens the Over and run-line tails.

Usage:
    python -m scripts.collapse_volatility_report DATE STATCAST_PKL
"""

from __future__ import annotations

import json
import sys
from datetime import date as Date

import numpy as np
import pandas as pd

from mlb_engine.config import load_config
from mlb_engine.output.audit_insight import to_mp3

FB = {"FF", "FA", "FT", "SI", "FC"}
WHIFF = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
CALLED = {"called_strike"}
K_EV = {"strikeout", "strikeout_double_play"}
WALK_EV = {"walk", "intent_walk"}


def load(pkl: str) -> pd.DataFrame:
    df = pd.read_pickle(pkl).copy()
    df["date"] = pd.to_datetime(df["game_date"])
    top = df["inning_topbot"].astype(str).str.startswith("Top")
    df["pitch_team"] = np.where(top, df["home_team"], df["away_team"])
    desc = df["description"].astype(str)
    ev = df["events"].astype(str)
    df["is_whiff"] = desc.isin(WHIFF)
    df["is_csw"] = desc.isin(WHIFF | CALLED)
    df["is_pa"] = df["events"].notna()
    df["is_k"] = ev.isin(K_EV)
    df["is_bb"] = ev.isin(WALK_EV)
    df["is_hr"] = ev.eq("home_run")
    df["ls"] = pd.to_numeric(df["launch_speed"], errors="coerce")
    df["is_barrel"] = pd.to_numeric(df["launch_speed_angle"], errors="coerce") == 6
    df["is_hardhit"] = df["ls"] >= 95.0
    # Starter = the inning-1 pitcher for each (date, team); relievers are the rest.
    i1 = df[df["inning"] == 1]
    cnt = i1.groupby(["date", "pitch_team", "pitcher"]).size().reset_index(name="n")
    st = cnt.sort_values("n").groupby(["date", "pitch_team"]).tail(1)
    starter_keys = set(map(tuple, st[["date", "pitcher"]].to_numpy()))
    df["is_starter"] = [
        (d, p) in starter_keys
        for d, p in zip(df["date"], df["pitcher"], strict=False)
    ]
    return df


def _team_components(rows: pd.DataFrame) -> dict[str, float]:
    n = len(rows)
    pa = int(rows["is_pa"].sum())
    bip = int(rows["ls"].notna().sum())
    if n == 0 or pa == 0 or bip == 0:
        return {}
    return {
        "bb": rows["is_bb"].sum() / pa,
        "barrel": rows["is_barrel"].sum() / bip,
        "hardhit": rows["is_hardhit"].sum() / bip,
        "hr": rows["is_hr"].sum() / pa,
        "csw": rows["is_csw"].sum() / n,
    }


def _collapse_by_role(df: pd.DataFrame, starters: bool) -> pd.Series:
    sub = df[df["is_starter"]] if starters else df[~df["is_starter"]]
    comp = (
        sub.groupby("pitch_team")
        .apply(_team_components, include_groups=False)
        .apply(pd.Series)
        .dropna()
    )
    z = (comp - comp.mean()) / comp.std(ddof=0)
    return (z["bb"] + z["barrel"] + z["hardhit"] + z["hr"] - z["csw"]).round(2)


def build_index(pkl: str) -> pd.DataFrame:
    df = load(pkl)
    sp = _collapse_by_role(df, starters=True).rename("sp")
    rp = _collapse_by_role(df, starters=False).rename("rp")
    return pd.concat([sp, rp], axis=1)


def game_rows(idx: pd.DataFrame, previews: list[dict]) -> list[dict]:
    rows = []
    for g in previews:
        away, home = g["away"], g["home"]
        a = idx.loc[away] if away in idx.index else None
        h = idx.loc[home] if home in idx.index else None
        if a is None or h is None:
            continue
        vol = float(a["sp"] + a["rp"] + h["sp"] + h["rp"])
        rows.append(
            {
                "matchup": g["matchup"],
                "away": away,
                "home": home,
                "away_sp": float(a["sp"]),
                "away_rp": float(a["rp"]),
                "home_sp": float(h["sp"]),
                "home_rp": float(h["rp"]),
                "volatility": round(vol, 2),
                "total_mean": g.get("total_mean"),
                "p_blowout": g.get("p_blowout"),
                "park": g.get("park_name"),
            }
        )
    rows.sort(key=lambda r: r["volatility"], reverse=True)
    return rows


# ---- house-style rendering ------------------------------------------------
NAVY = "#16324f"
RUST = "#b5471f"
MUTED = "#6b7785"
PAPER = "#faf8f4"


def _read(r: dict) -> str:
    hi = r["volatility"] >= 6
    lo = r["volatility"] <= -4
    if hi:
        return (
            "Fireworks risk. Both staffs skew toward traffic and loud contact; "
            "lean the Over and take the +1.5 dog rather than laying -1.5. A shaky "
            "bullpen behind a decent arm favors F5 sides/Unders (cash before the pen)."
        )
    if lo:
        return (
            "Collapse-resistant. Fewer big innings to fear; Unders and laying -1.5 "
            "are more reliable here."
        )
    return (
        "Middle of the board. No strong volatility edge either way; let price and "
        "matchup drive it."
    )


def build_html(day: Date, rows: list[dict], window: str) -> str:
    trs = []
    for i, r in enumerate(rows, 1):
        band = RUST if r["volatility"] >= 6 else (NAVY if r["volatility"] <= -4 else MUTED)
        tot = f"{r['total_mean']:.1f}" if r["total_mean"] is not None else "-"
        blow = f"{r['p_blowout'] * 100:.0f}%" if r["p_blowout"] is not None else "-"
        trs.append(
            f"<tr><td class='rk'>{i}</td><td class='mu'>{r['matchup']}</td>"
            f"<td class='v' style='color:{band}'><b>{r['volatility']:+.2f}</b></td>"
            f"<td>{r['away']} {r['away_sp']:+.1f}/{r['away_rp']:+.1f}</td>"
            f"<td>{r['home']} {r['home_sp']:+.1f}/{r['home_rp']:+.1f}</td>"
            f"<td>{tot}</td><td>{blow}</td></tr>"
        )
    read_blocks = []
    for r in rows:
        band = RUST if r["volatility"] >= 6 else (NAVY if r["volatility"] <= -4 else MUTED)
        tot = f"{r['total_mean']:.1f}" if r["total_mean"] is not None else "-"
        blow = f"{r['p_blowout'] * 100:.0f}%" if r["p_blowout"] is not None else "-"
        read_blocks.append(
            f"<div class='game'><div class='gh'>{r['matchup']} "
            f"<span style='color:{band}'>({r['volatility']:+.2f})</span></div>"
            f"<div class='gsub'>{r['away']} SP {r['away_sp']:+.2f} / pen {r['away_rp']:+.2f}"
            f" &nbsp;&bull;&nbsp; {r['home']} SP {r['home_sp']:+.2f} / pen {r['home_rp']:+.2f}"
            f" &nbsp;&bull;&nbsp; proj total {tot}"
            f" &nbsp;&bull;&nbsp; P(blowout) {blow}</div>"
            f"<div class='gr'>{_read(r)}</div></div>"
        )
    reads = "".join(read_blocks)
    return f"""<html><head><meta charset='utf-8'><style>
    @page {{ size: letter; margin: 1.5cm 1.6cm; }}
    body {{ font-family: Georgia, 'Times New Roman', serif; color:{NAVY}; background:{PAPER}; }}
    .kick {{ letter-spacing:.22em; text-transform:uppercase; font-size:10px; color:{RUST}; font-family:Helvetica,Arial,sans-serif; }}
    h1 {{ font-size:26px; margin:2px 0 2px; }}
    .dek {{ color:{MUTED}; font-size:12.5px; margin-bottom:14px; }}
    .lead {{ font-size:12.5px; line-height:1.5; margin:10px 0 14px; }}
    table {{ width:100%; border-collapse:collapse; font-family:Helvetica,Arial,sans-serif; font-size:10.5px; }}
    th {{ text-align:left; border-bottom:2px solid {NAVY}; padding:5px 4px; font-size:9px; text-transform:uppercase; letter-spacing:.06em; color:{MUTED}; }}
    td {{ padding:5px 4px; border-bottom:1px solid #e5ded2; }}
    td.rk {{ color:{MUTED}; width:18px; }} td.mu {{ font-weight:bold; }} td.v {{ text-align:right; }}
    .sec {{ letter-spacing:.14em; text-transform:uppercase; font-size:11px; color:{RUST}; font-family:Helvetica,Arial,sans-serif; margin:20px 0 6px; border-top:1px solid #e5ded2; padding-top:12px; }}
    .game {{ margin:9px 0; }} .gh {{ font-size:13px; font-weight:bold; }}
    .gsub {{ font-family:Helvetica,Arial,sans-serif; font-size:9.5px; color:{MUTED}; margin:1px 0 3px; }}
    .gr {{ font-size:11.5px; line-height:1.45; }}
    .foot {{ color:{MUTED}; font-size:9px; font-family:Helvetica,Arial,sans-serif; margin-top:16px; border-top:1px solid #e5ded2; padding-top:8px; }}
    </style></head><body>
    <div class='kick'>Payoff Pitch &nbsp;·&nbsp; Volatility Report</div>
    <h1>Collapse &amp; Crooked-Inning Risk</h1>
    <div class='dek'>{day:%A, %B %-d, %Y} slate &nbsp;·&nbsp; {len(rows)} games ranked by blow-up propensity</div>
    <div class='lead'>The collapse index scores each staff on the <i>leading indicators</i> of a
    multi-run inning &mdash; walks and loud contact (BB%, Barrel%, HardHit%, HR%) net of command
    (CSW%) &mdash; z-scored across the league over the trailing {window}, separately for
    starters and bullpens. A game's score sums both staffs' starter and bullpen risk. It measures
    <i>variance</i>, not who wins: a high score widens the Over and the run-line tails, and a shaky
    bullpen argues for getting out in the F5.</div>

    <div class='sec'>Fireworks board &mdash; most to least volatile</div>
    <table><tr><th>#</th><th>Game</th><th>Vol</th><th>Away SP/pen</th><th>Home SP/pen</th><th>Proj tot</th><th>P(blow)</th></tr>
    {''.join(trs)}</table>

    <div class='sec'>Game-by-game read</div>
    {reads}

    <div class='foot'>Leading-indicator proxy from Statcast pitch data, not a measured runs-per-inning
    rate; bullpen figures are noisier (smaller samples). Team-level starter aggregate, not the specific
    probable starter. Model preview, not investment advice.</div>
    </body></html>"""


def build_narration(day: Date, rows: list[dict]) -> str:
    intro = (
        f"Payoff Pitch volatility report for {day:%A, %B %-d}. "
        "This is the collapse index: the leading indicators of a multi-run inning, "
        "walks and loud contact net of command, scored for every starter and bullpen."
    )
    if not rows:
        return intro + " No games could be scored today."

    top = rows[0]
    parts = [intro]
    # Only pitch the Over/dog when the top game actually clears the volatility
    # band used in the written report; otherwise describe the board honestly.
    if top["volatility"] >= 6:
        parts.append(
            f"The most volatile game on the board is {top['matchup']}, "
            f"at a fireworks score of {top['volatility']:+.1f} — "
            "lean the over and the plus one and a half dog."
        )
    else:
        parts.append(
            f"No game clears the fireworks threshold today; the most volatile is "
            f"{top['matchup']} at only {top['volatility']:+.1f}, so there is no "
            "over lean from volatility alone."
        )
    hi = [r for r in rows if r["volatility"] >= 6]
    if len(hi) > 1:
        parts.append(
            "Other high-variance spots: "
            + ", ".join(r["matchup"] for r in hi[1:])
            + "."
        )
    calm = rows[-1]
    # With a single game, top and calm are the same row — don't describe it twice.
    if calm is not top and calm["volatility"] <= -4:
        parts.append(
            f"The calmest, most collapse-resistant game is {calm['matchup']}, "
            f"at {calm['volatility']:+.1f} — unders and laying the run line are safer there."
        )
    parts.append(
        "Remember this measures variance, not edge: a volatile game is only a bet "
        "when the price is wrong."
    )
    return " ".join(parts)


def main() -> None:
    day = Date.fromisoformat(sys.argv[1])
    pkl = sys.argv[2] if len(sys.argv) > 2 else "statcast_2026-06-21_2026-08-01.pkl"
    window = sys.argv[3] if len(sys.argv) > 3 else "six weeks"
    cfg = load_config()
    pkl_path = pkl if pkl.startswith("/") else str(cfg.cache_dir / pkl)
    previews = json.load(open(cfg.audit_dir / f"previews_{day.isoformat()}.json"))

    idx = build_index(pkl_path)
    rows = game_rows(idx, previews)

    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    csv = out / f"Collapse_Volatility_{day.isoformat()}.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    from weasyprint import HTML

    html = build_html(day, rows, window)
    pdf = out / f"PayoffPitch_Volatility_{day.isoformat()}.pdf"
    pdf.write_bytes(HTML(string=html).write_pdf())
    mp3 = out / f"PayoffPitch_Volatility_{day.isoformat()}.mp3"
    to_mp3(build_narration(day, rows), mp3)

    for r in rows:
        print(f"  {r['volatility']:+6.2f}  {r['matchup']:12}  "
              f"{r['away']} {r['away_sp']:+.1f}/{r['away_rp']:+.1f}  "
              f"{r['home']} {r['home_sp']:+.1f}/{r['home_rp']:+.1f}")
    print("CSV:", csv)
    print("PDF:", pdf)
    print("MP3:", mp3)


if __name__ == "__main__":
    main()
