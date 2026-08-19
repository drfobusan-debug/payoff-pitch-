"""Starting-pitcher regression article for a slate.

For every starter on a slate this profiles the skill + luck signals the engine
already trusts and ranks the staff by *expected regression*:

* 6-week SIERA (skill), with a recent-vs-prior trend arrow,
* Stuff via xK% (CSW%/SwStr% proxy -- FanGraphs Stuff+ is only populated when a
  subscription feed is present in ~/.mlb_engine/fangraphs, which it is not here),
  with a trend arrow,
* 3-week average fastball velocity (vFA, FF/SI), with a trend arrow,
* the luck signals that drive regression -- BABIP allowed and the xwOBA-minus-wOBA
  gap (dxwOBA) -- plus release biomechanics (extension, IVB, spin, release scatter),
* the engine's pitcher bets tied to that starter.

Pitchers are ranked positive-regression (most -> least likely to *improve*) then
negative-regression (most -> least likely to *decline*), where the luck index is
z(BABIP above .290) + z(wOBA above xwOBA). Renders a house-style PDF + MP3.

This is a model preview, not betting advice.
"""

from __future__ import annotations

import json
from datetime import date as Date

import pandas as pd

from mlb_engine.config import load_config
from mlb_engine.output.audit_insight import to_mp3, to_pdf

# The profiles this report ranks are the engine's own, so the card and the
# article ``mlb-engine run`` emails read one implementation.
from mlb_engine.output.regression_profiles import (  # noqa: F401 (re-exported)
    FB,
    RECENT_DAYS,
    _bets_for,
    _biomech,
    _pitcher_id_map,
    _siera_val,
    _starter_games,
    _stuff_xk,
    _vfa,
    analyze,
    build_profiles,
)


# --- rendering -------------------------------------------------------------
def _arrow(delta: float, good_up: bool) -> str:
    """HTML arrow: green when the move is good for the pitcher, red when bad."""
    if delta != delta or abs(delta) < 1e-9:  # NaN or flat
        return "<span class='flat'>&#9644;</span>"
    up = delta > 0
    good = up if good_up else not up
    glyph = "&#9650;" if up else "&#9660;"
    cls = "pos" if good else "neg"
    return f"<span class='{cls}'>{glyph}</span>"



def _bet_line(r: dict) -> str:
    odds = "" if r.get("market_american") is None else f" ({r['market_american']:+.0f})"
    ev = "" if r.get("ev") is None else f", EV {r['ev']:+.2f}"
    tier = r["tier"]
    cls = "buy" if tier in ("Strong buy", "Moderate buy") else "pass"
    return (
        f"<li class='{cls}'><b>{r['selection']}</b>{odds} — model "
        f"{r['model_prob'] * 100:.0f}%{ev} · <i>{tier}</i></li>"
    )


def _verdict(p: dict, positive: bool) -> str:
    siera = p["siera"]
    tier = (
        "front-line" if siera < 3.4 else "mid-rotation" if siera < 4.3 else "back-end"
    )
    if positive:
        why = []
        if p["unlucky_babip"] > 0.015:
            why.append(f"BABIP-against .{int(round(p['babip'] * 1000)):03d} well over the .290 norm")
        if p["dxwoba"] < -0.010:
            why.append(
                f"results (wOBA .{int(round(p['woba'] * 1000)):03d}) worse than contact "
                f"quality earned (xwOBA .{int(round(p['xwoba'] * 1000)):03d})"
            )
        tail = "; ".join(why) or "modest luck drag"
        return (
            f"A {tier} arm (SIERA {siera:.2f}) who's been unlucky — {tail}. "
            "Expect the hits/runs allowed to fall back toward his skill."
        )
    why = []
    if p["unlucky_babip"] < -0.015:
        why.append(f"BABIP-against a tidy .{int(round(p['babip'] * 1000)):03d} (under .290)")
    if p["dxwoba"] > 0.010:
        why.append(
            f"getting bailed out (xwOBA .{int(round(p['xwoba'] * 1000)):03d} > wOBA "
            f".{int(round(p['woba'] * 1000)):03d})"
        )
    tail = "; ".join(why) or "modest luck tailwind"
    return (
        f"A {tier} arm (SIERA {siera:.2f}) who's been fortunate — {tail}. "
        "Expect more hard contact and hits to leak through going forward."
    )


def _expect_today(p: dict, ctx: dict | None) -> str:
    if ctx is None:
        return ""
    opp_x = ctx["opp_lineup_xwoba"]
    strength = "a soft" if opp_x < 0.310 else "a dangerous" if opp_x > 0.335 else "an average"
    env = []
    if ctx.get("park_factor") is not None:
        pf = ctx["park_factor"]
        env.append("hitter-friendly park" if pf > 101 else "pitcher-friendly park" if pf < 99 else "neutral park")
    if ctx.get("wx_hr_mult") is not None and ctx["wx_hr_mult"] >= 1.03:
        env.append("HR-boosting air")
    elif ctx.get("wx_hr_mult") is not None and ctx["wx_hr_mult"] <= 0.97:
        env.append("HR-suppressing air")
    envs = ", in a " + " & ".join(env) if env else ""
    return (
        f"Today he draws {strength} {ctx['opp']} lineup (xwOBA .{int(round(opp_x * 1000)):03d}){envs}. "
        f"Sim total ~{ctx['total_mean']:.1f}."
    )


def _card(p: dict, ctx: dict | None, bets: list[dict], positive: bool) -> str:
    bm = p["biomech"]
    trends = (
        f"<span class='metric'>SIERA <b>{p['siera']:.2f}</b> {_arrow(p['d_siera'], good_up=False)}</span>"
        f"<span class='metric'>Stuff xK% <b>{p['xk'] * 100:.0f}</b> {_arrow(p['d_xk'], good_up=True)}</span>"
        f"<span class='metric'>vFA <b>{p['vfa']:.1f}</b> {_arrow(p['d_vfa'], good_up=True)}</span>"
    )
    luck = (
        f"BABIP-against <b>.{int(round(p['babip'] * 1000)):03d}</b> "
        f"(norm .290) · xwOBA−wOBA gap <b>{p['dxwoba'] * 1000:+.0f}</b> pts · "
        f"barrel% {p['barrel'] * 100:.0f} · K% {p['k_pct'] * 100:.0f} / BB% {p['bb_pct'] * 100:.0f}"
    )
    biomech = (
        f"Extension <b>{bm['ext']:.1f} ft</b> · IVB <b>{bm['ivb']:.1f} in</b> · "
        f"FB spin <b>{bm['spin']:.0f} rpm</b> · release scatter {bm['scatter']:.1f} in"
    )
    bet_html = (
        "<ul class='bets'>" + "".join(_bet_line(b) for b in bets[:4]) + "</ul>"
        if bets
        else "<p class='nobet'>No qualifying pitcher bet — the model passes his markets.</p>"
    )
    where = f" — {p['name']}'s bets" if bets else ""
    return (
        f"<div class='card'><h3>{p['name']} <span class='mu'>({ctx['matchup'] if ctx else ''})</span></h3>"
        f"<div class='trends'>{trends}</div>"
        f"<p class='luck'>{luck}</p>"
        f"<p class='bio'>{biomech}</p>"
        f"<p class='verdict'>{_verdict(p, positive)}</p>"
        f"<p class='today'>{_expect_today(p, ctx)}</p>"
        f"<p class='betlbl'><b>Pitcher bets{where}</b></p>{bet_html}</div>"
    )


CSS = """
@page { size: A4; margin: 1.4cm 1.5cm 1.6cm; }
* { box-sizing: border-box; }
body{font-family:Georgia,'Times New Roman',serif;color:#1a1a1a;line-height:1.5;font-size:10.5pt;margin:0;}
.masthead{border-bottom:3px solid #16324f;padding-bottom:8px;margin-bottom:4px;}
.brand{font-size:12pt;letter-spacing:2px;color:#c8102e;font-weight:bold;text-transform:uppercase;}
.brand .pp{color:#16324f;}
h1{font-size:22pt;color:#16324f;margin:6px 0 2px;line-height:1.08;}
.sub{color:#6b7280;font-style:italic;font-size:10.5pt;margin:0 0 2px;}
.dateline{font-size:8.5pt;color:#6b7280;letter-spacing:1px;text-transform:uppercase;margin-top:4px;}
h2{font-size:15pt;color:#16324f;border-bottom:1px solid #d7dbe0;padding-bottom:3px;margin:22px 0 4px;}
h2 .rk{font-size:9pt;color:#6b7280;font-style:italic;font-weight:normal;}
h3{font-size:12pt;color:#16324f;margin:2px 0;}
h3 .mu{font-size:9pt;color:#6b7280;font-weight:normal;}
p{margin:5px 0;}
.lead{font-size:11pt;}
.card{page-break-inside:avoid;border:1px solid #e6e8ec;border-left:4px solid #16324f;border-radius:4px;padding:9px 12px;margin:9px 0;}
.card.up{border-left-color:#2e7d32;} .card.down{border-left-color:#b23b3b;}
.trends{font-family:'DejaVu Sans',Helvetica,sans-serif;font-size:9.6pt;margin:4px 0 6px;}
.trends .metric{display:inline-block;background:#eef2f6;border-radius:10px;padding:2px 10px;margin-right:6px;}
.luck,.bio{font-family:'DejaVu Sans',Helvetica,sans-serif;font-size:8.9pt;color:#333;margin:3px 0;}
.verdict{font-size:10pt;} .today{font-size:9.6pt;color:#374151;font-style:italic;}
.pos{color:#2e7d32;font-weight:bold;} .neg{color:#b23b3b;font-weight:bold;} .flat{color:#9aa0a8;}
.betlbl{margin:6px 0 1px;font-size:9.6pt;color:#16324f;}
ul.bets{margin:1px 0 2px 0;font-size:9.2pt;font-family:'DejaVu Sans',Helvetica,sans-serif;}
ul.bets b{color:#111;} li.buy{color:#15612b;} li.pass{color:#6b7280;}
.nobet{font-size:9pt;color:#6b7280;font-style:italic;font-family:'DejaVu Sans',Helvetica,sans-serif;}
.fine{font-size:7.6pt;color:#9aa0a8;font-family:'DejaVu Sans',sans-serif;border-top:1px solid #e6e8ec;margin-top:16px;padding-top:6px;line-height:1.35;}
"""


def build_html(day: Date, pos: list, neg: list, ctxs: dict, preds: list[dict]) -> str:
    nice = day.strftime("%A, %B %-d, %Y")
    masthead = (
        "<div class='masthead'>"
        "<div class='brand'><span class='pp'>Payoff</span> Pitch · Mound Report</div>"
        "<h1>Starting Pitchers & Regression</h1>"
        "<p class='sub'>Skill (SIERA / Stuff / velo) vs. luck (BABIP, xwOBA gap) — who's due to bounce back, "
        "who's due to fall back.</p>"
        f"<div class='dateline'>Slate · {nice}</div></div>"
    )
    lead = (
        f"Good morning. Here's every starter on today's {len(pos) + len(neg)}-arm board, sorted by the story "
        "the underlying numbers tell. We pull each guy's 6-week SIERA (his true run-prevention skill), his Stuff "
        "read via xK% off CSW%/SwStr%, and his 3-week fastball velocity — each carries an arrow for which way it's "
        "trending. Then we line the skill up against the luck: BABIP-against and the xwOBA-minus-wOBA gap tell us "
        "whether results have gotten ahead of, or behind, the actual contact he's allowing. Arms that have been "
        "unlucky are due to improve; arms that have been bailed out are due to give it back. Bets tied to each "
        "starter are listed under him. Model preview, not betting advice."
    )

    def section(title: str, rows: list, positive: bool) -> str:
        cls = "up" if positive else "down"
        cards = "".join(
            f"<div class='card {cls}'>" + _card(p, ctxs.get(p["name"]), _bets_for(idmap[p["name"]], preds), positive)[len("<div class='card'>"):]
            for p in rows
        )
        return f"<h2>{title} <span class='rk'>most → least likely</span></h2>{cards}"

    idmap = _pitcher_id_map(preds)
    body = section("Positive regression — due to improve", pos, True)
    body += section("Negative regression — due to decline", neg, False)
    fine = (
        "<p class='fine'>Methodology: SIERA and rates are computed from each pitcher's trailing 6-week (42-day) "
        "Statcast slice; vFA is the mean four-seam/sinker velocity over the last 3 weeks. Stuff is the engine's "
        "xK% proxy (a CSW%/SwStr% fit) — FanGraphs Stuff+ is only shown when a subscription feed is loaded, which "
        "it is not in this environment. Regression rank is z(BABIP−.290) + z(wOBA−xwOBA): positive = unlucky (due "
        "to improve), negative = fortunate (due to decline). Biomechanics are release-tracking proxies (extension, "
        "induced vertical break, fastball spin, release scatter), not lab motion-capture. Model preview, not "
        "investment advice.</p>"
    )
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head>"
        f"<body>{masthead}<p class='lead'>{lead}</p>{body}{fine}</body></html>"
    )


def build_narration(day: Date, pos: list, neg: list, ctxs: dict, preds: list[dict]) -> str:
    idmap = _pitcher_id_map(preds)
    parts = [
        f"What's up everybody, welcome into the Payoff Pitch Mound Report for {day.strftime('%A, %B %-d')}. "
        "We're breaking down every starting pitcher by skill versus luck, and who's due to regress which way. "
    ]

    def read(rows, positive):
        verb = "due to improve" if positive else "due to give it back"
        for i, p in enumerate(rows, 1):
            b = _bets_for(idmap[p["name"]], preds)
            bet = ""
            buys = [x for x in b if x["tier"] in ("Strong buy", "Moderate buy")]
            if buys:
                bet = f" The bet: {buys[0]['selection']}, model {buys[0]['model_prob'] * 100:.0f} percent."
            parts.append(
                f"Number {i}, {p['name']}, SIERA {p['siera']:.2f}, BABIP against "
                f"{int(round(p['babip'] * 1000))}, {verb}.{bet} "
            )

    parts.append("First, the arms due to improve, most likely first. ")
    read(pos, True)
    parts.append("Now the flip side, the arms due to decline, most likely first. ")
    read(neg, False)
    parts.append(
        "That's the mound report. Skill wins over a season, luck wins tonight. Payoff Pitch, out."
    )
    return "".join(parts)



def main() -> None:
    cfg = load_config()
    day = Date(2026, 7, 31)
    previews = json.load(open(cfg.audit_dir / f"previews_{day.isoformat()}.json"))
    preds = json.load(open(cfg.audit_dir / f"predictions_{day.isoformat()}.json"))
    df = pd.read_pickle(cfg.cache_dir / "statcast_2026-06-19_2026-07-30.pkl")
    pos, neg, ctxs = build_profiles(previews, preds, df)

    html = build_html(day, pos, neg, ctxs, preds)
    narr = build_narration(day, pos, neg, ctxs, preds)
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / f"PayoffPitch_Mound_{day.isoformat()}.html").write_text(html)
    pdf = out / f"PayoffPitch_Mound_{day.isoformat()}.pdf"
    pdf.write_bytes(to_pdf(html))
    mp3 = out / f"PayoffPitch_Mound_{day.isoformat()}.mp3"
    to_mp3(narr, mp3)

    print(f"positive={len(pos)}  negative={len(neg)}")
    print("POSITIVE (improve):")
    for p in pos:
        print(f"  {p['reg_index']:+.2f}  {p['name']:22} SIERA {p['siera']:.2f}  BABIP .{int(round(p['babip']*1000)):03d}  dxwoba {p['dxwoba']*1000:+.0f}")
    print("NEGATIVE (decline):")
    for p in neg:
        print(f"  {p['reg_index']:+.2f}  {p['name']:22} SIERA {p['siera']:.2f}  BABIP .{int(round(p['babip']*1000)):03d}  dxwoba {p['dxwoba']*1000:+.0f}")
    print("PDF:", pdf)
    print("MP3:", mp3)


if __name__ == "__main__":
    main()
