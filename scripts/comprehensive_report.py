"""One comprehensive slate article: daily previews + pitcher regression + batter
regression, rendered to a single PDF and a single MP3.

Part 1 -- Daily slate previews (reuses ``mlb_engine.output.daily_preview``):
          per-game arms-vs-bats, regression watch, game shape, weather, best bets.
Part 2 -- Mound report (reuses ``scripts.pitcher_slate_analysis``): every starter
          ranked by expected regression with SIERA / Stuff / vFA trends.
Part 3 -- Batter regression (this file): the top-10 hitters due to heat up
          (positive regression) and the top-10 due to cool off (negative), ranked
          most -> least by the engine's xwOBA-minus-wOBA gap, each with a 6-week ->
          3-week wOBA trend arrow and the model's bet.

The three articles keep their own house styling; WeasyPrint renders each and the
page lists are concatenated into one document, and the three narrations are joined
into one MP3.  Model preview, not investment advice.
"""

from __future__ import annotations

import dataclasses
import json
import re
from datetime import date as Date

import pandas as pd

import scripts.pitcher_slate_analysis as psa
from mlb_engine.config import load_config
from mlb_engine.features.regression import BL_XSLG, build_batter_regression
from mlb_engine.market.tiers import Tier
from mlb_engine.output.audit_insight import to_mp3
from mlb_engine.output.daily_preview import build_preview_report
from mlb_engine.preview import load_previews
from mlb_engine.recommendations import Recommendation

DAY = Date(2026, 7, 31)
STATCAST_PKL = "statcast_2026-06-19_2026-07-30.pkl"
RECENT_DAYS = 21
MIN_BBE = 25
TOPN = 10
BATTER_STATS = ("hr", "tb", "h", "1b", "r", "rbi", "hrr")


# --- schema-tolerant rec loader (predictions JSON may carry PR#48 fields) ---
def load_recs(path) -> list[Recommendation]:
    fields = {f.name for f in dataclasses.fields(Recommendation)}
    out: list[Recommendation] = []
    for d in json.loads(path.read_text()):
        d = {k: v for k, v in d.items() if k in fields}
        d["game_date"] = Date.fromisoformat(d["game_date"])
        d["tier"] = Tier(d["tier"])
        out.append(Recommendation(**d))
    return out


# --- batter regression -----------------------------------------------------
# selection is "{name} {stat} {side}{line}" ("Matt McLain 1B o0.5", "Carlos
# Narvaez H+R+RBI u1.5"); strip the trailing market off to leave the hitter.
# Both sides: every prop has had its under priced since #144, and a side this
# misses leaves the market glued to the name, which then reads as a separate
# hitter carrying identical contact -- ten of them fill the top ten.
_SEL_RE = re.compile(r"\s+[A-Za-z0-9+]+\s+[ou]\d.*$")


def _batter_name(sel: str) -> str:
    return _SEL_RE.sub("", sel)


def _batter_id_map(preds: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in preds:
        if r["market"].startswith("batter_") and r.get("player_id"):
            out[_batter_name(r["selection"])] = r["player_id"]
    return out


def _batter_ctx(preds: list[dict], pv_by_pk: dict[int, dict]) -> dict[str, dict]:
    ctx: dict[str, dict] = {}
    for r in preds:
        if not r["market"].startswith("batter_"):
            continue
        sel = _batter_name(r["selection"])
        if sel in ctx:
            continue
        pk = r.get("game_pk")
        g = pv_by_pk.get(pk, {})
        ctx[sel] = {
            "matchup": r.get("matchup", ""),
            "park_factor": g.get("park_factor"),
            "wx_hr_mult": g.get("wx_hr_mult"),
        }
    return ctx


def _woba(slice_df: pd.DataFrame) -> float:
    if slice_df.empty:
        return float("nan")
    return build_batter_regression(slice_df).woba


def analyze_batter(name: str, pid: int, df: pd.DataFrame, cutoff: Date) -> dict:
    sl = df[df["batter"] == pid]
    reg = build_batter_regression(sl)
    recent = sl[pd.to_datetime(sl["game_date"]).dt.date > cutoff]
    return {
        "name": name,
        "bbe": reg.bbe,
        "woba": reg.woba,
        "xwoba": reg.xwoba,
        "dxwoba": reg.dxwoba,  # xwoba - woba: + => underperforming (heat up)
        "xslg": reg.xslg,
        "barrel": reg.barrel_rate,
        "babip": reg.babip,
        "hard_hit": reg.hard_hit,
        "woba6": reg.woba,
        "woba3": _woba(recent),
    }


def _best_batter_bet(pid: int, preds: list[dict]) -> dict | None:
    cands = [
        r for r in preds
        if r.get("player_id") == pid and r["market"].startswith("batter_")
    ]
    if not cands:
        return None
    tier_rank = {"Strong buy": 0, "Moderate buy": 1, "Pass": 2}
    cands.sort(key=lambda r: (tier_rank.get(r["tier"], 3), -(r.get("ev") or -9)))
    return cands[0]


def build_batter_profiles(preds: list[dict], df: pd.DataFrame):
    idmap = _batter_id_map(preds)
    maxd = pd.to_datetime(df["game_date"]).dt.date.max()
    cutoff = maxd - pd.Timedelta(days=RECENT_DAYS)
    cutoff = cutoff if isinstance(cutoff, Date) else cutoff.date()
    profs = []
    seen: set[int] = set()
    for name, pid in idmap.items():
        # A hitter is one hitter however his name reaches the sheet: two spellings
        # of the same id must not both be ranked.
        if pid in seen:
            continue
        seen.add(pid)
        p = analyze_batter(name, pid, df, cutoff)
        if p["bbe"] < MIN_BBE:
            continue
        p["pid"] = pid
        profs.append(p)
    pos = sorted([p for p in profs if p["dxwoba"] > 0], key=lambda p: -p["dxwoba"])[:TOPN]
    neg = sorted([p for p in profs if p["dxwoba"] < 0], key=lambda p: p["dxwoba"])[:TOPN]
    return pos, neg


def _bat_card(p: dict, ctx: dict | None, bet: dict | None, positive: bool) -> str:
    trend_d = p["woba3"] - p["woba6"]
    trend = psa._arrow(trend_d, good_up=True)
    gap = p["dxwoba"] * 1000
    power = "plus" if p["xslg"] > BL_XSLG + 0.03 else "light" if p["xslg"] < BL_XSLG - 0.03 else "average"
    if positive:
        verdict = (
            f"Underperforming his contact: xwOBA .{int(round(p['xwoba'] * 1000)):03d} vs actual wOBA "
            f".{int(round(p['woba'] * 1000)):03d} (gap +{gap:.0f}), {power} expected power (xSLG "
            f".{int(round(p['xslg'] * 1000)):03d}). Barrels and hard contact say the results are due to catch up — a buy-low bat."
        )
    else:
        verdict = (
            f"Overperforming his contact: actual wOBA .{int(round(p['woba'] * 1000)):03d} vs xwOBA "
            f".{int(round(p['xwoba'] * 1000)):03d} (gap {gap:.0f}), {power} expected power (xSLG "
            f".{int(round(p['xslg'] * 1000)):03d}). Some of the hits are fool's gold — regression back down is coming."
        )
    env = ""
    if ctx:
        bits = []
        if ctx.get("park_factor") is not None:
            pf = ctx["park_factor"]
            bits.append("hitter park" if pf > 101 else "pitcher park" if pf < 99 else "neutral park")
        if ctx.get("wx_hr_mult") is not None and ctx["wx_hr_mult"] >= 1.03:
            bits.append("HR-boosting air")
        env = f" · {', '.join(bits)}" if bits else ""
    bet_html = "<span class='nobet'>model passes his props</span>"
    if bet is not None:
        odds = "" if bet.get("market_american") is None else f" ({bet['market_american']:+.0f})"
        ev = "" if bet.get("ev") is None else f", EV {bet['ev']:+.2f}"
        cls = "buy" if bet["tier"] in ("Strong buy", "Moderate buy") else "pass"
        bet_html = (
            f"<span class='{cls}'><b>{bet['selection']}</b>{odds} — model "
            f"{bet['model_prob'] * 100:.0f}%{ev} · {bet['tier']}</span>"
        )
    return (
        f"<div class='bcard'><div class='bhead'><span class='bn'>{p['name']}</span>"
        f"<span class='bmu'>{ctx['matchup'] if ctx else ''}{env}</span></div>"
        f"<div class='bmetrics'>wOBA <b>.{int(round(p['woba'] * 1000)):03d}</b> "
        f"<span class='tr'>6wk&#8594;3wk {trend}</span> · xwOBA .{int(round(p['xwoba'] * 1000)):03d} · "
        f"gap <b>{gap:+.0f}</b> · barrel% {p['barrel'] * 100:.0f} · BABIP .{int(round(p['babip'] * 1000)):03d}</div>"
        f"<div class='bverdict'>{verdict}</div>"
        f"<div class='bbet'><b>Bet:</b> {bet_html}</div></div>"
    )


BATTER_CSS = """
@page { size: A4; margin: 1.4cm 1.5cm 1.6cm; }
* { box-sizing: border-box; }
body{font-family:Georgia,'Times New Roman',serif;color:#1a1a1a;line-height:1.5;font-size:10.5pt;margin:0;}
.masthead{border-bottom:3px solid #6b21a8;padding-bottom:8px;margin-bottom:4px;}
.brand{font-size:12pt;letter-spacing:2px;color:#c8102e;font-weight:bold;text-transform:uppercase;}
.brand .pp{color:#6b21a8;}
h1{font-size:22pt;color:#6b21a8;margin:6px 0 2px;line-height:1.08;}
.sub{color:#6b7280;font-style:italic;font-size:10.5pt;margin:0 0 2px;}
.dateline{font-size:8.5pt;color:#6b7280;letter-spacing:1px;text-transform:uppercase;margin-top:4px;}
h2{font-size:15pt;color:#6b21a8;border-bottom:1px solid #d7dbe0;padding-bottom:3px;margin:20px 0 6px;}
h2 .rk{font-size:9pt;color:#6b7280;font-style:italic;font-weight:normal;}
p.lead{font-size:11pt;}
.bcard{page-break-inside:avoid;border:1px solid #e6e8ec;border-left:4px solid #6b21a8;border-radius:4px;padding:7px 11px;margin:7px 0;}
.bcard.up{border-left-color:#2e7d32;} .bcard.down{border-left-color:#b23b3b;}
.bhead{display:flex;justify-content:space-between;align-items:baseline;}
.bn{font-size:11.5pt;font-weight:bold;color:#16324f;font-family:Georgia,serif;}
.bmu{font-size:8.4pt;color:#6b7280;font-family:'DejaVu Sans',sans-serif;}
.bmetrics{font-family:'DejaVu Sans',Helvetica,sans-serif;font-size:8.9pt;color:#333;margin:3px 0;}
.bmetrics .tr{color:#555;}
.bverdict{font-size:9.6pt;margin:2px 0;}
.bbet{font-family:'DejaVu Sans',sans-serif;font-size:8.9pt;margin-top:2px;}
.pos{color:#2e7d32;font-weight:bold;} .neg{color:#b23b3b;font-weight:bold;} .flat{color:#9aa0a8;}
.buy{color:#15612b;} .pass{color:#6b7280;} .nobet{color:#6b7280;font-style:italic;}
.fine{font-size:7.6pt;color:#9aa0a8;font-family:'DejaVu Sans',sans-serif;border-top:1px solid #e6e8ec;margin-top:16px;padding-top:6px;line-height:1.35;}
"""


def build_batter_html(day: Date, pos: list, neg: list, ctxs: dict, preds: list[dict]) -> str:
    nice = day.strftime("%A, %B %-d, %Y")
    masthead = (
        "<div class='masthead'><div class='brand'><span class='pp'>Payoff</span> Pitch · Batter Desk</div>"
        "<h1>Batter Regression Watch</h1>"
        "<p class='sub'>Who the Statcast contact says is about to heat up, and who's living on borrowed hits.</p>"
        f"<div class='dateline'>Slate · {nice}</div></div>"
    )
    lead = (
        "Same idea as the arms, flipped to the bats. For every hitter in today's projected lineups we take the "
        "engine's 6-week Statcast read and compare the quality of contact (xwOBA) to what's actually landed (wOBA). "
        "A positive gap means he's hitting the ball better than the box score shows — a buy-low bat due to heat up; "
        "a negative gap means results have outrun the contact — regression down is coming. Each hitter carries his "
        "6-week&#8594;3-week wOBA trend (green up = getting hot) and the model's bet on him. Top 10 each way, most "
        "likely to regress first. Model preview, not betting advice."
    )
    idmap = _batter_id_map(preds)

    def section(title, rows, positive):
        cls = "up" if positive else "down"

        def _card(p) -> str:
            inner = _bat_card(
                p, ctxs.get(p["name"]),
                _best_batter_bet(idmap.get(p["name"], -1), preds), positive,
            )[len("<div class='bcard'>"):]
            return f"<div class='bcard {cls}'>{inner}"

        cards = "".join(_card(p) for p in rows)
        return f"<h2>{title} <span class='rk'>most &#8594; least likely</span></h2>{cards}"

    body = section("Positive regression — due to heat up (buy-low)", pos, True)
    body += section("Negative regression — due to cool off", neg, False)
    fine = (
        "<p class='fine'>Methodology: each hitter's wOBA, xwOBA and xSLG come from his trailing 6-week (42-day) "
        "Statcast batted-ball slice; the trend arrow compares that 6-week wOBA to the last 3 weeks. Regression rank "
        "is the xwOBA-minus-wOBA gap (positive = underperforming contact, due to improve). Minimum 25 batted-ball "
        "events to qualify. Model preview, not investment advice.</p>"
    )
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{BATTER_CSS}</style></head>"
        f"<body>{masthead}<p class='lead'>{lead}</p>{body}{fine}</body></html>"
    )


def build_batter_narration(day: Date, pos: list, neg: list, preds: list[dict]) -> str:
    idmap = _batter_id_map(preds)
    parts = ["Now to the bats, the batter regression watch. "]

    def read(rows, positive):
        verb = "due to heat up" if positive else "due to cool off"
        for i, p in enumerate(rows, 1):
            bet = _best_batter_bet(idmap.get(p["name"], -1), preds)
            btxt = ""
            if bet and bet["tier"] in ("Strong buy", "Moderate buy"):
                btxt = f" The bet: {bet['selection']}."
            parts.append(
                f"Number {i}, {p['name']}, wOBA {int(round(p['woba'] * 1000))} against expected "
                f"{int(round(p['xwoba'] * 1000))}, {verb}.{btxt} "
            )

    parts.append("First, the hitters due to heat up, the buy-low bats. ")
    read(pos, True)
    parts.append("Now the hitters due to cool off. ")
    read(neg, False)
    parts.append("That wraps the comprehensive slate report. Payoff Pitch, out.")
    return "".join(parts)


# --- merge -----------------------------------------------------------------
def merge_pdf(htmls: list[str]):
    from weasyprint import HTML

    docs = [HTML(string=h).render() for h in htmls]
    pages = [pg for d in docs for pg in d.pages]
    return docs[0].copy(pages).write_pdf()


def main() -> None:
    cfg = load_config()
    day = DAY
    pv_raw = json.load(open(cfg.audit_dir / f"previews_{day.isoformat()}.json"))
    preds_raw = json.load(open(cfg.audit_dir / f"predictions_{day.isoformat()}.json"))
    pv_by_pk = {g["game_pk"]: g for g in pv_raw}

    previews = load_previews(cfg.audit_dir / f"previews_{day.isoformat()}.json")
    recs = load_recs(cfg.audit_dir / f"predictions_{day.isoformat()}.json")
    df = pd.read_pickle(cfg.cache_dir / STATCAST_PKL)

    # Part 1: daily slate previews
    slate_html, slate_narr = build_preview_report(day, previews, recs)

    # Part 2: pitcher regression
    ppos, pneg, pctxs = psa.build_profiles(pv_raw, preds_raw, df)
    pitcher_html = psa.build_html(day, ppos, pneg, pctxs, preds_raw)
    pitcher_narr = psa.build_narration(day, ppos, pneg, pctxs, preds_raw)

    # Part 3: batter regression
    bpos, bneg = build_batter_profiles(preds_raw, df)
    bctxs = _batter_ctx(preds_raw, pv_by_pk)
    batter_html = build_batter_html(day, bpos, bneg, bctxs, preds_raw)
    batter_narr = build_batter_narration(day, bpos, bneg, preds_raw)

    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / f"PayoffPitch_Comprehensive_{day.isoformat()}.pdf"
    pdf.write_bytes(merge_pdf([slate_html, pitcher_html, batter_html]))
    mp3 = out / f"PayoffPitch_Comprehensive_{day.isoformat()}.mp3"
    to_mp3(slate_narr + " " + pitcher_narr + " " + batter_narr, mp3)

    print(f"batters: positive={len(bpos)} negative={len(bneg)}")
    print("POSITIVE (heat up):")
    for p in bpos:
        print(f"  gap {p['dxwoba'] * 1000:+.0f}  {p['name']:22} wOBA .{int(round(p['woba'] * 1000)):03d} x .{int(round(p['xwoba'] * 1000)):03d}  bbe {p['bbe']}")
    print("NEGATIVE (cool off):")
    for p in bneg:
        print(f"  gap {p['dxwoba'] * 1000:+.0f}  {p['name']:22} wOBA .{int(round(p['woba'] * 1000)):03d} x .{int(round(p['xwoba'] * 1000)):03d}  bbe {p['bbe']}")
    print("PDF:", pdf)
    print("MP3:", mp3)


if __name__ == "__main__":
    main()
