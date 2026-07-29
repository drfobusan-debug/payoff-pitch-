"""Nightly audit *insight* report — the deep-dive that ships with every audit.

On top of the Excel ledger, the nightly audit emails a Morningstar-style
"Audit Desk" article (PDF) plus an MP3 narration. This module builds both.

What the report answers, in order:

1. **Daily slate** — whole-engine PPV / NPV for the graded slate.
2. **All slates** — cumulative PPV / NPV across the persisted history.
3. **Prop families** — moneyline / run line / totals / batter / pitcher.
4. **Every specific prop** — PPV / NPV per market.
5. **Why the failing props fail** — for each leaking market it takes the
   favored picks (true positives vs false positives) and the faded picks
   (true negatives vs false negatives) and asks *which engine metric actually
   separates the winners from the losers*. Point-biserial correlation with a
   significance test surfaces the discriminating metrics, so the finding is a
   real statistic, not a vibe.

PPV / NPV use the whole-engine convention already used across the audit: a pick
is *favored* when ``model_prob >= 0.5`` and *positive* when the selection won
(pushes dropped). TP/FP are favored wins/losses; TN/FN are faded losses/wins.

Every night we append that slate's graded picks (with the engine's own metric
columns) to ``graded_metrics.csv`` so the discriminant analysis runs on the full
history, not just one noisy slate.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from datetime import date as Date
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_engine.recommendations import Recommendation

logger = logging.getLogger(__name__)

# --- thresholds ------------------------------------------------------------
BREAKEVEN = 0.524  # -110 juice: below this, a "buy" side bleeds money
MIN_FAVORED = 10  # favored picks needed before we judge a prop's PPV
MIN_FADED = 10  # faded picks needed before we judge a prop's NPV
MIN_GROUP = 6  # per-outcome sample needed to run a discriminant test
NPV_FLOOR = 0.62  # faded winners above this rate → reclaimable FN pocket
PVAL = 0.05

STORE_NAME = "graded_metrics.csv"

# Numeric engine metrics tested as discriminators. Not every prop populates
# every column; the analysis skips columns without enough signal per prop.
METRIC_COLS: list[str] = [
    "model_prob",
    "raw_prob",
    "ev",
    "edge",
    "fair_prob",
    "bet_prob",
    "handle_pct",
    "bets_pct",
    "factor",
    "score",
    "line",
    "bat_xslg",
    "bat_k_pct",
    "bat_bb_pct",
    "bat_singles_under",
    "opp_starter_siera",
    "park_factor",
    "carry_factor",
    "wx_hr_mult",
    "xrd",
    "xrd_sd",
]

METRIC_LABELS: dict[str, str] = {
    "model_prob": "Model probability",
    "raw_prob": "Raw (pre-calibration) probability",
    "ev": "Expected value / $1",
    "edge": "Edge vs market",
    "fair_prob": "Devigged fair probability",
    "bet_prob": "Bet probability",
    "handle_pct": "Handle %",
    "bets_pct": "Bets %",
    "factor": "Signal factor",
    "score": "Selection score",
    "line": "Posted line",
    "bat_xslg": "Batter xSLG",
    "bat_k_pct": "Batter K%",
    "bat_bb_pct": "Batter BB%",
    "bat_singles_under": "Singles-under lean",
    "opp_starter_siera": "Opp. starter SIERA",
    "park_factor": "Park factor",
    "carry_factor": "Ball-carry factor",
    "wx_hr_mult": "Weather HR multiplier",
    "xrd": "Expected run differential",
    "xrd_sd": "xRD volatility",
}

MARKET_LABEL: dict[str, str] = {
    "game_ml": "Game moneyline",
    "game_rl": "Game run line",
    "game_total": "Game total",
    "f5_ml": "First-5 moneyline",
    "f5_rl": "First-5 run line",
    "f5_total": "First-5 total",
    "pitcher_k": "Pitcher strikeouts",
    "pitcher_outs": "Pitcher outs",
    "pitcher_h": "Pitcher hits allowed",
    "pitcher_bb": "Pitcher walks",
    "pitcher_er": "Pitcher earned runs",
    "batter_h": "Batter hits",
    "batter_tb": "Batter total bases",
    "batter_hr": "Batter home run",
    "batter_hrr": "Batter H+R+RBI",
    "batter_1b": "Batter singles",
    "batter_2b": "Batter doubles",
    "batter_3b": "Batter triples",
    "batter_r": "Batter runs",
    "batter_rbi": "Batter RBI",
    "batter_bb": "Batter walks",
    "batter_k": "Batter strikeouts",
}

FAMILY_LABEL: dict[str, str] = {
    "moneyline": "Moneyline",
    "runline": "Run line",
    "totals": "Totals",
    "batter": "Batter props",
    "pitcher": "Pitcher props",
    "other": "Other",
}


def market_label(market: str) -> str:
    return MARKET_LABEL.get(market, market)


def family_of(market: str) -> str:
    if market.endswith("_ml"):
        return "moneyline"
    if market.endswith("_rl"):
        return "runline"
    if market.endswith("_total"):
        return "totals"
    if market.startswith("batter"):
        return "batter"
    if market.startswith("pitcher"):
        return "pitcher"
    return "other"


# --- persistence -----------------------------------------------------------
def graded_to_frame(
    graded: list[tuple[Recommendation, str]], audit_date: Date
) -> pd.DataFrame:
    """Flatten the day's graded (rec, result) tuples into a metric frame."""
    rows: list[dict[str, object]] = []
    for rec, result in graded:
        row: dict[str, object] = {
            "date": audit_date.isoformat(),
            "game_pk": rec.game_pk,
            "matchup": rec.matchup,
            "category": rec.category,
            "market": rec.market,
            "selection": rec.selection,
            "player_id": rec.player_id,
            "result": result,
        }
        for col in METRIC_COLS:
            row[col] = getattr(rec, col, None)
        rows.append(row)
    return pd.DataFrame(rows)


def update_store(store_path: Path, day: pd.DataFrame, audit_date: Date) -> pd.DataFrame:
    """Replace this date's rows in the cumulative store and return the whole store."""
    iso = audit_date.isoformat()
    if store_path.exists():
        prior = pd.read_csv(store_path)
        prior = prior[prior["date"] != iso]
        combined = pd.concat([prior, day], ignore_index=True)
    else:
        combined = day.copy()
    store_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(store_path, index=False)
    return combined


# --- classification & stats ------------------------------------------------
def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Add favored/won columns; drop pushes and ungradeable rows."""
    out = df[df["result"].isin(["win", "loss"])].copy()
    out["won"] = (out["result"] == "win").astype(int)
    out["favored"] = (pd.to_numeric(out["model_prob"], errors="coerce") >= 0.5).astype(int)
    out["family"] = out["market"].astype(str).map(family_of)
    return out


@dataclass
class PropStat:
    key: str
    label: str
    family: str
    n: int
    n_fav: int
    n_fade: int
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def ppv(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def npv(self) -> float:
        d = self.tn + self.fn
        return self.tn / d if d else float("nan")

    @property
    def fav_win_pct(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def leaks_ppv(self) -> bool:
        return self.n_fav >= MIN_FAVORED and self.ppv < BREAKEVEN

    @property
    def reclaimable_fn(self) -> bool:
        # faded picks that keep winning → NPV low → false-negative pocket
        d = self.tn + self.fn
        if self.n_fade < MIN_FADED or not d:
            return False
        return self.npv < NPV_FLOOR


def _counts(sub: pd.DataFrame) -> tuple[int, int, int, int]:
    tp = int(((sub["favored"] == 1) & (sub["won"] == 1)).sum())
    fp = int(((sub["favored"] == 1) & (sub["won"] == 0)).sum())
    fn = int(((sub["favored"] == 0) & (sub["won"] == 1)).sum())
    tn = int(((sub["favored"] == 0) & (sub["won"] == 0)).sum())
    return tp, fp, fn, tn


def whole_engine_stat(df: pd.DataFrame, label: str = "Whole engine") -> PropStat:
    tp, fp, fn, tn = _counts(df)
    return PropStat(
        key="ALL",
        label=label,
        family="all",
        n=len(df),
        n_fav=tp + fp,
        n_fade=tn + fn,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def group_stats(df: pd.DataFrame, by: str) -> list[PropStat]:
    stats: list[PropStat] = []
    for key, sub in df.groupby(by):
        tp, fp, fn, tn = _counts(sub)
        if by == "family":
            label = FAMILY_LABEL.get(str(key), str(key))
            fam = str(key)
        else:
            label = market_label(str(key))
            fam = family_of(str(key))
        stats.append(
            PropStat(
                key=str(key),
                label=label,
                family=fam,
                n=len(sub),
                n_fav=tp + fp,
                n_fade=tn + fn,
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
            )
        )
    stats.sort(key=lambda s: s.n, reverse=True)
    return stats


# --- discriminant analysis -------------------------------------------------
@dataclass
class Discriminator:
    metric: str
    label: str
    mean_win: float
    mean_loss: float
    r: float
    p: float
    n_win: int
    n_loss: int

    @property
    def direction(self) -> str:
        return "higher in winners" if self.mean_win > self.mean_loss else "higher in losers"


def _discriminate(sub: pd.DataFrame) -> list[Discriminator]:
    """Point-biserial: which metrics separate wins (1) from losses (0)?"""
    from scipy.stats import pointbiserialr

    found: list[Discriminator] = []
    y = sub["won"].to_numpy()
    if y.sum() < MIN_GROUP or (len(y) - y.sum()) < MIN_GROUP:
        return found
    for col in METRIC_COLS:
        if col not in sub.columns:
            continue
        x = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(x)
        if mask.sum() < 2 * MIN_GROUP:
            continue
        xv, yv = x[mask], y[mask]
        if yv.sum() < MIN_GROUP or (len(yv) - yv.sum()) < MIN_GROUP:
            continue
        if not np.isfinite(np.nanstd(xv)) or np.nanstd(xv) < 1e-9:
            continue
        try:
            r, p = pointbiserialr(yv, xv)
        except Exception:  # noqa: BLE001
            continue
        if np.isnan(r) or p > PVAL:
            continue
        found.append(
            Discriminator(
                metric=col,
                label=METRIC_LABELS.get(col, col),
                mean_win=float(xv[yv == 1].mean()),
                mean_loss=float(xv[yv == 0].mean()),
                r=float(r),
                p=float(p),
                n_win=int(yv.sum()),
                n_loss=int(len(yv) - yv.sum()),
            )
        )
    found.sort(key=lambda d: d.p)
    return found


@dataclass
class PropDiag:
    stat: PropStat
    fp_discriminators: list[Discriminator] = field(default_factory=list)  # favored: TP vs FP
    fn_discriminators: list[Discriminator] = field(default_factory=list)  # faded: FN vs TN


def diagnose_failing(cum: pd.DataFrame, market_stats: list[PropStat]) -> list[PropDiag]:
    diags: list[PropDiag] = []
    for st in market_stats:
        if not (st.leaks_ppv or st.reclaimable_fn):
            continue
        sub = cum[cum["market"] == st.key]
        diag = PropDiag(stat=st)
        if st.leaks_ppv:
            diag.fp_discriminators = _discriminate(sub[sub["favored"] == 1])
        if st.reclaimable_fn:
            diag.fn_discriminators = _discriminate(sub[sub["favored"] == 0])
        diags.append(diag)
    # worst PPV first
    diags.sort(key=lambda d: (d.stat.ppv if not np.isnan(d.stat.ppv) else 1.0))
    return diags


# --- charts ----------------------------------------------------------------
NAVY = "#16324f"
RED = "#c8102e"
GOLD = "#b8860b"
POS = "#2e7d32"
NEG = "#b23b3b"
INK = "#1a1a1a"
MUTE = "#6b7280"


def _fig_b64(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _family_chart(fams: list[PropStat]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = [f for f in fams if f.n_fav + f.n_fade > 0]
    labels = [f.label for f in fams]
    ppv = [f.ppv * 100 if not np.isnan(f.ppv) else 0 for f in fams]
    npv = [f.npv * 100 if not np.isnan(f.npv) else 0 for f in fams]
    y = np.arange(len(labels))
    h = 0.38
    fig, ax = plt.subplots(figsize=(7.6, max(1.8, 0.62 * len(labels) + 0.8)))
    ax.barh(y - h / 2, ppv, height=h, color=NAVY, zorder=3, label="PPV (buy side)")
    ax.barh(y + h / 2, npv, height=h, color=GOLD, zorder=3, label="NPV (fade side)")
    ax.axvline(BREAKEVEN * 100, color=RED, lw=1.2, ls="--", zorder=4)
    ax.text(BREAKEVEN * 100 + 0.4, len(labels) - 0.4, "52.4% breakeven", color=RED, fontsize=8)
    for yi, (pv, nv) in enumerate(zip(ppv, npv, strict=False)):
        ax.text(pv + 0.6, yi - h / 2, f"{pv:.0f}", va="center", fontsize=8, color=INK)
        ax.text(nv + 0.6, yi + h / 2, f"{nv:.0f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Predictive value (%)", fontsize=9, color=MUTE)
    ax.set_xlim(0, 105)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.grid(axis="x", color="#e6e8ec", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("PPV vs NPV by prop family (all slates)", fontsize=11, color=NAVY, loc="left")
    return _fig_b64(fig)


def _market_chart(mkts: list[PropStat]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mkts = [m for m in mkts if m.n_fav >= 3]
    mkts = sorted(mkts, key=lambda m: (m.ppv if not np.isnan(m.ppv) else 0))
    labels = [f"{m.label}" for m in mkts]
    ppv = [m.ppv * 100 if not np.isnan(m.ppv) else 0 for m in mkts]
    colors = [NEG if p < BREAKEVEN * 100 else POS for p in ppv]
    fig, ax = plt.subplots(figsize=(7.6, max(1.8, 0.42 * len(labels) + 0.8)))
    y = np.arange(len(labels))
    ax.barh(y, ppv, color=colors, height=0.66, zorder=3)
    ax.axvline(BREAKEVEN * 100, color=RED, lw=1.2, ls="--", zorder=4)
    for yi, (p, m) in enumerate(zip(ppv, mkts, strict=False)):
        ax.text(p + 0.6, yi, f"{p:.0f}% (n={m.n_fav})", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Buy-side win rate — PPV (%)", fontsize=9, color=MUTE)
    ax.grid(axis="x", color="#e6e8ec", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Every prop, worst buy-side first (red = below breakeven)", fontsize=11, color=NAVY, loc="left")
    return _fig_b64(fig)


def _discriminator_chart(title: str, discs: list[Discriminator]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    discs = discs[:6]
    labels = [d.label for d in discs]
    # z-score the win/loss means per metric so they're comparable on one axis
    fig, ax = plt.subplots(figsize=(7.6, max(1.6, 0.5 * len(labels) + 0.8)))
    y = np.arange(len(labels))
    h = 0.38
    win_vals, loss_vals = [], []
    for d in discs:
        span = abs(d.mean_win) + abs(d.mean_loss)
        scale = span if span else 1.0
        win_vals.append(d.mean_win / scale)
        loss_vals.append(d.mean_loss / scale)
    ax.barh(y - h / 2, win_vals, height=h, color=POS, zorder=3, label="winners (mean)")
    ax.barh(y + h / 2, loss_vals, height=h, color=NEG, zorder=3, label="losers (mean)")
    lo = min([0.0, *win_vals, *loss_vals])
    hi = max([0.0, *win_vals, *loss_vals])
    ann_x = hi + 0.06
    for yi, d in enumerate(discs):
        ax.text(
            ann_x,
            yi,
            f"r={d.r:+.2f}, p={d.p:.3f}",
            va="center",
            fontsize=7.5,
            color=MUTE,
        )
    ax.set_xlim(lo - 0.08, hi + 0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.axvline(0, color="#c9ccd1", lw=0.8)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, -0.04), ncol=2, frameon=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([])
    ax.set_title(title, fontsize=10.5, color=NAVY, loc="left")
    return _fig_b64(fig)


# --- HTML shell (Morningstar house style) ----------------------------------
CSS = """
@page { size: A4; margin: 1.4cm 1.5cm 1.6cm; }
* { box-sizing: border-box; }
body{font-family:Georgia,'Times New Roman',serif;color:#1a1a1a;line-height:1.5;font-size:10.5pt;margin:0;}
.masthead{border-bottom:3px solid #16324f;padding-bottom:8px;margin-bottom:4px;}
.brand{font-family:Georgia,serif;font-size:12pt;letter-spacing:2px;color:#c8102e;font-weight:bold;text-transform:uppercase;}
.brand .pp{color:#16324f;}
h1{font-family:Georgia,serif;font-size:22pt;color:#16324f;margin:6px 0 2px;line-height:1.08;}
.sub{color:#6b7280;font-style:italic;font-size:10.5pt;margin:0 0 2px;}
.dateline{font-size:8.5pt;color:#6b7280;letter-spacing:1px;text-transform:uppercase;margin-top:4px;}
h2{font-family:Georgia,serif;font-size:14pt;color:#16324f;border-bottom:1px solid #d7dbe0;padding-bottom:3px;margin:20px 0 8px;}
h3{font-size:11pt;color:#16324f;margin:14px 0 4px;}
p{margin:6px 0;}
.lead{font-size:11pt;}
table{border-collapse:collapse;width:100%;font-family:'DejaVu Sans',Helvetica,sans-serif;font-size:8.6pt;margin:8px 0;}
th{background:#16324f;color:#fff;padding:5px 6px;text-align:center;font-weight:600;}
td{border-bottom:1px solid #e6e8ec;padding:5px 6px;text-align:center;}
tr:nth-child(even) td{background:#f5f6f8;}
td.l,th.l{text-align:left;}
.pos{color:#2e7d32;font-weight:bold;}.neg{color:#b23b3b;font-weight:bold;}
img.chart{width:100%;margin:6px 0 2px;}
.callout{background:#eef2f6;border-left:4px solid #16324f;padding:8px 12px;margin:10px 0;font-size:9.6pt;}
.warn{background:#fbeeee;border-left:4px solid #c8102e;padding:8px 12px;margin:10px 0;font-size:9.6pt;}
.fine{font-size:7.6pt;color:#9aa0a8;font-family:'DejaVu Sans',sans-serif;border-top:1px solid #e6e8ec;margin-top:16px;padding-top:6px;line-height:1.35;}
.kpi{display:flex;gap:10px;margin:10px 0;}
.kpi .box{flex:1;border:1px solid #d7dbe0;border-radius:6px;padding:8px 9px;text-align:center;background:#fff;}
.kpi .box .n{font-size:17pt;color:#16324f;font-weight:bold;font-family:Georgia,serif;}
.kpi .box .k{font-size:7.6pt;color:#6b7280;text-transform:uppercase;letter-spacing:1px;font-family:'DejaVu Sans',sans-serif;}
"""


def _pct(x: float) -> str:
    return "—" if x is None or np.isnan(x) else f"{x * 100:.1f}%"


def _pct_cls(x: float) -> str:
    if x is None or np.isnan(x):
        return ""
    return "pos" if x >= BREAKEVEN else "neg"


def _kpi(day_stat: PropStat, cum_stat: PropStat) -> str:
    return (
        "<div class='kpi'>"
        f"<div class='box'><div class='n'>{_pct(day_stat.ppv)}</div><div class='k'>Slate PPV</div></div>"
        f"<div class='box'><div class='n'>{_pct(day_stat.npv)}</div><div class='k'>Slate NPV</div></div>"
        f"<div class='box'><div class='n'>{_pct(cum_stat.ppv)}</div><div class='k'>All-slate PPV</div></div>"
        f"<div class='box'><div class='n'>{_pct(cum_stat.npv)}</div><div class='k'>All-slate NPV</div></div>"
        f"<div class='box'><div class='n'>{cum_stat.n:,}</div><div class='k'>Graded picks</div></div>"
        "</div>"
    )


def _stat_table(stats: list[PropStat], first_col: str) -> str:
    head = (
        f"<tr><th class='l'>{first_col}</th><th>Picks</th><th>Favored</th>"
        "<th>PPV</th><th>NPV</th><th>TP</th><th>FP</th><th>FN</th><th>TN</th></tr>"
    )
    body = ""
    for s in stats:
        body += (
            f"<tr><td class='l'>{s.label}</td><td>{s.n}</td><td>{s.n_fav}</td>"
            f"<td class='{_pct_cls(s.ppv)}'>{_pct(s.ppv)}</td>"
            f"<td class='{_pct_cls(s.npv)}'>{_pct(s.npv)}</td>"
            f"<td>{s.tp}</td><td>{s.fp}</td><td>{s.fn}</td><td>{s.tn}</td></tr>"
        )
    return f"<table>{head}{body}</table>"


def _disc_sentence(discs: list[Discriminator]) -> str:
    if not discs:
        return "no engine metric separated the winners from the losers at p&lt;0.05"
    bits = []
    for d in discs[:3]:
        bits.append(
            f"<b>{d.label}</b> ({d.direction}, r={d.r:+.2f}, p={d.p:.3f})"
        )
    return "; ".join(bits)


def build_report(
    day: Date,
    day_df: pd.DataFrame,
    cum_df: pd.DataFrame,
) -> tuple[str, str]:
    day_engine = whole_engine_stat(day_df)
    cum_engine = whole_engine_stat(cum_df)
    fams = group_stats(cum_df, "family")
    mkts = group_stats(cum_df, "market")
    diags = diagnose_failing(cum_df, mkts)

    nice = day.strftime("%A, %B %-d, %Y")
    masthead = (
        "<div class='masthead'>"
        "<div class='brand'><span class='pp'>Payoff</span> Pitch · Audit Desk</div>"
        "<h1>The Nightly Audit</h1>"
        "<p class='sub'>What the model got right, where it leaked, and which metric is to blame.</p>"
        f"<div class='dateline'>Slate graded · {nice}</div></div>"
    )

    lead = (
        "Every night the engine grades itself against the box scores, and tonight's report keeps the "
        "scoring honest: a pick is a <i>buy</i> when the model likes it (probability at or above 50%) and a "
        "<i>fade</i> when it doesn't. Positive predictive value is how often the buys actually cash; negative "
        "predictive value is how often the fades were right to sit out. "
        f"On the {nice.split(',')[0]} slate the model hit <b>{_pct(day_engine.ppv)}</b> on the buy side and "
        f"<b>{_pct(day_engine.npv)}</b> on the fade side across {day_engine.n:,} graded picks. Across the full "
        f"{cum_df['date'].nunique()}-slate history that settles to <b>{_pct(cum_engine.ppv)}</b> PPV and "
        f"<b>{_pct(cum_engine.npv)}</b> NPV on {cum_engine.n:,} picks — the number that actually matters, since "
        "one slate is mostly noise."
    )

    fam_chart = _family_chart(fams)
    mkt_chart = _market_chart(mkts)

    diag_html = ""
    if diags:
        diag_html += (
            "<h2>Why the Failing Props Fail</h2>"
            "<p>For every market leaking on the buy side (PPV under the 52.4% breakeven) or sitting on a "
            "reclaimable false-negative pocket, we split the picks into winners and losers and ask which "
            "engine metric actually separates them — point-biserial correlation, run on the full history so "
            "the sample is real. A significant metric is a lever: tighten the buy on it, or stop fading it.</p>"
        )
        for dg in diags:
            st = dg.stat
            diag_html += f"<h3>{st.label} — PPV {_pct(st.ppv)} (n={st.n_fav}), NPV {_pct(st.npv)} (n={st.n_fade})</h3>"
            if st.leaks_ppv:
                diag_html += (
                    "<p><b>Buy side (false positives).</b> Separating the winning buys from the losing buys: "
                    f"{_disc_sentence(dg.fp_discriminators)}.</p>"
                )
                if dg.fp_discriminators:
                    c = _discriminator_chart(
                        f"{st.label}: winning vs losing buys", dg.fp_discriminators
                    )
                    diag_html += f"<img class='chart' src='data:image/png;base64,{c}'/>"
            if st.reclaimable_fn:
                diag_html += (
                    "<p><b>Fade side (false negatives).</b> Separating the faded winners from the faded losers: "
                    f"{_disc_sentence(dg.fn_discriminators)}.</p>"
                )
                if dg.fn_discriminators:
                    c = _discriminator_chart(
                        f"{st.label}: faded winners vs faded losers", dg.fn_discriminators
                    )
                    diag_html += f"<img class='chart' src='data:image/png;base64,{c}'/>"
    else:
        diag_html = (
            "<h2>Why the Failing Props Fail</h2>"
            "<div class='callout'>No market cleared the sample bar for a discriminant test tonight "
            "(need a leaking prop with enough graded picks on each side). As the history grows, this section "
            "fills in with the specific metrics that separate winners from losers.</div>"
        )

    body = (
        f"<p class='lead'>{lead}</p>"
        + _kpi(day_engine, cum_engine)
        + "<div class='callout'><b>How to read PPV / NPV:</b> PPV is the buy-side batting average — of the picks "
        "the model backed, how many won. NPV is the discipline score — of the picks it passed on, how many it "
        "was right to pass. The 52.4% line is the break-even at standard -110 juice.</div>"
        "<h2>The Slate vs. The Record</h2>"
        + _stat_table([day_engine, cum_engine], "Window").replace(
            "<td class='l'>Whole engine</td>", "<td class='l'>Tonight's slate</td>", 1
        )
        + "<h2>By Prop Family</h2>"
        f"<img class='chart' src='data:image/png;base64,{fam_chart}'/>"
        + _stat_table(fams, "Family")
        + "<h2>Every Prop</h2>"
        f"<img class='chart' src='data:image/png;base64,{mkt_chart}'/>"
        + _stat_table(mkts, "Market")
        + diag_html
        + "<p class='fine'>Methodology: PPV/NPV computed at the model's 0.5 decision boundary (favored = model "
        "probability &ge; 0.5), pushes excluded. Discriminant tests are point-biserial correlations of each "
        "engine metric against the win/loss outcome, run on the cumulative graded-pick history; a metric is "
        "reported only at p&lt;0.05 with at least "
        f"{MIN_GROUP} picks on each side. P-values are unadjusted for multiple comparisons — treat single "
        "hits as leads to confirm, not laws. Model self-audit, not betting advice.</p>"
    )
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{masthead}{body}</body></html>"

    narr = _narration(day, day_engine, cum_engine, cum_df, fams, diags)
    return html, narr


def _narration(
    day: Date,
    day_engine: PropStat,
    cum_engine: PropStat,
    cum_df: pd.DataFrame,
    fams: list[PropStat],
    diags: list[PropDiag],
) -> str:
    nice = day.strftime("%A, %B %-d")
    parts = [
        f"What's up everybody, welcome into the Payoff Pitch Audit Desk for {nice}. "
        "Let's grade the machine against the box scores. ",
        f"On tonight's slate the engine hit {day_engine.ppv * 100:.0f} percent positive predictive value on the "
        f"buy side and {day_engine.npv * 100:.0f} percent negative predictive value on the fades, across "
        f"{day_engine.n} graded picks. ",
        f"But one slate is noise. Across the full {cum_df['date'].nunique()}-slate record, the engine settles at "
        f"{cum_engine.ppv * 100:.0f} percent PPV and {cum_engine.npv * 100:.0f} percent NPV on "
        f"{cum_engine.n} picks. That's the honest number. ",
    ]
    graded_fams = [f for f in fams if f.n_fav >= MIN_FAVORED]
    if graded_fams:
        best = max(graded_fams, key=lambda f: f.ppv)
        worst = min(graded_fams, key=lambda f: f.ppv)
        parts.append(
            f"By family, the strongest buy side is {best.label} at {best.ppv * 100:.0f} percent, "
            f"while {worst.label} is the soft spot at {worst.ppv * 100:.0f} percent. "
        )
    if diags:
        dg = diags[0]
        if dg.fp_discriminators:
            d = dg.fp_discriminators[0]
            parts.append(
                f"Now the fun part, why the losers lose. Take {dg.stat.label}. When we split the winning buys "
                f"from the losing buys, the metric that actually separates them is {d.label}, which runs "
                f"{d.direction}, with a correlation of {d.r:+.2f} and a p-value of {d.p:.3f}. "
                "That's a real lever the engine can tighten. "
            )
        elif dg.fn_discriminators:
            d = dg.fn_discriminators[0]
            parts.append(
                f"On {dg.stat.label}, the faded winners and faded losers split cleanest on {d.label}, "
                f"{d.direction}, correlation {d.r:+.2f}, p-value {d.p:.3f}. "
                "That's a false-negative pocket worth reclaiming. "
            )
    else:
        parts.append(
            "Not enough sample yet to isolate a single guilty metric on the failing props, but the history is "
            "growing every night and that call-out is coming. "
        )
    parts.append(
        "Bottom line: trust the cumulative number over any single slate, lean into the families that clear "
        "break-even, and fade the ones that don't until the metric that fixes them shows up in the data. "
        "That's your audit. We'll see you tomorrow night."
    )
    return "".join(parts)


# --- render + audio --------------------------------------------------------
def to_pdf(html: str) -> bytes:
    from weasyprint import HTML

    return bytes(HTML(string=html).write_pdf())


def to_mp3(text: str, path: Path) -> bytes:
    """Neural sportscaster voice via edge-tts, gTTS fallback."""
    try:
        import asyncio

        import edge_tts

        async def _go() -> None:
            comm = edge_tts.Communicate(
                text, voice="en-US-ChristopherNeural", rate="+12%", pitch="+2Hz"
            )
            await comm.save(str(path))

        asyncio.run(_go())
    except Exception as exc:  # noqa: BLE001
        logger.warning("edge-tts failed (%s); falling back to gTTS", exc)
        from gtts import gTTS

        gTTS(text=text, lang="en", tld="com", slow=False).save(str(path))
    return path.read_bytes()


# --- top-level entry point -------------------------------------------------
def generate_audit_insight(
    graded: list[tuple[Recommendation, str]],
    audit_date: Date,
    cfg,
    *,
    email: bool,
    to: str | None,
    extra_attachments: list[tuple[str, bytes]] | None = None,
) -> dict[str, Path | None]:
    """Build the insight PDF + MP3, and (optionally) email them with the ledger.

    ``extra_attachments`` (e.g. the Excel ledger) are attached alongside the
    PDF and MP3 so a single email carries all three deliverables.
    """
    out: dict[str, Path | None] = {"pdf": None, "mp3": None, "html": None}
    day = graded_to_frame(graded, audit_date)
    if day.empty:
        logger.warning("no graded picks for %s; skipping insight report", audit_date)
        return out

    store_path = cfg.audit_dir / STORE_NAME
    cum_raw = update_store(store_path, day, audit_date)
    day_c = classify(day)
    cum_c = classify(cum_raw)
    if day_c.empty or cum_c.empty:
        logger.warning("no gradeable picks after filtering pushes for %s", audit_date)
        return out

    html, narr = build_report(audit_date, day_c, cum_c)
    iso = audit_date.isoformat()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    html_path = cfg.output_dir / f"audit_insight_{iso}.html"
    html_path.write_text(html)
    out["html"] = html_path

    attachments: list[tuple[str, bytes]] = list(extra_attachments or [])
    pdf_bytes: bytes | None = None
    try:
        pdf_bytes = to_pdf(html)
        pdf_path = cfg.output_dir / f"PayoffPitch_Audit_{iso}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        out["pdf"] = pdf_path
        attachments.insert(0, (pdf_path.name, pdf_bytes))
    except Exception as exc:  # noqa: BLE001
        logger.warning("audit insight PDF not written: %s", exc)

    try:
        mp3_path = cfg.output_dir / f"PayoffPitch_Audit_{iso}.mp3"
        mp3_bytes = to_mp3(narr, mp3_path)
        out["mp3"] = mp3_path
        attachments.append((mp3_path.name, mp3_bytes))
    except Exception as exc:  # noqa: BLE001
        logger.warning("audit insight MP3 not written: %s", exc)

    print(
        "Audit insight -> "
        + ", ".join(str(p) for p in (out["pdf"], out["mp3"]) if p)
    )

    if email and attachments:
        from mlb_engine.output.email import EmailNotConfigured, send_card_email

        body_html = (
            "<div style='font-family:Georgia,serif;color:#1a1a1a;max-width:640px'>"
            "<h2 style='color:#16324f'>Payoff Pitch — Audit Desk</h2>"
            f"<p>Your nightly audit for <b>{audit_date.strftime('%A, %B %-d, %Y')}</b> is attached:</p>"
            "<ul><li><b>Excel ledger</b> — every graded pick, PPV/NPV, and CLV.</li>"
            "<li><b>Audit article (PDF)</b> — slate &amp; all-time PPV/NPV, by family and prop, plus the "
            "metric-level diagnosis of the failing props.</li>"
            "<li><b>Audio narration (MP3)</b> — the same read, sportscaster style.</li></ul>"
            "<p style='color:#6b7280;font-size:13px'>Model self-audit, not investment advice.</p></div>"
        )
        try:
            recipient = send_card_email(
                cfg,
                subject=f"Payoff Pitch — Nightly Audit ({iso})",
                html_body=body_html,
                text_body="Your Payoff Pitch nightly audit (Excel + PDF + audio) is attached.",
                to=to,
                attachments=attachments,
            )
            print(f"Emailed audit ({len(attachments)} attachments) to {recipient}")
        except EmailNotConfigured as exc:
            print(f"Audit email not sent: {exc}")
    return out
