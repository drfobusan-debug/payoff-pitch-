"""Daily Regression Radar newsletter generator.

Mirrors the style of the user's Regression Radar PDF: casual but analytical,
listing the top 10 positive/negative regression pitchers and batters on the
daily slate using wOBA vs. xwOBA gaps.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Any

import pandas as pd

from mlb_engine.config import Config
from mlb_engine.output.card import render_pdf
from mlb_engine.output.email import EmailNotConfigured, send_card_email

log = logging.getLogger(__name__)


@dataclass
class RegressionTarget:
    name: str
    team: str
    woba: float
    xwoba: float
    k_pct: float | None
    bb_pct: float | None
    barrel_pct: float | None
    pa: int

    @property
    def gap(self) -> float:
        """xwOBA - wOBA (positive means underlying contact is better)."""
        return self.xwoba - self.woba


def _standardize_name(name: str) -> str:
    """Convert 'Last, First' or 'First Last' to 'First Last' title case."""
    name = " ".join(name.split()).title()
    if "," in name:
        last, first = name.split(",", 1)
        name = f"{first.strip()} {last.strip()}"
    return name


def _pct(value: Any) -> float | None:
    """Convert a percentage value to a float; None if missing."""
    try:
        v = float(value)
        return v
    except (TypeError, ValueError):
        return None


def _load_batter_expected_stats(year: int, min_pa: int = 60) -> pd.DataFrame:
    """Merge season-to-date wOBA/xwOBA with K%, BB%, barrel% for batters."""
    from pybaseball import statcast_batter_expected_stats, statcast_batter_percentile_ranks

    try:
        exp = statcast_batter_expected_stats(year)
    except Exception as exc:
        log.warning("Could not load batter expected stats: %s", exc)
        return pd.DataFrame()

    try:
        pct = statcast_batter_percentile_ranks(year)
    except Exception as exc:
        log.warning("Could not load batter percentile ranks: %s", exc)
        pct = pd.DataFrame()

    if exp.empty:
        return exp

    exp = exp.rename(
        columns={
            "last_name, first_name": "name_raw",
            "pa": "pa",
            "woba": "woba",
            "est_woba": "xwoba",
            "est_woba_minus_woba_diff": "xwoba_diff",
        }
    )
    for c in ("woba", "xwoba", "pa"):
        if c in exp.columns:
            exp[c] = pd.to_numeric(exp[c], errors="coerce")

    if not pct.empty:
        pct = pct[["player_id", "k_percent", "bb_percent", "brl_percent"]].copy()
        for c in ("k_percent", "bb_percent", "brl_percent"):
            pct[c] = pd.to_numeric(pct[c], errors="coerce")
        exp = exp.merge(pct, on="player_id", how="left")

    exp = exp[exp["pa"] >= min_pa]
    return exp


def _load_pitcher_expected_stats(year: int, min_bf: int = 80) -> pd.DataFrame:
    """Merge season-to-date wOBA/xwOBA with K%, BB% for pitchers."""
    from pybaseball import statcast_pitcher_expected_stats, statcast_pitcher_percentile_ranks

    try:
        exp = statcast_pitcher_expected_stats(year)
    except Exception as exc:
        log.warning("Could not load pitcher expected stats: %s", exc)
        return pd.DataFrame()

    try:
        pct = statcast_pitcher_percentile_ranks(year)
    except Exception as exc:
        log.warning("Could not load pitcher percentile ranks: %s", exc)
        pct = pd.DataFrame()

    if exp.empty:
        return exp

    exp = exp.rename(
        columns={
            "last_name, first_name": "name_raw",
            "pa": "bf",
            "woba": "woba",
            "est_woba": "xwoba",
            "est_woba_minus_woba_diff": "xwoba_diff",
        }
    )
    for c in ("woba", "xwoba", "bf"):
        if c in exp.columns:
            exp[c] = pd.to_numeric(exp[c], errors="coerce")

    if not pct.empty:
        pct = pct[["player_id", "k_percent", "bb_percent"]].copy()
        for c in ("k_percent", "bb_percent"):
            pct[c] = pd.to_numeric(pct[c], errors="coerce")
        exp = exp.merge(pct, on="player_id", how="left")

    exp = exp[exp["bf"] >= min_bf]
    return exp


def _to_targets(df: pd.DataFrame, is_pitcher: bool) -> list[RegressionTarget]:
    """Convert a merged expected-stats DataFrame to RegressionTarget objects."""
    out: list[RegressionTarget] = []
    for _, row in df.iterrows():
        name = _standardize_name(str(row.get("name_raw", "")))
        team = str(row.get("team", "") or "")
        if not name:
            continue
        out.append(
            RegressionTarget(
                name=name,
                team=team,
                woba=float(row["woba"]) if pd.notna(row.get("woba")) else 0.0,
                xwoba=float(row["xwoba"]) if pd.notna(row.get("xwoba")) else 0.0,
                k_pct=_pct(row.get("k_percent")),
                bb_pct=_pct(row.get("bb_percent")),
                barrel_pct=_pct(row.get("brl_percent")),
                pa=int(row.get("bf" if is_pitcher else "pa", 1)) if pd.notna(row.get("bf" if is_pitcher else "pa")) else 0,
            )
        )
    return out


def load_regression_targets(
    slate_pitcher_names: set[str],
    slate_batter_names: set[str],
    year: int = 2026,
    min_pa: int = 60,
    min_bf: int = 80,
) -> tuple[list[RegressionTarget], list[RegressionTarget]]:
    """Return (pitcher_targets, batter_targets) filtered to today's slate."""
    slate_pitcher_names = {n.title() for n in slate_pitcher_names if n}
    slate_batter_names = {n.title() for n in slate_batter_names if n}

    p_df = _load_pitcher_expected_stats(year, min_bf=min_bf)
    b_df = _load_batter_expected_stats(year, min_pa=min_pa)

    if not p_df.empty:
        p_df["std_name"] = p_df["name_raw"].apply(_standardize_name)
        p_df = p_df[p_df["std_name"].isin(slate_pitcher_names)]
    if not b_df.empty:
        b_df["std_name"] = b_df["name_raw"].apply(_standardize_name)
        b_df = b_df[b_df["std_name"].isin(slate_batter_names)]

    return _to_targets(p_df, is_pitcher=True), _to_targets(b_df, is_pitcher=False)


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "0.0%"
    return f"{v:.1f}%"


def _pitcher_blurb(t: RegressionTarget) -> str:
    """One paragraph for a pitcher regression target."""
    gap = t.woba - t.xwoba
    k = _fmt_pct(t.k_pct)
    bb = _fmt_pct(t.bb_pct)
    name_team = f"{t.name} ({t.team})" if t.team else t.name
    if gap > 0:
        return (
            f"{name_team} has allowed a {t.woba:.3f} wOBA but his expected wOBA sits at "
            f"{t.xwoba:.3f} — that gap means he's been worse than his contact profile. "
            f"With {k} Ks and {bb} BBs, he's a classic buy-low."
        )
    return (
        f"{name_team} has allowed a {t.woba:.3f} wOBA but his expected wOBA is {t.xwoba:.3f} "
        f"— he's out-pitched his contact and is due to give up more damage. "
        f"{k} Ks and {bb} BBs back that up."
    )


def _batter_blurb(t: RegressionTarget, positive: bool) -> str:
    """One paragraph for a batter regression target."""
    k = _fmt_pct(t.k_pct)
    bb = _fmt_pct(t.bb_pct)
    brl = _fmt_pct(t.barrel_pct)
    name_team = f"{t.name} ({t.team})" if t.team else t.name
    if positive:
        return (
            f"{name_team} is running a {t.woba:.3f} wOBA but his xwOBA is {t.xwoba:.3f} — "
            f"the underlying contact says the box score is lying to the downside. "
            f"{k} Ks, {bb} BBs, {brl} barrels. Buy-low candidate."
        )
    return (
        f"{name_team} is posting a {t.woba:.3f} wOBA but his xwOBA is only {t.xwoba:.3f} — "
        f"he's been finding holes and gaps that contact doesn't support. "
        f"{k} Ks, {bb} BBs, {brl} barrels. Regression risk."
    )


def build_radar(
    pitcher_targets: list[RegressionTarget],
    batter_targets: list[RegressionTarget],
    top_n: int = 10,
) -> dict[str, Any]:
    """Sort and slice into the four radar buckets."""
    # Pitchers: wOBA allowed > xwOBA allowed -> positive regression (buy low)
    pitcher_pos = sorted(pitcher_targets, key=lambda x: x.woba - x.xwoba, reverse=True)[:top_n]
    pitcher_neg = sorted(pitcher_targets, key=lambda x: x.woba - x.xwoba)[:top_n]

    # Batters: xwOBA > wOBA -> positive regression (buy low)
    batter_pos = sorted(batter_targets, key=lambda x: x.gap, reverse=True)[:top_n]
    batter_neg = sorted(batter_targets, key=lambda x: x.gap)[:top_n]

    return {
        "pitchers_positive": pitcher_pos,
        "pitchers_negative": pitcher_neg,
        "batters_positive": batter_pos,
        "batters_negative": batter_neg,
    }


def render_markdown(radar: dict[str, Any], slate_date: Date) -> str:
    """Markdown version of the Regression Radar."""
    lines: list[str] = [
        f"# Regression Radar — {slate_date.isoformat()}",
        "",
        "A quick look at today's slate through the wOBA/xwOBA lens. Lower wOBA/xwOBA "
        "is better for pitchers; higher is better for hitters.",
        "",
    ]

    lines.extend(["## Pitchers", ""])
    lines.extend(["### Positive Regression Targets — better days ahead", ""])
    for t in radar["pitchers_positive"]:
        lines.append(f"- {_pitcher_blurb(t)}")
    lines.append("")
    lines.extend(["### Negative Regression Targets — the bill is coming due", ""])
    for t in radar["pitchers_negative"]:
        lines.append(f"- {_pitcher_blurb(t)}")
    lines.append("")
    lines.append(
        "_Among today's slate pitchers; min. 80 batters faced. "
        "Sorted by the gap between xwOBA allowed and wOBA allowed._"
    )
    lines.append("")

    lines.extend(["## Batters", ""])
    lines.extend(["### Positive Regression Targets — the box score is lying", ""])
    for t in radar["batters_positive"]:
        lines.append(f"- {_batter_blurb(t, positive=True)}")
    lines.append("")
    lines.extend(["### Negative Regression Targets — holes and gaps won't last", ""])
    for t in radar["batters_negative"]:
        lines.append(f"- {_batter_blurb(t, positive=False)}")
    lines.append("")
    lines.append(
        "_Among today's slate batters; min. 60 plate appearances. "
        "Sorted by the gap between xwOBA and wOBA._"
    )
    return "\n".join(lines)


def render_html(radar: dict[str, Any], slate_date: Date) -> str:
    """HTML version suitable for PDF rendering."""
    style = (
        "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
        "max-width:760px;margin:24px auto;padding:0 18px;color:#1a1a1a;line-height:1.55}"
        "h1{font-size:24px;border-bottom:3px solid #0b6;padding-bottom:8px}"
        "h2{font-size:20px;margin-top:28px;color:#0a5}"
        "h3{font-size:16px;margin-top:22px;background:#0b6;color:#fff;padding:8px 12px;"
        "border-radius:6px}ul{padding-left:20px}li{margin:6px 0}em{color:#555;font-size:13px}"
    )

    def section(title: str, items: list[RegressionTarget], blurb_fn) -> str:
        if not items:
            return f"<h3>{title}</h3><p><em>No qualified players on today's slate.</em></p>"
        list_items = "".join(f"<li>{blurb_fn(t)}</li>" for t in items)
        return f"<h3>{title}</h3><ul>{list_items}</ul>"

    blocks: list[str] = [
        f"<h1>Regression Radar — {slate_date.isoformat()}</h1>",
        "<p>A quick look at today's slate through the wOBA/xwOBA lens. Lower wOBA/xwOBA "
        "is better for pitchers; higher is better for hitters.</p>",
        "<h2>Pitchers</h2>",
        section("Positive Regression Targets — better days ahead", radar["pitchers_positive"], _pitcher_blurb),
        section("Negative Regression Targets — the bill is coming due", radar["pitchers_negative"], _pitcher_blurb),
        "<p><em>Among today's slate pitchers; min. 80 batters faced. "
        "Sorted by the gap between xwOBA allowed and wOBA allowed.</em></p>",
        "<h2>Batters</h2>",
        section("Positive Regression Targets — the box score is lying", radar["batters_positive"], lambda t: _batter_blurb(t, True)),
        section("Negative Regression Targets — holes and gaps won't last", radar["batters_negative"], lambda t: _batter_blurb(t, False)),
        "<p><em>Among today's slate batters; min. 60 plate appearances. "
        "Sorted by the gap between xwOBA and wOBA.</em></p>",
    ]
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{style}</style></head>"
        f"<body>{''.join(blocks)}</body></html>"
    )


def generate_radar_pdf(
    slate_pitcher_names: set[str],
    slate_batter_names: set[str],
    slate_date: Date,
    year: int = 2026,
    min_pa: int = 60,
    min_bf: int = 80,
) -> tuple[bytes | None, dict[str, Any]]:
    """Return (pdf_bytes, radar_dict) for the given slate names."""
    pitchers, batters = load_regression_targets(
        slate_pitcher_names, slate_batter_names, year=year, min_pa=min_pa, min_bf=min_bf
    )
    radar = build_radar(pitchers, batters, top_n=10)
    if not any(radar.values()):
        log.warning("No regression radar targets found for %s", slate_date)
    html = render_html(radar, slate_date)
    try:
        return render_pdf(html), radar
    except Exception as exc:
        log.warning("Regression radar PDF not rendered: %s", exc)
        return None, radar


def write_radar(
    slate_pitcher_names: set[str],
    slate_batter_names: set[str],
    slate_date: Date,
    cfg: Config,
    year: int = 2026,
    email: bool = False,
    to: str | None = None,
) -> Path | None:
    """Write the Regression Radar md + html + pdf to the output dir and optionally email it."""
    pdf_bytes, radar = generate_radar_pdf(
        slate_pitcher_names, slate_batter_names, slate_date, year=year
    )
    md = render_markdown(radar, slate_date)
    md_path = cfg.output_dir / f"regression_radar_{slate_date.isoformat()}.md"
    html_path = cfg.output_dir / f"regression_radar_{slate_date.isoformat()}.html"
    pdf_path = cfg.output_dir / f"regression_radar_{slate_date.isoformat()}.pdf"
    md_path.write_text(md)
    html_path.write_text(render_html(radar, slate_date))
    if pdf_bytes is not None:
        pdf_path.write_bytes(pdf_bytes)
    print(f"Regression Radar: {md_path}")

    if email and pdf_bytes is not None:
        subject = f"PayoffPitch Regression Radar — {slate_date.isoformat()}"
        try:
            recipient = send_card_email(
                cfg,
                subject=subject,
                html_body=render_html(radar, slate_date),
                text_body="Today's Regression Radar is attached as a PDF.\n",
                to=to,
                attachments=[(pdf_path.name, pdf_bytes)],
            )
            print(f"Emailed regression radar to {recipient}")
        except EmailNotConfigured as exc:
            print(f"Email not sent: {exc}")
    return pdf_path if pdf_bytes is not None else None
