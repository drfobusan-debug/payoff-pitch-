"""Weekly PDF newsletter: what to buy, what to watch, when to check the shelves.

Sections:
  * deals      -- shop listings under TCGplayer market this week
  * cards      -- top singles to watch under a price cap, with 6m/1y/2y/3y outlook
  * sealed     -- top booster boxes / ETBs to watch under a price cap, same outlook
  * drops      -- best Richmond restock windows for the coming week

The long-range outlook is a *scenario*, not a fit: the ledger only carries a
few days of history per product, so the curve is the typical Pokemon release
cycle (prices soften while a set is in print, recover once it goes out of
print) anchored at today's market price. Where the ledger does have 7+ days,
the fitted Holt trend tilts the first year. Bands widen with the square root
of time. Everything is labelled as such on the page.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from cardscout import deals as deals_mod
from cardscout import drops as drops_mod
from cardscout import forecast as forecast_mod
from cardscout.config import HOME_TZ, HOME_ZIP, data_dir
from cardscout.store import Ledger

HORIZONS = {"6 mo": 6, "1 yr": 12, "2 yr": 24, "3 yr": 36}

# Release-cycle priors: (months a set stays in print, drift/yr while in print,
# drift/yr after, band width per sqrt(year)).
CYCLE = {
    "sealed": (18, -0.08, 0.18, 0.20),
    "card": (15, -0.20, 0.08, 0.30),
    "promo": (0, 0.0, 0.05, 0.25),
}

SKIP_NAME_RE = re.compile(
    r"prerelease|\[staff\]|world championships|jumbo|oversized|code card|"
    r"error|misprint|miscut|test print|sample|japanese",
    re.I,
)
# main expansions are named "SV08: Surging Sparks", "ME: Ascended Heroes"; promos,
# POP series, fast-food sets and the misc bucket are not.
EXPANSION_RE = re.compile(r"^[A-Z]{2,4}\d{0,2}: ")
BOX_RE = re.compile(r"booster box|elite trainer box|booster display", re.I)
NOT_BOX_RE = re.compile(r"\bcase\b|half booster|lot of|bundle of|\bx\d", re.I)

DEFAULT_WATCHLIST = {
    "max_card_price": 300,
    "max_box_price": 500,
    "months_back": 24,
    "per_set": 2,
    "pins": [{"name": "Charmander - 038", "set": "Mega Evolution Promo"}],
}


@dataclass
class Outlook:
    kind: str
    months: list[int]
    base: list[float]
    low: list[float]
    high: list[float]
    fitted: bool

    def at(self, month: int) -> tuple[float, float, float]:
        i = self.months.index(month)
        return self.base[i], self.low[i], self.high[i]


@dataclass
class Pick:
    product_id: int
    name: str
    set_name: str
    printing: str
    market: float
    low: float | None
    published_on: str
    is_sealed: bool
    url: str
    outlook: Outlook
    offers: list[tuple[str, float, str]] = field(default_factory=list)  # shop, price, url

    @property
    def best_offer(self) -> tuple[str, float, str] | None:
        return min(self.offers, key=lambda o: o[1]) if self.offers else None

    @property
    def why(self) -> str:
        age = months_since(self.published_on)
        kind = self.outlook.kind
        window, _, _, _ = CYCLE[kind]
        if kind == "promo":
            return "Promo: print run fixed, demand-driven; expect slow drift."
        if age < window:
            left = window - age
            return f"Set is ~{age} mo old and likely in print ~{left} more mo; buy on dips."
        return f"Set is ~{age} mo old and past the usual print window; supply is fixed."


def months_since(published_on: str, today: date | None = None) -> int:
    today = today or date.today()
    try:
        d = date.fromisoformat(published_on[:10])
    except ValueError:
        return 0
    return max(0, (today.year - d.year) * 12 + today.month - d.month)


def scenario(
    price: float,
    age_months: int,
    kind: str,
    fitted_daily_trend: float | None = None,
    months: int = 36,
) -> Outlook:
    window, drift_in, drift_out, band = CYCLE[kind]
    base, low, high = [price], [price], [price]
    lp = math.log(price)
    fitted_year = None
    if fitted_daily_trend is not None:
        # a fitted trend is only trusted for the first year and only halfway
        fitted_year = max(-0.5, min(0.5, ((1 + fitted_daily_trend) ** 365 - 1) * 0.5))
    for m in range(1, months + 1):
        drift = drift_in if age_months + m <= window else drift_out
        if fitted_year is not None and m <= 12:
            drift = 0.5 * drift + 0.5 * fitted_year
        lp += math.log1p(drift) / 12
        years = m / 12
        base.append(math.exp(lp))
        low.append(math.exp(lp - band * math.sqrt(years)))
        high.append(math.exp(lp + band * math.sqrt(years)))
    return Outlook(kind, list(range(months + 1)), base, low, high, fitted_daily_trend is not None)


def load_watchlist(path: Path | None = None) -> dict:
    dest = path or (data_dir() / "watchlist.json")
    if not dest.exists():
        dest.write_text(json.dumps(DEFAULT_WATCHLIST, indent=2))
    cfg = dict(DEFAULT_WATCHLIST)
    cfg.update(json.loads(dest.read_text()))
    return cfg


def _fitted_trend(ledger: Ledger, product_id: int, printing: str) -> float | None:
    rows = [(r["snapshot"], r["market"]) for r in ledger.market_history(product_id, printing)]
    fc = forecast_mod.forecast(rows, 30)
    return None if fc is None or fc.flat else fc.daily_trend


def _offers(ledger: Ledger, product_id: int) -> list[tuple[str, float, str]]:
    rows = ledger.conn.execute(
        "SELECT shop, price, url FROM listings WHERE product_id = ? AND available = 1 "
        "AND snapshot = (SELECT MAX(snapshot) FROM listings l2 WHERE l2.shop = listings.shop)",
        (product_id,),
    ).fetchall()
    return [(r["shop"], r["price"], r["url"]) for r in rows]


def _latest_rows(ledger: Ledger, sealed: bool, since: str, until: str, max_price: float):
    # tcgcsv stamps undated legacy groups (POP Series, Burger King promos...)
    # with today's date, so "released before today" also drops those.
    return ledger.conn.execute(
        "SELECT p.product_id, p.name, p.url, g.name AS set_name, g.published_on, "
        "       m.printing, m.market, m.low "
        "FROM products p JOIN groups g USING (group_id) "
        "JOIN market_prices m ON m.product_id = p.product_id "
        "WHERE m.snapshot = (SELECT MAX(snapshot) FROM market_prices m2 "
        "                    WHERE m2.product_id = m.product_id AND m2.printing = m.printing) "
        "AND p.is_sealed = ? AND g.published_on >= ? AND g.published_on < ? "
        "AND m.market > 0 AND m.market <= ? "
        "ORDER BY m.market DESC",
        (int(sealed), since, until, max_price),
    ).fetchall()


def _pick(ledger: Ledger, row, kind: str) -> Pick:
    trend = _fitted_trend(ledger, row["product_id"], row["printing"])
    return Pick(
        product_id=row["product_id"],
        name=row["name"],
        set_name=row["set_name"],
        printing=row["printing"],
        market=row["market"],
        low=row["low"],
        published_on=row["published_on"] or "",
        is_sealed=kind == "sealed",
        url=row["url"] or "",
        outlook=scenario(row["market"], months_since(row["published_on"] or ""), kind, trend),
        offers=_offers(ledger, row["product_id"]),
    )


def select_cards(ledger: Ledger, cfg: dict, n: int = 10, today: date | None = None) -> list[Pick]:
    today = today or date.today()
    since = (today - timedelta(days=30 * int(cfg["months_back"]))).isoformat()
    picks: list[Pick] = []
    seen: set[int] = set()
    for pin in cfg.get("pins", []):
        row = ledger.conn.execute(
            "SELECT p.product_id, p.name, p.url, g.name AS set_name, g.published_on, "
            "       m.printing, m.market, m.low FROM products p JOIN groups g USING (group_id) "
            "JOIN market_prices m ON m.product_id = p.product_id "
            "WHERE p.name = ? AND g.name LIKE ? AND p.is_sealed = 0 "
            "ORDER BY m.snapshot DESC, m.market DESC LIMIT 1",
            (pin["name"], f"%{pin.get('set', '')}%"),
        ).fetchone()
        if row is not None and row["product_id"] not in seen:
            kind = "promo" if "promo" in row["set_name"].lower() else "card"
            picks.append(_pick(ledger, row, kind))
            seen.add(row["product_id"])
    per_set: dict[str, int] = {}
    for row in _latest_rows(ledger, False, since, today.isoformat(), float(cfg["max_card_price"])):
        if len(picks) >= n:
            break
        if row["product_id"] in seen or SKIP_NAME_RE.search(row["name"]):
            continue
        if not EXPANSION_RE.match(row["set_name"]) or "promo" in row["set_name"].lower():
            continue
        if per_set.get(row["set_name"], 0) >= int(cfg["per_set"]):
            continue
        per_set[row["set_name"]] = per_set.get(row["set_name"], 0) + 1
        picks.append(_pick(ledger, row, "card"))
        seen.add(row["product_id"])
    return picks


def select_sealed(ledger: Ledger, cfg: dict, n: int = 10, today: date | None = None) -> list[Pick]:
    today = today or date.today()
    since = (today - timedelta(days=30 * int(cfg["months_back"]))).isoformat()
    picks: list[Pick] = []
    per_set: dict[str, int] = {}
    for row in _latest_rows(ledger, True, since, today.isoformat(), float(cfg["max_box_price"])):
        if len(picks) >= n:
            break
        if not BOX_RE.search(row["name"]) or NOT_BOX_RE.search(row["name"]):
            continue
        if SKIP_NAME_RE.search(row["name"]) or not EXPANSION_RE.match(row["set_name"]):
            continue
        if per_set.get(row["set_name"], 0) >= int(cfg["per_set"]):
            continue
        per_set[row["set_name"]] = per_set.get(row["set_name"], 0) + 1
        picks.append(_pick(ledger, row, "sealed"))
    return picks


# -- rendering ----------------------------------------------------------------


def _money(v: float | None) -> str:
    return f"${v:,.0f}" if v is not None and v >= 100 else (f"${v:,.2f}" if v is not None else "--")


def charts_for(picks: list[Pick], stem: Path, title: str, per_page: int = 6) -> list[Path]:
    pages = [picks[i : i + per_page] for i in range(0, len(picks), per_page)]
    return [
        chart(page, stem.with_name(f"{stem.name}-{i + 1}.png"), f"{title} ({i + 1}/{len(pages)})")
        for i, page in enumerate(pages)
    ]


def chart(picks: list[Pick], path: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 2
    rows = math.ceil(len(picks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(8.0, 2.3 * rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, p in zip(axes.flat, picks, strict=False):
        ax.set_visible(True)
        o = p.outlook
        yrs = [m / 12 for m in o.months]
        ax.fill_between(yrs, o.low, o.high, color="#9ec5fe", alpha=0.45, linewidth=0)
        ax.plot(yrs, o.base, color="#0b5ed7", linewidth=1.8)
        for m in HORIZONS.values():
            b, _, _ = o.at(m)
            ax.plot([m / 12], [b], "o", color="#0b5ed7", markersize=3)
            ax.annotate(
                _money(b), (m / 12, b), textcoords="offset points", xytext=(0, 5), fontsize=6.5
            )
        ax.axhline(p.market, color="#888", linewidth=0.6, linestyle=":")
        name = p.name if len(p.name) <= 34 else p.name[:33] + "…"
        ax.set_title(f"{name}\n{p.set_name} — now {_money(p.market)}", fontsize=7.5, loc="left")
        ax.set_xticks([0, 0.5, 1, 2, 3])
        ax.set_xticklabels(["now", "6m", "1y", "2y", "3y"], fontsize=6.5)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _money(v)))
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(title, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _pick_rows(picks: list[Pick]) -> str:
    out = []
    for i, p in enumerate(picks, 1):
        cells = "".join(f"<td class=n>{_money(p.outlook.at(m)[0])}</td>" for m in HORIZONS.values())
        offer = p.best_offer
        shop = f"{_esc(offer[0])} {_money(offer[1])}" if offer else "—"
        fit = " <span class=tag>fitted</span>" if p.outlook.fitted else ""
        out.append(
            f"<tr><td>{i}</td><td><b>{_esc(p.name)}</b><br><span class=dim>{_esc(p.set_name)}"
            f" · {_esc(p.printing)}</span></td><td class=n>{_money(p.market)}</td>"
            f"<td class=n>{_money(p.low)}</td>{cells}<td>{shop}</td>"
            f"<td class=why>{_esc(p.why)}{fit}</td></tr>"
        )
    return "\n".join(out)


def render_html(
    issue: date,
    deals: list[deals_mod.Deal],
    cards: list[Pick],
    sealed: list[Pick],
    windows: dict[str, list[drops_mod.Window]],
    charts: dict[str, list[Path]],
    cfg: dict,
    n_listings: int,
) -> str:
    week = issue.isocalendar()
    deal_rows = (
        "\n".join(
            f"<tr><td class=n>{_money(d.price)}</td><td class=n>{_money(d.market)}</td>"
            f"<td class=n><b>{d.edge:+.0%}</b></td><td><b>{_esc(d.product_name)}</b> "
            f"<span class=dim>{_esc(d.set_name)}</span></td>"
            f"<td><a href='{_esc(d.url)}'>{_esc(d.shop)}</a></td></tr>"
            for d in deals
        )
        or "<tr><td colspan=5>No listings under market this week.</td></tr>"
    )
    drop_rows = []
    for retailer, ws in windows.items():
        stores = drops_mod.stores(retailer=retailer)[:2]
        where = "; ".join(s.name for s in stores) or "—"
        best = " · ".join(f"{w.label} ({w.probability:.0%})" for w in ws[:2]) or "—"
        drop_rows.append(
            f"<tr><td><b>{_esc(retailer)}</b></td><td>{best}</td><td>{where}</td></tr>"
        )
    heads = "".join(f"<th class=n>{h}</th>" for h in HORIZONS)
    pick_head = (
        f"<tr><th>#</th><th>Product</th><th class=n>Market</th><th class=n>Low</th>{heads}"
        "<th>Best shop</th><th>Why watch</th></tr>"
    )
    img = {
        k: "".join(f"<img src='{p.resolve().as_uri()}' class=chart>" for p in v)
        for k, v in charts.items()
    }
    return f"""<!doctype html><html><head><meta charset=utf-8><style>
@page {{ size: Letter; margin: 14mm 12mm; @bottom-center {{ content: "cardscout weekly · {issue:%b %d, %Y} · page " counter(page); font-size: 7pt; color: #777; }} }}
body {{ font: 8.6pt/1.3 -apple-system, Helvetica, Arial, sans-serif; color: #111; }}
h1 {{ font-size: 18pt; margin: 0 0 2px; }} h2 {{ font-size: 11.5pt; margin: 14px 0 4px; border-bottom: 1.5px solid #0b5ed7; padding-bottom: 2px; }}
.sub {{ color: #555; margin-bottom: 8px; }}
table {{ width: 100%; border-collapse: collapse; }} th {{ text-align: left; font-size: 7.5pt; color: #444; border-bottom: 1px solid #bbb; padding: 3px 4px; }}
tr {{ page-break-inside: avoid; }} td {{ padding: 3px 4px; border-bottom: 1px solid #eee; vertical-align: top; }} td.n, th.n {{ text-align: right; white-space: nowrap; }}
.dim {{ color: #666; font-size: 7.5pt; }} .why {{ font-size: 7.5pt; color: #333; max-width: 150px; }}
.tag {{ background: #e7f1ff; color: #0b5ed7; border-radius: 3px; padding: 0 3px; font-size: 6.5pt; }}
.chart {{ width: 100%; max-height: 235mm; page-break-inside: avoid; margin-top: 6px; }}
.note {{ background: #fff8e1; border-left: 3px solid #f0ad4e; padding: 5px 8px; margin: 8px 0; font-size: 7.8pt; }}
a {{ color: #0b5ed7; text-decoration: none; }}
</style></head><body>
<h1>cardscout weekly — Pokémon market brief</h1>
<div class=sub>Issue {week.year}-W{week.week:02d} · {issue:%A, %B %d, %Y} · Richmond, VA {HOME_ZIP} · market = TCGplayer market price via tcgcsv · {n_listings:,} shop listings scanned</div>

<h2>1. Deals this week — listings under market (cards ≤ {_money(float(cfg["max_card_price"]))}, boxes ≤ {_money(float(cfg["max_box_price"]))})</h2>
<table><tr><th class=n>Shop price</th><th class=n>Market</th><th class=n>Edge</th><th>Product</th><th>Shop</th></tr>
{deal_rows}</table>
<div class=dim>Edge = (market − price) / market. Verify condition and shipping before buying; matches are automatic.</div>

<h2>2. Top {len(cards)} cards to watch (≤ {_money(float(cfg["max_card_price"]))})</h2>
<table>{pick_head}
{_pick_rows(cards)}</table>
{img.get("cards", "")}

<h2>3. Top {len(sealed)} sealed boxes to watch (≤ {_money(float(cfg["max_box_price"]))})</h2>
<table>{pick_head}
{_pick_rows(sealed)}</table>
{img.get("sealed", "")}

<div class=note><b>How to read the outlook.</b> 6 mo / 1 yr / 2 yr / 3 yr columns and charts are <b>scenarios</b>, not predictions: today's market price
projected along the typical Pokémon release cycle (prices soften while a set is in print, recover once it goes out of print), with the shaded band
widening with time. Where cardscout has 7+ days of its own history for a product the fitted trend tilts year one and the row is tagged <span class=tag>fitted</span>.
Run <code>cardscout sync</code> weekly and the fitted share grows.</div>

<h2>4. Richmond restock windows — next 7 days</h2>
<table><tr><th>Retailer</th><th>Most likely windows (share of the week's stocking)</th><th>Nearest stores</th></tr>
{"".join(drop_rows)}</table>
<div class=dim>Windows blend reported retailer patterns with your logged sightings (<code>cardscout drops observe …</code>). They are probabilities, not guarantees.</div>
</body></html>"""


def build(
    ledger: Ledger,
    out_dir: Path | None = None,
    cfg: dict | None = None,
    issue: date | None = None,
    top_deals: int = 12,
) -> tuple[Path, Path]:
    """Write the HTML and PDF for this week's issue; return (pdf, html)."""
    cfg = cfg or load_watchlist()
    issue = issue or datetime.now(ZoneInfo(HOME_TZ)).date()
    out_dir = out_dir or (data_dir() / "newsletters")
    out_dir.mkdir(parents=True, exist_ok=True)
    week = issue.isocalendar()
    stem = out_dir / f"cardscout-{week.year}-W{week.week:02d}"

    listings = ledger.latest_listings()
    market = ledger.latest_market()
    found = [
        d
        for d in deals_mod.find_deals(listings, market, min_edge=0.10)
        if d.market <= float(cfg["max_box_price" if d.is_sealed else "max_card_price"])
    ][:top_deals]
    cards = select_cards(ledger, cfg)
    sealed = select_sealed(ledger, cfg)
    windows = {
        r: drops_mod.forecast_windows(r, ledger.drop_observations(r), days=7, top=2)
        for r in drops_mod.PRIORS
    }
    charts = {
        "cards": charts_for(cards, stem.with_name(stem.name + "-cards"), "Cards — 3-year outlook"),
        "sealed": charts_for(
            sealed, stem.with_name(stem.name + "-sealed"), "Sealed — 3-year outlook"
        ),
    }
    html = render_html(issue, found, cards, sealed, windows, charts, cfg, len(listings))
    html_path = stem.with_suffix(".html")
    html_path.write_text(html)
    pdf_path = stem.with_suffix(".pdf")
    from weasyprint import HTML  # type: ignore[import-untyped]

    HTML(string=html, base_url=str(out_dir)).write_pdf(pdf_path)
    return pdf_path, html_path


# -- weekly schedule ------------------------------------------------------------

LAUNCHD_LABEL = "com.cardscout.weekly"


def schedule_text(cardscout_bin: str, weekday: int = 1, hour: int = 7) -> str:
    """Cron line (Linux) and launchd plist (macOS) that run sync + newsletter weekly.

    weekday: 0=Sunday .. 6=Saturday (cron and launchd agree on this)."""
    cron = f"0 {hour} * * {weekday} {cardscout_bin} newsletter >> $HOME/.cardscout/weekly.log 2>&1"
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{LAUNCHD_LABEL}</string>
  <key>ProgramArguments</key><array><string>{cardscout_bin}</string><string>newsletter</string></array>
  <key>StartCalendarInterval</key><dict><key>Weekday</key><integer>{weekday}</integer><key>Hour</key><integer>{hour}</integer><key>Minute</key><integer>0</integer></dict>
  <key>StandardOutPath</key><string>{Path.home()}/.cardscout/weekly.log</string>
  <key>StandardErrorPath</key><string>{Path.home()}/.cardscout/weekly.log</string>
</dict></plist>
"""
    return (
        "Linux / any machine with cron -- add this line with `crontab -e`:\n\n"
        f"  {cron}\n\n"
        f"macOS -- save the plist below as ~/Library/LaunchAgents/{LAUNCHD_LABEL}.plist and run\n"
        f"  launchctl load ~/Library/LaunchAgents/{LAUNCHD_LABEL}.plist\n\n{plist}"
    )
