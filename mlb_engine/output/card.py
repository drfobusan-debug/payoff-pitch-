"""Daily betting card: hybrid narrative + top plays per game.

Turns the pipeline's :class:`Recommendation` list into a reader-facing card --
one section per game with a casual-but-quantitative read (starter metrics and
regression, ballpark, live weather, and the market/value picture) followed by
the top positive-EV plays with model probability, market-implied probability,
edge, and EV.

The card is built purely from the persisted recommendations, so it can be
regenerated from a prior slate's ``predictions_*.json`` without re-simulating.
Two renderers are provided -- Markdown (console/attachment) and HTML (email) --
both driven off the same :class:`GameCard` structures.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import date as Date

from mlb_engine.market.odds import american_to_prob
from mlb_engine.market.tiers import Tier
from mlb_engine.recommendations import Recommendation

# Markets that make coherent stand-alone plays on the card, best-first.
_GAME_MARKETS = ("game_total", "game_rl", "game_ml", "f5_rl", "f5_ml", "f5_total")
_MAX_PLAYS = 5


@dataclass
class Play:
    market: str
    selection: str
    odds: float | None
    model_prob: float
    implied_prob: float | None
    edge: float | None
    ev: float | None
    tier: Tier

    def _odds_str(self) -> str:
        return "n/a" if self.odds is None else f"{self.odds:+.0f}"

    @property
    def is_dart(self) -> bool:
        """High-variance longshot -- a plus-money HR or any big-price prop."""
        return self.market == "batter_hr" or (self.odds is not None and self.odds >= 250)


@dataclass
class Starter:
    name: str
    k5_prob: float | None = None  # P(5+ strikeouts)
    er3_prob: float | None = None  # P(3+ earned runs)


@dataclass
class GameCard:
    matchup: str
    starters: list[Starter] = field(default_factory=list)
    narrative: list[str] = field(default_factory=list)  # paragraphs
    plays: list[Play] = field(default_factory=list)


def _pitcher_name(selection: str, token: str) -> str:
    # e.g. "Tarik Skubal Ks o5.5" -> "Tarik Skubal"
    return selection.split(f" {token} ")[0].strip()


def _starters(recs: list[Recommendation]) -> list[Starter]:
    """Recover the two starters and their K/ER projections from prop rows."""
    order: list[str] = []
    k5: dict[str, float] = {}
    k5_line: dict[str, float] = {}
    er3: dict[str, float] = {}
    for r in recs:
        if r.market == "pitcher_k" and " Ks o" in r.selection and r.line is not None:
            name = _pitcher_name(r.selection, "Ks")
            if name not in order:
                order.append(name)
            # Keep the line closest to 4.5 (== "5 or more") as the 5+ K proxy.
            if name not in k5_line or abs(r.line - 4.5) < abs(k5_line[name] - 4.5):
                k5[name] = r.model_prob
                k5_line[name] = r.line
        # o2.5 ER line == "3 or more earned".
        elif r.market == "pitcher_er" and r.line is not None and abs(r.line - 2.5) < 0.01:
            er3[_pitcher_name(r.selection, "ER")] = r.model_prob
    return [Starter(name=n, k5_prob=k5.get(n), er3_prob=er3.get(n)) for n in order[:2]]


def _pct(p: float | None) -> str:
    return f"{p * 100:.0f}%" if p is not None else "n/a"


def _park_clause(recs: list[Recommendation]) -> str:
    ctx = recs[0]
    if ctx.park_name is None or ctx.park_factor is None:
        return ""
    pf = ctx.park_factor
    carry = ctx.carry_factor if ctx.carry_factor is not None else 1.0
    nums = f"{ctx.park_name} ({pf:.0f} park factor, {carry:.2f} carry)"
    if pf >= 103 or carry >= 1.10:
        return f"They're playing in {nums} — a genuine hitter's park that pushes run scoring up"
    if pf <= 97 or carry <= 0.92:
        return f"They're in {nums} — a pitcher-friendly yard where fly balls go to die"
    return f"They're in {nums}, a roughly neutral run environment"


def _weather_clause(recs: list[Recommendation]) -> str:
    ctx = recs[0]
    note = (ctx.wx_note or "").lower()
    if "closed roof" in note or ctx.roof in ("closed", "dome"):
        return "The roof is closed, so weather is a non-factor tonight"
    if ctx.wx_summary is None:
        return ""
    lead = f"Live weather: {ctx.wx_summary}"
    hr = ctx.wx_hr_mult
    if hr is not None and hr >= 1.05:
        lead += " — the air is helping the ball carry"
    elif hr is not None and hr <= 0.95:
        lead += " — the air is knocking balls down"
    else:
        lead += " — close to neutral for carry"
    if "retractable" in note:
        lead += " (retractable roof, effect damped)"
    return lead


def _xrd_clause(recs: list[Recommendation]) -> str:
    """Expected run differential (xRD/G): the sim's sequencing-luck-free margin."""
    ctx = recs[0]
    if ctx.xrd is None:
        return ""
    parts = ctx.matchup.split(" @ ")
    if len(parts) != 2:
        return ""
    away_abbr, home_abbr = parts
    team = home_abbr if ctx.xrd >= 0 else away_abbr
    mag = abs(ctx.xrd)
    if mag >= 1.0:
        tier = "an elite, run-line-reliable edge"
    elif mag >= 0.5:
        tier = "a structurally sound edge"
    else:
        tier = "a coin-flip margin, matchup-dependent"
    sd = f" ±{ctx.xrd_sd:.1f}" if ctx.xrd_sd is not None else ""
    return f"Expected run differential: {team} +{mag:.1f}{sd} runs/game — {tier}"


def _narrative(recs: list[Recommendation], starters: list[Starter]) -> list[str]:
    read: list[str] = []

    # 1) Starters + regression (rotate phrasing so 15 games don't read identically).
    ranked = [s for s in starters if s.k5_prob is not None]
    if len(ranked) == 2:
        ranked.sort(key=lambda s: s.k5_prob or 0.0, reverse=True)
        a, b = ranked
        openers = (
            f"{a.name} is the arm to trust — the regression gives him {_pct(a.k5_prob)} to clear "
            f"5 strikeouts, against {_pct(b.k5_prob)} for {b.name}",
            f"On the mound, {a.name} grades out ahead: {_pct(a.k5_prob)} to punch out 5-plus, "
            f"versus {_pct(b.k5_prob)} for {b.name}",
            f"This one tilts on the starters — {a.name} projects {_pct(a.k5_prob)} to reach 5 Ks "
            f"while {b.name} sits at {_pct(b.k5_prob)}",
        )
        sent = openers[len(a.name) % len(openers)]
        if a.er3_prob is not None:
            sent += f", and he's the stingier arm too ({_pct(a.er3_prob)} to allow 3+ earned)"
        read.append(sent + ".")
    elif len(ranked) == 1:
        a = ranked[0]
        read.append(
            f"{a.name} headlines the mound matchup with a {_pct(a.k5_prob)} shot at 5+ strikeouts."
        )

    # 2) Park + weather.
    env = [c for c in (_park_clause(recs), _weather_clause(recs)) if c]
    if env:
        read.append(". ".join(env) + ".")

    read_para = " ".join(read)

    # 3) Market read: favored side, whether the ML is a fade, and the total lean.
    market: list[str] = []
    ml = [r for r in recs if r.market == "game_ml"]
    if ml:
        fav = max(ml, key=lambda r: r.model_prob)
        fav_team = fav.selection.replace(" ML", "")
        if fav.ev is not None and fav.ev <= 0:
            market.append(
                f"The model likes {fav_team} to win ({_pct(fav.model_prob)}), but the moneyline "
                "is priced too rich to back — skip it and take the value on the run line and total"
            )
        else:
            market.append(f"The model makes {fav_team} a {_pct(fav.model_prob)} winner")
    tot = [r for r in recs if r.market == "game_total" and r.ev is not None]
    if tot:
        best = max(tot, key=lambda r: r.ev or -1.0)
        if (best.ev or 0) > 0:
            market.append(
                f"and it leans {best.selection} ({_pct(best.model_prob)}), where the edge lives"
            )
    market_para = (", ".join(market) + ".") if market else ""

    xrd_para = _xrd_clause(recs)
    if xrd_para:
        xrd_para += "."

    return [p for p in (read_para, market_para, xrd_para) if p]


def _plays(recs: list[Recommendation]) -> list[Play]:
    buys = [
        r
        for r in recs
        if r.tier in (Tier.STRONG, Tier.MODERATE) and r.ev is not None and r.ev > 0
    ]

    # Prefer coherent game-level plays first, then best props; dedupe selections.
    def sort_key(r: Recommendation) -> tuple[int, float]:
        rank = _GAME_MARKETS.index(r.market) if r.market in _GAME_MARKETS else len(_GAME_MARKETS)
        return (rank, -(r.ev or 0.0))

    buys.sort(key=sort_key)

    seen: set[tuple[str, str]] = set()
    out: list[Play] = []
    for r in buys:
        key = (r.market, r.selection)
        if key in seen:
            continue
        seen.add(key)
        implied = american_to_prob(r.market_american) if r.market_american is not None else None
        out.append(
            Play(
                market=r.market,
                selection=r.selection,
                odds=r.market_american,
                model_prob=r.model_prob,
                implied_prob=implied,
                edge=r.edge,
                ev=r.ev,
                tier=r.tier,
            )
        )
        if len(out) >= _MAX_PLAYS:
            break
    return out


def build_cards(recs: list[Recommendation]) -> list[GameCard]:
    """Group recommendations by game and build a card for each."""
    by_game: dict[str, list[Recommendation]] = {}
    for r in recs:
        by_game.setdefault(r.matchup, []).append(r)

    cards: list[GameCard] = []
    for grp in by_game.values():
        plays = _plays(grp)
        if not plays:
            continue
        starters = _starters(grp)
        cards.append(
            GameCard(
                matchup=grp[0].matchup,
                starters=starters,
                narrative=_narrative(grp, starters),
                plays=plays,
            )
        )
    # Order games by their strongest edge, best first.
    cards.sort(key=lambda c: max((p.ev or 0.0) for p in c.plays), reverse=True)
    return cards


def _play_bits(p: Play) -> str:
    bits = [f"model {_pct(p.model_prob)}"]
    if p.implied_prob is not None:
        bits.append(f"{_pct(p.implied_prob)} implied")
    if p.edge is not None:
        bits.append(f"{p.edge * 100:+.0f}% edge")
    if p.ev is not None:
        bits.append(f"+{p.ev:.2f} EV")
    return ", ".join(bits)


def _play_line_md(p: Play) -> str:
    tail = " 🎯 *dart — bet small*" if p.is_dart else ""
    return f"- **{p.selection} ({p._odds_str()})** — *{_play_bits(p)}*{tail}"


def render_markdown(cards: list[GameCard], slate_date: Date) -> str:
    lines = [
        f"# PayoffPitch — Betting Card for {slate_date.isoformat()}",
        "",
        "*The read on each game — starters, ballpark, weather, and where the market "
        "is off — then the plays with model probability, market-implied probability, "
        "edge, and EV.*",
        "",
    ]
    for c in cards:
        lines += ["---", "", f"## {c.matchup}", ""]
        for para in c.narrative:
            lines += [para, ""]
        lines.append("**🔒 PLAYS**")
        lines += [_play_line_md(p) for p in c.plays]
        lines.append("")
    lines += [
        "---",
        "",
        "*EV = expected value per $1 staked at the listed price; edge = model "
        "probability minus the book's implied probability. 🎯 darts are "
        "high-variance longshots — bet small. Prices move; shop the number.*",
    ]
    return "\n".join(lines)


def _play_line_html(p: Play) -> str:
    tail = " 🎯 <em>dart — bet small</em>" if p.is_dart else ""
    sel = html.escape(f"{p.selection} ({p._odds_str()})")
    return f"<li><strong>{sel}</strong> — <em>{html.escape(_play_bits(p))}</em>{tail}</li>"


def render_html(cards: list[GameCard], slate_date: Date) -> str:
    blocks = [
        "<h1>PayoffPitch — Betting Card for "
        f"{html.escape(slate_date.isoformat())}</h1>",
        "<p><em>The read on each game — starters, ballpark, weather, and where the "
        "market is off — then the plays with model probability, market-implied "
        "probability, edge, and EV.</em></p>",
    ]
    for c in cards:
        blocks.append(f"<h2>{html.escape(c.matchup)}</h2>")
        for para in c.narrative:
            blocks.append(f"<p>{html.escape(para)}</p>")
        items = "".join(_play_line_html(p) for p in c.plays)
        blocks.append(f"<p><strong>🔒 PLAYS</strong></p><ul>{items}</ul>")
    blocks.append(
        "<hr><p><em>EV = expected value per $1 staked; edge = model minus the "
        "book's implied probability. 🎯 darts are high-variance longshots — bet "
        "small. Prices move; shop the number.</em></p>"
    )
    style = (
        "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
        "max-width:760px;margin:24px auto;padding:0 18px;color:#1a1a1a;line-height:1.55}"
        "h1{font-size:24px;border-bottom:3px solid #0b6;padding-bottom:8px}"
        "h2{font-size:18px;margin-top:26px;background:#0b6;color:#fff;padding:8px 12px;"
        "border-radius:6px}ul{padding-left:20px}li{margin:5px 0}em{color:#555;font-size:13px}"
        "strong{color:#0a5}"
    )
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{style}</style></head>"
        f"<body>{''.join(blocks)}</body></html>"
    )
