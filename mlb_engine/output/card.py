"""Daily betting card: hybrid narrative + top plays per game.

Turns the pipeline's :class:`Recommendation` list into a reader-facing card --
one section per game with a short analytical narrative (which starter owns the
strikeout edge, which side the model favors, whether the moneyline is a fade,
and the total lean) followed by the top positive-EV plays with model
probability, market-implied probability, edge, and EV.

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
        if self.odds is None:
            return "n/a"
        return f"{self.odds:+.0f}"


@dataclass
class Starter:
    name: str
    k5_prob: float | None = None  # P(5+ strikeouts)
    er3_prob: float | None = None  # P(3+ earned runs)


@dataclass
class GameCard:
    matchup: str
    starters: list[Starter] = field(default_factory=list)
    narrative: str = ""
    plays: list[Play] = field(default_factory=list)


def _first(recs: list[Recommendation], market: str) -> Recommendation | None:
    for r in recs:
        if r.market == market:
            return r
    return None


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


def _narrative(recs: list[Recommendation], starters: list[Starter]) -> str:
    parts: list[str] = []

    # Strikeout edge between the two starters.
    ranked = [s for s in starters if s.k5_prob is not None]
    if len(ranked) == 2:
        ranked.sort(key=lambda s: s.k5_prob or 0.0, reverse=True)
        a, b = ranked
        sent = (
            f"{a.name} profiles as the sharper strikeout arm "
            f"({_pct(a.k5_prob)} to clear 5 Ks vs {_pct(b.k5_prob)} for {b.name})"
        )
        if a.er3_prob is not None:
            sent += f", and grades stingier on runs ({_pct(a.er3_prob)} to allow 3+ earned)"
        parts.append(sent + ".")
    elif len(ranked) == 1:
        a = ranked[0]
        parts.append(f"{a.name} headlines with a {_pct(a.k5_prob)} shot at 5+ strikeouts.")

    # Favored side + whether the moneyline is a fade.
    ml = [r for r in recs if r.market == "game_ml"]
    if ml:
        fav = max(ml, key=lambda r: r.model_prob)
        fav_team = fav.selection.replace(" ML", "")
        clause = f"The model makes {fav_team} a {_pct(fav.model_prob)} winner"
        if fav.ev is not None and fav.ev <= 0:
            clause += ", but the moneyline is priced too rich to back -- the value sits on the run line and total"
        parts.append(clause + ".")

    # Total lean.
    tot = [r for r in recs if r.market == "game_total"]
    priced = [r for r in tot if r.ev is not None]
    if priced:
        best = max(priced, key=lambda r: r.ev or -1.0)
        if (best.ev or 0) > 0:
            parts.append(f"It leans {best.selection} ({_pct(best.model_prob)}).")

    return " ".join(parts)


def _plays(recs: list[Recommendation]) -> list[Play]:
    buys = [
        r
        for r in recs
        if r.tier in (Tier.STRONG, Tier.MODERATE) and r.ev is not None and r.ev > 0
    ]
    buys.sort(key=lambda r: r.ev or 0.0, reverse=True)

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
    for matchup, grp in by_game.items():
        plays = _plays(grp)
        if not plays:
            continue
        starters = _starters(grp)
        cards.append(
            GameCard(
                matchup=matchup,
                starters=starters,
                narrative=_narrative(grp, starters),
                plays=plays,
            )
        )
    # Order games by their strongest edge, best first.
    cards.sort(key=lambda c: max((p.ev or 0.0) for p in c.plays), reverse=True)
    return cards


def _play_line_md(p: Play) -> str:
    bits = [f"model {_pct(p.model_prob)}"]
    if p.implied_prob is not None:
        bits.append(f"{_pct(p.implied_prob)} implied")
    if p.ev is not None:
        bits.append(f"+{p.ev:.2f} EV")
    return f"- **{p.selection} ({p._odds_str()})** — *{', '.join(bits)}*"


def render_markdown(cards: list[GameCard], slate_date: Date) -> str:
    lines = [
        f"# PayoffPitch — Betting Card for {slate_date.isoformat()}",
        "",
        "*Model-vs-market edges. Each game: the read, then the plays with model "
        "probability, market-implied probability, and EV. Shop the listed number.*",
        "",
    ]
    for c in cards:
        lines += ["---", "", f"## {c.matchup}", ""]
        if c.narrative:
            lines += [c.narrative, ""]
        lines.append("**🔒 PLAYS**")
        lines += [_play_line_md(p) for p in c.plays]
        lines.append("")
    lines += [
        "---",
        "",
        "*EV = expected value per $1 staked at the listed price; edge = model "
        "probability minus the book's implied probability. Big plus-money props "
        "are positive-EV lottery tickets — bet small. Prices move; shop the number.*",
    ]
    return "\n".join(lines)


def _play_line_html(p: Play) -> str:
    bits = [f"model {_pct(p.model_prob)}"]
    if p.implied_prob is not None:
        bits.append(f"{_pct(p.implied_prob)} implied")
    if p.ev is not None:
        bits.append(f"+{p.ev:.2f} EV")
    sel = html.escape(f"{p.selection} ({p._odds_str()})")
    return f"<li><strong>{sel}</strong> — <em>{html.escape(', '.join(bits))}</em></li>"


def render_html(cards: list[GameCard], slate_date: Date) -> str:
    blocks = [
        "<h1>PayoffPitch — Betting Card for "
        f"{html.escape(slate_date.isoformat())}</h1>",
        "<p><em>Model-vs-market edges. Each game: the read, then the plays with "
        "model probability, market-implied probability, and EV.</em></p>",
    ]
    for c in cards:
        blocks.append(f"<h2>{html.escape(c.matchup)}</h2>")
        if c.narrative:
            blocks.append(f"<p>{html.escape(c.narrative)}</p>")
        items = "".join(_play_line_html(p) for p in c.plays)
        blocks.append(f"<p><strong>🔒 PLAYS</strong></p><ul>{items}</ul>")
    style = (
        "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
        "max-width:760px;margin:24px auto;padding:0 18px;color:#1a1a1a;line-height:1.5}"
        "h1{font-size:24px;border-bottom:3px solid #0b6;padding-bottom:8px}"
        "h2{font-size:18px;margin-top:26px;background:#0b6;color:#fff;padding:8px 12px;"
        "border-radius:6px}ul{padding-left:20px}li{margin:5px 0}em{color:#666;font-size:13px}"
        "strong{color:#0a5}"
    )
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{style}</style></head>"
        f"<body>{''.join(blocks)}</body></html>"
    )
