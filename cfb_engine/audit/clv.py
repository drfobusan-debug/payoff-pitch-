"""Closing-line value: snapshot the closing market and score each bet against it.

``cfb-engine close`` captures the board near kickoff and writes a per-selection
closing snapshot (best American price + no-vig probability). The audit then
compares each bet's entry price to the close:

    clv     = closing no-vig prob - bet-time no-vig prob   (market moved our way)
    clv_ev  = EV of our bet price under the closing probability

Positive CLV is the durable signal that a model is beating the market even
before games are graded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cfb_engine.market import keys
from cfb_engine.market.board import GameOdds
from cfb_engine.market.ev import MarketQuote, ev_per_dollar, evaluate
from cfb_engine.schemas import Slate


@dataclass(frozen=True)
class ClosingQuote:
    american: float
    no_vig_prob: float


def _side_close(quotes: list[MarketQuote]) -> ClosingQuote | None:
    if not quotes:
        return None
    res = evaluate(0.5, quotes)
    return ClosingQuote(res.best_quote.american, res.fair_prob)


def closing_quotes(slate: Slate, board: dict[str, GameOdds]) -> dict[str, ClosingQuote]:
    """Map ``"<market>|<selection>"`` -> closing quote for every priced side."""
    out: dict[str, ClosingQuote] = {}
    for game in slate.games:
        odds = board.get(game.matchup())
        if odds is None:
            continue
        home, away = game.home.abbrev, game.away.abbrev
        for ab in (home, away):
            cq = _side_close(odds.ml.get(ab, []))
            if cq is not None:
                out[f"game_ml|{keys.game_ml(ab)}"] = cq
        point = odds.main_spread()
        if point is not None and point in odds.spreads:
            for ab, pt in ((home, point), (away, -point)):
                cq = _side_close(odds.spreads[point].get(ab, []))
                if cq is not None:
                    out[f"game_ats|{keys.game_ats(ab, pt)}"] = cq
        line = odds.main_total()
        if line is not None and line in odds.totals:
            for is_over, key in ((True, "over"), (False, "under")):
                cq = _side_close(odds.totals[line].get(key, []))
                if cq is not None:
                    out[f"game_total|{keys.game_total(is_over, line)}"] = cq
    return out


def merge_closing(
    existing: dict[str, ClosingQuote], fresh: dict[str, ClosingQuote]
) -> dict[str, ClosingQuote]:
    """Later capture wins per selection, but nothing already captured is dropped.

    A Saturday runs from noon to past midnight, so any single capture sees the
    close of one kickoff window and the long-stale opener of the rest. Repeat
    captures therefore merge instead of replacing: an in-progress game has left
    the pre-match board, and its captured close must survive later snapshots.
    """
    return {**existing, **fresh}


def save_closing(quotes: dict[str, ClosingQuote], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: {"american": q.american, "no_vig_prob": q.no_vig_prob} for k, q in quotes.items()}
    path.write_text(json.dumps(payload, indent=2))


def load_closing(path: Path) -> dict[str, ClosingQuote]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {k: ClosingQuote(v["american"], v["no_vig_prob"]) for k, v in raw.items()}


def compute_clv(
    market: str, selection: str, bet_american: float | None, bet_fair_prob: float | None,
    closing: dict[str, ClosingQuote],
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return ``(close_odds, close_prob, clv, clv_ev)`` for one bet."""
    cq = closing.get(f"{market}|{selection}")
    if cq is None:
        return None, None, None, None
    clv = None if bet_fair_prob is None else round(cq.no_vig_prob - bet_fair_prob, 4)
    clv_ev = None if bet_american is None else round(ev_per_dollar(cq.no_vig_prob, bet_american), 4)
    return cq.american, cq.no_vig_prob, clv, clv_ev


@dataclass
class ClvSummary:
    label: str
    n: int
    mean_clv: float
    beat_close_pct: float
    mean_clv_ev: float

    @property
    def positive(self) -> bool:
        return self.mean_clv >= 0


def clv_summary(rows: list[tuple[str, float | None, float | None]]) -> list[ClvSummary]:
    """Summarize ``(category, clv, clv_ev)`` rows by category plus an ALL row."""
    by_cat: dict[str, list[tuple[float, float]]] = {}
    allrows: list[tuple[float, float]] = []
    for cat, clv, clv_ev in rows:
        if clv is None:
            continue
        pair = (clv, clv_ev if clv_ev is not None else 0.0)
        by_cat.setdefault(cat, []).append(pair)
        allrows.append(pair)

    def summarize(label: str, pairs: list[tuple[float, float]]) -> ClvSummary:
        n = len(pairs)
        mean_clv = sum(c for c, _ in pairs) / n if n else 0.0
        beat = sum(1 for c, _ in pairs if c > 0) / n if n else 0.0
        mean_ev = sum(e for _, e in pairs) / n if n else 0.0
        return ClvSummary(label, n, round(mean_clv, 4), round(beat, 4), round(mean_ev, 4))

    out = [summarize(cat, by_cat[cat]) for cat in sorted(by_cat)]
    if allrows:
        out.append(summarize("ALL", allrows))
    return out
