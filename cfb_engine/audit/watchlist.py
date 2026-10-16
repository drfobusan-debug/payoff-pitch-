"""What the CFB engine is watching: every graded-only rule and screen, for the slate PDF.

These rules are graded on the ledger but never staked. The audit report shows
them after each slate; the slate carries the same snapshot so it can be read
alongside the card.
"""

from __future__ import annotations

from dataclasses import dataclass

from cfb_engine.audit import shop
from cfb_engine.audit.ledger import LedgerEntry
from cfb_engine.audit.probation import (
    CANDIDATE_RULES,
    CANDIDATE_SCREENS,
    CANDIDATE_UPGRADES,
    CLEAR,
    UNIT_RULES,
    WATCHING,
    YPP_RULES,
    Probation,
    _min_n,
    _min_refused,
    probation_rows,
)
from cfb_engine.market import gradedonly

# What each kind is waiting to see, and how the slate names it.
_KIND = {
    "screen": ("Graded-only market", "rows win"),
    "candidate": ("Skip screen", "rows lose"),
    "upgrade": ("Strong upgrade", "rows win"),
    "rule": ("Bet rule", "rows win"),
}
_GRADED_ONLY_KINDS = ("candidate", "upgrade", "rule")
_MARKET_TAG = {"ats": "ATS", "ml": "ML"}


def _rationales() -> dict[str, str]:
    out = {c.name: c.rationale for c in CANDIDATE_SCREENS}
    out.update({u.name: u.rationale for u in CANDIDATE_UPGRADES})
    out.update({r.name: r.rationale for r in (*CANDIDATE_RULES, *YPP_RULES)})
    for r in UNIT_RULES:
        tag = _MARKET_TAG.get(r.name.rsplit("_", 1)[-1], "")
        out[r.name] = f"{r.rationale} ({tag})" if tag else r.rationale
    return out


@dataclass(frozen=True)
class WatchRow:
    name: str
    what: str
    kind: str  # display label
    wants: str  # "rows win" | "rows lose"
    status: str
    n: int
    roi: float
    se: float
    first_half: float
    second_half: float
    needs: str

    @property
    def leaning(self) -> bool:
        """Whether the record so far points toward the verdict the rule wants."""
        return self.roi < 0 if self.wants == "rows lose" else self.roi > 0


def _watched(p: Probation) -> bool:
    return p.kind in _GRADED_ONLY_KINDS or (p.kind == "screen" and gradedonly.is_gate(p.name))


def _needs(p: Probation) -> str:
    bar = _min_refused() if p.kind == "screen" else _min_n()
    if p.status == WATCHING:
        return f"{bar - p.n} more graded"
    if p.status == CLEAR:
        return "judged: no edge yet"
    return f"verdict: {p.status}"


def _what(p: Probation, rationales: dict[str, str]) -> str:
    if p.kind == "screen":
        market = p.name.removeprefix("graded_only_").replace("game_", "")
        return f"would-be {market} buys, graded at the price they'd have taken"
    return rationales.get(p.name, p.name.replace("_", " "))


def watch_rows(entries: list[LedgerEntry]) -> list[WatchRow]:
    """Every graded-only rule and market, biggest samples first within each kind."""
    rationales = _rationales()
    order = {k: i for i, k in enumerate(("screen", "rule", "candidate", "upgrade"))}
    picked = sorted(
        (p for p in probation_rows(entries) if _watched(p)),
        key=lambda p: (order[p.kind], -p.n, p.name),
    )
    return [
        WatchRow(
            p.name,
            _what(p, rationales),
            _KIND[p.kind][0],
            _KIND[p.kind][1],
            p.status,
            p.n,
            p.roi,
            p.se,
            p.first_half,
            p.second_half,
            _needs(p),
        )
        for p in picked
    ]


@dataclass(frozen=True)
class Watchlist:
    rules: list[WatchRow]
    shopping: list[shop.Record]


def build(entries: list[LedgerEntry]) -> Watchlist:
    return Watchlist(watch_rows(entries), shop.records(entries, graded_only=False))
