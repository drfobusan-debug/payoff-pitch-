"""The recommendation record produced by the pipeline and consumed by outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date as Date
from pathlib import Path

from mlb_engine.market.odds import prob_to_american
from mlb_engine.market.tiers import Tier


@dataclass
class Recommendation:
    game_date: Date
    game_pk: int
    matchup: str
    category: str  # "game" | "f5" | "batter" | "pitcher"
    market: str
    selection: str
    model_prob: float
    line: float | None = None
    book: str | None = None
    market_american: float | None = None
    ev: float | None = None
    edge: float | None = None
    handle_pct: float | None = None
    bets_pct: float | None = None
    tier: Tier = Tier.PASS
    reasons: list[str] = field(default_factory=list)
    # --- structured grading metadata (used by the nightly audit) ---
    team_side: str | None = None  # "home" | "away"
    player_id: int | None = None
    stat: str | None = None  # e.g. "H", "HR", "K", "outs", "ER"
    side: str | None = None  # "win" | "cover" | "over" | "under"

    @property
    def model_american(self) -> float:
        return prob_to_american(self.model_prob)

    @property
    def display_category(self) -> str:
        """Human-facing market group used in outputs and the ledger."""
        m = self.market
        if m == "game_ml":
            return "Moneyline"
        if m == "game_total":
            return "Totals"
        if m == "game_rl":
            return "Run Lines"
        if m.startswith("f5"):
            return "First-5 (F5)"
        if m.startswith("batter_"):
            return "Batter Props"
        if m.startswith("pitcher_"):
            return "Pitcher Props"
        if m == "comeback":
            return "Comeback (info)"
        return m

    def as_row(self) -> dict[str, object]:
        return {
            "Date": self.game_date.isoformat(),
            "Matchup": self.matchup,
            "Category": self.category,
            "Market": self.market,
            "Selection": self.selection,
            "Line": self.line if self.line is not None else "",
            "Model %": round(self.model_prob * 100, 1),
            "Fair Odds": round(self.model_american),
            "Book": self.book or "",
            "Book Odds": round(self.market_american) if self.market_american is not None else "",
            "EV": round(self.ev, 3) if self.ev is not None else "",
            "Edge": round(self.edge, 3) if self.edge is not None else "",
            "Handle %": self.handle_pct if self.handle_pct is not None else "",
            "Bets %": self.bets_pct if self.bets_pct is not None else "",
            "Tier": self.tier.value,
            "Notes": "; ".join(self.reasons),
        }


def save_json(recs: list[Recommendation], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for r in recs:
        d = asdict(r)
        d["game_date"] = r.game_date.isoformat()
        d["tier"] = r.tier.value
        payload.append(d)
    path.write_text(json.dumps(payload, indent=2))


def load_json(path: Path) -> list[Recommendation]:
    raw = json.loads(path.read_text())
    out: list[Recommendation] = []
    for d in raw:
        d = dict(d)
        d["game_date"] = Date.fromisoformat(d["game_date"])
        d["tier"] = Tier(d["tier"])
        out.append(Recommendation(**d))
    return out
