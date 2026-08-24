"""The recommendation record produced by the pipeline and consumed by outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date as Date
from pathlib import Path

from cfb_engine.market.odds import prob_to_american
from cfb_engine.market.tiers import Tier


@dataclass
class Recommendation:
    game_date: Date
    game_id: str
    matchup: str
    market: str  # "game_ml" | "game_ats" | "game_total"
    selection: str
    model_prob: float
    raw_prob: float | None = None  # pre-calibration model probability (audit trail)
    line: float | None = None  # spread point (ATS) or total line
    book: str | None = None
    market_american: float | None = None
    # American price of the *other* side of this two-way market at the same book
    # (the under for a total, the opposing team for ML/ATS). Persisted so the
    # audit can grade/devig the fade side without re-pricing.
    opposite_american: float | None = None
    ev: float | None = None
    edge: float | None = None
    # Devigged market consensus probability and the probability the EV screen
    # actually bet on (they differ only when CFBE_MARKET_ANCHOR pulls the model
    # toward the market). bet_prob is the baseline for closing line value.
    fair_prob: float | None = None
    bet_prob: float | None = None
    tier: Tier = Tier.PASS
    reasons: list[str] = field(default_factory=list)
    # No-vig probability points the market has moved *toward* this side since the
    # first board the engine saw for the slate (negative: it walked away). The
    # pre-kickoff half of closing-line value; see ``cfb_engine.market.drift``.
    drift: float | None = None
    # The screen that demoted this row to Pass, if one did. Attribution is what
    # lets the audit grade a screen on the bets it refused rather than only on
    # the ones it let through.
    pass_gate: str | None = None
    # --- structured grading metadata (used by the nightly audit) ---
    team_side: str | None = None  # "home" | "away"
    side: str | None = None  # "win" | "cover" | "over" | "under"
    # --- game context (same for every rec in a game; for the card/preview) ---
    home_abbrev: str | None = None
    away_abbrev: str | None = None
    # Expected point differential (home perspective) = mean simulated margin,
    # and the total the sim expects, with their spreads.
    exp_margin: float | None = None
    exp_margin_sd: float | None = None
    exp_total: float | None = None
    exp_total_sd: float | None = None

    @property
    def model_american(self) -> float:
        return prob_to_american(self.model_prob)

    @property
    def display_category(self) -> str:
        m = self.market
        if m == "game_ml":
            return "Moneyline"
        if m == "game_ats":
            return "Spread (ATS)"
        if m == "game_total":
            return "Totals"
        return m

    def as_row(self) -> dict[str, object]:
        return {
            "Date": self.game_date.isoformat(),
            "Matchup": self.matchup,
            "Market": self.display_category,
            "Selection": self.selection,
            "Line": self.line if self.line is not None else "",
            "Model %": round(self.model_prob * 100, 1),
            "Market %": round(self.fair_prob * 100, 1) if self.fair_prob is not None else "",
            "Fair Odds": round(self.model_american),
            "Book": self.book or "",
            "Book Odds": round(self.market_american) if self.market_american is not None else "",
            "EV": round(self.ev, 3) if self.ev is not None else "",
            "Edge": round(self.edge, 3) if self.edge is not None else "",
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
