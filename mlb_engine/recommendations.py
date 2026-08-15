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
    raw_prob: float | None = None  # pre-calibration model probability (audit trail)
    line: float | None = None
    book: str | None = None
    market_american: float | None = None
    # American price of the *other* side of this two-way market at the same book
    # (the under for an O/U prop, the opposing team for ML/RL). Already fetched to
    # devig; persisted so the audit can grade the fade side without re-pricing.
    opposite_american: float | None = None
    ev: float | None = None
    edge: float | None = None
    # Devigged market consensus probability for this selection, and the
    # probability the EV screen actually bet on. They differ from model_prob only
    # when MLBE_MARKET_ANCHOR pulls the model toward the market; model_prob stays
    # the model's own output so PPV/NPV and the calibration refit keep measuring
    # the model rather than the blend. bet_prob is also the baseline for closing
    # line value: the audit compares it against the closing no-vig price.
    fair_prob: float | None = None
    bet_prob: float | None = None
    handle_pct: float | None = None
    bets_pct: float | None = None
    tier: Tier = Tier.PASS
    reasons: list[str] = field(default_factory=list)
    # --- structured grading metadata (used by the nightly audit) ---
    team_side: str | None = None  # "home" | "away"
    player_id: int | None = None
    stat: str | None = None  # e.g. "H", "HR", "K", "outs", "ER"
    side: str | None = None  # "win" | "cover" | "over" | "under"
    # Name of the run-line NPV gate that vetoed this selection, if any. Kept so
    # the audit can grade the counterfactual: did the gate remove losers?
    veto_gate: str | None = None
    # Which screen turned this selection into a Pass ("" when it was bought).
    # `reasons` already says so in prose, but only a stable name makes the
    # decision gradeable: a gate that rejects winners is a false negative and
    # is invisible until its own rows can be pulled out of the ledger.
    pass_gate: str | None = None
    # V1-style selector metadata surfaced for prop recommendations.
    signal: str | None = None
    factor: float | None = None
    score: float | None = None
    profile: str | None = None
    # Batter contact-quality features stamped on prop recs (for audit tuning of
    # the power/contact floor). None on non-batter markets.
    bat_xslg: float | None = None
    bat_k_pct: float | None = None
    bat_bb_pct: float | None = None
    # Singles-Under NPV score (structural anti-singles red flags); None off-batter.
    bat_singles_under: float | None = None
    # Opposing starter's SIERA (Statcast) for the singles matchup gate.
    opp_starter_siera: float | None = None
    # --- game environment context (same for every rec in a game; for the card) ---
    park_name: str | None = None
    park_factor: float | None = None
    carry_factor: float | None = None
    roof: str | None = None
    wx_summary: str | None = None  # live weather string, None if roofed/unavailable
    wx_hr_mult: float | None = None  # weather HR multiplier (1.0 = neutral)
    wx_note: str | None = None
    # Bullpen depletion (0-100 StatsAPI workload proxy) for the team this rec
    # backs and for its opponent. Stamped on game-level recs so the audit can
    # grade the moneyline bullpen gate's counterfactual.
    pen_fatigue: float | None = None
    opp_pen_fatigue: float | None = None
    # Lineup provenance ("posted" | "projected") and hours to first pitch at
    # pricing time -- the late-information read (see features.lineup_lock).
    lineup_status: str | None = None
    hours_to_first_pitch: float | None = None
    # Expected run differential (home perspective) = mean of the simulated run
    # margin, and its spread -- the sequencing-luck-free per-game xRD/G.
    xrd: float | None = None
    xrd_sd: float | None = None
    # An outside model's read on this same selection (VSIN/Opta, see data.opta).
    # opta_stars is Opta's own 0-3 rating, and it belongs to the side *it* bet:
    # opta_agrees says whether that is our side, so three stars against us are
    # never displayed as three stars for us. None throughout where Opta had no
    # projection for the prop, which is most game-level markets.
    opta_prob: float | None = None
    opta_stars: int | None = None
    opta_agrees: bool | None = None
    # THE BAT X's probability for the side we are backing (see data.batx). It is
    # displayed and persisted to the ledger, and it is an input to nothing: the
    # head-to-head that would justify weighting it is itself run off this column.
    batx_prob: float | None = None

    @property
    def bet_group(self) -> tuple[int, str, int | None]:
        """The thing being bet, independent of side and line.

        A prop is one opinion, so a player's hit total is one bet whether it is
        expressed as o0.5 or o1.5, and a run line is one bet whether it is taken
        as the favourite at -1.5 or the dog at +1.5. Grouping on this is what
        :func:`enforce_one_buy_per_group` deduplicates.
        """
        return (self.game_pk, self.market, self.player_id)

    @property
    def opta_mark(self) -> str:
        """Opta's stars, shown only when it likes the side we are buying."""
        if not self.opta_stars or self.opta_agrees is None:
            return ""
        stars = "\u2605" * self.opta_stars
        return stars if self.opta_agrees else f"fade {stars}"

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
            "Market %": round(self.fair_prob * 100, 1) if self.fair_prob is not None else "",
            "Fair Odds": round(self.model_american),
            "Book": self.book or "",
            "Book Odds": round(self.market_american) if self.market_american is not None else "",
            "EV": round(self.ev, 3) if self.ev is not None else "",
            "Edge": round(self.edge, 3) if self.edge is not None else "",
            "Handle %": self.handle_pct if self.handle_pct is not None else "",
            "Bets %": self.bets_pct if self.bets_pct is not None else "",
            "Tier": self.tier.value,
            "Signal": self.signal or "",
            "Factor": round(self.factor, 3) if self.factor is not None else "",
            "Score": round(self.score, 2) if self.score is not None else "",
            "Profile": self.profile or "",
            "Opta %": round(self.opta_prob * 100, 1) if self.opta_prob is not None else "",
            "AI": self.opta_mark,
            "BAT X %": round(self.batx_prob * 100, 1) if self.batx_prob is not None else "",
            "Notes": "; ".join(self.reasons),
        }


ONE_BUY_GATE = "one_buy_per_group"


def enforce_one_buy_per_group(recs: list[Recommendation]) -> list[Recommendation]:
    """Leave at most one buy per prop: over, under, or pass, never two.

    The pipeline prices every side of every line so the audit can grade the side
    we declined, which is what makes a gate's false negatives measurable. But a
    priced row is not a recommendation, and two buys on one prop are never one
    opinion: both sides of a run line are mutually exclusive and lose the vig
    with certainty, while a hit total bought at o0.5 *and* o1.5 stakes a single
    view twice and loses both legs together on an 0-for-4.

    The losing sides stay in the ledger as Passes rather than being dropped, so
    they are still graded and this rule's own false negatives are visible in the
    ``pass_gate`` breakdown. Ranking is by edge, matching :mod:`market.tiers`,
    which ranks on departure from the no-vig price rather than on EV.

    Rows with no price are left alone: the comeback flags carry a tier without
    being bets, and both teams in a game can honestly be resilient.
    """
    best: dict[tuple[int, str, int | None], Recommendation] = {}
    for r in recs:
        if not _is_buy(r):
            continue
        cur = best.get(r.bet_group)
        if cur is None or _buy_rank(r) > _buy_rank(cur):
            best[r.bet_group] = r
    for r in recs:
        if not _is_buy(r) or best.get(r.bet_group) is r:
            continue
        kept = best[r.bet_group]
        r.tier = Tier.PASS
        r.pass_gate = ONE_BUY_GATE
        r.reasons.append(
            f"one buy per prop: {kept.selection} is the better side of this market"
        )
    return recs


def _is_buy(r: Recommendation) -> bool:
    return r.tier is not Tier.PASS and r.market_american is not None


def _buy_rank(r: Recommendation) -> tuple[int, float, float]:
    return (
        1 if r.tier is Tier.STRONG else 0,
        r.edge if r.edge is not None else float("-inf"),
        r.ev if r.ev is not None else float("-inf"),
    )


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
