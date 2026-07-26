"""Run-line PPV confidence layer.

The Monte Carlo already yields the full run-differential distribution, so the raw
cover probabilities for -1.5 / +1.5 (and alt lines) come straight from it. This
module adds the market-inefficiency signals that carry documented PPV for *beating
the run-line price*, used only to raise/lower an already-priced run-line
recommendation's tier -- never to fabricate probability:

- **xwOBA differential** (data-grounded, the ROI-proven signal): hard-contact edge
  correlates with multi-run margins, so it confirms a favorite -1.5 cover and
  flags false positives when it contradicts the pick; a dog that owns (or nearly
  matches) the xwOBA edge is an undervalued +1.5.
- **Hot-hand fade** (hook): a team that is cold by recent results but sound by
  xwOBA has an inflated line -> value on its +1.5. Neutral until a recent-form
  feed is supplied.
- **Bullpen volatility** (hook): a favorite with a depleted pen is prone to
  blowing a late lead -> value on the opponent's +1.5. Neutral until an
  availability feed is supplied.

Separately, :func:`runline_veto` applies *NPV gates*: conditions under which a
cover is so improbable that the selection is dropped outright rather than
downgraded. Gates never create a bet; they only remove one, so they trade bet
volume for realized NPV. Each is individually switchable via :class:`RunLineGates`
and tags itself by name so the audit ledger can grade the counterfactual (a gate
earns its keep only when the selections it removed lost at a materially higher
rate than the ones it kept).
"""

from __future__ import annotations

from dataclasses import dataclass

from mlb_engine.config import RunLineGates

XWOBA_DIFF_STRONG = 0.030  # meaningful team hard-contact edge
XWOBA_DIFF_CLOSE = 0.010  # near-even by underlying contact quality


@dataclass(frozen=True)
class RunLineSignal:
    xwoba_diff: float | None = None  # home team mean xwOBA - away team mean xwOBA
    cold_sound_side: str | None = None  # 'home'/'away': cold by results, sound by xwOBA
    fav_pen_depleted_side: str | None = None  # 'home'/'away': favorite with a thin pen
    sharp_money_side: str | None = None  # 'home'/'away': VSIN spread handle% >> bets%

    # --- NPV gate inputs (None whenever the underlying sample is too thin) ---
    fav_side: str | None = None  # 'home'/'away': the side the model favors
    fav_iso: float | None = None  # favored lineup's ISO (min of realized / expected)
    fav_opp_sp_gb_pct: float | None = None  # GB% of the starter the favorite faces
    dog_sp_whip_l3: float | None = None  # underdog starter WHIP, last 3 starts
    dog_sp_hard_hit_l3: float | None = None  # underdog starter hard-hit% allowed, last 3
    dog_pen_xwoba: float | None = None  # underdog bullpen xwOBA allowed
    dog_pen_k_pct: float | None = None  # underdog bullpen K rate
    model_total: float | None = None  # simulated mean total runs


@dataclass(frozen=True)
class RunLineVeto:
    """Outcome of the NPV gates for one run-line selection."""

    gate: str | None = None  # gate name, None when nothing fired
    detail: str | None = None  # human-readable trigger, surfaced in rec.reasons

    @property
    def triggered(self) -> bool:
        return self.gate is not None

    def reason(self) -> str:
        return f"NPV veto [{self.gate}]: {self.detail}" if self.triggered else "NPV gates clear"


def _is_dog_side(team_side: str | None, fav_side: str | None) -> bool:
    return fav_side in ("home", "away") and team_side != fav_side


def runline_veto(
    team_side: str | None,
    line: float | None,
    signal: RunLineSignal,
    gates: RunLineGates,
) -> RunLineVeto:
    """Return the first NPV gate that disqualifies this run-line selection.

    A gate fires only when every input it needs is present, so a missing feed
    leaves the selection untouched rather than silently killing bet volume.
    """
    if team_side not in ("home", "away") or line not in (-1.5, 1.5):
        return RunLineVeto()

    backing_fav = line == -1.5 and team_side == signal.fav_side
    backing_dog = line == 1.5 and _is_dog_side(team_side, signal.fav_side)

    if backing_fav and gates.iso_gb:
        iso, gb = signal.fav_iso, signal.fav_opp_sp_gb_pct
        if iso is not None and gb is not None and iso < gates.iso_max and gb > gates.gb_min:
            return RunLineVeto(
                "fav_iso_x_gb",
                f"favorite ISO {iso:.3f} < {gates.iso_max:.3f} vs GB% {gb:.3f} > "
                f"{gates.gb_min:.2f}: no multi-run-homer path to a 2+ margin",
            )

    if backing_fav and gates.low_total:
        tot = signal.model_total
        if tot is not None and tot <= gates.low_total_max:
            return RunLineVeto(
                "low_total",
                f"model total {tot:.1f} <= {gates.low_total_max:.1f}: "
                "low-scoring games trend to 1-run margins",
            )

    if backing_dog and gates.dog_sp:
        whip, hh = signal.dog_sp_whip_l3, signal.dog_sp_hard_hit_l3
        if (
            whip is not None
            and hh is not None
            and whip > gates.dog_sp_whip_max
            and hh > gates.dog_sp_hard_hit_max
        ):
            return RunLineVeto(
                "dog_sp_blowout",
                f"underdog starter WHIP {whip:.2f} > {gates.dog_sp_whip_max:.2f} and "
                f"hard-hit {hh:.3f} > {gates.dog_sp_hard_hit_max:.2f} over 3 starts: "
                "traffic plus hard contact is a multi-run-inning script",
            )

    if backing_dog and gates.dog_pen:
        xw, k = signal.dog_pen_xwoba, signal.dog_pen_k_pct
        if (
            xw is not None
            and k is not None
            and xw > gates.dog_pen_xwoba_max
            and k < gates.dog_pen_k_min
        ):
            return RunLineVeto(
                "dog_pen_leak",
                f"underdog bullpen xwOBA {xw:.3f} > {gates.dog_pen_xwoba_max:.3f} and "
                f"K% {k:.3f} < {gates.dog_pen_k_min:.2f}: cannot strand inherited runners",
            )

    return RunLineVeto()


def runline_adjustment(
    team_side: str | None, line: float | None, signal: RunLineSignal
) -> tuple[int, list[str]]:
    """Return (tier_steps, reasons) for a run-line selection."""
    if team_side not in ("home", "away") or line not in (-1.5, 1.5):
        return 0, []

    steps = 0
    reasons: list[str] = []
    diff = signal.xwoba_diff
    if diff is not None:
        mag = abs(diff)
        side_has_edge = (team_side == "home" and diff > 0) or (
            team_side == "away" and diff < 0
        )
        if line == -1.5:  # favorite cover
            if side_has_edge and mag >= XWOBA_DIFF_STRONG:
                steps += 1
                reasons.append(f"xwOBA diff {diff:+.3f} supports -1.5")
            elif not side_has_edge and mag >= XWOBA_DIFF_STRONG:
                steps -= 1
                reasons.append(f"xwOBA diff {diff:+.3f} contradicts -1.5")
        else:  # +1.5 dog
            if side_has_edge:
                steps += 1
                reasons.append(f"xwOBA diff {diff:+.3f}: dog owns edge -> +1.5 value")
            elif mag <= XWOBA_DIFF_CLOSE:
                steps += 1
                reasons.append(f"xwOBA diff {diff:+.3f}: near-even -> +1.5 value")
            elif mag >= XWOBA_DIFF_STRONG:
                steps -= 1
                reasons.append(f"xwOBA diff {diff:+.3f}: dog outclassed -> +1.5 risk")

    if line == 1.5 and signal.cold_sound_side == team_side:
        steps += 1
        reasons.append("cold-but-sound: fade line inflation -> +1.5")

    if line == 1.5 and signal.fav_pen_depleted_side is not None:
        opp = "away" if signal.fav_pen_depleted_side == "home" else "home"
        if team_side == opp:
            steps += 1
            reasons.append("favorite bullpen depleted -> back opponent +1.5")

    if signal.sharp_money_side == team_side:
        steps += 1
        reasons.append("VSIN sharp money (handle% > bets%) backs this side")

    return steps, reasons
