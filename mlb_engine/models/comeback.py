"""Comeback-resilience flag.

Generic live win-probability models price a trailing team purely on the base/out/
inning/deficit state and miss *why* a specific team is primed to erase a deficit.
This layer flags teams whose underlying signals carry documented PPV for
mid-to-late comebacks:

- **xwOBA differential** (data-grounded): a lineup out-hitting its opponent by
  contact quality is "due" -- hard-hit metrics normalize over nine innings, so
  unlucky lineouts turn into runs.
- **On-base pressure** (data-grounded): a high-OBP lineup keeps innings alive,
  the precondition for multi-run rallies.
- **TTTO exposure** (data-grounded proxy): a long-leash opponent starter left in
  to face the order a third time is in the high-penalty window.
- **Opponent bullpen fatigue** (hook): overworked high-leverage arms (fatigue
  score > 60) inflate blown-lead risk. Neutral until a fatigue feed is supplied.

Emitted as an informational per-team flag with a bounded 0-1 resilience score and
reasons -- not a betting EV line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_BASE = 0.40
XWOBA_EDGE = 0.020
OBP_BASELINE = 0.320
TTTO_LONG_LEASH = 26  # starter batters-faced cap above this = 3rd-time exposure
FATIGUE_HIGH = 60.0

RESILIENT_SCORE = 0.60
MODERATE_SCORE = 0.50


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class ComebackSignal:
    xwoba_diff: float | None = None  # team lineup xwOBA - opponent lineup xwOBA
    team_obp: float | None = None
    opp_starter_bf_cap: int = 24
    opp_bullpen_fatigue: float | None = None  # 0-100 (hook)


@dataclass
class ComebackAssessment:
    score: float
    resilient: bool
    reasons: list[str] = field(default_factory=list)


def evaluate(sig: ComebackSignal) -> ComebackAssessment:
    score = _BASE
    reasons: list[str] = []

    if sig.xwoba_diff is not None:
        contrib = _clip(sig.xwoba_diff * 4.0, -0.25, 0.25)
        score += contrib
        if sig.xwoba_diff >= XWOBA_EDGE:
            reasons.append(f"xwOBA edge {sig.xwoba_diff:+.3f}: hard contact due to normalize")

    if sig.team_obp is not None:
        score += _clip((sig.team_obp - OBP_BASELINE) * 1.5, -0.10, 0.12)
        if sig.team_obp >= 0.335:
            reasons.append(f"high on-base pressure (OBP {sig.team_obp:.3f})")

    if sig.opp_starter_bf_cap >= TTTO_LONG_LEASH:
        score += 0.10
        reasons.append("long-leash opp starter -> 3rd-time-through window")

    if sig.opp_bullpen_fatigue is not None and sig.opp_bullpen_fatigue >= FATIGUE_HIGH:
        score += _clip((sig.opp_bullpen_fatigue - FATIGUE_HIGH) / 40.0 * 0.15, 0.0, 0.15)
        reasons.append(f"opp bullpen fatigued ({sig.opp_bullpen_fatigue:.0f})")

    score = _clip(score, 0.0, 1.0)
    return ComebackAssessment(score=score, resilient=score >= RESILIENT_SCORE, reasons=reasons)
