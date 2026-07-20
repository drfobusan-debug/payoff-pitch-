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
"""

from __future__ import annotations

from dataclasses import dataclass

XWOBA_DIFF_STRONG = 0.030  # meaningful team hard-contact edge
XWOBA_DIFF_CLOSE = 0.010  # near-even by underlying contact quality


@dataclass(frozen=True)
class RunLineSignal:
    xwoba_diff: float | None = None  # home team mean xwOBA - away team mean xwOBA
    cold_sound_side: str | None = None  # 'home'/'away': cold by results, sound by xwOBA
    fav_pen_depleted_side: str | None = None  # 'home'/'away': favorite with a thin pen


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

    return steps, reasons
