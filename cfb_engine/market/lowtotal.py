"""Under buys on a low total: a candidate screen, graded but never applied.

Through 2026-10-07 the engine's Under buys went 3-11 on a total below 50 against
13-10 at 50 or more (Fisher p=0.048, one of about ten cuts tried), and every total
the model leaned Under went 23-36 below 50 against 55-47 above. FBS weeks 1-6 of
2014-25 lean the same way: Unders landed 46-48% on totals of 48 or less and 53-54%
above 56 (n=3,319). The screen accrues on probation, where the three tests rather
than that story decide whether it ever refuses a bet.
"""

from __future__ import annotations

RULE_NAME = "totals_refuse_under_below_50"
LABEL = "Low-total Under (graded only, not a bet)"
LINE_FLOOR = 50.0


def refuses(market: str, selection: str, line: float | None) -> bool:
    """True on an Under whose total sits below :data:`LINE_FLOOR`."""
    return (
        market == "game_total"
        and selection.startswith("Under")
        and line is not None
        and line < LINE_FLOOR
    )
