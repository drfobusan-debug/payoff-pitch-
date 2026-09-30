"""Canonical ``selection`` strings for the ``(matchup, market, selection)`` quote
key. Shared by the pipeline (which emits recommendations) and the odds client
(which prices them) so quotes line up with recommendations exactly.

Markets: ``game_ml`` (moneyline), ``game_ats`` (against-the-spread), and
``game_total`` (over/under).
"""

from __future__ import annotations


def _pt(point: float) -> str:
    return f"{point:+.1f}"


def game_ml(abbrev: str) -> str:
    return f"{abbrev} ML"


def game_ats(abbrev: str, point: float) -> str:
    return f"{abbrev} {_pt(point)}"


def game_total(over: bool, line: float) -> str:
    return f"{'Over' if over else 'Under'} {line}"


def side_of(selection: str) -> str:
    """The handicap-free half of a selection: ``"ALA -7.0"`` -> ``"ALA"``.

    Movement has to be tracked against a key that survives the line moving,
    because the line moving is the thing being tracked. Keying a market snapshot
    on the full selection means a spread that goes -7 to -7.5 files its two ends
    under different names and no comparison is ever made -- which is exactly the
    bug that made closing-line value silently absent for every ATS and totals bet
    whose number moved (see :func:`cfb_engine.audit.clv.compute_clv`).
    """
    head, _, tail = selection.rpartition(" ")
    if not head:
        return selection
    if tail == "ML":
        return head
    try:
        float(tail)
    except ValueError:
        return selection
    return head
