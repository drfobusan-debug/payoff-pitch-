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
