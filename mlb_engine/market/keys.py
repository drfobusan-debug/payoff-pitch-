"""Canonical ``selection`` strings for the ``(matchup, market, selection)`` quote
key. Shared by the pipeline (which emits recommendations) and the market-data
clients (which price them) so quotes line up with the recommendations exactly.
"""

from __future__ import annotations


def _pt(point: float) -> str:
    return f"{point:+.1f}"


def game_ml(abbrev: str) -> str:
    return f"{abbrev} ML"


def game_rl(abbrev: str, point: float) -> str:
    return f"{abbrev} {_pt(point)}"


def game_total(over: bool, line: float) -> str:
    return f"{'Over' if over else 'Under'} {line}"


def f5_ml(abbrev: str) -> str:
    return f"{abbrev} F5 ML"


def f5_total(over: bool, line: float) -> str:
    return f"F5 {'Over' if over else 'Under'} {line}"


def f5_rl(abbrev: str, point: float) -> str:
    return f"{abbrev} F5 {_pt(point)}"


def batter_prop(name: str, stat: str, line: float) -> str:
    return f"{name} {stat} o{line}"


def pitcher_prop(name: str, label: str, line: float) -> str:
    return f"{name} {label} o{line}"


# Engine pitcher stat symbol -> label used in the selection string.
PITCHER_LABEL = {"K": "Ks", "outs": "Outs", "H": "Hits", "BB": "Walks", "ER": "ER"}
