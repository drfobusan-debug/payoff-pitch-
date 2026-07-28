"""Derive market probabilities from simulation output.

Turns the Monte Carlo :class:`GameSimResult` and the Markov F5 result into
probabilities for every market the engine prices: game ML/total/run-line,
first-5 ML/total/run-line, and batter/pitcher props.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlb_engine.models.markov_f5 import F5Result
from mlb_engine.models.montecarlo import GameSimResult


def p_over(arr: np.ndarray, line: float) -> float:
    return float((arr > line).mean())


def p_under(arr: np.ndarray, line: float) -> float:
    return float((arr < line).mean())


@dataclass
class MarketProb:
    market: str
    selection: str
    prob: float
    line: float | None = None


def game_markets(res: GameSimResult, home_abbr: str, away_abbr: str) -> list[MarketProb]:
    out: list[MarketProb] = []
    h, a = res.home_runs_full, res.away_runs_full
    total = h + a
    margin = h - a  # home perspective

    p_home = float((margin > 0).mean())
    p_away = float((margin < 0).mean())
    out.append(MarketProb("game_ml", f"{home_abbr} ML", p_home))
    out.append(MarketProb("game_ml", f"{away_abbr} ML", p_away))

    # run line -1.5 / +1.5
    out.append(MarketProb("game_rl", f"{home_abbr} -1.5", float((margin > 1.5).mean()), -1.5))
    out.append(MarketProb("game_rl", f"{away_abbr} +1.5", float((margin < 1.5).mean()), 1.5))
    out.append(MarketProb("game_rl", f"{away_abbr} -1.5", float((-margin > 1.5).mean()), -1.5))
    out.append(MarketProb("game_rl", f"{home_abbr} +1.5", float((-margin < 1.5).mean()), 1.5))

    # common totals lines
    for line in (7.5, 8.5, 9.5, 10.5):
        out.append(MarketProb("game_total", f"Over {line}", p_over(total, line), line))
        out.append(MarketProb("game_total", f"Under {line}", p_under(total, line), line))
    return out


def f5_markets(f5: F5Result, home_abbr: str, away_abbr: str) -> list[MarketProb]:
    out: list[MarketProb] = []
    out.append(MarketProb("f5_ml", f"{home_abbr} F5 ML", f5.p_home_ml))
    out.append(MarketProb("f5_ml", f"{away_abbr} F5 ML", f5.p_away_ml))
    out.append(MarketProb("f5_ml", "F5 Tie", f5.p_tie))
    for line in (4.5, 5.5):
        po = f5.p_total_over(line)
        out.append(MarketProb("f5_total", f"F5 Over {line}", po, line))
        out.append(MarketProb("f5_total", f"F5 Under {line}", 1.0 - po, line))
    out.append(MarketProb("f5_rl", f"{home_abbr} F5 -0.5", f5.p_home_cover(0.5), -0.5))
    out.append(MarketProb("f5_rl", f"{away_abbr} F5 +0.5", 1.0 - f5.p_home_cover(0.5), 0.5))
    return out


BATTER_PROP_LINES = {
    "H": [0.5, 1.5],
    "1B": [0.5],
    "2B": [0.5],
    "HR": [0.5],
    "R": [0.5],
    "RBI": [0.5],
}


def batter_markets(
    res: GameSimResult,
    team: str,
    slot: int,
    player_name: str,
    rbi_factor: float = 1.0,
    tb_factor: float = 1.0,
) -> list[MarketProb]:
    out: list[MarketProb] = []
    bat = res.bat[team]
    for stat, lines in BATTER_PROP_LINES.items():
        arr = bat[stat][:, slot].astype(float)
        if stat == "RBI":
            arr = arr * rbi_factor
        for line in lines:
            out.append(
                MarketProb(
                    f"batter_{stat.lower()}",
                    f"{player_name} {stat} o{line}",
                    p_over(arr, line),
                    line,
                )
            )
    # combined hits+runs+RBI
    hrr = (bat["H"][:, slot] + bat["R"][:, slot] + bat["RBI"][:, slot]).astype(float)
    for line in (1.5, 2.5):
        out.append(MarketProb("batter_hrr", f"{player_name} H+R+RBI o{line}", p_over(hrr, line), line))
    # total bases = 1B + 2*2B + 3*3B + 4*HR
    tb = (
        bat["1B"][:, slot] + 2 * bat["2B"][:, slot] + 3 * bat["3B"][:, slot] + 4 * bat["HR"][:, slot]
    ).astype(float)
    tb = tb * tb_factor
    for line in (1.5, 2.5, 3.5):
        out.append(MarketProb("batter_tb", f"{player_name} TB o{line}", p_over(tb, line), line))
    return out


PITCHER_PROP_LINES = {
    "K": [4.5, 5.5, 6.5],
    "outs": [15.5, 17.5],
    "H": [4.5, 5.5],
    "BB": [1.5, 2.5],
    "ER": [2.5, 3.5],
}


def pitcher_markets(res: GameSimResult, team: str, pitcher_name: str) -> list[MarketProb]:
    out: list[MarketProb] = []
    pit = res.pit[team]
    label = {"K": "Ks", "outs": "Outs", "H": "Hits", "BB": "Walks", "ER": "ER"}
    for stat, lines in PITCHER_PROP_LINES.items():
        arr = pit[stat].astype(float)
        for line in lines:
            out.append(
                MarketProb(
                    f"pitcher_{stat.lower()}",
                    f"{pitcher_name} {label[stat]} o{line}",
                    p_over(arr, line),
                    line,
                )
            )
    return out
