"""Grade and close out CBB ledger rows, then render the six views (§4).

* close: the last ``close`` shard for the event supplies ``close_consensus`` (devigged
  fair prob on the bet side) and ``close_exec_american`` (same book, same line);
  ``clv_consensus`` = close fair − fair at buy, in probability points × 100 (cents of
  no-vig probability); ``clv_exec`` = implied(close exec) − implied(bought) × 100.
* outcome: ESPN finals matched on event id → tip + team names from the shard.

Grading writes a correction row in a new pass (``supersedes`` = the original
``row_hash``, still ``bet_of_record``) so ``ledger.record`` returns the graded row;
the original is never rewritten.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from cbb_engine.config import SPORT
from cbb_engine.data.espn import Game
from cbb_engine.market import consensus
from engine_common.odds import american_to_prob
from engine_runtime import ledger, quotes, views


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def match_final(row: ledger.Row, games: Iterable[Game], home: str, away: str) -> tuple[int, int] | None:
    for g in games:
        if g.final and _norm(g.home) == _norm(home) and _norm(g.away) == _norm(away):
            a, h = g.final
            return a, h
    return None


def outcome_total(row: ledger.Row, final: tuple[int, int]) -> str:
    if row.line is None:
        return ""
    total = sum(final)
    if total == row.line:
        return "push"
    won = total > row.line if row.side == "over" else total < row.line
    return "win" if won else "loss"


def pnl(row: ledger.Row, outcome: str) -> float | None:
    if row.american is None or not outcome:
        return None
    if outcome == "push":
        return 0.0
    if outcome == "loss":
        return -1.0
    return row.american / 100 if row.american > 0 else 100 / -row.american


def grade(
    root: Path, day: str, games: list[Game], teams: dict[str, tuple[str, str]], *, pass_id: str
) -> list[ledger.Row]:
    """Settle every ungraded bet-of-record row for the day; returns the audit rows written."""
    current = ledger.record(root, SPORT, day)
    closes = [q for q in quotes.read_day(root, SPORT, day) if q.pass_kind == "close"]
    anchors = consensus.build(closes)
    by_key = {(a.event_id, a.market, a.line): a for a in anchors}
    out: list[ledger.Row] = []
    for r in current.values():
        if r.outcome or (r.close_status and not games):
            continue
        g = replace(r, pass_id=pass_id, pass_kind="board", supersedes=r.row_hash, prev_hash="", row_hash="")
        a = by_key.get((r.event_id, r.market, r.line))
        if a is not None and r.p_fair is not None and r.american is not None:
            cf = a.p_fair.get(r.side)
            ex = next((e for e in a.exec_quotes if e.book == r.book and e.side == r.side), None)
            g.close_status = "pre_tip"
            g.close_consensus = cf
            g.close_consensus_shard = a.shard
            g.clv_consensus = round((cf - r.p_fair) * 100, 3) if cf is not None else None
            if ex is not None:
                g.close_exec_american = ex.american
                g.close_exec_shard = a.shard
                g.clv_exec = round((american_to_prob(ex.american) - american_to_prob(r.american)) * 100, 3)
        else:
            g.close_status = "pulled"
        home_away = teams.get(r.event_id)
        if home_away and r.market.endswith("total"):
            final = match_final(r, games, *home_away)
            if final:
                g.outcome = outcome_total(r, final)
                g.pnl = pnl(r, g.outcome)
        out.append(g)
    if out:
        ledger.write_once(root, SPORT, day, pass_id, out)
    return out


def report(rows: list[ledger.Row], rungs: list[str]) -> dict[str, object]:
    """Page one first: CLV, shape, rung, gate, prior; W-L/ROI underneath."""
    graded = [r for r in rows if r.outcome and r.is_buy and r.status == "live"]
    units = sum(r.pnl or 0.0 for r in graded)
    return {
        "v_clv": views.v_clv(rows),
        "v_shape": views.v_shape(rows),
        "v_rung": views.v_rung(rows, rungs),
        "v_gate": views.v_gate(rows),
        "v_prior": views.v_prior(rows),
        "record": {
            "n": len(graded),
            "w": sum(1 for r in graded if r.outcome == "win"),
            "l": sum(1 for r in graded if r.outcome == "loss"),
            "p": sum(1 for r in graded if r.outcome == "push"),
            "units": round(units, 3),
            "roi": round(units / len(graded), 4) if graded else None,
        },
    }


def render(rep: dict[str, object]) -> str:
    return json.dumps(rep, indent=2, default=str)


__all__ = ["grade", "match_final", "outcome_total", "pnl", "render", "report"]
