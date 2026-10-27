"""Grade and close out CBB ledger rows, then render the six views (§4).

* close: the event's last close pass taken before tip and within 45 min of it supplies
  ``close_consensus`` (devigged fair prob of the bet side at the bet's own line — totals
  through the fitted width, spreads/moneylines at the same number) and
  ``close_exec_american`` (same book, same line); ``clv_consensus`` = close fair − fair at
  buy, in probability points × 100; ``clv_exec`` = implied(close exec) − implied(bought) × 100.
  An early or after-tip close is ``early`` / ``post_tip`` and carries no CLV.
* outcome: full-game totals, spreads and moneylines, passes included, on the ESPN final
  matched by ``history.match_final`` (folded team-name tokens) on the shard's teams.

Grading writes a correction row in a new pass (``supersedes`` = the original
``row_hash``, still ``bet_of_record``) so ``ledger.record`` returns the graded row;
the original is never rewritten.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from cbb_engine import history, pilot
from cbb_engine.cadence import parse_utc
from cbb_engine.config import SPORT
from cbb_engine.data.espn import Game
from cbb_engine.market import consensus
from cbb_engine.models import distribution as dist
from engine_common.odds import american_to_prob
from engine_runtime import ledger, quotes, views

CLOSE_MAX_MIN = 45


def _closes(quotes_: Iterable[quotes.Quote]) -> tuple[dict[str, list[quotes.Quote]], dict[str, str]]:
    """Per event, the last close pass taken before tip and within ``CLOSE_MAX_MIN`` of it.

    Events with only an after-tip or an early close get ``post_tip`` / ``early`` instead,
    and no CLV.
    """
    timely: dict[str, list[quotes.Quote]] = defaultdict(list)
    late: set[str] = set()
    early: set[str] = set()
    for q in quotes_:
        if q.pass_kind != "close":
            continue
        tip, cap = parse_utc(q.tip_utc), parse_utc(q.captured_at)
        if tip is None or cap is None:
            continue
        if cap >= tip:
            late.add(q.event_id)
        elif tip - cap > timedelta(minutes=CLOSE_MAX_MIN):
            early.add(q.event_id)
        else:
            timely[q.event_id].append(q)
    out: dict[str, list[quotes.Quote]] = {}
    for eid, qs in timely.items():
        last = max((q.captured_at, q.pass_id) for q in qs)[1]
        out[eid] = [q for q in qs if q.pass_id == last]
    status = {e: "post_tip" for e in late} | {e: "early" for e in early - late}
    return out, status


def outcome(row: ledger.Row, final: tuple[int, int]) -> str:
    """Full-game total, spread or moneyline graded on the (away, home) final; other markets ""."""
    if row.market == "game_total":
        return outcome_total(row, final)
    away, home = final
    mine, theirs = (home, away) if row.side == "home" else (away, home) if row.side == "away" else (None, None)
    if mine is None or theirs is None:
        return ""
    if row.market == "game_ml":
        return "win" if mine > theirs else "loss"
    if row.market == "game_ats" and row.line is not None:
        margin = mine - theirs + row.line
        return "push" if margin == 0 else "win" if margin > 0 else "loss"
    return ""


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


def _close_fair(row: ledger.Row, anchors: list[consensus.Anchor], width: dist.Width | None) -> float | None:
    """Devigged close probability of the row's side at the row's own line.

    Totals read the fair close total through the fitted width, so a line move counts;
    spreads and moneylines (no fitted margin width yet) need a close at the same number.
    """
    if row.market == "game_total" and width is not None and row.line is not None:
        sd = width.sd(row.quality_bucket) or width.sd("all")
        mu = pilot.close_mu(anchors, row.event_id, sd) if sd else None
        if sd and mu is not None:
            return pilot.p_close(mu, sd, row.side, row.line)
    a = next((a for a in anchors if (a.event_id, a.market, a.line) == (row.event_id, row.market, row.line)), None)
    return a.p_fair.get(row.side) if a is not None else None


def grade(
    root: Path,
    day: str,
    games: list[Game],
    teams: dict[str, tuple[str, str]],
    *,
    pass_id: str,
    width: dist.Width | None = None,
) -> list[ledger.Row]:
    """Settle every ungraded bet-of-record row for the day (buys and passes); returns the rows written."""
    current = ledger.record(root, SPORT, day)
    closes, missed = _closes(quotes.read_day(root, SPORT, day))
    anchors = {eid: consensus.build(qs, shard=qs[0].pass_id) for eid, qs in closes.items()}
    out: list[ledger.Row] = []
    for r in current.values():
        if r.outcome or (r.close_status and not games):
            continue
        g = replace(
            r, pass_id=pass_id, pass_kind="board", supersedes=r.row_hash, prev_hash="", row_hash=""
        )
        held = anchors.get(r.event_id)
        if held is not None and r.p_fair is not None and r.american is not None:
            shard = held[0].shard
            cf = _close_fair(r, held, width)
            g.close_status = "pre_tip"
            g.close_consensus, g.close_consensus_shard = cf, shard
            g.clv_consensus = round((cf - r.p_fair) * 100, 3) if cf is not None else None
            a = next((a for a in held if (a.market, a.line) == (r.market, r.line)), None)
            ex = next((e for e in a.exec_quotes if e.book == r.book and e.side == r.side), None) if a else None
            if ex is not None:
                g.close_exec_american, g.close_exec_shard = ex.american, shard
                g.clv_exec = round(
                    (american_to_prob(ex.american) - american_to_prob(r.american)) * 100, 3
                )
        else:
            g.close_status = missed.get(r.event_id, "pulled")
        home_away = teams.get(r.event_id)
        if home_away:
            final = history.match_final(home_away[0], home_away[1], games)
            if final:
                g.outcome = outcome(r, final)
                g.pnl = pnl(r, g.outcome)
        out.append(g)
    if out:
        ledger.write_once(root, SPORT, day, pass_id, out)
    return out


def v_persist(rows: list[ledger.Row], shards: list[quotes.Quote]) -> list[dict[str, object]]:
    """Stale-flagged quotes still on the board at the next featured pass and at the next card, by book x market.

    "Still available" = same (event, market, side, line, book) at the same or better American
    price in the first featured shard captured after the row's shard; "next card" = the same test
    against the next bet-of-record pass on the day.
    """
    by_pass: dict[str, list[quotes.Quote]] = defaultdict(list)
    for q in shards:
        by_pass[q.pass_id].append(q)
    featured = sorted({(q.captured_at, q.pass_id) for q in shards if q.pass_kind == "featured"})
    card_rows: dict[str, list[ledger.Row]] = defaultdict(list)
    for r in rows:
        if r.bet_of_record:
            card_rows[r.pass_id].append(r)
    shard_time = {pid: qs[0].captured_at for pid, qs in by_pass.items()}
    acc: dict[tuple[str, str], list[tuple[bool | None, bool | None]]] = defaultdict(list)
    for r in rows:
        if not r.stale_flag or not r.bet_of_record:
            continue
        t0 = shard_time.get(r.quote_shard, "")
        nxt = next((pid for ts, pid in featured if ts > t0), None)
        still_feat: bool | None = None
        if nxt is not None:
            still_feat = any(_same(q, r) for q in by_pass[nxt])
        later_cards = [pid for pid in card_rows if pid != r.pass_id and pid > r.pass_id]
        still_card: bool | None = None
        if later_cards:
            still_card = any(_same_row(x, r) for x in card_rows[min(later_cards)])
        acc[(r.book, r.market)].append((still_feat, still_card))
    out = []
    for (book, market), xs in sorted(acc.items()):
        f = [a for a, _ in xs if a is not None]
        c = [b for _, b in xs if b is not None]
        out.append(
            {
                "book": book,
                "market": market,
                "n": len(xs),
                "next_featured": round(sum(f) / len(f), 3) if f else None,
                "next_card": round(sum(c) / len(c), 3) if c else None,
            }
        )
    return out


def _same(q: quotes.Quote, r: ledger.Row) -> bool:
    return (
        (q.event_id, q.market, q.side, q.line, q.book)
        == (r.event_id, r.market, r.side, r.line, r.book)
        and r.american is not None
        and q.american >= r.american
    )


def _same_row(x: ledger.Row, r: ledger.Row) -> bool:
    return (
        (x.event_id, x.market, x.side, x.line, x.book)
        == (r.event_id, r.market, r.side, r.line, r.book)
        and r.american is not None
        and x.american is not None
        and x.american >= r.american
    )


def report(
    rows: list[ledger.Row], rungs: list[str], shards: list[quotes.Quote] | None = None
) -> dict[str, object]:
    """Page one first: CLV, shape, rung, gate, prior; W-L/ROI underneath."""
    graded = [r for r in rows if r.outcome and r.is_buy and r.status == "live"]
    units = sum(r.pnl or 0.0 for r in graded)
    return {
        "v_clv": views.v_clv(rows),
        "v_shape": views.v_shape(rows),
        "v_rung": views.v_rung(rows, rungs),
        "v_gate": views.v_gate(rows),
        "v_prior": views.v_prior(rows),
        "v_persist": v_persist(rows, shards or []),
        "v_promotion": views.v_promotion(rows),
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


__all__ = ["CLOSE_MAX_MIN", "grade", "outcome", "outcome_total", "pnl", "render", "report", "v_persist"]
