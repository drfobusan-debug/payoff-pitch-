"""§4.e pre-launch pilot: the dense 2024-25 slates set the MDEs in preregistered.yaml.

Everything here is measured on bought tape — per-bet consensus CLV of every execution quote
on the 10-minute featured grid against that game's close, stale-line frequency and
persistence to the next featured pass, the cut-10 price delta against a 15-minute subset,
and the per-game log score under the fitted width. Numbers are written to the yaml by
`write_prereg`, never typed in.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import date as Date
from datetime import timedelta
from pathlib import Path
from statistics import NormalDist, mean, pstdev

from cbb_engine import study
from cbb_engine.history import HIST_PREFIX, _parse
from cbb_engine.market import consensus
from engine_runtime.quotes import Quote

Z_CI = 1.96          # two-sided 95% CI used by the futility rule
Z_POWER = 2.8        # 80% power / 5% two-sided, as in study.featured_value
MIN_WORTHWHILE_CENTS = 0.5
WEEK5_DAYS = 35
LIVE_MARKET = "game_total"


def _implied(american: float) -> float:
    return consensus.implied(american)


def cents(row: dict[str, object]) -> float:
    v = row["clv_cents"]
    assert isinstance(v, (int, float))
    return float(v)


def close_mus(shards: Sequence[Quote], sd: float) -> dict[str, float]:
    """Fair close total per event: the paired close anchor nearest the consensus line, shifted by its
    devigged Over lean under N(·, sd)."""
    by_event: dict[str, list[Quote]] = defaultdict(list)
    for q in shards:
        if q.pass_kind != "close" or not q.pass_id.startswith(HIST_PREFIX) or q.market != LIVE_MARKET:
            continue
        tip, cap = _parse(q.tip_utc), _parse(q.captured_at)
        if tip is not None and cap is not None and cap < tip:
            by_event[q.event_id].append(q)
    mus: dict[str, float] = {}
    for eid, qs in by_event.items():
        last = max(q.pass_id for q in qs)
        anchors = consensus.build([q for q in qs if q.pass_id == last])
        line = consensus.consensus_line(anchors, eid, LIVE_MARKET)
        if line is None:
            continue
        priced = [a for a in anchors if a.line is not None and a.paired_books and "over" in a.p_fair]
        if not priced:
            continue
        a = min(priced, key=lambda a: (abs((a.line or 0.0) - line), -a.paired_books))
        p_over = min(max(a.p_fair["over"], 1e-6), 1 - 1e-6)
        mus[eid] = (a.line or 0.0) + sd * NormalDist().inv_cdf(p_over)
    return mus


def p_close(mu: float, sd: float, side: str, line: float) -> float:
    """P(side wins | no push) at `line` under N(mu, sd) on integer totals."""
    nd = NormalDist(mu, sd)
    if line == int(line):
        over, under = 1 - nd.cdf(line + 0.5), nd.cdf(line - 0.5)
    else:
        over, under = 1 - nd.cdf(line), nd.cdf(line)
    return (over if side == "over" else under) / max(over + under, 1e-12)


def clv(mu: float, sd: float, side: str, line: float, american: float) -> float:
    return (p_close(mu, sd, side, line) - _implied(american)) * 100


def clv_rows(shards: Sequence[Quote], sd: float) -> list[dict[str, object]]:
    """One row per execution quote on a hist featured pass: its consensus CLV vs the game's close.

    CLV is `(p_close(side, bet line) - implied(bet price)) * 100` cents, where `p_close` is the
    fair close total (`close_mus`) read at the bet's own line through the fitted width — so a
    line move counts, not only a price move. Games with no pre-tip close are skipped. `stale` is
    the §3.7 flag against the previous featured pass; `held_next` says the same (line, price) was
    still posted by that book on the next featured pass.
    """
    feat = [q for q in shards if q.pass_kind == "featured" and q.pass_id.startswith(HIST_PREFIX)]
    mus = close_mus(shards, sd)
    if not feat or not mus:
        return []
    by_pass: dict[str, list[Quote]] = defaultdict(list)
    for q in feat:
        if q.market == LIVE_MARKET:
            by_pass[q.pass_id].append(q)
    order = sorted(by_pass, key=lambda pid: by_pass[pid][0].captured_at)
    posted = [
        {(q.event_id, q.line, q.book, q.side, q.american) for q in by_pass[pid]} for pid in order
    ]
    out: list[dict[str, object]] = []
    prev: list[consensus.Anchor] = []
    for i, pid in enumerate(order):
        anchors = consensus.build(by_pass[pid], shard=pid)
        consensus.flag_stale(anchors, prev)
        nxt = posted[i + 1] if i + 1 < len(posted) else None
        for a in anchors:
            mu = mus.get(a.event_id)
            if mu is None or a.line is None:
                continue
            for e in a.exec_quotes:
                out.append({
                    "game_date": a.game_date, "pass_id": pid, "event_id": a.event_id, "book": e.book,
                    "side": e.side, "line": a.line, "american": e.american,
                    "clv_cents": clv(mu, sd, e.side, a.line, e.american), "stale": e.stale,
                    "held_next": None if nxt is None else (a.event_id, a.line, e.book, e.side, e.american) in nxt,
                })
        prev = anchors
    return out


def featured_deltas(shards: Sequence[Quote], sd: float) -> list[float]:
    """Cut 10 on the tape: CLV cents gained by buying an execution quote on a 10-minute featured
    pass instead of that book's quote (any line) at the last pass a 15-minute clock kept."""
    feat = [q for q in shards if q.pass_kind == "featured" and q.pass_id.startswith(HIST_PREFIX) and q.market == LIVE_MARKET]
    mus = close_mus(shards, sd)
    keep = study.subset_15(feat)
    by_pass: dict[str, list[Quote]] = defaultdict(list)
    for q in feat:
        by_pass[q.pass_id].append(q)
    times = sorted((qs[0].captured_at, pid) for pid, qs in by_pass.items())
    kept: dict[str, dict[tuple[str, str, str], tuple[float, float]]] = {
        pid: {(q.event_id, q.book, q.side): (q.line, q.american) for q in by_pass[pid] if q.line is not None}
        for pid in keep
    }
    deltas: list[float] = []
    for ts, pid in times:
        if pid in keep:
            continue
        prior = [p for t, p in times if t <= ts and p in keep]
        if not prior:
            continue
        ref = kept[prior[-1]]
        for q in by_pass[pid]:
            mu = mus.get(q.event_id)
            if q.book not in consensus.EXECUTION_BOOKS or mu is None or q.line is None:
                continue
            cf = ref.get((q.event_id, q.book, q.side))
            if cf is not None:
                deltas.append(clv(mu, sd, q.side, q.line, q.american) - clv(mu, sd, q.side, cf[0], cf[1]))
    return deltas


def log_scores(rows: Iterable[dict[str, str]], sd: float, *, start: str, days: int = WEEK5_DAYS) -> list[float]:
    """Per-game log score of N(close, sd) at the final for games tipping in the first `days`."""
    end = Date.fromisoformat(start) + timedelta(days=days)
    out: list[float] = []
    for r in rows:
        try:
            c, f = float(r["close_total"]), float(r["final_total"])
            d = Date.fromisoformat(r["game_date"])
        except (KeyError, ValueError):
            continue
        if d < Date.fromisoformat(start) or d >= end:
            continue
        out.append(-0.5 * ((f - c) / sd) ** 2 - math.log(sd * math.sqrt(2 * math.pi)))
    return out


def _design_effect(clv: Sequence[tuple[str, float]]) -> float | None:
    """1 + (m̄-1)·ICC from a one-way slate-day ANOVA; None with fewer than 3 days."""
    by_day: dict[str, list[float]] = defaultdict(list)
    for day, v in clv:
        by_day[day].append(v)
    k = len(by_day)
    n = len(clv)
    if k < 3 or n <= k:
        return None
    grand = mean(v for _, v in clv)
    msb = sum(len(xs) * (mean(xs) - grand) ** 2 for xs in by_day.values()) / (k - 1)
    msw = sum((v - mean(by_day[d])) ** 2 for d, v in clv) / (n - k)
    m0 = (n - sum(len(xs) ** 2 for xs in by_day.values()) / n) / (k - 1)
    icc = max(0.0, (msb - msw) / (msb + (m0 - 1) * msw)) if msb + (m0 - 1) * msw > 0 else 0.0
    return round(1 + (n / k - 1) * icc, 3)


def summarize(
    rows: Sequence[dict[str, object]],
    deltas: Sequence[float],
    scores: Sequence[float],
    *,
    cluster_inflation: float,
) -> dict[str, object]:
    """The pilot numbers and the four preregistered values they imply.

    futility.min_n: buys at which a true-zero channel's 95% upper bound sits below +0.5¢
      (n = (z·c·sd / 0.5)², c the slate-day cluster inflation).
    H4.min_n: 80%-power n for a 0.5¢ featured-pass price delta.
    prior_decision.mde: 80%-power detectable difference in mean per-game log score over the
      week-5 sample, using the score SD itself (conservative: variant differences are correlated).
    """
    clv = [cents(r) for r in rows]
    n = len(clv)
    sd = pstdev(clv) if n > 1 else None
    stale = [r for r in rows if r["stale"]]
    held = [r for r in stale if r["held_next"] is not None]
    sd_delta = pstdev(deltas) if len(deltas) > 1 else None
    sd_score = pstdev(scores) if len(scores) > 1 else None
    deff = _design_effect([(str(r["game_date"]), cents(r)) for r in rows])
    out: dict[str, object] = {
        "n_quotes": n,
        "days": sorted({str(r["game_date"]) for r in rows}),
        "clv_mean_cents": round(mean(clv), 3) if clv else None,
        "per_bet_clv_sd_cents": round(sd, 3) if sd else None,
        "cluster_inflation_empirical": deff,
        "stale_share": round(len(stale) / n, 4) if n else None,
        "stale_clv_mean_cents": round(mean(cents(r) for r in stale), 3) if stale else None,
        "stale_persist_next_featured": round(sum(1 for r in held if r["held_next"]) / len(held), 4) if held else None,
        "featured_delta_n": len(deltas),
        "featured_delta_mean_cents": round(mean(deltas), 3) if deltas else None,
        "featured_delta_sd_cents": round(sd_delta, 3) if sd_delta else None,
        "week5_games": len(scores),
        "log_score_sd": round(sd_score, 4) if sd_score else None,
    }
    prereg: dict[str, object] = {}
    if sd:
        prereg["mde.per_bet_clv_sd_cents"] = round(sd, 2)
        prereg["futility.min_n"] = math.ceil((Z_CI * cluster_inflation * sd / MIN_WORTHWHILE_CENTS) ** 2)
    if sd_delta:
        prereg["H4.min_n"] = math.ceil((Z_POWER * sd_delta / MIN_WORTHWHILE_CENTS) ** 2)
    if sd_score and scores:
        prereg["prior_decision.mde"] = round(Z_POWER * sd_score / math.sqrt(len(scores)), 4)
    out["prereg"] = prereg
    return out


_SLOTS = {
    "H4.min_n": r"(\{id: H4,[^\n]*min_n: )TODO-PILOT",
    "prior_decision.mde": r"(prior_decision:\n(?:[^\n]*\n)*?  mde: )TODO-PILOT",
    "futility.min_n": r"(futility:\n(?:[^\n]*\n)*?  min_n: )TODO-PILOT",
    "mde.per_bet_clv_sd_cents": r"(per_bet_clv_sd_cents: )TODO-PILOT",
}


def write_prereg(path: Path, prereg: dict[str, object], *, note: str) -> list[str]:
    """Fill the TODO-PILOT slots the pilot produced; returns the keys written. Slots already
    holding a number are left alone (pre-registration is not re-fit)."""
    text = path.read_text()
    written: list[str] = []
    for key, pat in _SLOTS.items():
        if key not in prereg:
            continue
        new, k = re.subn(pat, rf"\g<1>{prereg[key]}", text, count=1)
        if k:
            text, written = new, written + [key]
    if written:
        text = text.rstrip("\n") + f"\n# pilot ({note}): " + ", ".join(f"{k}={prereg[k]}" for k in written) + "\n"
        path.write_text(text)
    return written


__all__ = ["cents", "close_mus", "clv", "clv_rows", "p_close", "featured_deltas", "log_scores", "summarize", "write_prereg"]
