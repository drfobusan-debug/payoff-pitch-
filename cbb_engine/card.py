"""Price a slate from the latest board/featured shard and write the card pass.

The card pass is the only ``bet_of_record``; every (event, market, side, line,
book) seen on the execution books becomes a row — buy, pass or refusal — with
``status`` from the market map. Two channels on the same anchor:

* market-only: ``p_fair`` vs the execution quote (rows carry ``stale_flag``);
* model (live totals only): ``p_model`` read off the solved v0 distribution,
  or a ``width_unfit`` / ``anchor_miss`` / ``se_cap`` refusal.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cbb_engine.cadence import parse_utc
from cbb_engine.config import SPORT
from cbb_engine.market import consensus
from cbb_engine.market.gates import Thresholds, implied, market_gates, tier
from cbb_engine.markets import BY_KEY, status_of
from cbb_engine.models import distribution as dist
from engine_runtime import ledger, quotes

PREREG = Path(__file__).resolve().parent / "params" / "preregistered.yaml"
MODEL_VERSION = "v0_shape"


def prereg_scalar(key: str, default: float) -> float:
    """Read one ``key: value`` scalar from preregistered.yaml (no YAML dependency in the repo)."""
    m = re.search(rf"^\s*{re.escape(key)}:\s*([-0-9.]+)", PREREG.read_text(), re.M)
    return float(m.group(1)) if m else default


def params_hash(width: dist.Width | None) -> str:
    h = hashlib.sha256(PREREG.read_bytes())
    if width is not None:
        h.update(json.dumps(width.sd_by_bucket, sort_keys=True).encode())
    return h.hexdigest()[:12]


@dataclass
class Priced:
    rows: list[ledger.Row]
    anchors: list[consensus.Anchor]
    shard: str


def latest_pricing_shard(root: Path, day: str) -> dict[str, object] | None:
    man = [m for m in quotes.read_manifest(root, SPORT, day) if m.get("pass_kind") in ("board", "featured")]
    return max(enumerate(man), key=lambda im: (str(im[1]["captured_at"]), im[0]))[1] if man else None


def price(
    root: Path,
    day: str,
    *,
    pass_id: str,
    pass_kind: str,
    code_sha: str = "",
    env_lock_hash: str = "",
    pending_events: set[str] | None = None,
    width: dist.Width | None = None,
    thr: Thresholds | None = None,
    previous: list[consensus.Anchor] | None = None,
    max_live_buys: int = 8,
    max_per_book: int = 6,
    now: datetime | None = None,
) -> Priced | None:
    """Price the latest board/featured shard; a game whose tip has passed is left out."""
    now = now or datetime.now(timezone.utc)
    shard = latest_pricing_shard(root, day)
    if shard is None:
        return None
    sid = str(shard["pass_id"])
    rows = [q for q in quotes.read_day(root, SPORT, day) if q.pass_id == sid]
    anchors = consensus.build(rows, shard=sid)
    if previous:
        consensus.flag_stale(anchors, previous)
    tol = prereg_scalar("anchor_tolerance_pts", 0.1)
    thr = thr or Thresholds()
    pending = pending_events or set()
    out: list[ledger.Row] = []
    bet_of_record = pass_kind != "close"
    for a in anchors:
        tip = parse_utc(a.tip_utc)
        if not a.exec_quotes or tip is None or tip <= now:
            continue
        status = status_of(a.market)
        spread_line = consensus.consensus_line(anchors, a.event_id, "game_ats")
        bucket = dist.quality_bucket(a.paired_books, abs(spread_line) if spread_line is not None else None)
        base_gates = market_gates(paired_books=a.paired_books, one_way=a.one_way, exec_quote_present=True, pending=a.event_id in pending, thr=thr)
        model: dist.TotalDist | None = None
        model_gates: list[str] = []
        if a.market in BY_KEY and BY_KEY[a.market].status == "live" and a.line is not None and a.p_fair:
            sd = width.sd(bucket) if width else None
            if sd is None:
                model_gates.append("width_unfit")
            else:
                target = _devigged_total(a, sd)
                model = dist.solve_total(target, sd, width, tol=tol) if width else None
                if model is not None and abs(model.median - target) > tol:
                    model_gates.append("anchor_miss")
        for e in a.exec_quotes:
            p_fair = a.p_fair.get(e.side)
            if p_fair is None:
                continue
            p_model: float | None = None
            if model is not None and not model_gates and a.line is not None:
                p_model = model.p_over(a.line) if e.side == "over" else model.p_under(a.line)
                push = model.push_mass(a.line)
                if push:
                    p_model = p_model / (1 - push)
            p_use = p_model if p_model is not None else p_fair
            t, edge, ev, tier_gates = tier(p_use, e.american, implied(e.american), thr)
            gates = base_gates + model_gates + tier_gates
            out.append(
                ledger.Row(
                    slate_date=day, event_id=a.event_id, tip_utc=a.tip_utc, market=a.market, side=e.side, line=a.line,
                    book=e.book, pass_id=pass_id, pass_kind=pass_kind, bet_of_record=bet_of_record, status=status,
                    stale_flag=e.stale, quote_shard=sid, american=e.american, opposite_american=e.opposite_american,
                    books=a.books, paired_books=a.paired_books, exec_books=len(a.exec_quotes), code_sha=code_sha,
                    env_lock_hash=env_lock_hash, params_hash=params_hash(width), model_version=MODEL_VERSION if p_model is not None else "market_only",
                    quality_bucket=bucket, bucket_inputs={"paired_books": float(a.paired_books), "spread_abs": abs(spread_line) if spread_line is not None else -1.0},
                    p_fair=p_fair, p_model=p_model, target_median=model.target_median if model else None,
                    sim_median=model.median if model else None, paths=model.paths if model else 0, se=model.se if model else None,
                    push_mass=model.push_mass(a.line) if model and a.line is not None else None,
                    parity_mass=model.parity_mass() if model else None, edge=round(edge, 5), ev=round(ev, 5),
                    tier=t if not gates else "Pass", edge_by_rung={"v0_shape": round((p_model - p_fair), 5) if p_model is not None else 0.0},
                    gates=gates,
                )
            )
    apply_exposure(out, max_live_buys=max_live_buys, max_per_book=max_per_book)
    return Priced(out, anchors, sid)


def apply_exposure(rows: list[ledger.Row], *, max_live_buys: int, max_per_book: int) -> None:
    """Cap live buys per day and per book; the lowest-edge extras get the `exposure_cap` gate (graded like any gate)."""
    buys = sorted((r for r in rows if r.is_buy and r.status == "live"), key=lambda r: -(r.edge or 0.0))
    kept_total = 0
    kept_book: dict[str, int] = {}
    for r in buys:
        if kept_total >= max_live_buys or kept_book.get(r.book, 0) >= max_per_book:
            r.gates.append("exposure_cap")
            r.tier = "Pass"
            continue
        kept_total += 1
        kept_book[r.book] = kept_book.get(r.book, 0) + 1


def _devigged_total(a: consensus.Anchor, sd: float) -> float:
    """Invert the devigged Over probability at the posted line into a continuous median (§3.6 ii)."""
    p_over = a.p_fair.get("over")
    line = a.line if a.line is not None else 0.0
    if p_over is None or not 0.01 < p_over < 0.99:
        return line
    # P(T > line) = 1 − Φ((line − μ)/σ)  ⇒  μ = line + σ·Φ⁻¹(p_over)
    return line + sd * math.sqrt(2) * _erfinv(2 * p_over - 1)


def _erfinv(y: float) -> float:
    a = 0.147
    ln = math.log(1 - y * y)
    t = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), y)


def render(rows: list[ledger.Row]) -> str:
    """Plain-text card: buys first, then refusals by gate, then the shadow/research count."""
    buys = [r for r in rows if r.is_buy and r.status == "live"]
    lines = [f"CBB card — {rows[0].slate_date if rows else ''} — {len(buys)} live buy(s)"]
    for r in sorted(buys, key=lambda r: -(r.edge or 0)):
        lines.append(f"  {r.tier:<12} {r.event_id} {r.market} {r.side} {r.line} @ {r.book} {r.american:+.0f}  edge {r.edge:+.3f}  p_fair {r.p_fair:.3f}" + ("  STALE" if r.stale_flag else ""))
    refused: dict[str, int] = {}
    for r in rows:
        for g in r.gates:
            refused[g] = refused.get(g, 0) + 1
    if refused:
        lines.append("  refusals: " + ", ".join(f"{g}={n}" for g, n in sorted(refused.items())))
    shadow = sum(1 for r in rows if r.status != "live")
    lines.append(f"  shadow/research rows priced: {shadow}; market-only stale flags: {sum(1 for r in rows if r.stale_flag)}")
    return "\n".join(lines)


__all__ = ["MODEL_VERSION", "PREREG", "Priced", "apply_exposure", "latest_pricing_shard", "params_hash", "prereg_scalar", "price", "render"]
