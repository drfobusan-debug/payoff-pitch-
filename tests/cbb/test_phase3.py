"""Phase 0b/3 studies: gate funnel, cut 10 featured value, time-travel check (tape only)."""

from __future__ import annotations

from cbb_engine import study
from engine_runtime import ledger
from engine_runtime.quotes import Quote
from tests.cbb.test_phase1 import _q
from tests.cbb.test_phase2 import _row


def test_funnel_counts_survivors_per_gate() -> None:
    a = _row("c1", "draftkings", 0.04)
    b = _row("c1", "betmgm", 0.01)
    b.gates.append("thin_market")
    b.event_id = "e2"
    out = study.funnel([a, b])
    assert len(out) == 1 and out[0]["entered"] == 2 and out[0]["thin_market"] == 1 and out[0]["buys"] == 1


def _feat(pid: str, ts: str, am: float) -> Quote:
    q = _q("draftkings", "over", am, -110, pid=pid, kind="featured")
    object.__setattr__(q, "captured_at", ts)
    return q


def test_subset_15_and_featured_value_delta() -> None:
    shards = [_feat("f00", "2026-01-10T15:00:00Z", -110), _feat("f10", "2026-01-10T15:10:00Z", -120), _feat("f20", "2026-01-10T15:20:00Z", -100)]
    assert study.subset_15(shards) == {"f00", "f20"}
    rows = [_row("c1", "draftkings", 0.03, shard="f10", am=-120)]
    out = study.featured_value(rows, shards)
    assert out[0]["n"] == 1 and out[0]["delta_pts"] < 0 and out[0]["mde_80"] is None
    rows = [_row("c1", "draftkings", 0.03, shard="f20", am=-100)]
    assert study.featured_value(rows, shards)[0]["delta_pts"] == 0.0


def test_time_travel_flags_future_stamps() -> None:
    bad = study.time_travel("2026-01-10", [("quote", "2026-01-10T23:00:00Z"), ("torvik", "2026-01-11T03:00:00Z")])
    assert bad == ["torvik 2026-01-11T03:00:00Z"]
    assert ledger.Row.__dataclass_fields__["quality_bucket"]
