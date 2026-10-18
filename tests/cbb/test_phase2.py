"""Phase 2: exposure caps, v_persist, email build (offline — nothing is sent)."""

from __future__ import annotations

from pathlib import Path

from cbb_engine import audit, card, email
from cbb_engine.config import Config, Delivery
from engine_runtime import ledger
from engine_runtime.quotes import Quote
from tests.cbb.test_phase1 import _q


def _row(pid: str, book: str, edge: float, *, stale: bool = False, shard: str = "", am: float = -110) -> ledger.Row:
    return ledger.Row(
        slate_date="2026-01-10", pass_id=pid, pass_kind="featured", bet_of_record=True,
        event_id="e1", tip_utc="2026-01-10T23:00:00Z", market="game_total", side="over", line=145.5,
        book=book, american=am, status="live", tier="B", edge=edge, stale_flag=stale, quote_shard=shard,
    )


def test_exposure_cap_gates_lowest_edge_extras() -> None:
    rows = [_row("p1", "draftkings", 0.05), _row("p1", "draftkings", 0.04), _row("p1", "betmgm", 0.03)]
    card.apply_exposure(rows, max_live_buys=2, max_per_book=1)
    gated = [r for r in rows if "exposure_cap" in r.gates]
    assert [r.edge for r in gated] == [0.04]
    assert gated[0].tier == "Pass" and not gated[0].is_buy
    assert sum(r.is_buy for r in rows) == 2


def test_v_persist_splits_next_featured_from_next_card() -> None:
    shards: list[Quote] = [_q("draftkings", "over", -110, -110, pid="s1", kind="featured")]
    shards += [_q("draftkings", "over", -115, -105, pid="s2", kind="featured")]
    for q, t in zip(shards, ("2026-01-10T15:00:00Z", "2026-01-10T15:10:00Z"), strict=True):
        object.__setattr__(q, "captured_at", t)
    rows = [_row("c1", "draftkings", 0.03, stale=True, shard="s1"), _row("c2", "draftkings", 0.03, am=-115)]
    out = audit.v_persist(rows, shards)
    assert out == [{"book": "draftkings", "market": "game_total", "n": 1, "next_featured": 0.0, "next_card": 0.0}]
    shards[1] = _q("draftkings", "over", -110, -110, pid="s2", kind="featured")
    object.__setattr__(shards[1], "captured_at", "2026-01-10T15:10:00Z")
    assert audit.v_persist(rows, shards)[0]["next_featured"] == 1.0


def test_email_build_attaches_card_and_requires_password(tmp_path: Path) -> None:
    p = tmp_path / "card.txt"
    p.write_text("hi")
    msg = email.build("a@x", "b@x", "CBB card", "body", {"card": p})
    assert msg["To"] == "b@x" and [a.get_filename() for a in msg.iter_attachments()] == ["card.txt"]
    cfg = Config(delivery=Delivery(gmail_user="a@x", gmail_app_password=None, email_to="b@x"))
    try:
        email.addresses(cfg)
    except email.EmailNotConfigured:
        pass
    else:
        raise AssertionError("expected EmailNotConfigured")
