from __future__ import annotations

from pathlib import Path

import pytest

from engine_runtime import ledger, quotes, views
from engine_runtime.devig import consensus, devig_power, novig


def _q(
    pass_id: str, kind: str, at: str, book: str = "pinnacle", american: float = -110
) -> quotes.Quote:
    return quotes.Quote(
        at,
        "2026-11-10",
        "ev1",
        "2026-11-11T00:00:00Z",
        "game_total",
        "over",
        "",
        145.5,
        book,
        american,
        -110,
        pass_id,
        kind,
    )


def test_shards_are_immutable_and_hashed(tmp_path: Path) -> None:
    rows = [_q("featured_1", "featured", "2026-11-10T20:00:00Z")]
    p = quotes.write_shard(
        tmp_path, "cbb", "2026-11-10", rows, pass_id="featured_1", pass_kind="featured"
    )
    assert p is not None and quotes.verify(tmp_path, "cbb", "2026-11-10") == []
    with pytest.raises(FileExistsError):
        quotes.write_shard(
            tmp_path, "cbb", "2026-11-10", rows, pass_id="featured_1", pass_kind="featured"
        )
    assert (
        quotes.write_shard(
            tmp_path, "cbb", "2026-11-10", [], pass_id="featured_2", pass_kind="featured"
        )
        is None
    )
    p.write_bytes(p.read_bytes() + b"x")
    assert quotes.verify(tmp_path, "cbb", "2026-11-10") == [p.name]


def test_duckdb_query_reads_shards(tmp_path: Path) -> None:
    quotes.write_shard(
        tmp_path,
        "cbb",
        "d",
        [_q("a", "board", "t1"), _q("b", "featured", "t2", "draftkings")],
        pass_id="a",
        pass_kind="board",
    )
    out = quotes.query(
        tmp_path, "cbb", "d", "SELECT book, count(*) AS n FROM quotes GROUP BY book ORDER BY book"
    )
    assert out == [{"book": "draftkings", "n": 1}, {"book": "pinnacle", "n": 1}]


def test_capture_gaps_counted_per_pass_kind() -> None:
    tips = {"ev1": "2026-11-11T00:00:00Z"}
    man = [
        {"pass_kind": "board", "captured_at": "2026-11-10T10:00:00Z"},
        {"pass_kind": "board", "captured_at": "2026-11-10T11:00:00Z"},  # 60-min clock, no gap
        {"pass_kind": "board", "captured_at": "2026-11-10T13:00:00Z"},  # one missing hour
        {"pass_kind": "featured", "captured_at": "2026-11-10T21:00:00Z"},
        {"pass_kind": "featured", "captured_at": "2026-11-10T21:10:00Z"},
        {
            "pass_kind": "featured",
            "captured_at": "2026-11-10T21:40:00Z",
        },  # two 10-min passes missing
    ]
    out = quotes.expected_passes(tips, man)
    assert out["board"] == {"passes": 3, "missing": 1}
    assert out["featured"] == {"passes": 3, "missing": 2}
    assert out["close"] == {"passes": 0, "missing": 0}


def test_power_devig_and_execution_excluded_consensus() -> None:
    probs = devig_power([0.5238, 0.5238])
    assert abs(sum(probs) - 1) < 1e-9 and abs(probs[0] - 0.5) < 1e-9
    assert abs(novig(-110, -110) - 0.5) < 1e-9
    fair, seen, used = consensus(
        [
            ("pinnacle", -115, -105),
            ("draftkings", -130, +110),
            ("betmgm", -125, +105),
            ("fanduel", -112, -108),
        ],
        exclude_books=("draftkings", "betmgm"),
    )
    assert (seen, used) == (4, 2) and fair is not None and 0.50 < fair < 0.53
    assert consensus([("draftkings", -110, -110)], exclude_books=("draftkings",))[0] is None


def _row(pass_id: str, kind: str, bor: bool, **kw) -> ledger.Row:
    base = dict(
        slate_date="d",
        event_id="ev1",
        tip_utc="t",
        market="game_total",
        side="over",
        line=145.5,
        book="draftkings",
        pass_id=pass_id,
        pass_kind=kind,
        bet_of_record=bor,
        status="live",
    )
    base.update(kw)
    return ledger.Row(**base)


def test_ledger_write_once_chain_and_close_pass_rule(tmp_path: Path) -> None:
    r1 = _row(
        "card_1",
        "featured",
        True,
        american=-110,
        tier="A",
        clv_consensus=0.012,
        placed="Y",
        push_mass=0.03,
        outcome="push",
    )
    r2 = _row("card_1", "featured", True, side="under", gates=["thin_market"], clv_consensus=-0.01)
    ledger.write_once(tmp_path, "cbb", "d", "card_1", [r1, r2])
    with pytest.raises(FileExistsError):
        ledger.write_once(tmp_path, "cbb", "d", "card_1", [r1])
    with pytest.raises(ValueError):
        ledger.write_once(tmp_path, "cbb", "d", "close_1", [_row("close_1", "close", True)])
    with pytest.raises(ValueError):
        ledger.write_once(
            tmp_path, "cbb", "d", "bad", [_row("bad", "featured", True, status="paper")]
        )
    rows = ledger.read_day(tmp_path, "cbb", "d")
    assert len(rows) == 2 and ledger.verify_chain(rows) and rows[0].prev_hash == ledger.GENESIS
    rows[0].tier = "B"
    assert not ledger.verify_chain(rows)
    rec = ledger.record(tmp_path, "cbb", "d")
    assert set(k[2] for k in rec) == {"over", "under"}
    clv = views.v_clv(ledger.read_day(tmp_path, "cbb", "d"))
    assert clv["n"] == 1 and clv["n_filled"] == 1 and clv["clv_consensus"] == 0.012
    assert views.v_gate(ledger.read_day(tmp_path, "cbb", "d"))[0]["gate"] == "thin_market"
    assert views.v_shape(ledger.read_day(tmp_path, "cbb", "d"))[0]["real_push"] == 1.0
