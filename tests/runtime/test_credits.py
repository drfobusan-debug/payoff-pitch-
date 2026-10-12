from __future__ import annotations

import logging
from pathlib import Path

from engine_runtime import credits


def _budget(root: Path, enforce: bool, **sports: int) -> None:
    lines = [
        f"enforce: {str(enforce).lower()}",
        "key_total: 1000",
        "reserve:",
        "  cbb_featured: 300",
        "sports:",
    ]
    lines += [f"  {k}: {v}" for k, v in sports.items()]
    (root / "credit_budget.yaml").write_text("\n".join(lines) + "\n")


def test_missing_budget_fails_open(tmp_path: Path, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        assert credits.debit("mlb", 50, root=tmp_path, day="2026-11-01") is True
    assert "falling through" in caplog.text
    assert not credits.ledger_path(tmp_path, "2026-11-01").exists()


def test_observe_only_logs_every_call_and_never_denies(tmp_path: Path) -> None:
    _budget(tmp_path, False, mlb=10)
    assert credits.debit("mlb", 50, root=tmp_path, day="2026-11-01", endpoint="odds")
    assert credits.debit("mlb", 50, root=tmp_path, day="2026-11-01", endpoint="odds")
    used = credits.spent(credits.ledger_path(tmp_path, "2026-11-01"))
    assert used == {"": 100, "mlb": 100}


def test_enforce_denies_over_sport_slice_and_logs_denial(tmp_path: Path) -> None:
    _budget(tmp_path, True, mlb=60, cbb=500)
    assert credits.debit("mlb", 50, root=tmp_path, day="d")
    assert credits.debit("mlb", 20, root=tmp_path, day="d") is False
    rows = credits.ledger_path(tmp_path, "d").read_text().splitlines()
    assert len(rows) == 3 and rows[-1].endswith("False")
    assert credits.spent(credits.ledger_path(tmp_path, "d"))["mlb"] == 50


def test_reserve_is_held_from_other_sports_not_from_cbb(tmp_path: Path) -> None:
    _budget(tmp_path, True, mlb=900, cbb=900)
    # key_total 1000 - reserve 300 = 700 headroom for mlb
    assert credits.debit("mlb", 650, root=tmp_path, day="d")
    assert credits.debit("mlb", 100, root=tmp_path, day="d") is False
    # cbb may use the reserve it owns
    assert credits.debit("cbb", 300, root=tmp_path, day="d")


def test_corrupt_ledger_fails_open(tmp_path: Path) -> None:
    _budget(tmp_path, True, mlb=10)
    path = credits.ledger_path(tmp_path, "d")
    path.parent.mkdir(parents=True)
    path.write_text("ts,sport,cost,allowed\nx,mlb,notanumber,True\n")
    assert credits.debit("mlb", 999, root=tmp_path, day="d") is True


def test_debit_for_costs_markets_times_regions_and_skips_free_endpoints(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(credits.ENV_ROOT, str(tmp_path))
    _budget(tmp_path, False, nhl=10)
    assert credits.debit_for("nhl", "https://x/v4/sports/k/events", {"regions": "us"})
    assert credits.debit_for(
        "nhl", "https://x/v4/sports/k/odds", {"regions": "us,uk", "markets": "h2h,totals"}
    )
    assert credits.debit_for(
        "nhl", "https://x/v4/sports/k/events/abc/odds", {"regions": "us", "markets": "totals_p1"}
    )
    import csv

    rows = list(csv.DictReader(credits.ledger_path(tmp_path, credits._today()).open()))
    assert [(r["endpoint"], r["cost"], r["events"]) for r in rows] == [
        ("odds", "4", "0"),
        ("event_odds", "1", "1"),
    ]
