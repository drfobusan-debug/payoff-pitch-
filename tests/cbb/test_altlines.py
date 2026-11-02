from __future__ import annotations

from pathlib import Path

from cbb_engine import altlines
from engine_runtime import altlines as rt


def test_table_clusters_by_game_and_splits_bought_sold() -> None:
    alts = [
        rt.Alt("g1", "spread", 1.0, True, "win", 0.8),
        rt.Alt("g1", "spread", 2.0, True, "win", 0.6),
        rt.Alt("g2", "spread", -1.0, False, "loss", -1.0),
        rt.Alt("g3", "total", 0.0, False, "win", 1.0),  # the main number itself is not an alternate
    ]
    rows = {r[0]: r for r in rt.table(alts)}
    assert rows["spread all: bought"][1:4] == ("1", "2", "2-0-0")
    assert rows["spread beside buys: bought"][2] == "2"
    assert rows["spread all: sold"][3] == "0-1-0"
    assert not any(k.startswith("total") for k in rows)
    assert rt.best_price([((1,), -120.0, "a"), ((1,), 110.0, "b"), ((2,), -105.0, "c")]) == ["b", "c"]


def test_cbb_favour_and_ledger_round_trip(tmp_path: Path) -> None:
    assert altlines.favour("game_total_alt", "over", 145.5, 143.5) == 2.0
    assert altlines.favour("game_total_alt", "under", 145.5, 143.5) == -2.0
    assert altlines.favour("game_ats_alt", "home", -6.5, -4.5) == 2.0
    path = tmp_path / "alt_ledger.csv"
    row = {"date": "2026-11-10", "event_id": "e1", "market": "game_ats_alt", "side": "home", "main_line": -6.5,
           "alt_line": -4.5, "favour": 2.0, "bought": True, "book": "draftkings", "american": -150.0,
           "result": "win", "pnl": 0.667}
    altlines.save_day(path, "2026-11-10", [row])
    altlines.save_day(path, "2026-11-11", [{**row, "date": "2026-11-11", "result": "loss", "pnl": -1.0}])
    got = altlines.alts(path, until="2026-11-10")
    assert len(got) == 1 and got[0].beside_buy and got[0].kind == "spread" and got[0].favour == 2.0
    assert len(altlines.alts(path)) == 2
