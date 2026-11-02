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


def test_grade_matches_a_later_card_and_drops_in_play_alts(tmp_path: Path) -> None:
    from dataclasses import replace
    from datetime import datetime, timezone

    from cbb_engine import card
    from cbb_engine.data.espn import Game
    from cbb_engine.models import distribution as dist
    from engine_runtime import ledger, quotes
    from engine_runtime.quotes import Quote

    day = "2026-11-10"

    def q(book: str, side: str, am: float, *, line: float = 145.5, market: str = "game_total", pid: str = "b1",
          kind: str = "board", cap: str = f"{day}T18:00:00Z", opp: float | None = None) -> Quote:
        return Quote(game_date=day, event_id="e1", tip_utc=f"{day}T23:00:00Z", market=market, side=side, line=line,
                     book=book, entity="", american=am, opposite_american=opp, captured_at=cap, pass_id=pid,
                     pass_kind=kind)

    board = []
    for b, over, under in (("pinnacle", -115, -105), ("fanduel", -110, -110), ("caesars", -112, -108), ("betrivers", -110, -110)):
        board += [q(b, "over", over, opp=under), q(b, "under", under, opp=over)]
    board += [q("draftkings", "over", 130, opp=-110), q("draftkings", "under", -110, opp=130)]
    quotes.write_shard(tmp_path, "cbb", day, board, pass_id="b1", pass_kind="board", extra={"teams": {"e1": ["Duke", "Kansas"]}})
    width = dist.Width(sd_by_bucket={"all": 11.0}, key_mass={}, parity_shift=0.0, fitted_on="test", n=999)
    priced = card.price(tmp_path, day, now=datetime(2026, 11, 10, 18, 5, tzinfo=timezone.utc), pass_id="card1",
                        pass_kind="featured", width=width)
    assert priced is not None and altlines.tracked(priced.rows) == ["e1"]
    ledger.write_once(tmp_path, "cbb", day, "card1", priced.rows)
    ledger.write_once(tmp_path, "cbb", day, "card2", [replace(r, pass_id="card2") for r in priced.rows])
    alts = [
        q("draftkings", "over", -150, line=143.5, market="game_total_alt", pid="card2_alt_e1", kind="alt", cap=f"{day}T21:00:00Z"),
        q("draftkings", "over", -200, line=141.5, market="game_total_alt", pid="card2_alt_e1", kind="alt", cap=f"{day}T23:10:00Z"),
    ]
    assert quotes.write_shard(tmp_path, "cbb", day, alts, pass_id="card2_alt_e1", pass_kind="alt") is not None
    game = Game(espn_id="x", start_utc="", away="Kansas", home="Duke", state="STATUS_FINAL", neutral=False,
                conference_game=False, away_halves=(40, 40), home_halves=(35, 36))
    rows = altlines.grade(tmp_path, day, [game], {"e1": ("Duke", "Kansas")})
    assert [(r["alt_line"], r["favour"], r["result"], r["bought"]) for r in rows] == [(143.5, 2.0, "win", True)]
