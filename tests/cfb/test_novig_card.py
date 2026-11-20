"""Novig's quote beside each buy on the card and workbook (reference only)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from openpyxl import load_workbook

from cfb_engine.audit import snapshot
from cfb_engine.audit.snapshot import SideQuote
from cfb_engine.market.tiers import Tier
from cfb_engine.output.card import NOVIG_NOTE, _game_best_block, _slate_best_block
from cfb_engine.output.excel import COLUMNS, write_workbook
from cfb_engine.pipeline import Pipeline
from cfb_engine.recommendations import Recommendation, load_json, save_json
from tests.cfb.test_strongest import DAY, _board, _ctx, _home, _rec, pipe  # noqa: F401


def _ats(
    line: float, book: float, novig_line: float | None, novig: float | None, side: str = "cover"
) -> Recommendation:
    rec = dataclasses.replace(
        _rec("IND", Tier.STRONG, is_strongest=False),
        market="game_ats",
        selection=f"IND {line:+g}",
        line=line,
        market_american=book,
        side=side,
    )
    return dataclasses.replace(rec, novig_line=novig_line, novig_american=novig)


def test_novig_counts_only_when_number_and_price_are_both_no_worse() -> None:
    assert _ats(-7.0, -110, -7.0, -105).novig_at_least_as_good
    assert _ats(-7.0, -110, -6.5, -110).novig_at_least_as_good
    assert not _ats(-7.0, -110, -7.5, -105).novig_at_least_as_good
    assert not _ats(-7.0, -110, -7.0, -115).novig_at_least_as_good
    assert not _ats(-7.0, -110, None, None).novig_at_least_as_good
    over = _ats(52.5, -110, 52.0, -110, side="over")
    under = _ats(52.5, -110, 52.0, -110, side="under")
    assert over.novig_at_least_as_good and not under.novig_at_least_as_good
    ml = dataclasses.replace(_rec("IND", Tier.STRONG, is_strongest=False), novig_american=-105.0)
    assert ml.novig_at_least_as_good and ml.novig_quote() == "-105"
    assert _ats(3.0, -110, 3.5, -110).novig_quote() == "+3.5 -110"
    assert dataclasses.replace(over, market="game_total").novig_quote() == "52 -110"


def test_the_card_and_workbook_show_the_novig_quote(tmp_path: Path) -> None:
    good, worse = _ats(-7.0, -110, -7.0, -105), _ats(-3.0, -110, -3.5, -110)
    worse = dataclasses.replace(worse, game_id="OSU", matchup="AWAY @ OSU", selection="OSU -3")
    for block in (_slate_best_block([good, worse]), _game_best_block([good, worse])):
        assert "Novig -7 -105 ✓" in block and "Novig -3.5 -110</span>" in block
        assert NOVIG_NOTE in block
    assert NOVIG_NOTE not in _slate_best_block([_ats(-7.0, -110, None, None)])

    assert "Novig" in COLUMNS
    ws = load_workbook(write_workbook([good, worse], tmp_path / "c.xlsx", DAY))["All"]
    header = [c.value for c in ws[1]]
    sel, col = header.index("Selection"), header.index("Novig")
    cells = {r[sel]: r[col] for r in ws.iter_rows(min_row=2, values_only=True) if r[sel]}
    assert cells["IND -7"] == "-7 -105 (bet here)" and cells["OSU -3"] == "-3.5 -110"

    path = tmp_path / "p.json"
    save_json([good], path)
    assert load_json(path)[0].novig_american == -105


def test_the_pipeline_stamps_novig_from_its_own_pull_without_moving_the_price(
    pipe: Pipeline,  # noqa: F811
) -> None:
    base = _home(pipe._price_ml(_ctx(0.56, None), _board()))
    assert base.novig_american is None
    pipe._novig = {snapshot.key(base.matchup, base.market, base.selection): SideQuote(-104.0, 0.5)}
    rec = _home(pipe._price_ml(_ctx(0.56, None), _board()))
    assert rec.novig_american == -104.0 and rec.novig_at_least_as_good
    assert dataclasses.replace(rec, novig_american=None) == base
