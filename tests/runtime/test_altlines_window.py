"""Each engine reads only the alternate rungs a few points either side of the main number."""

from __future__ import annotations

from engine_runtime import altlines as rt


def _alt(kind: str, favour: float) -> rt.Alt:
    return rt.Alt(game="g", kind=kind, favour=favour, beside_buy=True, outcome="win", pnl=0.5)


def test_the_window_is_per_market_and_both_directions() -> None:
    alts = [_alt("spread", 3.0), _alt("spread", -3.5), _alt("total", -4.0), _alt("total", 4.5)]
    kept = rt.within(alts, spread=3.0, total=4.0)
    assert [(a.kind, a.favour) for a in kept] == [("spread", 3.0), ("total", -4.0)]
