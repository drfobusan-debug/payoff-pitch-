"""Shared fixtures.

``legacy_anchor`` pins the market anchor at the 0.3 the engine shipped with
before every market was fitted to its own weight (``_MARKET_ANCHOR_BY_MARKET``).
Under the fitted weights the bet probability is the devigged price to within a
point on most markets, so a fixture priced -110 against -110 can no longer be a
buy at all -- and a screen that is exercised by flipping a buy to a pass has
nothing to flip. Those screens (conviction floor, price ceilings, lineup clock,
sharp-money gate, thin-starter gate, walks veto, both-sides pricing) are tested
under the old weight so they keep testing themselves rather than the anchor.
The anchor's own tests live in ``test_selection_guards.py`` and take the
shipped defaults.
"""

from __future__ import annotations

import pytest

from mlb_engine import config as _config


@pytest.fixture
def legacy_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLBE_MARKET_ANCHOR", "0.3")
    for market in _config._MARKET_ANCHOR_BY_MARKET:
        weight = "0.0" if market.endswith("_total") else "0.3"
        monkeypatch.setenv(f"MLBE_MARKET_ANCHOR_{market.upper()}", weight)
