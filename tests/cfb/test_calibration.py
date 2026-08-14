"""Calibration maps are only valid for the features that produced them."""

from __future__ import annotations

import json
from pathlib import Path

from cfb_engine.calibration import FEATURE_BASIS, Calibrator, IsotonicMap


def _map(path: Path, basis: str | None) -> None:
    payload: dict[str, object] = {
        "markets": {"game_ml": {"x": [0.4, 0.6], "y": [0.3, 0.5]}},
        "default": {"x": [0.4, 0.6], "y": [0.3, 0.5]},
    }
    if basis is not None:
        payload["basis"] = basis
    path.write_text(json.dumps(payload))


def test_a_map_from_this_basis_is_applied(tmp_path: Path) -> None:
    path = tmp_path / "cal.json"
    _map(path, FEATURE_BASIS)
    cal = Calibrator.from_json(path)
    assert cal.apply("game_ml", 0.6) < 0.6


def test_a_map_from_another_basis_is_ignored(tmp_path: Path) -> None:
    """A stale map re-imposes the bias the feature change corrected.

    In the MLB engine one silently cancelled a fix outright -- the corrected
    home-run probability came out 11.76% -> 11.62%, i.e. unchanged -- so a map
    fit on other features must not be applied, only refit.
    """
    path = tmp_path / "cal.json"
    _map(path, "some-older-basis")
    cal = Calibrator.from_json(path)
    assert cal.apply("game_ml", 0.6) == 0.6
    assert cal.maps == {}


def test_an_unstamped_map_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "cal.json"
    _map(path, None)
    assert Calibrator.from_json(path).apply("game_ml", 0.6) == 0.6


def test_a_saved_map_carries_the_basis(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    Calibrator(maps={}, default=IsotonicMap([0.4], [0.35])).to_json(path)
    assert json.loads(path.read_text())["basis"] == FEATURE_BASIS
