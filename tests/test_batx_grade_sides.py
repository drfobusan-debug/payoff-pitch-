"""The BAT X head-to-head has to grade the side that was actually bet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

bx = pytest.importorskip("scripts.batx_study")


@pytest.mark.parametrize(
    "selection,under",
    [
        ("Aaron Judge HR o0.5", False),
        ("Aaron Judge HR u0.5", True),
        ("Zack Wheeler Ks u6.5", True),
        ("Pete Crow-Armstrong H+R+RBI o1.5", False),
    ],
)
def test_the_side_is_read_off_the_selection(selection: str, under: bool) -> None:
    assert bx.is_under(selection) is under


def test_an_under_row_is_graded_against_p_under_not_p_over() -> None:
    """``batx_prob`` is always P(over); the ledger's row is the side we bet.

    Left unflipped, a fade is scored against the mirror image of the forecast,
    which turns a correct call into a wrong one. Every counting prop the engine
    fades is an under, so this is most of the fade book once it is graded.
    """
    priced = pd.DataFrame(
        {
            "date": ["2026-08-14", "2026-08-14"],
            "player": ["aaron judge", "aaron judge"],
            "market": ["batter_hr", "batter_h"],
            "line": [0.5, 0.5],
            "selection": ["Aaron Judge HR u0.5", "Aaron Judge H o0.5"],
            "batx_prob": [0.12, 0.71],
        }
    )
    priced["under"] = priced.selection.map(bx.is_under)
    priced.loc[priced.under, "batx_prob"] = 1.0 - priced.loc[priced.under, "batx_prob"]
    assert priced.batx_prob.tolist() == pytest.approx([0.88, 0.71])


def test_market_dummies_leave_the_forecast_columns_alone() -> None:
    """The per-market intercepts must not eat the three forecasts' coefficients."""
    frame = pd.DataFrame(
        {
            "model_prob": [0.4, 0.6, 0.3, 0.7],
            "fair_prob": [0.45, 0.55, 0.35, 0.65],
            "batx_prob": [0.42, 0.58, 0.33, 0.68],
            "market": ["batter_h", "batter_h", "batter_hr", "batter_hr"],
        }
    )
    design = bx._design(frame, ["batter_h", "batter_hr"])
    assert design.shape == (4, 4)  # three forecasts + one dropped-baseline dummy
    assert np.allclose(design[:, 0], bx.logit(frame.model_prob.to_numpy()))
    assert design[:, 3].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_grade_survives_a_ledger_that_carries_its_own_batx_column(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The engine stamps ``batx_prob`` on the ledger, and the join must not collide.

    Both frames name the column the same thing, so an unqualified merge suffixes
    them to ``_x``/``_y`` and every later reference raises ``KeyError``.
    """
    ledger = tmp_path / "ledger.csv"
    pd.DataFrame(
        {
            "date": ["2026-08-14", "2026-08-14"],
            "market": ["batter_hr", "batter_h"],
            "selection": ["Aaron Judge HR u0.5", "Aaron Judge H o0.5"],
            "line": [0.5, 0.5],
            "result": ["win", "loss"],
            "model_prob": [0.9, 0.55],
            "fair_prob": [0.88, 0.53],
            "batx_prob": [float("nan"), float("nan")],
        }
    ).to_csv(ledger, index=False)

    probs = tmp_path / "2026-08-14.csv"
    pd.DataFrame(
        {
            "date": ["2026-08-14", "2026-08-14"],
            "player": ["aaron judge", "aaron judge"],
            "market": ["batter_hr", "batter_h"],
            "line": [0.5, 0.5],
            "batx_prob": [0.12, 0.71],
        }
    ).to_csv(probs, index=False)

    bx.cmd_grade(argparse.Namespace(probs=str(probs), ledger=str(ledger)))
    out = capsys.readouterr().out
    assert "joined 2 graded rows" in out
    # Mean of the flipped HR under (0.88) and the hits over (0.71). Reading the
    # ledger's own empty column instead of the priced one gives no number at all.
    assert "0.795" in out


def test_a_single_market_needs_no_dummy() -> None:
    frame = pd.DataFrame(
        {
            "model_prob": [0.4, 0.6],
            "fair_prob": [0.45, 0.55],
            "batx_prob": [0.42, 0.58],
            "market": ["batter_h", "batter_h"],
        }
    )
    assert bx._design(frame, ["batter_h"]).shape == (2, 3)
