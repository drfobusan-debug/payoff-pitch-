"""A BAT X export has to be the slate it is filed under.

Nothing in the export states its date and ``--date`` is what the join keys on,
so a file saved for the next day and priced as today's joins by name and puts
another slate's projections beside tonight's bets. It is invisible to the eye:
consecutive days of a series carry the same matchups and nearly the same
hitters. Only the starting pitchers differ, and both files name them.
"""

from __future__ import annotations

import pandas as pd
import pytest

bx = pytest.importorskip("scripts.batx_study")

# The hitters export names each hitter's opposing starter; the pitchers export
# names the starters themselves.
HITTERS = pd.DataFrame(
    {
        "NAME": ["Wyatt Langford", "Shohei Ohtani"],
        "PITCHER": ["Jacob Lopez", "Logan Henderson"],
    }
)
PITCHERS = pd.DataFrame({"NAME": ["Jacob Lopez", "Logan Henderson"]})


def test_the_export_is_dated_by_who_is_starting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(bx, "starters_on", lambda _: {"jacob lopez", "logan henderson"})
    bx.check_slate_date(HITTERS, PITCHERS, "2026-08-16")
    assert "2/2 of 2026-08-16's starters" in capsys.readouterr().out


def test_another_days_export_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bx, "starters_on", lambda _: {"tarik skubal", "paul skenes"})
    with pytest.raises(SystemExit, match="not 2026-08-15"):
        bx.check_slate_date(HITTERS, PITCHERS, "2026-08-15")


def test_a_hitters_only_export_still_dates_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    """The opposing-starter column carries the check when no pitcher file is given."""
    monkeypatch.setattr(bx, "starters_on", lambda _: {"tarik skubal", "paul skenes"})
    with pytest.raises(SystemExit):
        bx.check_slate_date(HITTERS, None, "2026-08-15")


def test_a_check_that_cannot_run_does_not_block_pricing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unreachable schedule leaves the operator's date alone rather than failing."""

    def boom(_: str) -> set[str]:
        raise RuntimeError("no network")

    monkeypatch.setattr(bx, "starters_on", boom)
    bx.check_slate_date(HITTERS, PITCHERS, "2026-08-16")
    assert "could not read" in capsys.readouterr().out

    bx.check_slate_date(pd.DataFrame({"NAME": ["Aaron Judge"]}), None, "2026-08-16")
    assert "names no starters" in capsys.readouterr().out


def test_a_batters_walks_and_strikeouts_are_priced_off_the_feeds_own_columns() -> None:
    """BB and K are per-PA events the export states, so they need no dispersion."""
    row = pd.Series(
        {
            "PA": 4.0,
            "R": 0.6,
            "RBI": 0.6,
            "BB": 0.4,
            "K": 1.0,
            "1B": 0.6,
            "2B": 0.2,
            "3B": 0.0,
            "HR": 0.15,
        }
    )
    probs = bx.price_row(row, 1.5, 1.5)
    assert probs["batter_bb@0.5"] == pytest.approx(1 - (1 - 0.1) ** 4, abs=1e-6)
    assert probs["batter_k@0.5"] == pytest.approx(1 - (1 - 0.25) ** 4, abs=1e-6)
    # The harder line is strictly less likely, on both.
    assert probs["batter_bb@1.5"] < probs["batter_bb@0.5"]
    assert probs["batter_k@1.5"] < probs["batter_k@0.5"]
