"""The ledger records whether the card saw the lineup that batted.

``features.lineup_lock`` stamps provenance on every recommendation so that "do
projected-lineup rows underperform posted ones" can be answered from the history
-- the evidence its own demotion gate is waiting on. The columns stopped at the
recommendation, so the answer was unobtainable. These tests pin the persistence
and the honesty of the read: a gap inside its own standard error is reported as
no gap.
"""

from __future__ import annotations

import csv
from datetime import date as Date
from pathlib import Path

from mlb_engine.audit.analysis import lineup_findings, lineup_splits
from mlb_engine.audit.ledger import (
    LEDGER_FIELDS,
    LedgerEntry,
    load_ledger,
    update_ledger,
)
from mlb_engine.features.lineup_lock import POSTED, PROJECTED
from mlb_engine.market.tiers import Tier


def _row(
    status: str,
    result: str,
    *,
    model_prob: float = 0.5,
    tier: str = Tier.PASS.value,
    hours: float | None = 5.5,
) -> LedgerEntry:
    return LedgerEntry(
        date="2026-08-16",
        matchup="KC @ DET",
        category="Batter Props",
        market="batter_h",
        selection="Bobby Witt Jr. H o0.5",
        line=0.5,
        book="dk",
        odds=-130,
        tier=tier,
        model_prob=model_prob,
        ev=0.04,
        result=result,
        pnl=0.77 if result == "win" else -1.0,
        lineup_status=status,
        hours_to_first_pitch=hours,
    )


def test_provenance_survives_a_write_and_a_read(tmp_path: Path) -> None:
    """Both columns round-trip, which is the whole point of the change."""
    path = tmp_path / "ledger.csv"
    update_ledger(path, [_row(PROJECTED, "loss", hours=6.25)], Date(2026, 8, 16))
    back = load_ledger(path)
    assert [(e.lineup_status, e.hours_to_first_pitch) for e in back] == [
        (PROJECTED, 6.25)
    ]
    assert "lineup_status" in LEDGER_FIELDS
    assert "hours_to_first_pitch" in LEDGER_FIELDS


def test_a_ledger_written_before_the_columns_existed_still_loads(tmp_path: Path) -> None:
    """71,667 rows predate this; they read back blank, not as an error."""
    path = tmp_path / "old.csv"
    old_fields = [f for f in LEDGER_FIELDS
                  if f not in ("lineup_status", "hours_to_first_pitch")]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=old_fields)
        w.writeheader()
        w.writerow({k: "" for k in old_fields} | {
            "date": "2026-08-01", "matchup": "KC @ DET", "category": "Batter Props",
            "market": "batter_h", "selection": "x H o0.5", "tier": Tier.PASS.value,
            "model_prob": "0.5", "result": "loss", "pnl": "-1.0", "book": "dk",
            "veto_gate": "", "pass_gate": "",
        })
    (row,) = load_ledger(path)
    assert row.lineup_status == ""
    assert row.hours_to_first_pitch is None
    # And a row with no provenance is excluded rather than counted as posted.
    assert lineup_splits([row]) == []


def test_calibration_is_measured_on_passes_too() -> None:
    """A projected lineup damages the probability whether or not it was bought.

    Restricting the calibration to buys would select on the overstatement being
    measured, so passes count for bias and only buys count for ROI.
    """
    rows = [
        *[_row(POSTED, "win", model_prob=0.6) for _ in range(6)],
        *[_row(POSTED, "loss", model_prob=0.6) for _ in range(4)],
        *[_row(PROJECTED, "win", model_prob=0.6, tier=Tier.STRONG.value)
          for _ in range(3)],
        *[_row(PROJECTED, "loss", model_prob=0.6, tier=Tier.STRONG.value)
          for _ in range(7)],
    ]
    splits = {s.status: s for s in lineup_splits(rows)}
    assert splits[POSTED].n == 10 and splits[POSTED].n_buys == 0
    assert abs(splits[POSTED].bias - 0.0) < 1e-9  # model .60, realized .60
    assert splits[PROJECTED].n == 10 and splits[PROJECTED].n_buys == 10
    assert abs(splits[PROJECTED].bias - 0.30) < 1e-9  # model .60, realized .30
    assert splits[POSTED].roi == 0.0  # no priced buys, so no ROI to claim


def test_a_thin_sample_is_reported_as_thin() -> None:
    """The number is shown; acting on it is not invited."""
    rows = [
        *[_row(POSTED, "win") for _ in range(5)],
        *[_row(PROJECTED, "loss") for _ in range(5)],
    ]
    (finding,) = lineup_findings(rows)
    assert "not to act on" in finding
    assert "n=5" in finding


def test_one_sided_history_says_so_rather_than_comparing() -> None:
    """Every game projected means there is nothing to compare against."""
    (finding,) = lineup_findings([_row(PROJECTED, "loss") for _ in range(400)])
    assert "only one side" in finding and "projected n=400" in finding


def test_a_history_with_no_provenance_at_all_says_so() -> None:
    """The state of the existing ledger: every row predates the columns."""
    (finding,) = lineup_findings([_row("", "loss") for _ in range(400)])
    assert "no graded batter row carries it yet" in finding


def test_a_gap_inside_one_standard_error_is_not_a_finding() -> None:
    """Two groups of coin flips differ; that is not evidence they differ."""
    rows = [
        *[_row(POSTED, "win", model_prob=0.5) for _ in range(200)],
        *[_row(POSTED, "loss", model_prob=0.5) for _ in range(200)],
        *[_row(PROJECTED, "win", model_prob=0.5) for _ in range(198)],
        *[_row(PROJECTED, "loss", model_prob=0.5) for _ in range(202)],
    ]
    (finding,) = lineup_findings(rows)
    assert "indistinguishable" in finding
    assert "against the lineup-lock demotion" in finding


def test_a_real_gap_names_the_worse_side_and_the_cheaper_fix() -> None:
    """Priced after lineups post beats demoting the buys."""
    rows = [
        *[_row(POSTED, "win", model_prob=0.5) for _ in range(200)],
        *[_row(POSTED, "loss", model_prob=0.5) for _ in range(200)],
        *[_row(PROJECTED, "win", model_prob=0.5, tier=Tier.STRONG.value)
          for _ in range(120)],
        *[_row(PROJECTED, "loss", model_prob=0.5, tier=Tier.STRONG.value)
          for _ in range(280)],
    ]
    (finding,) = lineup_findings(rows)
    assert "projected rows are the worse-calibrated side" in finding
    assert "after lineups post" in finding
