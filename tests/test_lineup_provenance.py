"""The ledger records whether the card saw the lineup that batted.

``features.lineup_lock`` stamps provenance on every recommendation so that "do
projected-lineup rows underperform posted ones" can be answered from the history
-- the evidence its own demotion gate is waiting on. The columns stopped at the
recommendation, so the answer was unobtainable. These tests pin the persistence
and the honesty of the read: a gap inside its own standard error is reported as
no gap.

The read has since been answered, so they also pin what the answer bought: a
projected lineup caps a batter prop at Moderate.
"""

from __future__ import annotations

import csv
from datetime import date as Date
from pathlib import Path
from types import SimpleNamespace

from mlb_engine.audit.analysis import lineup_findings, lineup_splits
from mlb_engine.audit.ledger import (
    LEDGER_FIELDS,
    LedgerEntry,
    load_ledger,
    update_ledger,
)
from mlb_engine.config import Config
from mlb_engine.features.drift_gate import DriftGate
from mlb_engine.features.lineup_lock import POSTED, PROJECTED, LineupLockGate
from mlb_engine.features.ml_gate import MLPenGate, MLSharpGate
from mlb_engine.market.ev import MarketQuote
from mlb_engine.market.tiers import Tier
from mlb_engine.pipeline import Pipeline


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


# ---- the batter-prop provenance cap ----------------------------------------
# Graded batter buys run posted -1.6% (n=391) against projected -9.9% (n=774),
# an 8.3pp gap at ~1.4 SE that the newer half of the sample cannot reproduce
# (posted -14.7% vs projected -15.9%), so a projected lineup caps a batter prop
# at Moderate rather than refusing it.
MATCHUP = "MIA @ ATL"


class _Identity:
    def apply(self, market: str, prob: float) -> float:
        return prob


def _pipeline() -> Pipeline:
    p = Pipeline.__new__(Pipeline)
    p.cfg = Config()
    p._calibrator = _Identity()
    p._shrink = None
    p._splits = {}
    p._ml_gate = MLSharpGate.from_env()
    p._pen_gate = MLPenGate.from_env()
    p._lineup_gate = LineupLockGate()
    p._lineup_lock = None
    p._drift_gate = DriftGate.from_env()
    p._open_board = {}
    return p


def _rec(p: Pipeline, market: str, selection: str, model_prob: float = 0.61):
    """A buy threading every batter screen: -130 at model .61 is Strong.

    Above the conviction floor and the EV floor, under the edge ceiling and the
    .62 batter probability ceiling, so the provenance cap is the only thing that
    can move the tier.
    """
    game = SimpleNamespace(game_date="2026-08-08", game_pk=1)
    quotes = {
        (MATCHUP, market, selection): [
            MarketQuote(book="dk", american=-130.0, opposite_american=110.0)
        ]
    }
    return p._mk(
        game, MATCHUP, "batter", market, selection, model_prob,
        team_side="away", side="over", quotes=quotes,
    )


def test_the_provenance_cap_ships_on() -> None:
    assert LineupLockGate.from_env().provenance_cap is True


def test_a_projected_batter_prop_is_capped() -> None:
    gate = LineupLockGate()
    cap, reason = gate.caps_at_moderate(
        gate.read(projected=True, hours=1.0), "batter_2b"
    )
    assert cap is True
    assert "projected lineup" in reason and "-9.9%" in reason


def test_a_posted_batter_prop_is_left_alone() -> None:
    gate = LineupLockGate()
    lock = gate.read(projected=False, hours=1.0)
    assert gate.caps_at_moderate(lock, "batter_2b") == (False, "")


def test_a_pitcher_prop_is_never_capped_on_a_projected_lineup() -> None:
    """Both provenances profit there, and a starter is named days ahead."""
    gate = LineupLockGate()
    lock = gate.read(projected=True, hours=1.0)
    assert gate.caps_at_moderate(lock, "pitcher_k")[0] is False
    assert gate.caps_at_moderate(lock, "game_ml")[0] is False


def test_an_unrecorded_provenance_is_not_a_projected_one() -> None:
    """Every backtest lands here: the cap refuses a measured status, not a gap."""
    assert LineupLockGate().caps_at_moderate(None, "batter_2b")[0] is False


def test_the_cap_is_switchable(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_LINEUP_PROVENANCE_CAP", "0")
    gate = LineupLockGate.from_env()
    lock = gate.read(projected=True, hours=1.0)
    assert gate.caps_at_moderate(lock, "batter_2b")[0] is False


# ---- the cap inside the pipeline ------------------------------------------
def test_a_strong_batter_buy_on_a_projected_lineup_becomes_moderate() -> None:
    p = _pipeline()
    p._lineup_lock = p._lineup_gate.read(projected=False, hours=1.0)
    posted = _rec(p, "batter_2b", "Some Batter o0.5 2B")
    assert posted.tier is Tier.STRONG

    p._lineup_lock = p._lineup_gate.read(projected=True, hours=1.0)
    projected = _rec(p, "batter_2b", "Some Batter o0.5 2B")
    assert projected.tier is Tier.MODERATE
    # The reason travels with the row, and _attach_context stamps the status the
    # ledger splits on, so the cap is gradeable against the buys it shrank.
    assert any("MODERATE cap" in r for r in projected.reasons)
