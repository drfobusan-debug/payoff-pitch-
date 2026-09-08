"""What a refit must not throw away, and what it must actually ship.

Both are consequences of retiring maps per basis: ``load_calibrator`` hides the
stale curves from the refit, and the refit used to write the file from what it
could see.
"""

from __future__ import annotations

import argparse
import json
from datetime import date as Date
from pathlib import Path

import pytest

from mlb_engine import cli
from mlb_engine.calibration import FEATURE_BASIS, FEATURE_BASIS_SINCE, Calibrator


def _ledger(path: Path) -> None:
    """Two graded slates on the current basis: one to train on, one to hold out."""
    day_one = FEATURE_BASIS_SINCE
    day_two = day_one.replace(day=day_one.day + 1)
    header = (
        "date,matchup,category,market,selection,line,book,odds,tier,model_prob,ev,"
        "result,pnl,raw_prob,fair_prob\n"
    )
    rows = []
    for day in (day_one, day_two):
        for i in range(400):
            # Raw 0.80 wins only half the time: a curve exists to be found.
            won = "win" if i % 2 else "loss"
            rows.append(
                f"{day.isoformat()},AAA@BBB,batter,batter_hr,Bat {i} HR o0.5,0.5,dk,"
                f"-110,Pass,0.8,0.3,{won},0,0.8,0.5\n"
            )
    path.write_text(header + "".join(rows))


def _stale_map(path: Path) -> None:
    """A map file whose curves were fitted on features the engine no longer has."""
    path.write_text(
        json.dumps(
            {
                "basis": "some-older-basis",
                "markets": {
                    "pitcher_k": {"x": [0.1, 0.9], "y": [0.1, 0.7], "basis": "some-older-basis"},
                    "batter_hr": {"x": [0.1, 0.9], "y": [0.1, 0.6], "basis": "some-older-basis"},
                },
                "default": {"x": [0.1, 0.9], "y": [0.1, 0.8]},
            }
        )
    )


@pytest.fixture()
def refit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run ``calibrate`` over a two-slate ledger against a stale live map."""
    monkeypatch.setenv("MLBE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MLBE_CALIB_MIN_SAMPLES", "10000")  # nothing earns its own curve
    (tmp_path / "audit").mkdir(parents=True, exist_ok=True)
    _ledger(tmp_path / "audit" / "ledger.csv")
    live = tmp_path / "calibration_live.json"
    _stale_map(live)
    args = argparse.Namespace(holdout=1, min_holdout=1, force=True, revalidate=None)
    assert cli.cmd_calibrate(args) == 0
    return live


def test_a_refit_keeps_the_curves_it_did_not_replace(refit: Path) -> None:
    """Losing them would put them beyond ``calibrate --revalidate``'s reach."""
    written = json.loads(refit.read_text())
    assert set(written["markets"]) >= {"pitcher_k"}
    assert written["markets"]["pitcher_k"]["basis"] == "some-older-basis"
    # Carried at their old basis, so pricing still refuses them.
    assert "pitcher_k" in Calibrator.from_json(refit).retired


def test_a_refit_ships_the_pooled_curve_it_was_measured_on(refit: Path) -> None:
    """An adopted market too thin for its own map prices off the new pooled fit."""
    written = json.loads(refit.read_text())
    assert written["default"]["x"], "the pooled curve must not be the retired empty one"
    cal = Calibrator.from_json(refit)
    # batter_hr was adopted, so its stale own-map is gone and the pooled curve
    # applies -- and that curve knows the raw 0.80 only won half its rows.
    assert "batter_hr" not in written["markets"]
    assert cal.apply("batter_hr", 0.8) < 0.7


def test_the_pooled_curve_is_not_borrowed_from_the_retired_file(refit: Path) -> None:
    """The old file's default was fitted on features that are gone."""
    written = json.loads(refit.read_text())
    assert written["basis"] == FEATURE_BASIS
    # The retired file's default read 0.8 -> 0.8; the graded rows say 0.8 -> ~0.5.
    pooled = Calibrator.from_json(refit).default
    assert pooled.apply(0.8) == pytest.approx(0.5, abs=0.05)


def test_revalidate_re_stamps_without_discarding_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A survivor comes back onto the current basis; nothing else is touched."""
    monkeypatch.setenv("MLBE_DATA_DIR", str(tmp_path))
    (tmp_path / "audit").mkdir(parents=True, exist_ok=True)
    _ledger(tmp_path / "audit" / "ledger.csv")
    live = tmp_path / "calibration_live.json"
    live.write_text(
        json.dumps(
            {
                "basis": FEATURE_BASIS,
                "markets": {
                    # Knows the raw 0.8 is really 0.5, so it beats the raw prob.
                    "batter_hr": {"x": [0.2, 0.8], "y": [0.2, 0.5], "basis": "older"},
                    "pitcher_k": {"x": [0.2, 0.8], "y": [0.2, 0.4], "basis": "older"},
                },
                "default": {"x": [0.2, 0.8], "y": [0.2, 0.5]},
            }
        )
    )
    args = argparse.Namespace(
        holdout=1, min_holdout=1, force=False, revalidate=FEATURE_BASIS_SINCE.isoformat()
    )
    assert cli.cmd_calibrate(args) == 0
    after = json.loads(live.read_text())
    assert after["markets"]["batter_hr"]["basis"] == FEATURE_BASIS
    # No graded pitcher_k rows to measure, so it stays as it was rather than vanishing.
    assert after["markets"]["pitcher_k"]["basis"] == "older"
    assert after["default"]["y"] == [0.2, 0.5], "the pooled curve was measured; keep it"


def _paired_ledger(path: Path) -> None:
    """Two slates of complementary prop pairs whose raw probability is honest.

    The over is priced at 0.5 and wins half the time, so any refit curve is
    fitting noise -- the holdout delta must land inside its own standard error.
    """
    day_one = FEATURE_BASIS_SINCE
    day_two = day_one.replace(day=day_one.day + 1)
    header = (
        "date,matchup,category,market,selection,line,book,odds,tier,model_prob,ev,"
        "result,pnl,raw_prob,fair_prob\n"
    )
    rows = []
    for day in (day_one, day_two):
        for i in range(300):
            over_won = i % 2 == 1
            for side, prob, won in (
                ("o0.5", 0.5, over_won),
                ("u0.5", 0.5, not over_won),
            ):
                rows.append(
                    f"{day.isoformat()},AAA@BBB,batter,batter_hr,Bat {i} HR {side},0.5,dk,"
                    f"-110,Pass,{prob},0.0,{'win' if won else 'loss'},0,{prob},0.5\n"
                )
    path.write_text(header + "".join(rows))


def test_a_market_inside_its_own_noise_is_not_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A smaller Brier is not a better map; it has to clear a standard error."""
    monkeypatch.setenv("MLBE_DATA_DIR", str(tmp_path))
    (tmp_path / "audit").mkdir(parents=True, exist_ok=True)
    _paired_ledger(tmp_path / "audit" / "ledger.csv")
    live = tmp_path / "calibration_live.json"
    args = argparse.Namespace(holdout=1, min_holdout=1, force=False, revalidate=None)
    assert cli.cmd_calibrate(args) == 1
    assert not live.exists(), "nothing beat its incumbent, so nothing is written"


def test_the_holdout_counts_props_not_complements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both sides of a prop are one piece of evidence about the refit, not two."""
    monkeypatch.setenv("MLBE_DATA_DIR", str(tmp_path))
    (tmp_path / "audit").mkdir(parents=True, exist_ok=True)
    _paired_ledger(tmp_path / "audit" / "ledger.csv")
    args = argparse.Namespace(holdout=1, min_holdout=1, force=True, revalidate=None)
    assert cli.cmd_calibrate(args) == 0
    out = capsys.readouterr().out
    assert "(300 props)" in out, out
    assert "trained on 600 rows" in out, "the fit still sees both sides"


def test_calibration_source_prefers_the_local_refit(tmp_path: Path) -> None:
    from mlb_engine.pipeline import _CALIBRATION_FILE, calibration_source

    live = tmp_path / "calibration_live.json"
    assert calibration_source(live) == _CALIBRATION_FILE
    _stale_map(live)
    assert calibration_source(live) == live


def test_grading_window_is_the_current_basis_only() -> None:
    """Guard the assumption the fixture leans on."""
    assert FEATURE_BASIS_SINCE < Date.today()
