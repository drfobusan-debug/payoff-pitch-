from __future__ import annotations

from datetime import date as Date
from datetime import datetime, timezone
from pathlib import Path

from nhl_engine.audit.ledger import LedgerRow, load_rows, write_once
from nhl_engine.cli import _local_cutoff
from nhl_engine.pipeline import starting_before


def _row(matchup: str, tag: str) -> LedgerRow:
    return LedgerRow(
        slate_date="2026-10-10",
        matchup=matchup,
        home=matchup[-3:],
        away=matchup[:3],
        market="game_ml",
        side=matchup[-3:],
        entity="",
        line=None,
        ot_rule="incl_ot_so",
        book="dk",
        american=-110.0,
        books=5,
        consensus=0.5,
        model_prob=0.5,
        push_prob=0.0,
        edge=0.0,
        ev=0.0,
        tier="Pass",
        gates=[],
        pass_gate=False,
        kelly=0.0,
        card_tag=tag,
    )


def test_cutoff_is_eastern_on_the_slate_date():
    cut = _local_cutoff(Date(2026, 10, 10), "17:30")
    assert cut.astimezone(timezone.utc) == datetime(2026, 10, 10, 21, 30, tzinfo=timezone.utc)


def test_starting_before_keeps_unstarted_games_before_the_cutoff():
    starts = {
        "PHI @ BOS": "2026-10-10T17:00:00Z",
        "EDM @ SJS": "2026-10-10T20:00:00Z",
        "MIN @ FLA": "2026-10-10T22:00:00Z",
        "UTA @ BUF": "2026-10-10T23:00:00Z",
        "ANA @ CGY": "",
    }
    cutoff = datetime(2026, 10, 10, 21, 30, tzinfo=timezone.utc)
    morning = datetime(2026, 10, 10, 15, 40, tzinfo=timezone.utc)
    assert starting_before(starts, cutoff, morning) == {"PHI @ BOS", "EDM @ SJS"}
    after_first = datetime(2026, 10, 10, 18, 0, tzinfo=timezone.utc)
    assert starting_before(starts, cutoff, after_first) == {"EDM @ SJS"}
    assert starting_before({"UTA @ BUF": "2026-10-10T23:00:00Z"}, cutoff, morning) == set()


def test_ledger_keeps_the_early_games_and_adds_the_rest(tmp_path: Path):
    path = tmp_path / "predictions.json"
    _, wrote = write_once([_row("PHI @ BOS", "early")], path)
    assert wrote
    later = [_row("PHI @ BOS", "initial"), _row("UTA @ BUF", "initial")]
    _, wrote = write_once(later, path)
    assert wrote
    rows = load_rows(path)
    assert [(r.matchup, r.card_tag) for r in rows] == [
        ("PHI @ BOS", "early"),
        ("UTA @ BUF", "initial"),
    ]
    _, wrote = write_once(later, path)
    assert not wrote and len(load_rows(path)) == 2
