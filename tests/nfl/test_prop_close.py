"""The prop close: archived inside the hour before kickoff, attached at grading."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast

import pytest

from nfl_engine import cli, props, props_grade
from nfl_engine.data import capture
from nfl_engine.data.capture import QuoteRow
from nfl_engine.market.ev import ev
from nfl_engine.market.odds import american_to_decimal, american_to_prob

MATCHUP = "BUF @ KC"


def research_row(**overrides: str) -> dict[str, str]:
    row = {
        "captured_at": "2026-10-29T12:00:00Z",
        "season": "2026",
        "week": "8",
        "matchup": MATCHUP,
        "market": "player_receptions",
        "player": "Travis Kelce",
        "side": "over",
        "line": "5.5",
        "book": "draftkings",
        "american": "100.0",
        "stat": "receptions",
        "projection": "6.1",
        "model_prob": "0.58",
        "fair_prob": "0.50",
        "ev_fair": "0.0",
        "screens": "research_only",
        "basis": props.BASIS,
    }
    row.update(overrides)
    return row


def close_quote(american: float, opposite: float | None) -> QuoteRow:
    return QuoteRow(
        captured_at="2026-11-01T17:20:00Z",
        season=2026,
        week=8,
        game_date="2026-11-01",
        matchup=MATCHUP,
        market="player_receptions",
        side="over",
        line=5.5,
        book="draftkings",
        american=american,
        opposite_american=opposite,
        player="Travis Kelce",
    )


def test_rush_attempts_is_retired() -> None:
    assert "player_rush_attempts" in props.RETIRED_MARKETS
    assert "player_receptions" not in props.RETIRED_MARKETS


def test_grading_attaches_the_paired_close_and_its_ev(tmp_path) -> None:
    capture.write_snapshot(
        [close_quote(-120.0, 100.0)],
        season=2026,
        week=8,
        kind=capture.PROP_CLOSE_KIND,
        root=tmp_path,
    )
    closes = props_grade.prop_closes(2026, 8, root=tmp_path)
    graded = props_grade.grade_row(research_row(), {}, "now", closes)
    mine, theirs = american_to_prob(-120.0), american_to_prob(100.0)
    prob = mine / (mine + theirs)
    assert graded.close_american == -120.0
    assert graded.close_prob == pytest.approx(prob, abs=1e-6)
    assert graded.close_ev == pytest.approx(ev(prob, american_to_decimal(100.0)), abs=1e-6)
    assert graded.close_captured_at == "2026-11-01T17:20:00Z"


def test_an_unpaired_close_is_kept_but_not_devigged(tmp_path) -> None:
    capture.write_snapshot(
        [close_quote(-120.0, None)],
        season=2026,
        week=8,
        kind=capture.PROP_CLOSE_KIND,
        root=tmp_path,
    )
    graded = props_grade.grade_row(
        research_row(), {}, "now", props_grade.prop_closes(2026, 8, root=tmp_path)
    )
    assert graded.close_american == -120.0
    assert graded.close_prob is None and graded.close_ev is None


def test_a_row_at_another_line_gets_no_close(tmp_path) -> None:
    capture.write_snapshot(
        [close_quote(-120.0, 100.0)],
        season=2026,
        week=8,
        kind=capture.PROP_CLOSE_KIND,
        root=tmp_path,
    )
    graded = props_grade.grade_row(
        research_row(line="6.5"), {}, "now", props_grade.prop_closes(2026, 8, root=tmp_path)
    )
    assert graded.close_american is None


def test_close_ev_is_read_on_dk_mgm_leans_only() -> None:
    base = replace(
        props_grade.grade_row(research_row(), {}, "now"), result=props_grade.WIN, pnl=1.0
    )
    rows = [
        replace(base, close_ev=0.02),
        replace(base, book="fanduel", close_ev=0.5),
        replace(base, model_prob=0.40, close_ev=0.5),
        replace(base, close_ev=None),
    ]
    (tally,) = props_grade.tallies(rows)
    assert tally.close_n == 1 and tally.close_ev == pytest.approx(0.02)


class _Client:
    def __init__(self) -> None:
        self.games: list[Any] = []
        self.credits_remaining = None

    def fetch_props(self, games: list[Any], **_: object) -> list[QuoteRow]:
        self.games = list(games)
        return []


def test_the_prop_close_fetches_only_unstarted_games_inside_the_hour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client()
    monkeypatch.setattr(cli, "_client", lambda: client)
    now = datetime(2026, 11, 1, 17, 0, tzinfo=timezone.utc)
    games = [
        SimpleNamespace(kickoff_utc="2026-11-01T17:30:00Z"),  # due
        SimpleNamespace(kickoff_utc="2026-11-01T16:55:00Z"),  # started
        SimpleNamespace(kickoff_utc="2026-11-01T20:25:00Z"),  # later window
        SimpleNamespace(kickoff_utc=None),
    ]
    fetched = cli.Fetched(2026, 8, "2026-11-01T17:00:00Z", cast(Any, games), cast(Any, {}))
    cli._close_props(fetched, now, 45)
    assert [g.kickoff_utc for g in client.games] == ["2026-11-01T17:30:00Z"]


def test_the_close_command_takes_a_props_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(cli, "cmd_close", lambda args: seen.append(args) or 0)
    cli.main(["close", "--within", "45", "--props"])
    assert seen and seen[0].props is True and seen[0].within == 45
