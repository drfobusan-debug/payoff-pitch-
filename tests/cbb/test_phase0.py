from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from cbb_engine import cadence, cli, markets
from cbb_engine.config import Cadence
from cbb_engine.data import capture, espn, torvik, watch
from engine_runtime import quotes

UTC = timezone.utc
FIX = Path(__file__).parent / "fixtures"


def test_market_map_statuses_match_year1_plan() -> None:
    assert markets.LIVE_KEYS == ("game_total",)
    assert markets.BY_KEY["game_ats"].status == "research"
    assert markets.BY_KEY["h1_total"].status == "shadow"
    assert markets.BY_KEY["team_total"].status == "shadow"
    assert markets.BY_KEY["game_total_alt"].status == "shadow"
    assert all(markets.BY_PROVIDER[p].per_event is False for p in markets.FEATURED)
    assert markets.status_of("unknown") == "research"


def test_preregistered_yaml_has_fixed_rung_order_and_null_prior() -> None:
    text = (Path(__file__).parents[2] / "cbb_engine/params/preregistered.yaml").read_text()
    rungs = [
        "v0_shape",
        "foul_redistribution",
        "half_correlation",
        "blend_shift",
        "availability_residual",
    ]
    pos = [text.index(f"- {r}") for r in rungs]
    assert pos == sorted(pos)
    assert "all_zero, futures_weight: 0.0" in text
    assert "null_wins_unless_beaten_by_mde: true" in text
    for g in (
        "anchor_miss",
        "se_cap",
        "halves_sum",
        "team_totals_sum",
        "monotone_alt",
        "win_vs_cover",
    ):
        assert f"- {g}" in text


def test_cadence_clocks() -> None:
    cad = Cadence()
    now = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
    far, mid, near = now + timedelta(hours=8), now + timedelta(hours=4), now + timedelta(hours=1)
    assert cadence.board_interval(far, now, cad) == 60
    assert cadence.board_interval(mid, now, cad) == 30
    assert cadence.board_interval(near, now, cad) == 15
    assert cadence.slate_board_interval([far, mid], now, cad) == 30
    assert cadence.slate_board_interval([], now, cad) == 60
    assert cadence.featured_due([far], now, cad) is False and cadence.featured_due([near], now, cad)
    tips = {
        "a": now + timedelta(minutes=10),
        "b": now + timedelta(minutes=2),
        "c": now + timedelta(minutes=45),
    }
    assert cadence.close_due(tips, now, cad) == ["a"]
    assert cadence.due(None, 10, now) and not cadence.due(now - timedelta(minutes=9), 10, now)


def test_daily_budget_from_cadence_table() -> None:
    out = cadence.daily_budget(60, Cadence())
    # 24 + 6 + 12 board passes x 3 credits; 18 featured x 3; at most one 3-credit close per game
    assert out["board"] == 126 and out["featured"] == 54 and out["close"] == 180
    assert out["event_markets"] == 0 and out["total"] == 360


def _event() -> dict[str, object]:
    return json.loads((FIX / "oddsapi_event.json").read_text())


def test_payload_to_quotes_keeps_both_sides_and_drops_unknown_markets() -> None:
    rows = capture.rows_from_events(
        [_event()], captured_at="t", pass_id="board_1", pass_kind="board"
    )
    by = {(r.market, r.side, r.book): r for r in rows}
    over = by[("game_total", "over", "draftkings")]
    assert over.line == 145.5 and over.american == -108 and over.opposite_american == -112
    assert by[("game_ats", "home", "draftkings")].line == -3.5
    assert by[("game_ats", "away", "draftkings")].opposite_american == -110
    assert by[("game_ml", "away", "fanduel")].entity == "Duke Blue Devils"
    assert not any(r.market == "outrights" for r in rows)
    assert all(r.game_date == "2026-11-10" for r in rows)  # 00:00Z tip -> previous US evening
    assert capture.tips_of([_event()]) == {"ev1": "2026-11-11T00:00:00Z"}


def test_espn_scoreboard_parse_halves_and_neutral() -> None:
    games = espn.parse_scoreboard(json.loads((FIX / "espn_scoreboard.json").read_text()))
    assert len(games) == 2
    g = games[0]
    assert g.is_final and g.final == (71, 68) and g.away_halves == (33, 38) and g.neutral
    assert games[1].final is None and games[1].state == "STATUS_SCHEDULED"


def test_torvik_as_of_and_staleness(tmp_path: Path) -> None:
    d = date(2026, 11, 20)
    assert torvik.season_of(d) == 2027 and torvik.season_of(date(2027, 3, 1)) == 2027
    for day in ("2026-11-15", "2026-11-18", "2026-11-25"):
        p = torvik.file_path(tmp_path, "team_results", date.fromisoformat(day))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("1,Duke,ACC,5-0,120.1,1,90.2,1,0.95" + ",0" * 36 + "\n")
    got = torvik.as_of(tmp_path, "team_results", d)
    assert got.captured == date(2026, 11, 18) and got.stale_days == 2
    assert not got.usable(1) and got.usable(2)
    assert torvik.as_of(tmp_path, "super_sked", d).path is None
    assert got.path is not None
    assert torvik.read_csv(got.path)[0]["adjoe"] == "120.1"


def test_watch_tick_diffs_and_pulls_featured_forward(tmp_path: Path) -> None:
    d = date(2026, 11, 10)
    g = espn.Game("1", "t", "A", "B", "STATUS_SCHEDULED", False, False)
    assert watch.tick(tmp_path, d, [g], "<html>v1</html>", at="2026-11-10T20:00:00Z") == []
    assert watch.pull_forward(tmp_path, d) is False
    g2 = espn.Game("1", "t", "A", "B", "STATUS_POSTPONED", False, False)
    hits = watch.tick(tmp_path, d, [g2], "<html>v2</html>", at="2026-11-10T20:05:00Z")
    assert {h["kind"] for h in hits} == {"espn_status", "rotowire_change"}
    assert watch.pull_forward(tmp_path, d) is True and watch.pull_forward(tmp_path, d) is False
    assert len((tmp_path / "watch" / "2026-11-10" / "hits.jsonl").read_text().splitlines()) == 2


class FakeClient:
    """Serves the fixture; counts what a real client would have been charged."""

    def __init__(self) -> None:
        self.spent = 0
        self.credits_remaining = 999
        self.calls: list[str] = []

    def available(self) -> bool:
        return True

    def events(self, **params: str) -> list[dict[str, object]]:
        self.calls.append("events")
        return [_event()]

    def board(self, mk: tuple[str, ...], *, pass_id: str = "", **p: str) -> list[dict[str, object]]:
        self.calls.append("board")
        self.spent += len(mk)
        return [_event()]

    def event(
        self, eid: str, mk: tuple[str, ...], *, pass_id: str = "", **p: str
    ) -> dict[str, object] | None:
        self.calls.append(f"event:{eid}")
        self.spent += len(mk)
        return _event()


def test_capture_tick_runs_only_due_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENGINE_RUNTIME_DIR", str(tmp_path / "rt"))
    monkeypatch.setenv("CBBE_DATA_DIR", str(tmp_path / "data"))
    fake = FakeClient()
    monkeypatch.setattr(cli, "_client", lambda cache_ttl=None: fake)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return cls(2026, 11, 10, 23, 50, tzinfo=UTC)  # T-10 for the 00:00Z tip

    monkeypatch.setattr(cli, "datetime", Clock)
    assert cli.main(["capture", "--date", "2026-11-10"]) == 0
    man = quotes.read_manifest(tmp_path / "rt", "cbb", "2026-11-10")
    assert sorted(m["pass_kind"] for m in man) == ["board", "close", "featured"]
    assert fake.calls == ["events", "board", "board", "board"]
    close = next(m for m in man if m["pass_kind"] == "close")
    assert close["event_ids"] == ["ev1"]
    # second tick a minute later: nothing due (board 15, featured 10, close already taken)
    assert cli.main(["capture", "--date", "2026-11-10"]) == 0
    assert fake.calls[4:] == ["events"]
    assert quotes.verify(tmp_path / "rt", "cbb", "2026-11-10") == []
    assert not (tmp_path / "rt" / "ledger").exists()
