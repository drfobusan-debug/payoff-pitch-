"""§4.e historical close buy: snapshot grid, cost plan, resumable pull, fit_width export — offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbb_engine import history
from cbb_engine.config import SPORT
from engine_runtime import quotes
from engine_runtime.odds_client import HIST_MULT

DAY = "2025-01-11"
TIPS = [f"{DAY}T17:00:00Z", f"{DAY}T17:02:00Z", f"{DAY}T19:30:00Z", "2025-01-12T00:00:00Z"]


def _event(eid: str, tip: str, home: str, away: str, total: float = 145.5, spread: float = -4.5) -> dict[str, object]:
    def book(key: str, over: float = -110, under: float = -110) -> dict[str, object]:
        return {
            "key": key,
            "markets": [
                {"key": "totals", "outcomes": [
                    {"name": "Over", "point": total, "price": over},
                    {"name": "Under", "point": total, "price": under},
                ]},
                {"key": "spreads", "outcomes": [
                    {"name": home, "point": spread, "price": -110},
                    {"name": away, "point": -spread, "price": -110},
                ]},
            ],
        }
    return {
        "id": eid, "commence_time": tip, "home_team": home, "away_team": away,
        "bookmakers": [book("pinnacle"), book("fanduel"), book("caesars"), book("draftkings", -115, -105)],
    }


class FakeHist:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def historical_board(self, markets: tuple[str, ...], *, at: str, pass_id: str = "", **p: str) -> tuple[str, list[dict[str, object]]]:
        self.calls.append(at)
        return at, [
            _event("e1", TIPS[0], "Duke Blue Devils", "North Carolina Tar Heels"),
            _event("e2", TIPS[1], "Kansas Jayhawks", "Baylor Bears", total=150.5, spread=-9.5),
            _event("e3", TIPS[2], "Gonzaga Bulldogs", "Saint Mary's Gaels", total=138.0),
            _event("e4", TIPS[3], "UCLA Bruins", "USC Trojans"),
        ]


def test_close_times_one_snapshot_per_grid_cell_and_dense_grid() -> None:
    assert history.close_times(TIPS) == [
        f"{DAY}T16:50:00Z", f"{DAY}T19:20:00Z", f"{DAY}T23:50:00Z",
    ]
    dense = history.dense_times([TIPS[0]])
    assert dense[0] == f"{DAY}T14:00:00Z" and dense[-1] == f"{DAY}T16:50:00Z" and len(dense) == 18


def test_plan_costs_from_grid_before_any_spend() -> None:
    pl = history.plan({DAY: TIPS}, dense_days=(DAY,))
    assert pl["credits_per_snapshot"] == HIST_MULT * 2
    d = pl["days"][DAY]  # type: ignore[index]
    assert d["close_snapshots"] == 3 and d["dense_snapshots"] > 3
    assert pl["credits"] == HIST_MULT * 2 * (d["close_snapshots"] + d["dense_snapshots"])
    assert history.plan({DAY: TIPS})["days"][DAY]["dense_snapshots"] == 0  # type: ignore[index]


def test_pull_writes_close_shards_per_window_and_resumes(tmp_path: Path) -> None:
    fake = FakeHist()
    c = history.pull_day(fake, tmp_path, DAY, TIPS)
    assert c["snapshots"] == 3 and c["skipped"] == 0
    assert fake.calls == history.close_times(TIPS)
    man = quotes.read_manifest(tmp_path, SPORT, DAY)
    assert [m["pass_kind"] for m in man] == ["close"] * 3 and all(m["hist"] for m in man)
    # each snapshot keeps only the games tipping in its window; the 00:00Z tip is this slate (ET)
    by_pass = {str(m["pass_id"]): sorted(str(k) for k in dict(m["teams"])) for m in man}  # type: ignore[call-overload]
    assert by_pass[history._pass_id("close", f"{DAY}T16:50:00Z")] == ["e1", "e2"]
    assert by_pass[history._pass_id("close", f"{DAY}T23:50:00Z")] == ["e4"]
    again = history.pull_day(fake, tmp_path, DAY, TIPS)
    assert again["snapshots"] == 0 and again["skipped"] == 3 and len(fake.calls) == 3


def _scoreboard(data_dir: Path) -> None:
    def comp(home: str, away: str, hs: int, as_: int) -> dict[str, object]:
        return {
            "id": home[:3], "date": TIPS[0], "status": {"type": {"name": "STATUS_FINAL"}},
            "competitions": [{"competitors": [
                {"homeAway": "home", "team": {"displayName": home}, "linescores": [{"value": hs}]},
                {"homeAway": "away", "team": {"displayName": away}, "linescores": [{"value": as_}]},
            ]}],
        }
    payload = {"events": [
        comp("Duke Blue Devils", "North Carolina Tar Heels", 80, 70),
        comp("Kansas Jayhawks", "Baylor Bears", 77, 61),
    ]}
    p = data_dir / "espn" / DAY / "scoreboard_x.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(payload))


def test_export_joins_consensus_close_to_espn_final(tmp_path: Path) -> None:
    history.pull_day(FakeHist(), tmp_path / "rt", DAY, TIPS)
    _scoreboard(tmp_path / "data")
    rows = history.export_day(tmp_path / "rt", tmp_path / "data", DAY)
    by = {r["event_id"]: r for r in rows}
    assert set(by) == {"e1", "e2", "e3", "e4"}
    assert by["e1"]["close_total"] == 145.5 and by["e1"]["final_total"] == 150
    assert by["e2"]["spread_abs"] == 9.5 and by["e2"]["paired_books"] == 3  # DK excluded
    assert by["e3"]["final_total"] == ""  # no ESPN final -> left blank, never guessed
    n = history.write_csv(rows, tmp_path / "out.csv")
    text = (tmp_path / "out.csv").read_text().splitlines()
    assert n == 4 and text[0].split(",")[:1] == ["game_date"] and "close_total" in text[0]


def test_cli_plan_and_pull_without_yes_spend_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from cbb_engine import cli

    monkeypatch.setenv("ENGINE_RUNTIME_DIR", str(tmp_path / "rt"))
    monkeypatch.setenv("CBBE_DATA_DIR", str(tmp_path / "data"))
    _scoreboard(tmp_path / "data")
    monkeypatch.setattr(cli, "_client", lambda cache_ttl=None: pytest.fail("must not build a paid client"))
    assert cli.main(["history", "pull", "--start", DAY, "--end", DAY]) == 0
    out = capsys.readouterr()
    assert '"credits"' in out.out and "re-run with --yes" in out.err
    assert not (tmp_path / "rt").exists()


def test_match_final_folds_odds_api_abbreviations() -> None:
    from cbb_engine.data import espn

    def game(eid: str, away: str, home: str, final: tuple[int, int]) -> espn.Game:
        return espn.Game(
            espn_id=eid, start_utc="", away=away, home=home, state="STATUS_FINAL", neutral=False,
            conference_game=False, away_halves=(final[0],), home_halves=(final[1],),
        )

    games = [
        game("1", "Colorado State Rams", "Michigan State Spartans", (70, 71)),
        game("2", "Saint Mary's Gaels", "North Carolina Tar Heels", (60, 80)),
    ]
    assert history.match_final("Michigan St Spartans", "Colorado St Rams", games) == (70, 71)
    assert history.match_final("N Carolina Tar Heels", "St Mary's Gaels", games) == (60, 80)
    assert history.match_final("Michigan St Spartans", "St Mary's Gaels", games) is None
    assert history.match_final("Duke Blue Devils", "Kansas Jayhawks", games) is None
