"""Novig logging for CBB, NBA, NHL and NFL: capture, same-line pairing, and the report."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine_common import novig


class _Stop(Exception):
    pass


def _query_of(client: object, call: object) -> dict[str, str]:
    seen: dict[str, str] = {}

    def cache(url: str, query: dict[str, str]) -> None:
        seen.update(query)
        raise _Stop

    client._cache_path = cache  # type: ignore[attr-defined]
    with pytest.raises(_Stop):
        call()  # type: ignore[operator]
    return seen


@pytest.mark.parametrize("engine", ["nba", "nhl"])
def test_a_named_book_pull_asks_for_novig_and_the_card_pull_does_not(engine: str) -> None:
    mod = __import__(f"{engine}_engine.data.oddsapi", fromlist=["OddsAPIClient"])
    client = mod.OddsAPIClient("key")
    day = date(2026, 11, 20)
    q = _query_of(client, lambda: client.fetch_board(slate_date=day, bookmakers="novig"))
    assert q["bookmakers"] == "novig"
    q = _query_of(client, lambda: client.fetch_board(slate_date=day))
    assert "bookmakers" not in q


def test_the_nfl_named_book_pull_asks_for_novig() -> None:
    from nfl_engine.data.oddsapi import OddsAPIClient

    client = OddsAPIClient("key", cache_dir=None)
    first = date(2026, 11, 19)
    q = _query_of(
        client,
        lambda: client.fetch_board(season=2026, week=12, first_day=first, bookmakers="novig"),
    )
    assert q["bookmakers"] == "novig"
    q = _query_of(client, lambda: client.fetch_board(season=2026, week=12, first_day=first))
    assert "bookmakers" not in q


def test_only_pregame_quotes_are_kept_and_the_bet_time_quote_is_paired(tmp_path: Path) -> None:
    path = tmp_path / "novig.jsonl"
    k = novig.key("ev1", "game_total", "under")
    tip = "2026-11-20T00:00:00Z"
    kept = novig.append(
        path,
        [
            novig.record(k, 145.5, -104, -104, "2026-11-19T15:00:00Z", tip),
            novig.record(k, 145.5, -110, 100, "2026-11-19T23:00:00Z", tip),
            novig.record(k, 145.5, -150, 130, "2026-11-20T00:30:00Z", tip),  # in play
        ],
    )
    assert kept == 2
    index = novig.load(path)
    assert novig.as_of(index, k, 145.5, "2026-11-19T16:00:00Z")["american"] == -104  # type: ignore[index]
    assert novig.as_of(index, k, 145.5, "")["american"] == -110  # type: ignore[index]
    assert novig.as_of(index, k, 145.5, "2026-11-19T14:00:00Z") is None  # before any quote
    assert novig.as_of(index, k, 146.5, "2026-11-19T16:00:00Z") is None  # other line: unpaired


def test_a_line_novig_has_moved_off_is_not_paired_after_the_move(tmp_path: Path) -> None:
    path = tmp_path / "novig.jsonl"
    k, other = novig.key("ev1", "game_total", "under"), novig.key("ev2", "game_ml", "home")
    tip = "2026-11-20T00:00:00Z"
    novig.append(path, [novig.record(k, 145.5, -104, -104, "2026-11-19T15:00:00Z", tip)])
    novig.append(
        path,
        [
            novig.record(k, 146.5, -105, -103, "2026-11-19T17:00:00Z", tip),
            novig.record(other, None, 120, -125, "2026-11-19T19:00:00Z", tip),
        ],
    )
    index = novig.load(path)
    assert novig.as_of(index, k, 145.5, "2026-11-19T16:00:00Z")["american"] == -104  # type: ignore[index]
    assert novig.as_of(index, k, 145.5, "2026-11-19T18:00:00Z") is None  # moved to 146.5
    assert novig.as_of(index, k, 146.5, "2026-11-19T18:00:00Z")["american"] == -105  # type: ignore[index]
    assert novig.as_of(index, k, 146.5, "2026-11-19T20:00:00Z") is None  # not in the later pull


def test_novig_close_ev_reprices_the_book_row_at_the_same_line() -> None:
    # Book -110 worth -3% at the close; at even money the same close is worth more.
    at = novig.close_ev_at_novig(-110, -0.03, 100)
    assert at == pytest.approx(0.97 * 2.0 / (1 + 100 / 110) - 1)
    assert novig.close_ev_at_novig(-110, -0.03, -110) == pytest.approx(-0.03)


def test_the_comparison_clusters_by_game_and_holds_the_bar() -> None:
    pairs = [novig.Pair("game_total", f"g{i // 2}", True, -0.03, -0.01) for i in range(40)] + [
        novig.Pair("game_total", "gx", False, -0.04, None)
    ]
    rows = novig.compare(pairs)
    buys = rows[0]
    assert (buys.n, buys.of) == (40, 40)
    assert buys.gain == pytest.approx(0.02)
    assert buys.verdict == f"needs {novig.BAR_BUYS - 40} more"
    assert rows[-1].of == 41 and rows[-1].n == 40
    assert "Pre-registered" in novig.render(rows)
    assert novig.compare([])[0].verdict == "no Novig quotes yet"


def test_nba_capture_appends_only_novig_quotes_and_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nba_engine import cli

    row = SimpleNamespace(
        event_id="e1",
        market="game_ml",
        side="home",
        entity="",
        line=None,
        book="novig",
        american=-104.0,
        opposite_american=100.0,
    )
    other = SimpleNamespace(**{**vars(row), "book": "draftkings"})
    slate = SimpleNamespace(
        games=[SimpleNamespace(event_id="e1", start_utc="2026-11-21T00:00:00Z")]
    )
    seen: dict[str, object] = {}

    def fetch(**kw: object) -> tuple[object, list[object]]:
        seen.update(kw)
        return slate, [row, other]

    client = SimpleNamespace(fetch_board=fetch)
    cli._capture_novig(client, tmp_path, date(2026, 11, 20), 30, "2026-11-20T18:00:00Z")  # type: ignore[arg-type]
    assert seen["bookmakers"] == "novig"
    index = novig.load(tmp_path / "novig.jsonl")
    assert list(index.quotes) == [(novig.key("e1", "game_ml", "home", ""), None)]

    def boom(**kw: object) -> None:
        raise RuntimeError("down")

    cli._capture_novig(SimpleNamespace(fetch_board=boom), tmp_path, date(2026, 11, 20), 30, "x")  # type: ignore[arg-type]
    monkeypatch.setenv("NBAE_NOVIG", "0")
    cli._capture_novig(None, tmp_path, date(2026, 11, 20), 30, "x")  # type: ignore[arg-type]


def test_a_cbb_pass_id_carries_its_bet_time() -> None:
    from cbb_engine.cli import _pass_time

    assert _pass_time("featured_20261101T163017Z") == "2026-11-01T16:30:17Z"
    assert _pass_time("manual") == ""


def test_an_empty_book_placeholder_is_not_logged_as_a_price(tmp_path: Path) -> None:
    assert novig.tradable(-111, 100)
    assert novig.tradable(-178, 170)
    assert novig.tradable(-104, None)
    assert not novig.tradable(-9900, -100000)
    assert not novig.tradable(-99900, -99900)
    assert not novig.tradable(-212, -251)  # both sides favourites: no market
    k = novig.key("ev1", "game_ats", "home")
    tip = "2026-11-20T00:00:00Z"
    kept = novig.append(
        tmp_path / "n.jsonl",
        [
            novig.record(k, -32.5, -9900, -100000, "2026-11-19T15:00:00Z", tip),
            novig.record(k, -1.5, -111, 108, "2026-11-19T15:00:00Z", tip),
        ],
    )
    assert kept == 1
