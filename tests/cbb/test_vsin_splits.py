from __future__ import annotations

from datetime import date as Date
from pathlib import Path

import pytest

from cbb_engine import bands
from cbb_engine.data import vsin_splits as vs
from engine_runtime import ledger

PREVIEW = '<div class="sp-preview-more"><b>30 more games</b> for VSiN Pro subscribers.</div>'


def _row(code: str, team: str, cells: list[str]) -> str:
    badges = "".join(f'<span class="sp-badge">{c}</span>' for c in cells)
    return (
        '<tr class="sp-row">'
        f'<td><button data-gamecode="{code}"></button></td>'
        f'<td><a class="sp-team-link" href="/x">{team}</a></td>'
        f"{badges}</tr>"
    )


def _page(*rows: str) -> str:
    return "<table>" + "".join(rows) + "</table>"


def _game(code: str, away: str, home: str, away_cells: list[str], home_cells: list[str]) -> str:
    return _row(code, away, away_cells) + _row(code, home, home_cells)


def test_parse_reads_both_rows_with_date_number_and_shares() -> None:
    html = _page(_game("20261108CBB00693", "Manhattan", "Maine",
                       ["+3.5", "85%", "19%", "136.5", "0%", "29% <span>v</span>", "+130", "3%", "21%"],
                       ["-3.5", "15%", "81%", "136.5", "100%", "71%", "-155", "97%", "79%"]))
    a, h = vs.parse_splits(html, "draftkings")
    assert (a.game_date, a.gamecode, a.team, a.book) == ("2026-11-08", "20261108CBB00693", "Manhattan", "draftkings")
    assert (a.spread_line, a.spread_handle, a.spread_bets) == (3.5, 85.0, 19.0)
    assert (a.total_line, a.total_handle, a.total_bets) == (136.5, 0.0, 29.0)
    assert (a.ml_american, a.ml_handle, a.ml_bets) == (130.0, 3.0, 21.0)
    assert (h.spread_line, h.ml_american, h.ml_bets) == (-3.5, -155.0, 79.0)


def test_gated_preview_page_parses_to_nothing() -> None:
    html = _page(_game("20261108CBB00941", "Iona", "ST Johns", ["+26.5"] + ["1%"] * 8, ["-26.5"] + ["1%"] * 8)) + PREVIEW
    assert vs.gated(html)
    assert vs.parse_splits(html, "draftkings") == []


def test_vsin_spellings_resolve_to_the_engine_names() -> None:
    assert vs.vsin_name("C Conn ST") == "Central Connecticut St"
    assert vs.vsin_name("N Dakota ST") == "North Dakota ST"
    assert vs.vsin_name("Texas-El Paso") == "UTEP"
    assert vs.vsin_name("William &amp; Mary") == "William & Mary"
    assert vs.vsin_name("ST Thomas-MN") == "ST Thomas MN"


def _splits(at: str, book: str = "draftkings", code: str = "20261108CBB00693", **over: object) -> list[vs.SplitRow]:
    base = dict(game_date="2026-11-08", gamecode=code, book=book, captured_at=at, pass_id="p",
                total_line=136.5, total_handle=30.0, total_bets=40.0, ml_american=130.0, ml_handle=3.0, ml_bets=21.0)
    away = vs.SplitRow(**{**base, "team": "Manhattan", "spread_line": 3.5, "spread_handle": 85.0, "spread_bets": 19.0})  # type: ignore[arg-type]
    home = vs.SplitRow(**{**base, "team": "Maine", "spread_line": -3.5, "spread_handle": 15.0, "spread_bets": 81.0, **over})  # type: ignore[arg-type]
    return [away, home]


TEAMS = {"e1": ("Maine Black Bears", "Manhattan Jaspers")}


def test_public_maps_sides_and_puts_the_visitor_on_the_over() -> None:
    pub = vs.public(_splits("2026-11-08T15:00:00Z"), TEAMS, {"e1": "2026-11-08T18:00:00Z"})["e1"]
    assert pub[("game_ats", "home")].bets_pct == 81.0 and pub[("game_ats", "home")].majority is True
    assert pub[("game_ats", "away")].majority is False
    assert pub[("game_total", "over")].bets_pct == 40.0 and pub[("game_total", "under")].bets_pct == 40.0
    assert pub[("game_ml", "home")].book == "draftkings"


def test_public_takes_the_latest_capture_before_tip_and_none_after() -> None:
    rows = _splits("2026-11-08T15:00:00Z") + _splits("2026-11-08T17:00:00Z", spread_bets=60.0) + _splits("2026-11-08T18:30:00Z", spread_bets=90.0)
    pub = vs.public(rows, TEAMS, {"e1": "2026-11-08T18:00:00Z"})["e1"]
    assert pub[("game_ats", "home")].bets_pct == 60.0
    assert vs.public(_splits("2026-11-08T18:30:00Z"), TEAMS, {"e1": "2026-11-08T18:00:00Z"}) == {}


def test_public_skips_placeholder_shares_so_the_other_book_is_read() -> None:
    circa = _splits("2026-11-08T15:00:00Z", book="circa", spread_bets=50.0)
    circa[0] = vs.SplitRow(**{**circa[0].__dict__, "spread_bets": 50.0})
    pub = vs.public(circa + _splits("2026-11-08T15:00:00Z"), TEAMS, {})["e1"]
    assert pub[("game_ats", "home")].book == "draftkings"
    assert pub[("game_ml", "home")].book == "circa"
    assert vs.Public(50.0, 90.0, "circa", "").majority is None


def test_public_needs_both_teams_of_one_game_code() -> None:
    rows = _splits("2026-11-08T15:00:00Z")[:1] + _splits("2026-11-08T15:00:00Z", code="20261108CBB00999")[1:]
    assert vs.public(rows, TEAMS, {}) == {}


def test_capture_round_trips_and_refuses_a_rewrite(tmp_path: Path) -> None:
    rows = _splits("2026-11-08T15:00:00Z")
    p = vs.write_capture(tmp_path, "2026-11-08", "p", rows)
    assert p is not None and vs.read_day(tmp_path, "2026-11-08") == rows
    assert vs.write_capture(tmp_path, "2026-11-08", "q", []) is None
    with pytest.raises(FileExistsError):
        vs.write_capture(tmp_path, "2026-11-08", "p", rows)


def test_fetch_without_a_token_or_gated_stores_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(vs.TOKEN_ENV, raising=False)
    assert vs.fetch(Date(2026, 11, 8), captured_at="t", pass_id="p") == []
    monkeypatch.setenv(vs.TOKEN_ENV, "tok")
    seen: list[str] = []

    class Resp:
        text = _page(_game("20261108CBB00941", "Iona", "ST Johns", ["+26.5"] + ["1%"] * 8, ["-26.5"] + ["1%"] * 8)) + PREVIEW

        def raise_for_status(self) -> None:
            pass

    def get(url: str, **kw: object) -> Resp:
        seen.append(str(kw["headers"]["Cookie"]))  # type: ignore[index]
        return Resp()

    monkeypatch.setattr("cbb_engine.data.vsin_splits.http.get", get)
    assert vs.fetch(Date(2026, 11, 8), captured_at="t", pass_id="p") == []
    assert seen == ["__utp=tok"] * 2


def _ledger(side: str, line: float, market: str = "game_ats") -> ledger.Row:
    return ledger.Row(slate_date="2026-11-08", event_id="e1", tip_utc="2026-11-08T18:00:00Z", market=market, side=side,
                      line=line, book="dk", pass_id="p", pass_kind="board", bet_of_record=True, status="live",
                      american=-110, p_fair=0.5)


def test_rlm_needs_a_move_against_the_ticket_majority_and_a_split_on_record() -> None:
    pub = {("game_ats", "home"): vs.Public(81.0, 15.0, "draftkings", "t"), ("game_ats", "away"): vs.Public(19.0, 85.0, "draftkings", "t")}
    ctx = bands.GameCtx(opener={("game_ats", "home"): -3.5, ("game_ats", "away"): 3.5}, public=pub)
    rlm = dict(bands.BANDS)["reverse line move (>= 1 against the ticket majority) ATS"]
    assert rlm(_ledger("away", 2.5), ctx)  # dog took 19% of tickets, number came to it
    assert not rlm(_ledger("home", -4.5), ctx)  # moved toward the public side: steam, not reverse
    assert not rlm(_ledger("away", 3.0), ctx)  # half a point is below the bar
    assert not rlm(_ledger("away", 2.5), bands.GameCtx(opener=ctx.opener))  # no split: plain movement only
    dog = dict(bands.BANDS)["reverse line move (>= 1 against the ticket majority) underdog ATS"]
    fav = dict(bands.BANDS)["reverse line move (>= 1 against the ticket majority) favourite ATS"]
    assert dog(_ledger("away", 2.5), ctx) and not fav(_ledger("away", 2.5), ctx)


def test_context_carries_public_shares_into_the_game() -> None:
    pub = {"e1": {("game_ats", "home"): vs.Public(81.0, 15.0, "draftkings", "t")}}
    ctx = bands.context(TEAMS, Date(2026, 11, 8), {}, [], {}, None, None, None, pub)
    assert ctx["e1"].public[("game_ats", "home")].bets_pct == 81.0
