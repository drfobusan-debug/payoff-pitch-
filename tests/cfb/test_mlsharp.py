"""The moneyline sharp-money gate, and the splits feed it reads."""

from __future__ import annotations

from datetime import date

import pytest
import requests

from cfb_engine.data.vsin_splits import (
    Split,
    SplitsProvider,
    lookup,
    parse_splits,
    vsin_key,
)
from cfb_engine.market.mlsharp import GATE, SharpGate
from cfb_engine.schemas import Game, Slate, TeamGameInfo

DAY = date(2026, 8, 29)


def _page(rows: str) -> str:
    return f'<html><body><table class="sp-table">{rows}</table></body></html>'


def _row(gamecode: str, team: str, cells: list[str]) -> str:
    badges = "".join(f'<span class="sp-badge">{c}</span>' for c in cells)
    return (
        '<tr class="sp-row">'
        f'<td><button data-gamecode="{gamecode}"></button></td>'
        f'<td><a class="sp-team-link" href="/x">{team}</a></td>'
        f"{badges}</tr>"
    )


UNC = ["+7.5", "33%", "29%", "47.5", "34%", "56%", "+250", "27%", "16%"]
TCU = ["-7.5", "67%", "71%", "47.5", "66%", "44%", "-310", "73%", "84%"]
_GAME = _row("20260829CFB00215", "North Carolina Tar Heels", UNC) + _row(
    "20260829CFB00215", "TCU horned Frogs", TCU
)


def _slate() -> Slate:
    return Slate(
        slate_date=DAY,
        games=[
            Game(
                game_id="g1",
                game_date=DAY,
                home=TeamGameInfo(name="TCU Horned Frogs", abbrev="TCU", is_home=True),
                away=TeamGameInfo(
                    name="North Carolina Tar Heels", abbrev="UNC", is_home=False
                ),
            )
        ],
    )


# --- parsing ---------------------------------------------------------------
def test_a_team_row_carries_its_kickoff_date_and_all_three_markets() -> None:
    rows = parse_splits(_page(_GAME))
    assert [r.name for r in rows] == ["North Carolina Tar Heels", "TCU horned Frogs"]
    unc = rows[0]
    assert unc.game_date == DAY
    assert (unc.ml_american, unc.ml_handle, unc.ml_bets) == (250.0, 27.0, 16.0)
    assert (unc.spread_handle, unc.spread_bets) == (33.0, 29.0)
    assert (unc.total_handle, unc.total_bets) == (34.0, 56.0)


def test_a_market_the_page_leaves_blank_reads_as_no_split() -> None:
    cells = ["+38.5", "16%", "26%", "59.5", "94%", "82%", "-", "-", "-"]
    row = parse_splits(_page(_row("20260905CFB00229", "San Jose ST Spartans", cells)))[0]
    assert (row.ml_american, row.ml_handle, row.ml_bets) == (None, None, None)
    assert Split(row.ml_handle, row.ml_bets).divergence is None
    # A zero in one column beside a number in the other is a real reading.
    assert (row.spread_handle, row.spread_bets) == (16.0, 26.0)


def test_a_page_without_rows_is_empty_rather_than_an_error() -> None:
    assert parse_splits("<html><body>maintenance</body></html>") == []


def test_vsin_screen_spellings_resolve_to_the_boards_team_key() -> None:
    assert vsin_key("N Dakota ST") == vsin_key("North Dakota State")
    assert vsin_key("C Michigan Chippewas") == vsin_key("Central Michigan Chippewas")
    assert vsin_key("Middle Tenn") == vsin_key("Middle Tennessee Blue Raiders")


# --- assembling the book ---------------------------------------------------
def test_splits_are_filed_per_side_and_the_total_belongs_to_a_side_of_the_number(
    tmp_path,
) -> None:
    provider = SplitsProvider(tmp_path)
    (tmp_path / "vsin").mkdir()
    for source in ("circa", "DK"):
        (tmp_path / "vsin" / f"splits_{source}.html").write_text(_page(_GAME))

    book = provider.fetch(_slate())
    ml = lookup(book, "UNC @ TCU", "game_ml", "UNC ML")
    assert ml is not None and ml.divergence == pytest.approx(11.0)
    ats = lookup(book, "UNC @ TCU", "game_ats", "TCU -7.5")
    assert ats is not None and ats.divergence == pytest.approx(-4.0)
    # The visitor's row is the Over's; the home team's is the Under's.
    over = lookup(book, "UNC @ TCU", "game_total", "Over 45.5")
    assert over is not None and over.divergence == pytest.approx(-22.0)
    under = lookup(book, "UNC @ TCU", "game_total", "Under 45.5")
    assert under is not None and under.divergence == pytest.approx(22.0)


def test_a_placeholder_dead_heat_gives_way_to_a_book_that_published_a_split(
    tmp_path,
) -> None:
    """Circa prints identical shares (50/50, 100/100, 0/0) on markets it has no
    split for, and none of them can be told from a real dead heat -- so keeping
    them would hide DraftKings' actual reading behind a fabricated one."""
    provider = SplitsProvider(tmp_path)
    (tmp_path / "vsin").mkdir()
    placeholder = _row(
        "20260829CFB00215",
        "North Carolina Tar Heels",
        ["+7.5", "50%", "50%", "47.5", "0%", "0%", "+250", "100%", "100%"],
    )
    (tmp_path / "vsin" / "splits_circa.html").write_text(_page(placeholder))
    (tmp_path / "vsin" / "splits_DK.html").write_text(_page(_GAME))

    book = provider.fetch(_slate())
    ml = lookup(book, "UNC @ TCU", "game_ml", "UNC ML")
    assert ml is not None
    assert (ml.book, ml.divergence) == ("draftkings", pytest.approx(11.0))
    ats = lookup(book, "UNC @ TCU", "game_ats", "UNC +7.5")
    assert ats is not None and ats.book == "draftkings"
    assert lookup(book, "UNC @ TCU", "game_total", "Over 45.5") is not None


def test_the_same_teams_meeting_later_in_the_season_is_not_read_as_this_slate(
    tmp_path,
) -> None:
    """The page lists the whole season, so a name match alone prices the wrong game."""
    provider = SplitsProvider(tmp_path)
    (tmp_path / "vsin").mkdir()
    november = _row("20261121CFB00215", "North Carolina Tar Heels", UNC)
    for source in ("circa", "DK"):
        (tmp_path / "vsin" / f"splits_{source}.html").write_text(_page(november))
    assert provider.fetch(_slate()) == {}


def test_an_unreachable_page_costs_the_screen_its_opinion_not_the_slate_its_card(
    tmp_path, monkeypatch
) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        raise requests.RequestException("no network")

    monkeypatch.setattr("cfb_engine.data.vsin_splits.http.get", boom)
    assert SplitsProvider(tmp_path).fetch(_slate()) == {}


# --- the gate --------------------------------------------------------------
def test_money_keeping_pace_with_the_tickets_confirms_the_buy() -> None:
    keep, reason, gate = SharpGate().verdict(Split(27.0, 16.0, "circa"))
    assert (keep, gate) == (True, None)
    assert "confirms" in reason and "+11" in reason


def test_a_side_the_public_holds_more_heavily_than_the_money_is_refused() -> None:
    keep, reason, gate = SharpGate().verdict(Split(73.0, 84.0, "draftkings"))
    assert (keep, gate) == (False, GATE)
    assert "-11" in reason and "PASS" in reason


def test_an_exact_tie_is_kept_because_the_money_has_kept_pace() -> None:
    keep, _, gate = SharpGate().verdict(Split(50.0, 50.0))
    assert (keep, gate) == (True, None)


def test_a_missing_split_is_neutral_so_a_data_hole_is_not_a_veto() -> None:
    for split in (None, Split(None, None), Split(40.0, None)):
        keep, reason, gate = SharpGate().verdict(split)
        assert (keep, gate) == (True, None)
        assert "no public-money split" in reason


def test_the_gate_can_be_switched_off_and_then_only_annotates() -> None:
    keep, reason, gate = SharpGate(enabled=False).verdict(Split(73.0, 84.0))
    assert (keep, gate) == (True, None)
    assert "measuring" in reason


def test_the_threshold_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("CFBE_ML_MIN_DIVERGENCE", "10")
    gate = SharpGate.from_env()
    assert gate.min_divergence == pytest.approx(10.0)
    assert gate.verdict(Split(30.0, 25.0))[0] is False
    assert gate.verdict(Split(30.0, 15.0))[0] is True


def test_the_gate_ships_armed_and_reads_its_off_switch(monkeypatch) -> None:
    assert SharpGate.from_env().enabled is True
    monkeypatch.setenv("CFBE_ML_SHARP_GATE", "0")
    assert SharpGate.from_env().enabled is False
