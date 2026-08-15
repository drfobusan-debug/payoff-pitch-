"""VSiN's VOLT/JOLT cards, parsed and set beside our own picks.

The point of the column is disagreement. Two models on opposite sides of one
total mean one of them is wrong, and that is only visible if the join is on the
*bet* rather than on the selection string -- their Under 9.5 and our Under 9 are
one opinion about one game, and must not read as "no comparison available".
"""

from __future__ import annotations

from datetime import date as Date

import pytest
import requests

from mlb_engine.data import propicks
from mlb_engine.data.propicks import (
    ProPick,
    annotate,
    fetch,
    load_picks,
    merge_picks,
    parse_cards,
    save_picks,
)
from mlb_engine.market.tiers import Tier
from mlb_engine.output.card import build_cards, render_markdown
from mlb_engine.recommendations import Recommendation


def _card(
    *,
    league: str = "MLB",
    expert: str = "VSiN VOLT Model",
    edge: str = "7.9% EDGE",
    player: str = "White Sox at Tigers",
    label: str = "Game Total",
    side: str = "UNDER 7.5",
    side_class: str = "fp-side-under",
    market: str = "Total Runs",
    price: str = "-102",
    book: str = "DK",
    date: str = "Aug 15, 2026",
) -> str:
    return f"""
    <div class="fp-card">
      <div class="fp-band">
        <img src="/img/leagues/mlb.png" alt="{league}" class="fp-band-league-logo" />
        <span class="fp-band-expert">{expert}</span>
        <span class="fp-edge"><svg viewBox="0 0 10 10"></svg> {edge}</span>
      </div>
      <div class="fp-player-row">
        <div class="fp-player">{player}</div>
        <div class="fp-team-label">{label}</div>
      </div>
      <div class="fp-selection ">
        <span class="fp-side {side_class}">{side}</span>
        <span class="fp-market">{market}</span>
        <span class="fp-price">{price}</span>
      </div>
      <div class="fp-footer">
        <span class="fp-book">{book}</span>
        <span class="fp-date">{date}</span>
      </div>
    </div>
    """


def _rec(**kw) -> Recommendation:
    base = dict(
        game_date=Date(2026, 8, 15),
        game_pk=1,
        matchup="CWS @ DET",
        category="game",
        market="game_total",
        selection="Under 8.5",
        model_prob=0.55,
        line=8.5,
        side="under",
        tier=Tier.MODERATE,
        ev=0.05,
        edge=0.03,
        market_american=-105.0,
    )
    base.update(kw)
    return Recommendation(**base)


# --- parsing ---------------------------------------------------------------


def test_game_card_carries_side_line_price_and_matchup() -> None:
    (pick,) = parse_cards(_card(), "VOLT")
    assert pick.model == "VOLT"
    assert pick.league == "MLB"
    assert pick.date == "2026-08-15"
    assert pick.market == "game_total"
    assert pick.matchup == "CWS @ DET"
    assert (pick.side, pick.line, pick.price) == ("under", 7.5, -102.0)
    assert pick.edge == pytest.approx(0.079)
    assert pick.book == "DK"


def test_prop_card_keeps_the_player_and_leaves_the_matchup_empty() -> None:
    html = _card(
        expert="VSiN JOLT Model",
        player="Michael Busch",
        label="Chicago Cubs",
        market="Total Bases",
        side="OVER 1.5",
        side_class="fp-side-over",
        price="+134",
    )
    (pick,) = parse_cards(html, "JOLT")
    assert pick.market == "batter_tb"
    assert pick.subject == "Michael Busch"
    assert pick.matchup == ""
    assert (pick.side, pick.line, pick.price) == ("over", 1.5, 134.0)
    assert pick.summary == "JOLT OVER 1.5 (+134)"


def test_unmapped_market_is_kept_rather_than_dropped() -> None:
    """A label we do not recognise must show up in the capture, not vanish."""
    (pick,) = parse_cards(_card(market="Team Total Runs"), "VOLT")
    assert pick.raw_market == "Team Total Runs"
    assert pick.market == ""


def test_team_side_resolves_to_the_engine_abbreviation() -> None:
    html = _card(market="Run Line", side="Tigers -1.5", side_class="fp-side")
    (pick,) = parse_cards(html, "VOLT")
    assert (pick.market, pick.side, pick.line) == ("game_rl", "DET", -1.5)


def test_moneyline_card_has_a_team_and_no_line() -> None:
    html = _card(market="Moneyline", side="Tigers", side_class="fp-side")
    (pick,) = parse_cards(html, "VOLT")
    assert (pick.market, pick.side, pick.line) == ("game_ml", "DET", None)


def test_card_with_no_readable_side_is_dropped() -> None:
    assert parse_cards(_card(side="", market="Something Else"), "VOLT") == []


def test_nickname_pairs_that_need_the_map() -> None:
    for nick, abbrev in (
        ("Diamondbacks", "AZ"),
        ("White Sox", "CWS"),
        ("Athletics", "ATH"),
        ("Guardians", "CLE"),
    ):
        (pick,) = parse_cards(_card(player=f"{nick} at Braves"), "VOLT")
        assert pick.matchup == f"{abbrev} @ ATL"


def test_unknown_nickname_leaves_the_matchup_unset() -> None:
    (pick,) = parse_cards(_card(player="Fresno Grizzlies at Braves"), "VOLT")
    assert pick.matchup == ""


# --- fetching --------------------------------------------------------------


def test_fetch_reads_both_models_and_filters_by_league(monkeypatch) -> None:
    pages = {
        propicks.VOLT_URL: _card(),
        propicks.JOLT_URL: _card(
            league="NBA", expert="VSiN JOLT Model", player="Someone", market="Points"
        ),
    }

    class Resp:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(propicks.http, "get", lambda url, **kw: Resp(pages[url]))
    picks = fetch()
    assert [p.model for p in picks] == ["VOLT"]


def test_fetch_survives_a_dead_page(monkeypatch) -> None:
    def boom(url: str, **kw: object) -> object:
        raise requests.RequestException("down")

    monkeypatch.setattr(propicks.http, "get", boom)
    assert fetch() == []


# --- agreement -------------------------------------------------------------


def test_star_when_the_direction_matches_even_at_a_different_number() -> None:
    """Their Under 7.5 and our Under 8.5 are one opinion about one game."""
    rec = _rec()
    assert annotate([rec], parse_cards(_card(), "VOLT")) == 1
    assert rec.vsin_agrees is True
    assert rec.vsin_mark == "\u2605"
    assert rec.vsin_pick == "VOLT UNDER 7.5 (-102)"
    assert rec.vsin_edge == pytest.approx(0.079)


def test_cross_when_they_take_the_other_side() -> None:
    rec = _rec(selection="Over 8.5", side="over")
    annotate([rec], parse_cards(_card(), "VOLT"))
    assert rec.vsin_agrees is False
    assert rec.vsin_mark == "\u2717"


def test_a_game_they_have_no_pick_on_is_left_blank() -> None:
    rec = _rec(matchup="NYY @ BOS")
    assert annotate([rec], parse_cards(_card(), "VOLT")) == 0
    assert (rec.vsin_pick, rec.vsin_agrees, rec.vsin_mark) == (None, None, "")


def test_the_pick_only_reaches_the_market_it_was_made_on() -> None:
    """A total is not an opinion about the moneyline."""
    rec = _rec(market="game_ml", selection="DET ML", side="win")
    assert annotate([rec], parse_cards(_card(), "VOLT")) == 0


def test_prop_joins_on_the_player_not_the_selection_string() -> None:
    html = _card(
        expert="VSiN JOLT Model",
        player="Michael Busch",
        label="Chicago Cubs",
        market="Total Bases",
        side="OVER 1.5",
        price="+134",
    )
    rec = _rec(
        market="batter_tb",
        selection="Michael Busch TB o2.5",
        matchup="CHC @ KC",
        side="over",
        category="batter",
    )
    assert annotate([rec], parse_cards(html, "JOLT")) == 1
    assert rec.vsin_agrees is True


def test_accents_and_suffixes_do_not_break_the_prop_join() -> None:
    html = _card(
        expert="VSiN JOLT Model",
        player="Jos\u00e9 Ram\u00edrez Jr.",
        market="Total Bases",
        side="OVER 1.5",
    )
    rec = _rec(
        market="batter_tb",
        selection="Jose Ramirez TB o1.5",
        side="over",
        category="batter",
    )
    assert annotate([rec], parse_cards(html, "JOLT")) == 1


def test_team_side_agreement_is_compared_as_teams() -> None:
    picks = parse_cards(_card(market="Run Line", side="Tigers -1.5"), "VOLT")
    ours = _rec(market="game_rl", selection="DET +1.5", side="cover", line=1.5)
    theirs = _rec(market="game_rl", selection="CWS -1.5", side="cover", line=-1.5)
    annotate([ours, theirs], picks)
    assert (ours.vsin_agrees, theirs.vsin_agrees) == (True, False)


def test_unmapped_market_matches_nothing() -> None:
    picks = parse_cards(_card(market="Team Total Runs"), "VOLT")
    assert annotate([_rec()], picks) == 0


# --- persistence -----------------------------------------------------------


def test_save_and_load_round_trip(tmp_path) -> None:
    picks = parse_cards(_card(), "VOLT")
    path = tmp_path / "propicks_2026-08-15.json"
    save_picks(path, picks)
    assert load_picks(path) == picks


def test_load_tolerates_a_missing_or_corrupt_file(tmp_path) -> None:
    assert load_picks(tmp_path / "nope.json") == []
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_picks(bad) == []


def test_load_skips_rows_with_unexpected_fields(tmp_path) -> None:
    path = tmp_path / "mixed.json"
    path.write_text('[{"model": "VOLT"}, "nonsense"]', encoding="utf-8")
    assert load_picks(path) == []


def test_merge_keeps_both_days_and_replaces_a_restated_pick() -> None:
    old = parse_cards(_card(price="-102"), "VOLT")
    new = parse_cards(_card(price="-115"), "VOLT")
    other = parse_cards(_card(player="Mets at Braves"), "VOLT")
    merged = merge_picks(old, new + other)
    assert len(merged) == 2
    assert {p.price for p in merged} == {-115.0, -102.0}


# --- the reader-facing card ------------------------------------------------


def test_the_mark_reaches_the_written_card() -> None:
    rec = _rec(tier=Tier.STRONG, ev=0.08, edge=0.05)
    annotate([rec], parse_cards(_card(), "VOLT"))
    md = render_markdown(build_cards([rec]), Date(2026, 8, 15))
    assert "\u2605 VOLT UNDER 7.5 (-102)" in md


def test_the_card_says_nothing_when_vsin_had_no_opinion() -> None:
    md = render_markdown(build_cards([_rec(tier=Tier.STRONG, ev=0.08)]), Date(2026, 8, 15))
    assert "VOLT" not in md
    assert "\u2605" not in md


def test_the_pick_is_display_only(monkeypatch) -> None:
    """Nothing about our own number may move because VSiN agrees or disagrees."""
    rec = _rec()
    before = (rec.model_prob, rec.ev, rec.edge, rec.tier, rec.fair_prob, rec.bet_prob)
    annotate([rec], parse_cards(_card(side="OVER 7.5", side_class="fp-side-over"), "VOLT"))
    assert rec.vsin_agrees is False
    after = (rec.model_prob, rec.ev, rec.edge, rec.tier, rec.fair_prob, rec.bet_prob)
    assert before == after


def test_row_and_grid_expose_the_mark() -> None:
    rec = _rec()
    annotate([rec], parse_cards(_card(), "VOLT"))
    row = rec.as_row()
    assert row["VSiN"] == "\u2605"
    assert row["VSiN Pick"] == "VOLT UNDER 7.5 (-102)"
    assert row["VSiN Edge"] == 7.9


def test_pick_key_separates_two_models_on_one_game() -> None:
    volt = ProPick(
        model="VOLT",
        league="MLB",
        date="2026-08-15",
        subject="White Sox at Tigers",
        label="Game Total",
        raw_market="Total Runs",
        market="game_total",
        matchup="CWS @ DET",
        side="under",
        line=7.5,
        price=-102.0,
        edge=0.079,
        book="DK",
    )
    jolt = ProPick(**{**volt.__dict__, "model": "JOLT"})
    assert volt.key != jolt.key
    assert len(merge_picks([volt], [jolt])) == 2
