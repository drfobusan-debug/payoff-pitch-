"""The prop layer's promises: research only, prior weeks only, no invented partner."""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_engine import props
from nfl_engine.data.capture import PROP_KIND, QuoteRow
from nfl_engine.features import usage
from nfl_engine.models.player import (
    CARRIES,
    MIN_GAMES,
    RECEPTIONS,
    RUSHING_YARDS,
    Projection,
    count_prob,
    prob_over,
    shrunk_mean,
    yards_prob,
)


def quote(
    *,
    player: str = "Puka Nacua",
    market: str = "player_receptions",
    side: str = "over",
    line: float | None = 4.5,
    book: str = "draftkings",
    american: float = -115.0,
    opposite: float | None = 105.0,
    matchup: str = "SEA @ LA",
) -> QuoteRow:
    return QuoteRow(
        captured_at="2026-09-10T15:00:00Z",
        season=2026,
        week=2,
        game_date="2026-09-13",
        matchup=matchup,
        market=market,
        side=side,
        line=line,
        book=book,
        american=american,
        opposite_american=opposite,
        player=player,
        event_id="evt",
        source="oddsapi",
    )


def projection(
    *,
    player: str = "Puka Nacua",
    stat: str = RECEPTIONS,
    mean: float = 6.4,
    games: int = 6,
    team: str = "LA",
) -> Projection:
    return Projection(
        player=player,
        player_id="00-0001",
        position="WR",
        team=team,
        stat=stat,
        games=games,
        mean=mean,
        prior_mean=5.9,
    )


def book_of(name: str, american: float, opposite: float | None) -> QuoteRow:
    return quote(book=name, american=american, opposite=opposite)


def projections_for(*items: Projection) -> dict[tuple[str, str], Projection]:
    return {(usage.normalise(p.player), p.stat): p for p in items}


class TestResearchOnly:
    def test_every_row_is_research_and_stamped(self) -> None:
        rows = [book_of("draftkings", -115, 105), book_of("fanduel", -110, -110)]
        priced = props.price_props(rows, projections_for(projection()))
        assert priced
        for prop in priced:
            assert props.RESEARCH_ONLY in prop.screens
            assert prop.mode == props.RESEARCH
            assert prop.basis == props.BASIS

    def test_research_veto_is_first_and_never_alone_removed(self) -> None:
        rows = [book_of("draftkings", 100, -120), book_of("fanduel", 102, -122)]
        priced = props.price_props(rows, projections_for(projection(mean=9.0)))
        assert priced[0].screens[0] == props.RESEARCH_ONLY

    def test_research_rows_go_to_their_own_file_not_the_ledger(self, tmp_path) -> None:
        priced = props.price_props(
            [book_of("draftkings", -115, 105)], projections_for(projection())
        )
        path = props.write_research(priced, season=2026, week=2, root=tmp_path)
        assert path is not None
        assert path.name == "research_2026_wk02.csv"
        assert not (tmp_path / "nfl_ledger.csv").exists()
        assert "research_only" in path.read_text()

    def test_unwritable_root_reports_instead_of_raising(self, tmp_path) -> None:
        blocked = tmp_path / "blocked"
        blocked.write_text("not a directory")
        priced = props.price_props(
            [book_of("draftkings", -115, 105)], projections_for(projection())
        )
        assert props.write_research(priced, season=2026, week=2, root=blocked) is None


class TestRetirement:
    @pytest.mark.parametrize(
        "market",
        ["player_pass_attempts", "player_pass_completions", "player_pass_yds", "player_rush_yds"],
    )
    def test_markets_worse_than_the_base_rate_are_retired(self, market: str) -> None:
        stat = props.stat_for(market)
        assert stat is not None
        rows = [
            quote(market=market, player="Matthew Stafford", line=250.5, book="draftkings"),
            quote(market=market, player="Matthew Stafford", line=250.5, book="fanduel"),
        ]
        priced = props.price_props(
            rows, projections_for(projection(player="Matthew Stafford", stat=stat, mean=260.0))
        )
        assert all(props.RETIRED in prop.screens for prop in priced)

    def test_a_surviving_market_is_not_retired(self) -> None:
        rows = [book_of("draftkings", -115, 105), book_of("fanduel", -110, -110)]
        priced = props.price_props(rows, projections_for(projection()))
        assert all(props.RETIRED not in prop.screens for prop in priced)

    def test_unknown_market_is_dropped_not_priced_blind(self) -> None:
        priced = props.price_props(
            [quote(market="player_field_goals")], projections_for(projection())
        )
        assert priced == []


class TestPairing:
    def test_an_unpaired_quote_gets_no_invented_partner(self) -> None:
        rows = [book_of("draftkings", -115, None), book_of("fanduel", -110, None)]
        priced = props.price_props(rows, projections_for(projection()))
        assert all(prop.fair_prob is None for prop in priced)
        assert all(props.UNPAIRED in prop.screens for prop in priced)

    def test_one_paired_book_is_thin(self) -> None:
        priced = props.price_props(
            [book_of("draftkings", -115, 105)], projections_for(projection())
        )
        assert all(props.THIN in prop.screens for prop in priced)

    def test_two_paired_books_clear_the_pairing_screens(self) -> None:
        rows = [book_of("draftkings", -115, 105), book_of("fanduel", -110, -110)]
        priced = props.price_props(rows, projections_for(projection()))
        assert all(prop.paired_books == 2 for prop in priced)
        assert all(props.THIN not in prop.screens for prop in priced)
        assert all(props.UNPAIRED not in prop.screens for prop in priced)

    def test_the_best_price_wins_the_position(self) -> None:
        rows = [book_of("draftkings", -115, 105), book_of("fanduel", 100, -120)]
        priced = props.price_props(rows, projections_for(projection()))
        assert len(priced) == 1
        assert priced[0].book == "fanduel"


class TestScreens:
    def test_no_projection_is_a_refusal_not_a_guess(self) -> None:
        priced = props.price_props([book_of("draftkings", -115, 105)], {})
        assert all(props.NO_PROJECTION in prop.screens for prop in priced)
        assert all(prop.projection is None for prop in priced)

    def test_below_the_usage_floor_is_a_refusal(self) -> None:
        priced = props.price_props(
            [book_of("draftkings", -115, 105)], projections_for(projection(mean=1.0))
        )
        assert all(props.BELOW_FLOOR in prop.screens for prop in priced)

    def test_a_longshot_price_is_capped(self) -> None:
        rows = [
            quote(american=450.0, opposite=-600.0, line=9.5, book="draftkings"),
            quote(american=440.0, opposite=-580.0, line=9.5, book="fanduel"),
        ]
        priced = props.price_props(rows, projections_for(projection()))
        assert all(props.LONGSHOT in prop.screens for prop in priced)

    def test_a_wild_disagreement_reads_as_our_error(self) -> None:
        rows = [
            quote(american=-110.0, opposite=-110.0, book="draftkings"),
            quote(american=-110.0, opposite=-110.0, book="fanduel"),
        ]
        priced = props.price_props(rows, projections_for(projection(mean=14.0)))
        assert all(props.DISAGREES in prop.screens for prop in priced)

    def test_no_execution_edge_when_the_price_is_the_consensus(self) -> None:
        rows = [
            quote(american=-110.0, opposite=-110.0, book="draftkings"),
            quote(american=-110.0, opposite=-110.0, book="fanduel"),
        ]
        priced = props.price_props(rows, projections_for(projection()))
        assert all(props.NO_EDGE in prop.screens for prop in priced)

    def test_execution_edge_and_model_disagreement_stay_separate(self) -> None:
        rows = [book_of("draftkings", -115, 105), book_of("fanduel", -110, -110)]
        prop = props.price_props(rows, projections_for(projection()))[0]
        assert prop.ev_fair is not None
        assert prop.edge_vs_fair is not None
        assert prop.ev_fair != prop.edge_vs_fair


class TestCorrelation:
    """Rows built to clear every other screen, so the guard is what stops them.

    An outlier book quoting one side only is the shape that produces execution
    edge: the de-vigged consensus comes from the paired books, and the price taken
    is the outlier's.
    """

    def _clean(self, player: str, market: str, line: float) -> list[QuoteRow]:
        return [
            quote(
                player=player, market=market, line=line, american=150.0, opposite=None, book="ci"
            ),
            quote(player=player, market=market, line=line, american=-110, opposite=-110, book="dk"),
            quote(player=player, market=market, line=line, american=-108, opposite=-112, book="fd"),
        ]

    def _priced(self, rows: list[QuoteRow], *projections: Projection) -> list[props.PricedProp]:
        return props.price_props(rows, projections_for(*projections))

    def test_the_fixture_clears_every_screen_but_research(self) -> None:
        priced = self._priced(
            self._clean("Puka Nacua", "player_receptions", 4.5), projection(mean=4.7)
        )
        assert [p.screens for p in priced] == [(props.RESEARCH_ONLY,)]

    def test_one_leg_per_player(self) -> None:
        rows = self._clean("Puka Nacua", "player_receptions", 4.5) + self._clean(
            "Puka Nacua", "player_rush_attempts", 12.5
        )
        priced = self._priced(
            rows,
            projection(mean=4.7),
            projection(player="Puka Nacua", stat=CARRIES, mean=13.0),
        )
        assert [props.DUPLICATE_PLAYER in p.screens for p in priced].count(True) == 1

    def test_one_leg_per_team_and_direction(self) -> None:
        rows = self._clean("Puka Nacua", "player_receptions", 4.5) + self._clean(
            "Kyren Williams", "player_rush_attempts", 12.5
        )
        priced = self._priced(
            rows,
            projection(mean=4.7),
            projection(player="Kyren Williams", stat=CARRIES, mean=13.0),
        )
        assert [props.CORRELATED in p.screens for p in priced].count(True) == 1

    def test_a_correlated_leg_stays_a_row(self) -> None:
        rows = self._clean("Puka Nacua", "player_receptions", 4.5) + self._clean(
            "Kyren Williams", "player_rush_attempts", 12.5
        )
        priced = self._priced(
            rows,
            projection(mean=4.7),
            projection(player="Kyren Williams", stat=CARRIES, mean=13.0),
        )
        assert len(priced) == 2

    def test_vetoed_rows_do_not_consume_the_team_slot(self) -> None:
        rows = [quote(player="Tyler Higbee", market="player_receptions", line=2.5, book="dk")]
        rows += self._clean("Puka Nacua", "player_receptions", 4.5)
        priced = self._priced(
            rows, projection(mean=4.7), projection(player="Tyler Higbee", mean=3.4)
        )
        nacua = [p for p in priced if p.player == "Puka Nacua"][0]
        higbee = [p for p in priced if p.player == "Tyler Higbee"][0]
        assert props.THIN in higbee.screens
        assert props.CORRELATED not in higbee.screens
        assert nacua.screens == (props.RESEARCH_ONLY,)


class TestDistributions:
    def test_an_integer_count_line_pushes(self) -> None:
        prob = count_prob(4.6, 2.1, 4.0)
        assert prob.push > 0.0

    def test_a_half_count_line_cannot_push(self) -> None:
        assert count_prob(4.6, 2.1, 4.5).push == 0.0

    def test_counts_are_overdispersed_relative_to_poisson(self) -> None:
        wide = count_prob(10.0, 6.0, 14.5).conditional
        tight = count_prob(10.0, 3.2, 14.5).conditional
        assert wide > tight

    def test_yardage_is_right_skewed(self) -> None:
        """More weight above 2x the mean than a normal would give it."""
        assert yards_prob(50.0, 25.0, 100.0).win > 0.03
        assert yards_prob(50.0, 25.0, 0.0).win == 1.0

    def test_the_family_follows_the_stat(self) -> None:
        assert prob_over(RECEPTIONS, 4.6, 4.0).push > 0.0
        assert prob_over(RUSHING_YARDS, 60.0, 60.0).push == 0.0

    def test_shrinkage_pulls_toward_the_anchor(self) -> None:
        assert shrunk_mean(0.0, 4, 6.0) > 0.0
        assert shrunk_mean(40.0, 4, 2.0) < 10.0


class TestUsageLeakage:
    def _frame(self) -> pd.DataFrame:
        rows = []
        for week in range(1, 8):
            rows.append(
                {
                    "player_id": "00-0001",
                    "player_display_name": "Puka Nacua",
                    "position": "WR",
                    "season": 2026,
                    "week": week,
                    "team": "LA",
                    "season_type": "REG",
                    "receptions": 30.0 if week == 6 else 5.0,
                    "targets": 8.0,
                    "carries": 0.0,
                    "attempts": 0.0,
                    "completions": 0.0,
                    "receiving_yards": 70.0,
                    "rushing_yards": 0.0,
                    "passing_yards": 0.0,
                }
            )
        return pd.DataFrame(rows)

    def test_the_week_being_priced_is_not_in_its_own_projection(self, monkeypatch) -> None:
        frame = self._frame()
        monkeypatch.setattr(
            usage.nflverse,
            "player_week",
            lambda season: frame if season == 2026 else frame.iloc[:0],
        )
        out = usage.projections(2026, 6)
        mean = out[("puka nacua", "receptions")].mean
        assert mean < 6.0  # week 6's 30 receptions cannot be in a week 6 projection

    def test_too_few_prior_games_is_no_projection(self, monkeypatch) -> None:
        frame = self._frame()
        monkeypatch.setattr(
            usage.nflverse,
            "player_week",
            lambda season: frame if season == 2026 else frame.iloc[:0],
        )
        assert usage.projections(2026, MIN_GAMES) == {}

    def test_names_match_across_feeds_without_matching_strangers(self) -> None:
        assert usage.normalise("Marvin Harrison Jr.") == usage.normalise("marvin harrison")
        assert usage.normalise("D.K. Metcalf") == usage.normalise("DK Metcalf")
        assert usage.normalise("Michael Pittman Jr.") != usage.normalise("Michael Thomas")


class TestSummary:
    def test_summary_says_nothing_is_bettable(self) -> None:
        rows = [book_of("draftkings", -115, 105), book_of("fanduel", -110, -110)]
        lines = props.summary(props.price_props(rows, projections_for(projection())))
        assert any("nothing bettable" in line for line in lines)
        assert any("research_only" in line for line in lines)

    def test_summary_survives_an_empty_board(self) -> None:
        assert props.summary([]) == ["props: nothing priced (no archived quotes for this week)"]


class TestCommand:
    def test_props_command_is_offline_and_reads_the_archive(self, tmp_path, monkeypatch) -> None:
        from nfl_engine import cli

        monkeypatch.setattr(cli.capture, "latest_snapshot", lambda *a, **k: None)
        monkeypatch.setattr(cli, "_client", lambda: pytest.fail("props must not fetch"))
        assert cli.main(["props", "--season", "2026", "--week", "2"]) == 1

    def test_props_prices_the_archive_it_finds(self, tmp_path, monkeypatch) -> None:
        from nfl_engine import cli
        from nfl_engine.data import capture

        rows = [quote(book="draftkings"), quote(book="fanduel", american=-110, opposite=-110)]
        written = capture.write_snapshot(rows, season=2026, week=2, kind=PROP_KIND, root=tmp_path)
        assert written is not None
        monkeypatch.setattr(cli.capture, "latest_snapshot", lambda *a, **k: written)
        monkeypatch.setattr(
            cli.usage, "projections", lambda season, week: projections_for(projection())
        )
        monkeypatch.setattr(cli, "_client", lambda: pytest.fail("props must not fetch"))
        monkeypatch.setattr(props, "write_research", lambda *a, **k: tmp_path / "research.csv")
        assert cli.main(["props", "--season", "2026", "--week", "2"]) == 0
