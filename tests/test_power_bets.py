"""The priced end of the screen.

The stage adds no model, so the tests pin the two things it can get wrong: the
median it reads off the board's own thresholds, and what it is willing to call a
recommendation. A Pass is not a buy however large its expected value, and the
biggest expected values on any slate belong to rows the price screen vetoed.
"""

from __future__ import annotations

import math
from datetime import date as Date

from mlb_engine.market.tiers import Tier
from mlb_engine.output import power_bets, power_report
from mlb_engine.output.power_bets import PITCHER_STATS, median_at
from mlb_engine.output.power_screen import FinalScore, HalfLine
from mlb_engine.recommendations import Recommendation
from tests.test_power_screen import _result


def _rec(
    name: str,
    stat: str,
    line: float,
    side: str = "over",
    *,
    category: str = "batter",
    player_id: int | None = None,
    model: float = 0.55,
    american: float | None = -115.0,
    ev: float | None = 0.04,
    fair: float | None = 0.50,
    tier: Tier = Tier.MODERATE,
) -> Recommendation:
    return Recommendation(
        game_date=Date(2026, 8, 17),
        game_pk=1,
        matchup="ATL @ MIN",
        category=category,
        market=f"{category}_{stat.lower()}",
        selection=f"{name} {stat} {'o' if side == 'over' else 'u'}{line}",
        model_prob=model,
        line=line,
        book="DraftKings",
        market_american=american,
        opposite_american=-105.0,
        ev=ev,
        edge=0.05,
        fair_prob=fair,
        tier=tier,
        player_id=player_id,
        stat=stat,
        side=side,
    )


def _pid(result) -> int:
    return result.sections[0].hitters[0].line.mlbam_id


def _final(result) -> list[FinalScore]:
    """The composite list a scored screen would carry, for the kept pool."""
    return [
        FinalScore(
            name=v.line.name,
            team=v.line.team,
            versus=v.line.versus,
            slot=v.line.slot,
            early=HalfLine(half="early", points=13),
            late=HalfLine(half="late", points=13),
            context=v.context,
            edge=v.edge,
        )
        for s in result.sections
        for v in s.hitters
    ]


# --- the median off the board's own thresholds ----------------------------


def test_the_median_is_the_highest_threshold_the_model_clears() -> None:
    assert median_at({0.5: 0.77, 1.5: 0.52, 2.5: 0.21}) == 2
    assert median_at({0.5: 0.68, 1.5: 0.31}) == 1
    assert median_at({0.5: 0.44}) == 0


def test_a_stat_the_board_never_priced_has_no_median_rather_than_a_zero() -> None:
    assert math.isnan(median_at({}))


def test_a_median_under_the_boards_lowest_line_is_a_bound_and_not_a_zero() -> None:
    """Under 4.5 strikeouts is not zero strikeouts, and the board never said."""
    overs = {4.5: 0.44, 5.5: 0.26}
    assert math.isnan(median_at(overs))
    assert power_bets.below(overs) == 5


def test_a_missed_half_line_is_a_true_zero() -> None:
    assert median_at({0.5: 0.44}) == 0
    assert math.isnan(power_bets.below({0.5: 0.44}))
    assert math.isnan(power_bets.below({4.5: 0.61}))


def test_the_report_prints_the_bound_rather_than_a_zero() -> None:
    result = _result()
    arm = result.sections[0].starter
    result.bets = power_bets.build(
        result,
        [
            _rec(arm.name, "K", 4.5, category="pitcher", player_id=arm.mlbam_id, model=0.44),
        ],
    )
    assert "&lt;5" in power_report.render_html(result)


def test_the_median_stops_at_the_first_line_the_model_misses() -> None:
    """A non-monotone series is a pricing artefact, not a higher median."""
    assert median_at({0.5: 0.60, 1.5: 0.40, 2.5: 0.55}) == 1


def test_the_projection_reads_every_stat_the_board_carried() -> None:
    result = _result()
    pid = _pid(result)
    card = power_bets.build(
        result,
        [
            _rec("Matt Olson", "H", 0.5, player_id=pid, model=0.78),
            _rec("Matt Olson", "H", 1.5, player_id=pid, model=0.34),
            _rec("Matt Olson", "HR", 0.5, player_id=pid, model=0.53),
            _rec("Matt Olson", "HRR", 1.5, player_id=pid, model=0.71),
        ],
    )
    hitter = card.hitters[0]
    assert hitter.median == {"H": 1, "HR": 1, "HRR": 2}
    assert hitter.reach["H"] == 0.78
    assert "TB" not in hitter.median


def test_a_pitcher_row_is_read_on_the_pitcher_stats() -> None:
    result = _result()
    arm = result.sections[0].starter
    card = power_bets.build(
        result,
        [
            _rec(arm.name, "K", 5.5, category="pitcher", player_id=arm.mlbam_id, model=0.63),
            _rec(arm.name, "outs", 17.5, category="pitcher", player_id=arm.mlbam_id, model=0.42),
            _rec(arm.name, "outs", 15.5, category="pitcher", player_id=arm.mlbam_id, model=0.71),
        ],
    )
    assert card.arms[0].median["K"] == 6
    assert card.arms[0].median["outs"] == 16  # the board prices 15.5 and 17.5, not 16.4
    assert set(card.arms[0].median) <= set(PITCHER_STATS)


def test_a_batter_row_never_reaches_the_arms_and_the_reverse() -> None:
    result = _result()
    arm = result.sections[0].starter
    card = power_bets.build(
        result,
        [
            _rec("Matt Olson", "K", 1.5, player_id=_pid(result)),
            _rec(arm.name, "K", 5.5, category="pitcher", player_id=arm.mlbam_id),
        ],
    )
    assert [s.stat for s in card.arms[0].sides] == ["K"]
    assert card.arms[0].median["K"] == 6
    assert card.hitters[0].median == {}  # a batter's own K is not one of his stats


def test_another_players_row_is_not_priced_onto_this_one() -> None:
    result = _result()
    card = power_bets.build(
        result, [_rec("Matt Olson", "HR", 0.5, player_id=_pid(result) + 1)]
    )
    assert card.hitters[0].sides == ()
    assert card.hitters[0].median == {}


# --- what counts as a recommendation --------------------------------------


def test_a_pass_is_not_a_buy_however_large_its_expected_value() -> None:
    """The vetoed rows carry the biggest edges on any slate, by construction."""
    result = _result()
    pid = _pid(result)
    card = power_bets.build(
        result,
        [
            _rec("Matt Olson", "HR", 0.5, player_id=pid, ev=0.72, tier=Tier.PASS),
            _rec("Matt Olson", "TB", 1.5, player_id=pid, ev=0.05, tier=Tier.STRONG),
        ],
    )
    assert [b.selection for b in card.batter_buys] == ["Matt Olson TB o1.5"]


def test_the_buys_are_ordered_by_the_devigged_price_not_by_expected_value() -> None:
    """The market's own probability ranks the card; our EV against it does not.

    Graded buys carrying a devigged price lose 11.2% below .45 fair and 3.8% at
    .60-.65, while return *falls* as the claimed edge grows -- so the row with
    four times the EV goes second when the market thinks less of it.
    """
    result = _result()
    pid = _pid(result)
    card = power_bets.build(
        result,
        [
            _rec("Matt Olson", "H", 0.5, player_id=pid, ev=0.03, fair=0.62, tier=Tier.MODERATE),
            _rec("Matt Olson", "R", 0.5, player_id=pid, ev=0.12, fair=0.41, tier=Tier.STRONG),
        ],
    )
    assert [b.stat for b in card.batter_buys] == ["H", "R"]


def test_the_same_bet_at_two_books_is_one_recommendation_at_the_better_price() -> None:
    result = _result()
    pid = _pid(result)
    card = power_bets.build(
        result,
        [
            _rec("Matt Olson", "TB", 1.5, player_id=pid, american=280.0, ev=0.30, tier=Tier.STRONG),
            _rec("Matt Olson", "TB", 1.5, player_id=pid, american=255.0, ev=0.22, tier=Tier.STRONG),
        ],
    )
    assert [b.odds for b in card.batter_buys] == [280.0]


def test_a_home_run_is_priced_and_shown_and_never_bought() -> None:
    result = _result()
    pid = _pid(result)
    card = power_bets.build(
        result,
        [
            _rec("Matt Olson", "HR", 0.5, player_id=pid, american=280.0, ev=0.30, tier=Tier.STRONG),
        ],
    )
    assert [s.stat for s in card.hitters[0].sides] == ["HR"]
    assert card.batter_buys == ()


def test_the_probability_shown_is_the_one_the_bet_was_priced_from() -> None:
    result = _result()
    pid = _pid(result)
    rec = _rec("Matt Olson", "TB", 1.5, player_id=pid, ev=0.10, tier=Tier.STRONG)
    rec.bet_prob = 0.61
    card = power_bets.build(result, [rec])
    assert card.batter_buys[0].prob == 0.61


def test_both_sides_of_one_prop_can_be_read_and_only_one_can_be_bought() -> None:
    result = _result()
    pid = _pid(result)
    card = power_bets.build(
        result,
        [
            _rec("Matt Olson", "1B", 0.5, "over", player_id=pid, tier=Tier.PASS),
            _rec("Matt Olson", "1B", 0.5, "under", player_id=pid, tier=Tier.STRONG),
        ],
    )
    assert len(card.hitters[0].sides) == 2
    assert [b.side for b in card.batter_buys] == ["under"]


def test_the_hitters_are_read_in_the_composite_order() -> None:
    result = _result()
    result.final = _final(result)
    card = power_bets.build(result, [])
    assert [h.name for h in card.hitters] == [f.name for f in result.final]
    assert card.hitters[0].mlbam_id == _pid(result)


def test_a_screen_whose_halves_never_ran_falls_back_to_the_kept_pool() -> None:
    result = _result()
    assert result.final == []
    assert [h.name for h in power_bets.build(result, []).hitters] == ["Matt Olson"]


# --- the report -----------------------------------------------------------


def test_the_note_prints_the_projection_and_both_bet_tables() -> None:
    result = _result()
    pid = _pid(result)
    arm = result.sections[0].starter
    result.bets = power_bets.build(
        result,
        [
            _rec("Matt Olson", "TB", 1.5, player_id=pid, model=0.53, tier=Tier.STRONG),
            _rec(arm.name, "K", 5.5, category="pitcher", player_id=arm.mlbam_id, tier=Tier.STRONG),
        ],
    )
    html = power_report.render_html(result)
    assert "Matt Olson TB o1.5" in html
    assert f"{arm.name} K o5.5" in html
    assert "Batter props" in html and "Pitcher props" in html


def test_a_screen_that_was_never_priced_prints_no_bet_section() -> None:
    result = _result()
    assert result.bets is None
    assert "Batter props" not in power_report.render_html(result)


def test_a_priced_slate_with_no_buy_says_so_rather_than_printing_nothing() -> None:
    result = _result()
    result.bets = power_bets.build(
        result, [_rec("Matt Olson", "HR", 0.5, player_id=_pid(result), tier=Tier.PASS)]
    )
    html = power_report.render_html(result)
    assert "No side survived the EV screen" in html
