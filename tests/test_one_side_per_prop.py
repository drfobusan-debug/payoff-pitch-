"""A prop's two ledger rows are one wager, and a measurement counts it once.

Both sides of every prop are priced, so each prop lands in the ledger twice: the
side that was bought, and its complement as a Pass so the fade stays graded.
Counting both in a measurement is not a mild double-count -- the rows are
complements, so one of them is whichever side is nearly certain, and grading
"Ohtani HR u0.5" at .96 as a correct positive prediction is free credit for the
base rate.

Measured on the real 86,762-row ledger by synthesising the complement of every
prop row: whole-engine PPV .531 -> .706, batter_hr .20 -> .90, five markets
lifted off Fade and pitcher_k promoted to Play, with NPV landing at .707 against
a PPV of .706 -- the tell that it is no longer independent information.
"""

from __future__ import annotations

from mlb_engine.audit.analysis import false_positive_insights
from mlb_engine.audit.clv import clv_rows, summarize
from mlb_engine.audit.ledger import (
    LedgerEntry,
    engine_metrics,
    one_side_per_prop,
    prop_subject,
)
from mlb_engine.market.tiers import Tier


def _prop(
    side: str,
    model_prob: float,
    result: str,
    *,
    tier: str = Tier.PASS.value,
    stat: str = "HR",
    line: float = 0.5,
    market: str = "batter_hr",
    date: str = "2026-08-20",
) -> LedgerEntry:
    return LedgerEntry(
        date=date,
        matchup="LAD @ SD",
        category="Batter Props",
        market=market,
        selection=f"Shohei Ohtani {stat} {side}{line:g}",
        line=line,
        book="dk",
        odds=270 if side == "o" else -350,
        tier=tier,
        model_prob=model_prob,
        ev=0.03,
        result=result,
        pnl=(2.7 if side == "o" else 0.2857) if result == "win" else -1.0,
    )


def test_the_two_sides_of_one_prop_collapse_to_one_row() -> None:
    """Nothing was bought, so the over is kept: the side the ledger always held."""
    rows = [_prop("o", 0.04, "loss"), _prop("u", 0.96, "win")]
    (kept,) = one_side_per_prop(rows)
    assert kept.selection.endswith("o0.5")


def test_the_bought_side_is_kept_even_when_the_model_fades_it() -> None:
    """A +560 double is bought at p<.5; dropping it would erase a real bet."""
    rows = [
        _prop("o", 0.20, "loss", tier=Tier.STRONG.value),
        _prop("u", 0.80, "win"),
    ]
    (kept,) = one_side_per_prop(rows)
    assert kept.tier == Tier.STRONG.value
    assert kept.selection.endswith("o0.5")
    assert kept.pnl == -1.0  # the loss survives into the P&L

    # And the same when the bought side is the under.
    rows = [_prop("o", 0.60, "win"), _prop("u", 0.40, "loss", tier=Tier.MODERATE.value)]
    (kept,) = one_side_per_prop(rows)
    assert kept.tier == Tier.MODERATE.value
    assert kept.selection.endswith("u0.5")


def test_the_near_certain_complement_cannot_credit_itself() -> None:
    """The whole point: adding the fade rows must not move the measurement.

    Ten props the model rates .04 and gets wrong. Counting the .96 unders as well
    would turn a 0% record into 50% -- and into 100% on the positive side, since
    every one of those unders is favored and wins.
    """
    overs = [_prop("o", 0.04, "loss", stat=f"HR{i}") for i in range(10)]
    unders = [_prop("u", 0.96, "win", stat=f"HR{i}") for i in range(10)]
    before = engine_metrics(overs)
    polluted = engine_metrics(overs + unders)
    after = engine_metrics(one_side_per_prop(overs + unders))
    # `n` counts the favored rows -- none of the overs, all of the unders. So the
    # bug invents ten perfect positive predictions out of the base rate.
    assert (before.n, before.npv) == (0, 1.0)
    assert (polluted.n, polluted.ppv) == (10, 1.0)  # what the bug looks like
    assert (after.n, after.ppv, after.npv) == (before.n, before.ppv, before.npv)


def test_closing_line_value_cannot_cancel_itself_out() -> None:
    """CLV is the worst-hit metric, because the cancellation there is exact.

    ``clv = close_prob - bet_prob`` and both are devigged, so the fade side's CLV
    is ``-clv`` to the decimal. Counting both sides drives every market's mean CLV
    to exactly zero and "beat the close" to 50% -- arithmetic, not evidence. On
    the real ledger's 6,633 prop rows with a close attached: ``batter_tb``
    +0.113pts -> +0.000, ``pitcher_k`` +0.100 -> +0.000.
    """
    over = _prop("o", 0.40, "loss")
    over.close_odds, over.close_prob, over.clv, over.clv_ev = 250, 0.46, 0.06, 0.05
    under = _prop("u", 0.60, "win")
    under.close_odds, under.close_prob, under.clv, under.clv_ev = -320, 0.54, -0.06, -0.02

    both = {s.label: s for s in summarize(clv_rows([over, under]))}["batter_hr"]
    assert (both.n, both.mean_clv, both.beat_close_pct) == (2, 0.0, 0.5)  # the bug

    kept = {s.label: s for s in summarize(clv_rows(one_side_per_prop([over, under])))}
    assert (kept["batter_hr"].n, kept["batter_hr"].mean_clv) == (1, 0.06)


def test_an_over_only_history_is_untouched() -> None:
    """79,103 rows predate both-sided pricing; none of them may move."""
    rows = [_prop("o", 0.30, "loss", stat=f"HR{i}") for i in range(50)]
    assert one_side_per_prop(rows) == rows


def test_the_same_player_at_two_lines_stays_two_props() -> None:
    """H o0.5 and H o1.5 are different questions, not two sides of one."""
    rows = [_prop("o", 0.60, "win", stat="H", line=0.5, market="batter_h"),
            _prop("o", 0.25, "loss", stat="H", line=1.5, market="batter_h")]
    assert len(one_side_per_prop(rows)) == 2


def test_a_repeated_key_is_left_alone_rather_than_collapsed() -> None:
    """Only a proved pair is collapsed, so nothing can be silently swallowed.

    Two overs and one under under the same key is not one wager seen twice, so
    all three rows survive.
    """
    rows = [_prop("o", 0.6, "win"), _prop("o", 0.6, "loss"), _prop("u", 0.4, "loss")]
    assert one_side_per_prop(rows) == rows


def test_a_selection_with_no_side_marker_is_never_grouped() -> None:
    """Legacy prop rows written without the o/u token stay one row each."""
    rows = [
        LedgerEntry(
            date="2026-07-23", matchup="AAA @ BBB", category="Pitcher Props",
            market="pitcher_k", selection="Over 7.5", line=7.5, book="dk", odds=-110,
            tier=Tier.PASS.value, model_prob=0.7, ev=0.0, result="win", pnl=0.91,
        )
    ] * 10
    assert one_side_per_prop(rows) == rows


def test_game_markets_pass_straight_through() -> None:
    """A total's two sides are separate games' rows, not a prop pair."""
    game = LedgerEntry(
        date="2026-08-20", matchup="LAD @ SD", category="Totals", market="game_total",
        selection="Over 8.5", line=8.5, book="dk", odds=-110,
        tier=Tier.STRONG.value, model_prob=0.55, ev=0.02, result="win", pnl=0.91,
    )
    assert one_side_per_prop([game, game]) == [game, game]


def test_a_line_finding_names_the_side_it_is_talking_about() -> None:
    """"Stop buying line 1.5" tells the reader to stop buying both sides of it.

    A losing over and a winning under at the same number are opposite bets, so
    the pockets are keyed on side as well as line, and printed with it.
    """
    rows = [
        *[_prop("o", 0.60, "loss", stat="H", line=1.5, market="batter_h") for _ in range(25)],
        *[_prop("u", 0.60, "win", stat="H2", line=1.5, market="batter_h") for _ in range(25)],
    ]
    findings = [i.finding for i in false_positive_insights(rows)]
    assert any("line o1.5" in f for f in findings)
    assert not any("line 1.5:" in f for f in findings)
    # Pooled, the two sides average to 50% and the pocket disappears entirely.
    assert any("win 0.0%" in f for f in findings)


def test_the_subject_of_a_prop_ignores_which_way_it_is_bet() -> None:
    assert prop_subject("Bobby Witt Jr. H o0.5") == "Bobby Witt Jr. H"
    assert prop_subject("Bobby Witt Jr. H u0.5") == "Bobby Witt Jr. H"
    assert prop_subject("Bobby Witt Jr. H+R+RBI o1.5") == "Bobby Witt Jr. H+R+RBI"
    # Not a prop marker: left alone rather than mangled.
    assert prop_subject("KC ML") == "KC ML"
    assert prop_subject("Over 8.5") == "Over 8.5"
