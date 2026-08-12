"""Every Pass is attributed to the screen that caused it, and each is gradeable.

The engine passes on ~98% of the board, and until now the ledger recorded only
*that* it passed. A screen that rejects winners is a false negative, and a false
negative is invisible unless its own rows can be pulled back out -- so the gate
name is what turns "should we raise the edge ceiling?" from an argument into a
query.
"""

from __future__ import annotations

from mlb_engine.audit.ledger import LedgerEntry, gate_metrics
from mlb_engine.config import EVThresholds
from mlb_engine.market.ev import EVResult, MarketQuote
from mlb_engine.market.tiers import Tier, classify, price_screen


def _res(ev: float, edge: float) -> EVResult:
    return EVResult(
        model_prob=0.6,
        best_quote=MarketQuote(book="dk", american=-110),
        decimal=1.91,
        ev=ev,
        fair_prob=0.5,
        edge=edge,
        sharp_divergence=None,
    )


# --- the screens name themselves -------------------------------------------
def test_each_price_screen_reports_its_own_name() -> None:
    thr = EVThresholds()
    assert price_screen(_res(ev=-0.10, edge=0.05), thr)[0] == "ev_floor"
    assert price_screen(_res(ev=0.05, edge=0.001), thr)[0] == "thin_edge"
    assert price_screen(_res(ev=0.90, edge=0.40), thr)[0] == "edge_ceiling"


def test_a_price_that_clears_every_screen_is_named_by_none_of_them() -> None:
    assert price_screen(_res(ev=0.10, edge=0.05), EVThresholds()) is None


def test_the_screen_name_and_the_prose_reason_cannot_disagree() -> None:
    """``classify`` renders the same screen it reports, so the two never drift."""
    thr = EVThresholds()
    for res in (
        _res(ev=-0.10, edge=0.05),
        _res(ev=0.05, edge=0.001),
        _res(ev=0.90, edge=0.40),
    ):
        screened = price_screen(res, thr)
        assert screened is not None
        tier, reasons = classify(res, thr)
        assert tier is Tier.PASS
        assert screened[1] in reasons


# --- the ledger grades what each screen threw away -------------------------
def _entry(gate: str, result: str, *, odds: float = -110) -> LedgerEntry:
    return LedgerEntry(
        date="2026-08-11",
        matchup="TB @ ATH",
        category="Batter Props",
        market="batter_h",
        selection="Junior Caminero H o1.5",
        line=1.5,
        book="dk",
        odds=odds,
        tier="Pass" if gate else "Strong buy",
        model_prob=0.50,
        ev=0.36,
        result=result,
        pnl=(0.909 if odds == -110 else abs(odds) / 100.0) if result == "win" else -1.0,
        pass_gate=gate,
    )


def test_a_screen_is_graded_against_the_price_it_declined_not_against_a_coin_flip() -> None:
    """-110 demands 52.4%; a screen rejecting 1-of-3 winners deleted losers."""
    rows = {m.tier: m for m in gate_metrics([
        _entry("edge_ceiling", "win"),
        _entry("edge_ceiling", "loss"),
        _entry("edge_ceiling", "loss"),
        _entry("", "win"),
    ])}
    ceiling = rows["GATE edge_ceiling"]
    assert (ceiling.wins, ceiling.losses) == (1, 2)
    assert round(ceiling.required_win_pct, 3) == 0.524
    assert ceiling.win_pct < ceiling.required_win_pct  # earned its keep
    assert ceiling.units < 0
    assert rows["BOUGHT (no gate)"].n == 1


def test_a_screen_that_deletes_winners_shows_up_as_a_positive_row() -> None:
    """The false-negative case: the same rows, the other way round."""
    rows = {m.tier: m for m in gate_metrics([
        _entry("edge_ceiling", "win"),
        _entry("edge_ceiling", "win"),
        _entry("edge_ceiling", "loss"),
    ])}
    ceiling = rows["GATE edge_ceiling"]
    assert ceiling.win_pct > ceiling.required_win_pct
    assert ceiling.units > 0
    assert ceiling.roi > 0


def test_a_dog_the_model_faded_still_counts_as_a_forgone_bet() -> None:
    """Gate rows count every rejected pick, not only the ones over 50%.

    An ``edge_ceiling`` veto often sits on a plus-money selection whose model
    probability is under a half -- edge is measured against the devigged price,
    not against a coin flip -- so keying these rows on ``model_prob >= 0.5``
    would silently drop the very picks the screen exists to remove.
    """
    dog = _entry("edge_ceiling", "win", odds=200)
    dog.model_prob = 0.38
    rows = {m.tier: m for m in gate_metrics([dog])}
    assert rows["GATE edge_ceiling"].n == 1
    assert round(rows["GATE edge_ceiling"].required_win_pct, 3) == 0.333


def test_an_unpriced_pass_is_not_a_forgone_bet() -> None:
    unpriced = _entry("unpriced", "loss")
    unpriced.odds = None
    assert gate_metrics([unpriced]) == []


def test_gates_are_listed_busiest_first() -> None:
    rows = gate_metrics([
        _entry("edge_ceiling", "loss"),
        _entry("ev_floor", "loss"),
        _entry("ev_floor", "win"),
        _entry("ev_floor", "loss"),
    ])
    assert [m.tier for m in rows] == ["GATE ev_floor", "GATE edge_ceiling"]


def test_the_ledger_round_trips_the_gate_name(tmp_path) -> None:
    from datetime import date

    from mlb_engine.audit.ledger import load_ledger, update_ledger

    path = tmp_path / "ledger.csv"
    update_ledger(
        path,
        [_entry("edge_ceiling", "loss"), _entry("", "win")],
        date(2026, 8, 11),
    )
    back = load_ledger(path)
    assert [e.pass_gate for e in back] == ["edge_ceiling", ""]
