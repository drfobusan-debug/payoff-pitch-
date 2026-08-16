"""BAT X beside our own number: the side, the join, and never the price."""

from __future__ import annotations

import csv
from datetime import date as Date
from pathlib import Path

from mlb_engine.audit.ledger import LEDGER_FIELDS, entries_from_graded, load_ledger, update_ledger
from mlb_engine.data.batx import BatxRow, annotate, load_rows
from mlb_engine.market.tiers import Tier
from mlb_engine.recommendations import Recommendation


def rec(selection: str, market: str, side: str, line: float | None = 0.5) -> Recommendation:
    return Recommendation(
        game_date=Date(2026, 8, 15),
        game_pk=1,
        matchup="MIL@LAD",
        category="batter",
        market=market,
        selection=selection,
        model_prob=0.55,
        line=line,
        side=side,
        tier=Tier.MODERATE,
    )


def test_the_feed_quotes_the_over_so_an_under_is_flipped() -> None:
    rows = [BatxRow(player="Aaron Judge", market="batter_h", line=0.5, prob=0.72)]
    over = rec("Aaron Judge H o0.5", "batter_h", "over")
    under = rec("Aaron Judge H u0.5", "batter_h", "under")

    assert annotate([over, under], rows) == 2
    assert over.batx_prob == 0.72
    assert under.batx_prob == 0.28


def test_the_line_is_part_of_the_join() -> None:
    """o0.5 and o1.5 hits are different bets and get different probabilities."""
    rows = [
        BatxRow(player="Aaron Judge", market="batter_h", line=0.5, prob=0.72),
        BatxRow(player="Aaron Judge", market="batter_h", line=1.5, prob=0.31),
    ]
    one, two = rec("Aaron Judge H o0.5", "batter_h", "over"), rec(
        "Aaron Judge H o1.5", "batter_h", "over", line=1.5
    )
    annotate([one, two], rows)
    assert (one.batx_prob, two.batx_prob) == (0.72, 0.31)


def test_names_join_across_accents_and_suffixes() -> None:
    rows = [BatxRow(player="Andres Gimenez", market="batter_1b", line=0.5, prob=0.4)]
    r = rec("Andrés Giménez 1B o0.5", "batter_1b", "over")
    assert annotate([r], rows) == 1
    assert r.batx_prob == 0.4


def test_every_prop_market_reads_its_player_back_out_of_the_selection() -> None:
    """A market whose stat token is unknown to the reader joins to nothing.

    Batter walks and strikeouts were priced without their "BB"/"K" tokens being
    added here, so the player of "A.J. Ewing BB o0.5" came back with the token
    still on it and every row of two whole markets missed.
    """
    from mlb_engine.market import keys

    cases = {
        keys.batter_prop("A.J. Ewing", "BB", 0.5): "A.J. Ewing",
        keys.batter_prop("A.J. Ewing", "K", 1.5, "under"): "A.J. Ewing",
        keys.batter_prop("Aaron Judge", "H+R+RBI", 2.5): "Aaron Judge",
        keys.batter_prop("Aaron Judge", "TB", 1.5): "Aaron Judge",
        keys.pitcher_prop("Tarik Skubal", "Ks", 5.5): "Tarik Skubal",
        keys.pitcher_prop("Tarik Skubal", "Walks", 1.5, "under"): "Tarik Skubal",
    }
    for selection, player in cases.items():
        assert keys.player_from_selection(selection) == player, selection

    rows = [BatxRow(player="A.J. Ewing", market="batter_bb", line=0.5, prob=0.11)]
    r = rec(keys.batter_prop("A.J. Ewing", "BB", 0.5), "batter_bb", "over")
    assert annotate([r], rows) == 1
    assert r.batx_prob == 0.11


def test_an_unmatched_selection_is_left_alone() -> None:
    r = rec("Shohei Ohtani HR o0.5", "batter_hr", "over")
    assert annotate([r], [BatxRow("Aaron Judge", "batter_hr", 0.5, 0.2)]) == 0
    assert r.batx_prob is None


def test_a_game_market_is_never_annotated() -> None:
    """The feed prices players; a moneyline has no BAT X opinion to carry."""
    r = rec("Dodgers ML", "game_ml", "win", line=None)
    assert annotate([r], [BatxRow("Dodgers", "game_ml", 0.5, 0.6)]) == 0
    assert r.batx_prob is None


def test_the_annotation_moves_no_price(tmp_path: Path) -> None:
    r = rec("Aaron Judge H o0.5", "batter_h", "over")
    before = (r.model_prob, r.bet_prob, r.ev, r.edge, r.tier)
    annotate([r], [BatxRow("Aaron Judge", "batter_h", 0.5, 0.99)])
    assert (r.model_prob, r.bet_prob, r.ev, r.edge, r.tier) == before


def test_load_skips_rows_it_cannot_join(tmp_path: Path) -> None:
    path = tmp_path / "2026-08-15.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "player", "team", "market", "line", "batx_prob"])
        w.writerow(["2026-08-15", "Aaron Judge", "NYY", "batter_h", "0.5", "0.72"])
        w.writerow(["2026-08-15", "Broken Row", "NYY", "batter_h", "", "0.5"])
        w.writerow(["2026-08-15", "Also Broken", "NYY", "batter_h", "0.5", "n/a"])
    rows = load_rows(path)
    assert [r.player for r in rows] == ["Aaron Judge"]


def test_a_missing_export_is_not_an_error(tmp_path: Path) -> None:
    assert load_rows(tmp_path / "nope.csv") == []


def test_the_probability_reaches_the_ledger_and_survives_a_reload(tmp_path: Path) -> None:
    r = rec("Aaron Judge H o0.5", "batter_h", "over")
    annotate([r], [BatxRow("Aaron Judge", "batter_h", 0.5, 0.72)])

    entries = entries_from_graded([(r, "win")], Date(2026, 8, 15))
    assert entries[0].batx_prob == 0.72
    assert "batx_prob" in LEDGER_FIELDS

    path = tmp_path / "ledger.csv"
    update_ledger(path, entries, Date(2026, 8, 15))
    assert load_ledger(path)[0].batx_prob == 0.72


def test_a_ledger_written_before_the_column_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "ledger.csv"
    fields = [f for f in LEDGER_FIELDS if f != "batx_prob"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerow(
            {
                **{f: "" for f in fields},
                "date": "2026-08-13",
                "market": "batter_h",
                "selection": "Aaron Judge H o0.5",
                "model_prob": "0.55",
                "result": "win",
                "pnl": "0.9",
            }
        )
    loaded = load_ledger(path)
    assert loaded[0].batx_prob is None
