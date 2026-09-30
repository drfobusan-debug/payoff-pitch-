"""The batter regression report ranks hitters, so it has to see one per hitter."""

from __future__ import annotations

import pandas as pd

import mlb_engine.output.regression_profiles as cr


def _pred(selection: str, player_id: int) -> dict:
    return {
        "market": "batter_h",
        "selection": selection,
        "player_id": player_id,
        "tier": "Pass",
        "ev": 0.0,
        "game_pk": 1,
        "matchup": "BAL @ TB",
    }


def test_batter_name_strips_the_market_off_both_sides():
    # Under selections have been priced since #144; a regex that only knows the
    # over leaves the market on the name and invents a hitter per prop.
    assert cr._batter_name("Carlos Narvaez H o0.5") == "Carlos Narvaez"
    assert cr._batter_name("Carlos Narvaez H u0.5") == "Carlos Narvaez"
    assert cr._batter_name("Carlos Narvaez H+R+RBI u2.5") == "Carlos Narvaez"
    assert cr._batter_name("Jac Caglianone TB o1.5") == "Jac Caglianone"


def test_one_hitter_is_ranked_once_however_many_props_he_has(monkeypatch):
    preds = [
        _pred("Carlos Narvaez H u0.5", 1),
        _pred("Carlos Narvaez H+R+RBI u1.5", 1),
        _pred("Carlos Narvaez TB u2.5", 1),
        _pred("Eli White H o1.5", 2),
    ]
    assert cr._batter_id_map(preds) == {"Carlos Narvaez": 1, "Eli White": 2}

    gaps = {1: 0.109, 2: -0.262}

    def fake_analyze(name, pid, df, cutoff):
        return {"name": name, "bbe": 40, "dxwoba": gaps[pid]}

    monkeypatch.setattr(cr, "analyze_batter", fake_analyze)
    df = pd.DataFrame({"batter": [1, 2], "game_date": ["2026-08-01", "2026-08-01"]})
    pos, neg = cr.build_batter_profiles(preds, df)

    assert [p["pid"] for p in pos] == [1]
    assert [p["pid"] for p in neg] == [2]


def test_name_left_alone_when_there_is_no_side_marker() -> None:
    """The guard the shared parser needs: no marker means chop nothing, or a
    plain name loses its surname."""
    assert cr._batter_name("Matt McLain") == "Matt McLain"
    assert cr._batter_name("Over 7.5") == "Over 7.5"


def test_context_is_keyed_on_the_hitter_not_the_prop() -> None:
    preds = [
        {**_pred("Moises Ballesteros H u0.5", 1), "matchup": "KC @ LAA", "game_pk": 7},
        {**_pred("Moises Ballesteros HR u0.5", 1), "matchup": "KC @ LAA", "game_pk": 7},
    ]
    ctx = cr._batter_ctx(preds, {7: {"park_factor": 97.0, "wx_hr_mult": 1.05}})
    assert list(ctx) == ["Moises Ballesteros"]
    assert ctx["Moises Ballesteros"]["park_factor"] == 97.0
