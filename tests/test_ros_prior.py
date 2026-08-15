"""Rest-of-season projections as the batter prior, and per-outcome shrinkage."""

from __future__ import annotations

from datetime import date as Date

import pandas as pd
import pytest

from mlb_engine.features.rolling import (
    LEAGUE_RATES,
    OUTCOME_PRIOR_STRENGTH,
    OUTCOMES_ORDER,
    PEN_PRIOR_STRENGTH,
    build_batter_profile,
    build_bullpen_profile,
    load_ros_priors,
    pa_outcome_counts,
    rates_from_events,
    ros_rates_from_projection,
)


def _projection(**over: float) -> pd.DataFrame:
    row = {
        "PA": 600.0,
        "H": 150.0,
        "1B": 100.0,
        "2B": 30.0,
        "3B": 2.0,
        "HR": 18.0,
        "BB": 60.0,
        "SO": 120.0,
        "HBP": 6.0,
        "MLBAMID": 12345,
    }
    row.update(over)
    return pd.DataFrame([row])


def test_projection_becomes_a_normalized_rate_vector() -> None:
    rates = ros_rates_from_projection(_projection())
    assert len(rates) == 1
    vec = rates.iloc[0]
    assert vec["mlbam_id"] == 12345
    assert pytest.approx(sum(vec[oc] for oc in OUTCOMES_ORDER), abs=1e-9) == 1.0
    # 100 singles in 600 PA, and HBP folded in with the walks.
    assert vec["1B"] == pytest.approx(100 / 600, abs=1e-6)
    assert vec["BB"] == pytest.approx(66 / 600, abs=1e-6)


def test_missing_singles_column_is_derived_from_hits() -> None:
    df = _projection().drop(columns=["1B"])
    vec = ros_rates_from_projection(df).iloc[0]
    assert vec["1B"] == pytest.approx(100 / 600, abs=1e-6)


def test_a_missing_priors_file_is_not_an_error(tmp_path) -> None:
    assert load_ros_priors(tmp_path / "nope.csv") == {}


def test_priors_round_trip(tmp_path) -> None:
    path = tmp_path / "ros.csv"
    ros_rates_from_projection(_projection()).to_csv(path, index=False)
    loaded = load_ros_priors(path)
    assert set(loaded) == {12345}
    assert pytest.approx(sum(loaded[12345].values()), abs=1e-9) == 1.0
    assert loaded[12345]["HR"] == pytest.approx(18 / 600, abs=1e-6)


def test_per_outcome_strength_shrinks_doubles_harder_than_strikeouts() -> None:
    """The whole point: a hot doubles streak should barely move, a K rate should."""
    # 100 PA in which the hitter doubled and struck out at double the league rate.
    events = ["double"] * 9 + ["strikeout"] * 45 + ["field_out"] * 46
    flat = rates_from_events(pd.Series(events))
    per_outcome = rates_from_events(pd.Series(events), LEAGUE_RATES, OUTCOME_PRIOR_STRENGTH)

    def lift(r: object, oc: str) -> float:
        return r.as_dict()[oc] / LEAGUE_RATES[oc]  # type: ignore[attr-defined]

    # Under the flat prior the doubles rate is carried most of the way to the
    # observed .09; under the fitted strengths it is almost entirely prior.
    assert lift(flat, "2B") > 1.5
    assert lift(per_outcome, "2B") < 1.1
    # Strikeouts are the bucket a six-week window genuinely measures, so the
    # fitted strength leaves them close to where the flat prior had them.
    assert lift(per_outcome, "K") == pytest.approx(lift(flat, "K"), rel=0.15)
    assert pytest.approx(sum(per_outcome.as_dict().values()), abs=1e-9) == 1.0


def _statcast(events: list[str], batter: int = 12345) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "batter": [batter] * len(events),
            "events": events,
            "game_date": [Date(2026, 8, 1)] * len(events),
            "inning_topbot": ["Bot"] * len(events),
            "p_throws": ["R"] * len(events),
        }
    )


def test_profile_without_a_projection_is_unchanged() -> None:
    """Off by default: no projection means the league prior at a flat strength."""
    df = _statcast(["single"] * 20 + ["field_out"] * 80)
    a = build_batter_profile(df, 12345, Date(2026, 8, 13), 21, 21, 42)
    b = build_batter_profile(df, 12345, Date(2026, 8, 13), 21, 21, 42, True, None)
    assert a.overall.as_dict() == b.overall.as_dict()


def _relief_frame(events: list[str]) -> pd.DataFrame:
    """A pen's late-inning relief rows, plus a starter's 1st-inning row to exclude."""
    rows = [
        {
            "game_date": Date(2026, 8, 1),
            "pitcher": 200,
            "inning": 7,
            "inning_topbot": "Top",
            "events": ev,
            "batter": 1,
            "home_team": "NYY",
            "away_team": "BOS",
        }
        for ev in events
    ]
    rows.append({**rows[0], "pitcher": 100, "inning": 1, "events": "single"})
    return pd.DataFrame(rows)


def test_pen_counts_ignore_the_starter() -> None:
    frame = _relief_frame(["double"] * 5 + ["field_out"] * 5)
    counts = pa_outcome_counts(frame)
    assert counts["2B"] == 5
    assert counts["1B"] == 1  # the starter row is in the frame until it is filtered


def test_pen_doubles_are_shrunk_to_the_league() -> None:
    """Across 30 pens the doubles-allowed spread is all noise, so ignore the pen's."""
    hot = build_bullpen_profile(
        _relief_frame(["double"] * 40 + ["field_out"] * 60),
        "NYY",
        Date(2026, 8, 13),
        21,
        prior_strength=PEN_PRIOR_STRENGTH,
    )
    # 40 doubles in 100 relief PA, and the read still lands near the league's
    # rate: nine tenths of the pen's own doubles signal is discarded. It is not
    # exactly league because each bucket has its own denominator and the vector
    # is renormalized afterwards, which redistributes a little.
    assert hot.allowed.p_2b < LEAGUE_RATES["2B"] * 1.3
    # The strikeout rate is the one bucket a three-week pen read does carry, so a
    # pen that struck out nobody should still read below league.
    assert hot.allowed.p_k < LEAGUE_RATES["K"]


def test_pen_default_is_the_flat_prior() -> None:
    frame = _relief_frame(["double"] * 40 + ["field_out"] * 60)
    flat = build_bullpen_profile(frame, "NYY", Date(2026, 8, 13), 21)
    fitted = build_bullpen_profile(
        frame, "NYY", Date(2026, 8, 13), 21, prior_strength=PEN_PRIOR_STRENGTH
    )
    assert flat.allowed.p_2b > fitted.allowed.p_2b * 3
    assert pytest.approx(sum(fitted.allowed.as_dict().values()), abs=1e-9) == 1.0


def test_projection_moves_the_hitter_toward_his_own_line() -> None:
    """A weak window on a good hitter lands nearer his projection than the league's."""
    df = _statcast(["single"] * 5 + ["strikeout"] * 30 + ["field_out"] * 65)
    ros = ros_rates_from_projection(_projection()).iloc[0]
    prior = {oc: float(ros[oc]) for oc in OUTCOMES_ORDER}

    league = build_batter_profile(df, 12345, Date(2026, 8, 13), 21, 21, 42).overall
    with_ros = build_batter_profile(
        df, 12345, Date(2026, 8, 13), 21, 21, 42, True, prior
    ).overall

    # His projection has him hitting singles well above the league mean, and the
    # thin cold window should not be allowed to erase that.
    assert prior["1B"] > LEAGUE_RATES["1B"]
    assert with_ros.p_1b > league.p_1b
    assert pytest.approx(sum(with_ros.as_dict().values()), abs=1e-9) == 1.0
