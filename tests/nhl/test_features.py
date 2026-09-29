"""Phase 1: shared shrinkage/odds math, MoneyPuck parsing, team strength, goalie, preseason prior."""

from __future__ import annotations

import json
import math
from datetime import date as Date
from pathlib import Path

import pandas as pd
import pytest

from engine_common import odds, shrink
from nhl_engine.data import preseason
from nhl_engine.data.moneypuck import (
    FRANCHISE_PREDECESSOR,
    MoneyPuckClient,
    as_of,
    mp_code,
    parse_goalie_games,
    parse_team_games,
    season_of,
)
from nhl_engine.data.oddsapi import outright_prices
from nhl_engine.features import goalie, strength

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def tor() -> pd.DataFrame:
    return parse_team_games((FIX / "mp_team_TOR.csv").read_text())


@pytest.fixture
def woll() -> pd.DataFrame:
    return parse_goalie_games((FIX / "mp_goalie_8480045.csv").read_text())


# --- engine_common -----------------------------------------------------------


def test_eb_posterior_endpoints() -> None:
    assert shrink.eb_posterior(2.5, 4.0, n=0, k=100) == 2.5
    assert shrink.eb_posterior(2.5, 4.0, n=100, k=100) == pytest.approx(3.25)
    assert shrink.eb_posterior(2.5, 4.0, n=1e12, k=100) == pytest.approx(4.0, abs=1e-6)
    assert shrink.reliability_at(0, 100) == 0.0
    assert shrink.reliability_at(300, 100) == pytest.approx(0.75)


def test_k_from_split_half_matches_reliability() -> None:
    k = shrink.k_from_split_half(0.5, n_half=41)
    assert k == pytest.approx(41)
    assert shrink.reliability_at(41, k) == pytest.approx(0.5)
    assert math.isinf(shrink.k_from_split_half(0.0, 41))
    assert shrink.k_from_split_half(1.0, 41) == 0.0


def test_pearson_and_spearman_brown() -> None:
    assert shrink.pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert math.isnan(shrink.pearson([1, 2, 3], [5, 5, 5]))
    assert shrink.spearman_brown(0.5) == pytest.approx(2 / 3)


def test_odds_roundtrip_and_devig() -> None:
    assert odds.american_to_prob(-150) == pytest.approx(0.6)
    assert odds.american_to_prob(+200) == pytest.approx(1 / 3)
    assert odds.prob_to_american(odds.american_to_prob(-137)) == pytest.approx(-137, abs=1e-6)
    fair = odds.devig({"a": -120, "b": +100})
    assert sum(fair.values()) == pytest.approx(1.0)
    assert fair["a"] > fair["b"]
    assert odds.overround({"a": -110, "b": -110}) == pytest.approx(0.0476, abs=1e-3)


# --- moneypuck ---------------------------------------------------------------


def test_team_parse_normalises_codes_and_dates(tor: pd.DataFrame) -> None:
    assert set(tor["situation"]) == {"all", "5on5", "5on4", "4on5", "other"}
    assert tor["gameDate"].iloc[0] == Date(2023, 10, 11)
    assert set(tor["season"]) == {2023, 2024}
    assert mp_code("L.A") == "LAK" and mp_code("T.B") == "TBL" and mp_code("TOR") == "TOR"
    assert FRANCHISE_PREDECESSOR["UTA"] == "ARI"


def test_as_of_excludes_slate_day_and_future(tor: pd.DataFrame) -> None:
    dates = sorted(tor["gameDate"].unique())
    cut = as_of(tor, dates[7], season=2024)
    assert cut["gameDate"].max() < dates[7]
    assert set(cut["season"]) == {2024}
    assert as_of(tor, Date(2024, 10, 1), season=2024).empty


def test_season_of() -> None:
    assert season_of(Date(2026, 9, 29)) == 2026
    assert season_of(Date(2026, 1, 15)) == 2025
    assert season_of(Date(2026, 8, 31)) == 2025


def test_client_reads_cache_offline(tmp_path: Path, tor: pd.DataFrame) -> None:
    cache = tmp_path / "moneypuck"
    cache.mkdir()
    (cache / "careers_gameByGame_regular_teams_TOR.csv").write_text(
        (FIX / "mp_team_TOR.csv").read_text()
    )
    mp = MoneyPuckClient(cache_dir=tmp_path, cache_ttl=10**9)
    assert len(mp.team_games("TOR")) == len(tor)
    assert mp.team_games("ZZZ").empty  # no file, no network in tests -> typed empty


# --- strength ----------------------------------------------------------------


def test_game_table_one_row_per_game_with_rates(tor: pd.DataFrame) -> None:
    table = strength.game_table(tor)
    assert len(table) == 10
    r, exp = strength.rate(table, "xgf60_5v5")
    g5 = tor[tor["situation"] == "5on5"]
    assert exp == pytest.approx(g5["iceTime"].sum())
    assert r == pytest.approx(g5["scoreVenueAdjustedxGoalsFor"].sum() / exp * 3600)
    # PP metrics use the PP state's own exposure
    _, pp_exp = strength.rate(table, "pp_xgf60")
    assert pp_exp == pytest.approx(tor[tor["situation"] == "5on4"]["iceTime"].sum())


def test_estimate_zero_games_is_the_prior(tor: pd.DataFrame) -> None:
    est = strength.team_strength(
        tor, Date(2024, 10, 1), season=2024, priors={"xgf60_5v5": 2.6}, ks={"xgf60_5v5": 30155.0}
    )
    e = est["xgf60_5v5"]
    assert e.games == 0 and e.posterior == 2.6 and e.reliability == 0.0


def test_estimate_moves_toward_data_with_exposure(tor: pd.DataFrame) -> None:
    table = strength.game_table(as_of(tor, Date(2024, 10, 22), season=2024))
    e = strength.estimate(table, "xgf60_5v5", prior=2.6, k=30155.0)
    assert 0 < e.reliability < 1
    assert min(2.6, e.observed) <= e.posterior <= max(2.6, e.observed)
    assert e.posterior == pytest.approx(shrink.eb_posterior(2.6, e.observed, e.exposure, 30155.0))


def test_every_metric_has_a_fitted_k() -> None:
    from nhl_engine.config import ShrinkParams

    assert set(ShrinkParams().team_k) == {m.key for m in strength.METRICS}


# --- goalie ------------------------------------------------------------------


def test_goalie_table_tracks_rest_and_career(woll: pd.DataFrame) -> None:
    t = goalie.goalie_game_table(woll)
    assert len(t) == 8
    assert t["career_toi"].iloc[0] == 0.0
    assert t["career_toi"].is_monotonic_increasing
    assert t["days_rest"].iloc[1] == 3  # 2023-10-24 -> 10-27
    r, exp = goalie.gsax60(t)
    w = woll[woll["situation"] == "all"]
    assert exp == pytest.approx(w["icetime"].sum())
    assert r == pytest.approx((w["xGoals"].sum() - w["goals"].sum()) / exp * 3600)
    assert 0.8 < goalie.svpct(t) < 1.0


def test_goalie_skill_callup_anchor_and_reliability(woll: pd.DataFrame) -> None:
    # first appearance of 2024: zero season minutes, three prior-season games
    s = goalie.goalie_skill(
        woll,
        Date(2024, 10, 4),
        season=2024,
        league_gsax60=0.0,
        k_season=349639.0,
        k_career=349639.0,
        r_yy=0.14,
        min_minutes=180.0,
        callup_prior=-0.229,
    )
    assert s.games_season == 0 and s.season_toi == 0.0
    assert s.reliability == 0.0
    assert s.anchor in {"league", "callup"}
    assert s.svpct_season != s.svpct_season or s.svpct_season >= 0  # display-only, may be nan
    # a goalie with a huge floor is always a call-up and is anchored below league
    c = goalie.goalie_skill(
        woll,
        Date(2024, 10, 4),
        season=2024,
        league_gsax60=0.0,
        k_season=349639.0,
        k_career=349639.0,
        r_yy=0.14,
        min_minutes=1e9,
        callup_prior=-0.229,
    )
    assert c.is_callup and c.gsax60 < 0.0


# --- preseason prior ---------------------------------------------------------


class _FakeMP:
    def __init__(self, tor: pd.DataFrame) -> None:
        self._tor = tor

    def team_games(self, code: str) -> pd.DataFrame:
        if code in {"TOR", "ARI"}:
            return self._tor
        return self._tor.iloc[0:0]


def test_outright_prices_from_fixture() -> None:
    payload = json.loads((FIX / "outrights.json").read_text())
    prices = outright_prices(payload)
    assert len(prices) == 32
    assert all(len(v) == 2 for v in prices.values())
    probs = preseason.futures_probs(prices)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_build_prior_regresses_and_handles_relocation(tor: pd.DataFrame) -> None:
    prior = preseason.build(
        _FakeMP(tor),  # type: ignore[arg-type]
        2024,
        r_yy={"xgf60_5v5": 0.5},
        codes=frozenset({"TOR", "UTA", "BOS"}),
        today=Date(2024, 10, 1),
    )
    assert set(prior.teams) == {"TOR", "UTA"}  # BOS has no games -> league fallback
    t = prior.teams["TOR"]
    lg = prior.league["xgf60_5v5"]
    assert t.source_season == 2023 and t.source_games == 4
    assert t.rates["xgf60_5v5"] == pytest.approx(lg + 0.5 * (t.raw_prior_season["xgf60_5v5"] - lg))
    assert t.rates["xga60_5v5"] == pytest.approx(
        lg if False else prior.league["xga60_5v5"]
    )  # r_yy=0
    assert prior.teams["UTA"].source_code == "ARI"
    assert prior.rates_for("BOS") == prior.league
    assert prior.components["roster_carryover"].startswith("not_fitted")
    assert prior.components["futures"] == "not_available"


def test_prior_roundtrip_and_staleness(tor: pd.DataFrame, tmp_path: Path) -> None:
    prior = preseason.build(
        _FakeMP(tor),  # type: ignore[arg-type]
        2024,
        r_yy={},
        codes=frozenset({"TOR"}),
        today=Date(2024, 10, 1),
    )
    path = preseason.save(prior, tmp_path)
    assert path.name == "preseason_prior_2024.json"
    back = preseason.load(tmp_path, 2024)
    assert back == prior
    assert preseason.load(tmp_path, 2023) is None
    freeze = Date(2024, 10, 31)
    assert preseason.is_stale(None, Date(2024, 10, 5), freeze_after=freeze)
    assert not preseason.is_stale(prior, Date(2024, 10, 5), freeze_after=freeze)
    assert preseason.is_stale(prior, Date(2024, 10, 20), freeze_after=freeze)
    assert not preseason.is_stale(prior, Date(2025, 1, 20), freeze_after=freeze)  # frozen
