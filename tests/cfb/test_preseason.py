"""Makinen / TeamRankings preseason ratings and VSiN stability shrinkage."""

from __future__ import annotations

from cfb_engine.data.preseason import (
    MAKINEN,
    STABILITY,
    TEAMRANKINGS,
    stability_factor,
)
from cfb_engine.data.teamnames import school_key


def test_all_three_tables_cover_138_fbs_teams():
    assert len(MAKINEN) == 138
    assert len(TEAMRANKINGS) == 138
    assert len(STABILITY) == 138


def test_representative_makinen_values():
    assert MAKINEN[school_key("Ohio State")] == 71.0
    assert MAKINEN[school_key("Georgia")] == 67.5
    assert MAKINEN[school_key("Massachusetts")] == 16.0
    assert MAKINEN[school_key("Notre Dame")] == 68.5


def test_representative_teamrankings_values():
    assert TEAMRANKINGS[school_key("Ohio State")] == 32.3
    assert TEAMRANKINGS[school_key("UMass")] == -28.5
    assert TEAMRANKINGS[school_key("Iowa State")] == 0.0


def test_source_abbreviations_reconcile_to_cfbd_keys():
    # Guide abbreviations are canonicalized at build time; runtime lookups use
    # CFBD/odds names, which must land on those same keys.
    assert MAKINEN[school_key("UTSA")] == 41.0
    assert TEAMRANKINGS[school_key("San José State")] == TEAMRANKINGS[school_key("San Jose St")]
    assert MAKINEN[school_key("Louisiana-Monroe")] == 22.0
    # Mascot names that previously failed to normalize now resolve.
    assert school_key("North Dakota State Bison") == "north dakota state"
    assert school_key("Sacramento State Hornets") == "sacramento state"
    assert school_key("Hawaii Rainbow Warriors") == "hawaii"


def test_representative_stability_scores():
    assert STABILITY[school_key("Notre Dame")] == 18
    assert STABILITY[school_key("Arizona")] == 17
    assert STABILITY[school_key("Virginia Tech")] == 4


def test_stability_factor_bounds_and_direction():
    # Two fully-stable teams keep the whole gap (factor 1.0).
    high = stability_factor("Houston", "Houston")  # Houston stability 18
    # Two volatile teams shrink hardest (toward the min-keep floor of 0.75).
    low = stability_factor("Virginia Tech", "Virginia Tech")  # stability 4
    assert 0.75 <= low < high <= 1.0


def test_stability_factor_missing_team_uses_mid_reliability():
    factor = stability_factor("Some Nonexistent School", "Another Fake School")
    # Default reliability 0.6 -> 0.75 + 0.25*0.6 = 0.90.
    assert abs(factor - 0.90) < 1e-9


def test_stability_factor_disabled_is_noop():
    assert stability_factor("Virginia Tech", "Virginia Tech", enabled=False) == 1.0


def test_stability_haircut_ships_off(monkeypatch):
    # Measured worse at every dose (scripts/cfb/stability_study.py), so the
    # pipeline must not apply it unless the operator opts in.
    from cfb_engine.config import Config

    monkeypatch.delenv("CFBE_VSIN_STABILITY", raising=False)
    assert Config().vsin_stability is False
    monkeypatch.setenv("CFBE_VSIN_STABILITY", "1")
    assert Config().vsin_stability is True
