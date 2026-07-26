"""Run-line NPV gates, walk-off truncation, and gate attribution in the ledger."""

from __future__ import annotations

from datetime import date

import pandas as pd

from mlb_engine.audit.ledger import LedgerEntry, runline_metrics
from mlb_engine.config import RunLineGates
from mlb_engine.features.efficiency import recent_start_form
from mlb_engine.features.rolling import batter_iso, build_bullpen_profile, lineup_iso
from mlb_engine.market.runline import RunLineSignal, runline_veto
from mlb_engine.models.montecarlo import MonteCarlo, TeamSimConfig

BASE_RATES = {
    "1B": 0.14,
    "2B": 0.045,
    "3B": 0.004,
    "HR": 0.032,
    "BB": 0.085,
    "K": 0.225,
    "OUT": 0.469,
}


# --- walk-off truncation ---------------------------------------------------
def test_home_never_wins_by_more_than_the_walkoff_margin_late() -> None:
    """A game ending in the bottom of the 9th+ cannot post a blowout home margin.

    The half-inning stops the instant the home team takes the lead, so the only
    way home wins by 2+ is a walk-off homer with runners aboard -- never the
    phantom runs a fully played-out half-inning would add.
    """
    bat = [dict(BASE_RATES) for _ in range(9)]
    cfg = TeamSimConfig(bat_vs_starter=bat, bat_vs_pen=bat)
    res = MonteCarlo(4000, seed=7).simulate(cfg, cfg)
    margin = res.home_runs_full - res.away_runs_full
    # Even with identical lineups the margin distribution is asymmetric, exactly
    # as it is in real baseball: home wins pile up at +1 (the game ends there),
    # and the away side covers +1.5 more often than the home side does.
    assert float((margin == 1).mean()) > float((margin == -1).mean()) + 0.02
    assert float((margin < -1.5).mean()) > float((margin > 1.5).mean())
    # Moneyline stays a coin flip -- truncation removes runs, not wins.
    assert abs(float((margin > 0).mean()) - float((margin < 0).mean())) < 0.03


def test_home_scores_fewer_runs_than_an_identical_away_offense() -> None:
    """Home forfeits the bottom of the 9th whenever it is ahead or goes ahead.

    With the same lineup on both sides, that lost (or truncated) half-inning is
    the only difference between the two teams, so home must score less.
    """
    bat = [dict(BASE_RATES) for _ in range(9)]
    cfg = TeamSimConfig(bat_vs_starter=bat, bat_vs_pen=bat)
    res = MonteCarlo(4000, seed=3).simulate(cfg, cfg)
    assert res.home_runs_full.mean() < res.away_runs_full.mean()
    # F5 is untouched by the 9th-inning rule.
    assert abs(res.home_runs_f5.mean() - res.away_runs_f5.mean()) < 0.15


# --- gate predicates -------------------------------------------------------
def _fav_signal(**kw) -> RunLineSignal:
    base = dict(fav_side="home", fav_iso=0.120, fav_opp_sp_gb_pct=0.56)
    base.update(kw)
    return RunLineSignal(**base)


def test_gate_iso_gb_vetoes_only_the_favorite_when_enabled() -> None:
    gates = RunLineGates(iso_gb=True)
    veto = runline_veto("home", -1.5, _fav_signal(), gates)
    assert veto.gate == "fav_iso_x_gb"
    assert "0.120" in (veto.detail or "")
    # The dog's +1.5 in the same game is untouched by a favorite gate.
    assert not runline_veto("away", 1.5, _fav_signal(), gates).triggered


def test_gate_iso_gb_is_inert_when_disabled_or_below_threshold() -> None:
    assert not runline_veto("home", -1.5, _fav_signal(), RunLineGates()).triggered
    gates = RunLineGates(iso_gb=True)
    assert not runline_veto("home", -1.5, _fav_signal(fav_iso=0.190), gates).triggered
    assert not runline_veto("home", -1.5, _fav_signal(fav_opp_sp_gb_pct=0.41), gates).triggered


def test_missing_inputs_never_veto() -> None:
    """A thin sample leaves the selection alone rather than killing volume."""
    gates = RunLineGates(iso_gb=True, dog_sp=True, dog_pen=True, low_total=True)
    blank = RunLineSignal(fav_side="home")
    assert not runline_veto("home", -1.5, blank, gates).triggered
    assert not runline_veto("away", 1.5, blank, gates).triggered
    half_known = _fav_signal(fav_iso=None)
    assert not runline_veto("home", -1.5, half_known, gates).triggered


def test_gate_dog_sp_needs_both_whip_and_hard_hit() -> None:
    gates = RunLineGates(dog_sp=True)
    sig = RunLineSignal(fav_side="home", dog_sp_whip_l3=1.62, dog_sp_hard_hit_l3=0.48)
    assert runline_veto("away", 1.5, sig, gates).gate == "dog_sp_blowout"
    # Hard contact alone (good control) is not the blowout script.
    tidy = RunLineSignal(fav_side="home", dog_sp_whip_l3=1.10, dog_sp_hard_hit_l3=0.48)
    assert not runline_veto("away", 1.5, tidy, gates).triggered
    # The favorite's -1.5 is not a dog selection.
    assert not runline_veto("home", -1.5, sig, gates).triggered


def test_gate_dog_pen_vetoes_leaky_low_k_bullpen() -> None:
    gates = RunLineGates(dog_pen=True)
    sig = RunLineSignal(fav_side="away", dog_pen_xwoba=0.352, dog_pen_k_pct=0.163)
    assert runline_veto("home", 1.5, sig, gates).gate == "dog_pen_leak"
    missy = RunLineSignal(fav_side="away", dog_pen_xwoba=0.352, dog_pen_k_pct=0.240)
    assert not runline_veto("home", 1.5, missy, gates).triggered


def test_gate_low_total_is_off_by_default_and_fires_when_enabled() -> None:
    sig = RunLineSignal(fav_side="home", model_total=6.4)
    assert not runline_veto("home", -1.5, sig, RunLineGates()).triggered
    assert runline_veto("home", -1.5, sig, RunLineGates(low_total=True)).gate == "low_total"
    assert not runline_veto("away", 1.5, sig, RunLineGates(low_total=True)).triggered


def test_gates_ignore_non_runline_selections() -> None:
    gates = RunLineGates(iso_gb=True, low_total=True)
    sig = _fav_signal(model_total=5.0)
    assert not runline_veto("home", None, sig, gates).triggered
    assert not runline_veto(None, -1.5, sig, gates).triggered


def test_env_flags_drive_the_gates(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_RL_GATE_ISO_GB", "1")
    monkeypatch.setenv("MLBE_RL_ISO_MAX", "0.200")
    gates = RunLineGates()
    assert gates.iso_gb and gates.iso_max == 0.200
    assert runline_veto("home", -1.5, _fav_signal(fav_iso=0.180), gates).triggered


# --- feature inputs --------------------------------------------------------
def test_batter_iso_matches_slg_minus_avg_and_guards_thin_samples() -> None:
    events = pd.Series(["home_run"] * 10 + ["field_out"] * 30 + ["walk"] * 5)
    # 10 HR / 40 AB: SLG .1000... -> TB 40, hits 10 => ISO (40-10)/40 = .750
    assert batter_iso(events) == 0.75
    assert batter_iso(pd.Series(["home_run"] * 5)) is None


def test_lineup_iso_skips_batters_without_enough_at_bats() -> None:
    as_of = date(2024, 7, 20)
    rows = []
    for batter, n_hr in ((1, 10), (2, 0)):
        for i in range(45):
            rows.append(
                {
                    "batter": batter,
                    "game_date": date(2024, 7, 10),
                    "events": "home_run" if i < n_hr else "field_out",
                }
            )
    rows.append({"batter": 3, "game_date": date(2024, 7, 10), "events": "home_run"})
    df = pd.DataFrame(rows)
    iso = lineup_iso(df, [1, 2, 3], as_of, days=30)
    assert iso is not None
    # Batter 3 (1 AB) is excluded; mean of .667 and .000.
    assert abs(iso - (10 * 3 / 45) / 2) < 1e-9
    assert lineup_iso(df, [3], as_of, days=30) is None


def test_recent_start_form_uses_the_last_three_starts_only() -> None:
    as_of = date(2024, 7, 20)
    rows = []
    starts = {
        date(2024, 6, 1): ("single", 100.0),  # older start, must be excluded
        date(2024, 7, 5): ("single", 100.0),
        date(2024, 7, 11): ("single", 100.0),
        date(2024, 7, 17): ("single", 100.0),
    }
    for d, (ev, ls) in starts.items():
        for _ in range(6):
            rows.append({"game_date": d, "events": ev, "launch_speed": ls})
        for _ in range(9):
            rows.append({"game_date": d, "events": "field_out", "launch_speed": 70.0})
    df = pd.DataFrame(rows)
    form = recent_start_form(df, as_of)
    assert form is not None and form.starts == 3
    assert form.whip > 1.45 and form.hard_hit_pct == 0.4
    # Only two starts on record -> no signal.
    two = df[df["game_date"] > date(2024, 7, 6)]
    assert recent_start_form(two, as_of) is None


def test_bullpen_profile_exposes_xwoba_and_k_rate() -> None:
    rows = []
    for i in range(60):
        rows.append(
            {
                "game_date": date(2024, 7, 10),
                "inning": 7,
                "events": "strikeout" if i % 2 else "single",
                "pitcher": 100 + i,
                "home_team": "DET",
                "away_team": "KC",
                "inning_topbot": "Top",
                "estimated_woba_using_speedangle": 0.34,
                "zone": 5,
            }
        )
    prof = build_bullpen_profile(pd.DataFrame(rows), "DET", date(2024, 7, 20), 21, 6)
    assert prof.xwoba_allowed is not None and abs(prof.xwoba_allowed - 0.34) < 1e-9
    assert 0.0 < prof.k_pct < 1.0


# --- ledger attribution ----------------------------------------------------
def _entry(line: float, result: str, gate: str = "", prob: float = 0.6) -> LedgerEntry:
    return LedgerEntry(
        date="2024-07-19",
        matchup="KC @ DET",
        category="Run Lines",
        market="game_rl",
        selection="DET -1.5",
        line=line,
        book="dk",
        odds=-110,
        tier="Pass" if gate else "Strong Buy",
        model_prob=prob,
        ev=0.05,
        result=result,
        pnl=0.91 if result == "win" else -1.0,
        veto_gate=gate,
    )


def test_runline_metrics_grades_what_each_gate_removed() -> None:
    entries = [
        _entry(-1.5, "loss", "fav_iso_x_gb"),
        _entry(-1.5, "loss", "fav_iso_x_gb"),
        _entry(-1.5, "win", "fav_iso_x_gb"),
        _entry(1.5, "loss", "dog_pen_leak"),
        _entry(-1.5, "win"),
        _entry(1.5, "win"),
    ]
    rows = {m.tier: m for m in runline_metrics(entries)}
    assert rows["ALL RUN LINES"].n == 6
    assert rows["FAVORITE (-1.5)"].n == 4
    assert rows["UNDERDOG (+1.5)"].n == 2
    # The gate's own row: it removed two losers and one winner.
    assert (rows["VETO fav_iso_x_gb"].wins, rows["VETO fav_iso_x_gb"].losses) == (1, 2)
    assert rows["VETO dog_pen_leak"].n == 1
    assert (rows["KEPT (no veto)"].wins, rows["KEPT (no veto)"].losses) == (2, 0)
    assert runline_metrics([]) == []
