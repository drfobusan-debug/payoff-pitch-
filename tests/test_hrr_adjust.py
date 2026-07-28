"""Unit tests for the H+R+RBI probability adjuster."""

from __future__ import annotations

from mlb_engine.features.hrr_adjust import (
    HRR_SWEET_CENTER,
    HRR_XSLG_CENTER,
    HRRAdjuster,
)


def test_o15_overconfidence_shrink_pulls_high_prob_down() -> None:
    adj = HRRAdjuster(enabled=True, pivot=0.45, shrink=0.5, tilt_w=0.0)
    out = adj.apply(0.65, line=1.5, sweet_spot=None, xslg=None)
    # 0.45 + (0.65 - 0.45) * 0.5 = 0.55
    assert abs(out - 0.55) < 1e-9


def test_o15_shrink_leaves_low_prob_untouched() -> None:
    adj = HRRAdjuster(enabled=True, pivot=0.45, shrink=0.5, tilt_w=0.0)
    out = adj.apply(0.30, line=1.5, sweet_spot=None, xslg=None)
    assert out == 0.30


def test_o25_not_shrunk() -> None:
    adj = HRRAdjuster(enabled=True, pivot=0.45, shrink=0.5, tilt_w=0.0)
    out = adj.apply(0.65, line=2.5, sweet_spot=None, xslg=None)
    assert out == 0.65


def test_contact_quality_tilt_up_and_down() -> None:
    adj = HRRAdjuster(enabled=True, shrink=1.0, tilt_w=0.30, tilt_cap=0.03)
    hi = adj.apply(
        0.30, line=2.5,
        sweet_spot=HRR_SWEET_CENTER + 0.05, xslg=HRR_XSLG_CENTER + 0.05,
    )
    lo = adj.apply(
        0.30, line=2.5,
        sweet_spot=HRR_SWEET_CENTER - 0.05, xslg=HRR_XSLG_CENTER - 0.05,
    )
    assert hi > 0.30 > lo


def test_tilt_is_capped() -> None:
    adj = HRRAdjuster(enabled=True, shrink=1.0, tilt_w=0.30, tilt_cap=0.03)
    out = adj.apply(0.30, line=2.5, sweet_spot=0.90, xslg=0.90)
    assert abs(out - (0.30 + 0.03)) < 1e-9


def test_disabled_is_noop() -> None:
    adj = HRRAdjuster(enabled=False)
    out = adj.apply(0.65, line=1.5, sweet_spot=0.90, xslg=0.90)
    assert out == 0.65


def test_missing_metrics_only_shrinks() -> None:
    adj = HRRAdjuster(enabled=True, pivot=0.45, shrink=0.5, tilt_w=0.30)
    out = adj.apply(0.65, line=1.5, sweet_spot=None, xslg=None)
    assert abs(out - 0.55) < 1e-9


def test_from_env_defaults(monkeypatch) -> None:
    for k in (
        "MLBE_HRR_ADJUST",
        "MLBE_HRR_PIVOT",
        "MLBE_HRR_SHRINK",
        "MLBE_HRR_TILT_W",
        "MLBE_HRR_TILT_CAP",
    ):
        monkeypatch.delenv(k, raising=False)
    adj = HRRAdjuster.from_env()
    assert adj.enabled is True
    assert adj.pivot == 0.45
    assert adj.shrink == 0.70


def test_from_env_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_HRR_ADJUST", "0")
    assert HRRAdjuster.from_env().enabled is False
