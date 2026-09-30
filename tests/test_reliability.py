"""The measured reliability curves, and the screen's use of them.

What the tests pin is not the numbers -- those are measurements and will be
remeasured -- but the shape the code has to give them: more plate appearances is
never less signal, a metric that never repeats must never be readable, and a
metric the pool ranks a hitter first in must not carry him through a cut at a
sample where it is noise.
"""

from __future__ import annotations

import math

from mlb_engine.features import reliability as rel
from mlb_engine.output.power_screen import HitterLine, score_pool


def _line(name: str, pa: float, **kw: float) -> HitterLine:
    base: dict[str, float] = {
        "pa": pa,
        "wrc": 100.0,
        "woba": 0.320,
        "obp": 0.320,
        "slg": 0.400,
        "ops": 0.720,
        "ba": 0.250,
        "xba": 0.250,
        "xslg": 0.400,
        "xwoba_pa": 0.320,
        "xwoba_con": 0.370,
        "k": 0.220,
        "bb": 0.080,
        "brl": 0.070,
        "hh": 0.390,
        "ev90": 104.0,
        "osw": 0.300,
    }
    base.update(kw)
    return HitterLine(
        name=name,
        mlbam_id=abs(hash(name)) % 10**6,
        team="AAA",
        slot=3,
        bats="L",
        versus="Some Arm",
        **{k: float(v) for k, v in base.items()},  # type: ignore[arg-type]
    )


def test_reliability_never_falls_as_the_sample_grows() -> None:
    for metric in rel.CURVES:
        values = [rel.reliability(metric, pa) for pa in (5, 15, 40, 90, 180, 250, 600)]
        assert values == sorted(values), metric
        assert all(0.0 <= v <= 1.0 for v in values), metric


def test_reliability_interpolates_between_measured_points() -> None:
    lo = rel.reliability("hh", 130)
    hi = rel.reliability("hh", 180)
    mid = rel.reliability("hh", 155)
    assert lo < mid < hi


def test_a_sample_below_the_first_measured_point_is_not_extrapolated_down() -> None:
    assert rel.reliability("bat_speed", 1) == rel.reliability("bat_speed", 15)


def test_the_metrics_that_never_repeat_are_never_readable() -> None:
    for metric in ("ba", "slg", "ops", "wrc"):
        assert rel.never_readable(metric), metric
        assert not rel.readable(metric, 700), metric
    assert not rel.never_readable("hh")


def test_bat_speed_is_readable_at_a_sample_hard_hit_rate_is_not() -> None:
    assert rel.readable("bat_speed", 20)
    assert not rel.readable("hh", 20)
    assert rel.readable("hh", 250)


def test_an_unmeasured_metric_is_taken_at_face_value() -> None:
    assert rel.reliability("brand_new_stat", 30) == 1.0
    assert rel.readable("brand_new_stat", 1)


def test_a_top_finish_on_an_unreadable_metric_is_withheld_not_counted() -> None:
    """A hitter can lead the pool in batting average and get no credit for it.

    Batting average never reaches r=.50 with itself, so leading the pool in it
    says nothing about tonight. The finish is recorded in ``withheld`` rather
    than dropped, because the reader is entitled to know the screen saw it and
    refused it.
    """
    lucky = _line("Lucky", pa=60, ba=0.400, ops=1.100, wrc=180.0, slg=0.700)
    steady = _line("Steady", pa=260, ba=0.240, ops=0.700, wrc=95.0, slg=0.380)
    score_pool([lucky, steady], top_k=1)

    assert "BA" in lucky.withheld
    assert "BA" not in lucky.top_in
    assert "OPS" in lucky.withheld
    assert "wRC+" in lucky.withheld


def test_the_same_finish_counts_once_the_sample_makes_it_readable() -> None:
    heavy = _line("Heavy", pa=300, hh=0.520, ev90=108.0)
    light = _line("Light", pa=300, hh=0.300, ev90=101.0)
    score_pool([heavy, light], top_k=1)

    assert "HH%" in heavy.top_in
    assert "HH%" not in heavy.withheld
    assert "EV90" in heavy.top_in


def test_a_reliable_metric_outscores_a_noisy_one_by_the_measured_ratio() -> None:
    """Two hitters, one first in bat-speed-grade power, one first in results.

    The point of the weighting is that these are not worth the same. EV90 at 300
    PA is most of a real number; wRC+ at 300 PA is barely a fifth of one, and the
    score has to say so.
    """
    power = _line("Power", pa=300, ev90=109.0)
    results = _line("Results", pa=300, wrc=190.0, ops=1.050)
    score_pool([power, results], top_k=1)

    assert power.score > results.score
    assert math.isclose(power.points, round(power.score))
