import random

from engine_runtime import faststart
from engine_runtime.faststart import Obs


def _games(b: float, *, n: int = 600, slow_b: float | None = None, seed: int = 3) -> list[Obs]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        total = rng.uniform(130, 160)
        gap = rng.gauss(0, 8)
        slope = slow_b if (slow_b is not None and gap < 0) else b
        out.append(
            Obs(
                f"g{i}",
                f"2026-11-{1 + i // 20:02d}",
                total,
                0.47 * total + gap,
                total + slope * gap + rng.gauss(0, 2),
            )
        )
    return out


def test_share_uses_only_earlier_days() -> None:
    obs = [Obs(f"a{i}", "2026-11-01", 100.0, 50.0, 100.0) for i in range(30)]
    obs += [Obs("b", "2026-11-02", 100.0, 90.0, 140.0)]
    rows = faststart.gaps(obs)
    assert [(o.game_id, round(s, 3), round(g, 3)) for o, s, g in rows] == [("b", 0.5, 40.0)]


def test_no_read_before_min_prior_games() -> None:
    assert (
        faststart.read("half", [Obs(f"a{i}", "2026-11-01", 100.0, 50.0, 100.0) for i in range(40)])
        is None
    )


def test_slope_of_one_holds() -> None:
    r = faststart.read("half", _games(1.0))
    assert r is not None and abs(r.b - 1.0) < 0.03 and r.se < faststart.SE_TARGET
    assert r.verdict.startswith("holds") and r.converse == "same both ways"
    assert r.rmse_fit <= r.rmse_frozen + 1e-9


def test_carry_over_and_asymmetric_slow_starts_are_called() -> None:
    r = faststart.read("half", _games(1.2, slow_b=0.8))
    assert r is not None
    assert abs(r.b_fast - 1.2) < 0.05 and abs(r.b_slow - 0.8) < 0.05
    assert r.converse.startswith("differ")


def test_small_sample_says_underpowered_with_games_needed() -> None:
    obs = [o for o in _games(1.0, n=60) if True]
    noisy = [
        Obs(o.game_id, o.day, o.pregame, o.so_far, o.final + (i % 7 - 3) * 6)
        for i, o in enumerate(obs)
    ]
    r = faststart.read("half", noisy)
    assert r is not None and r.verdict.startswith("underpowered (needs ~")
    assert faststart.table([r])[0][0] == "half"


class _Q:
    def __init__(
        self, at: str, eid: str, line: float, side: str = "over", market: str = "game_total"
    ) -> None:
        self.captured_at, self.event_id, self.market, self.side, self.line = (
            at,
            eid,
            market,
            side,
            line,
        )


def test_close_totals_take_the_newest_timely_pre_tip_capture() -> None:
    tip = "2026-11-02T23:00:00Z"
    qs = [
        _Q("2026-11-02T22:20:00Z", "e", 220.0),
        _Q("2026-11-02T22:50:00Z", "e", 222.0),
        _Q("2026-11-02T22:50:00Z", "e", 224.0),
        _Q("2026-11-02T22:50:00Z", "e", 230.0, side="under"),
        _Q("2026-11-02T23:05:00Z", "e", 240.0),  # in play: not a close
        _Q("2026-11-02T21:00:00Z", "old", 210.0),  # 2 h before tip: stale
    ]
    tips = {"e": tip, "old": tip}
    assert faststart.close_totals(qs, tips) == {"e": 223.0}


def test_checkpoints_cut_the_periods() -> None:
    out = faststart.checkpoints(
        "g", "2026-11-02", 220.0, [60, 50, 55, 57, 10], 232.0, {"Q1": 1, "half": 2}
    )
    assert out["Q1"].so_far == 60 and out["half"].so_far == 110 and out["half"].final == 232.0
