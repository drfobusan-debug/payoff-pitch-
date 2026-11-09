"""Close vs results by price range: the devig check behind close-implied ROI."""

from engine_common import closecheck


def _g(p: float, american: float, won: bool) -> closecheck.Graded:
    dec = 1 + (american / 100 if american > 0 else 100 / -american)
    return closecheck.Graded(p, american, won, dec - 1 if won else -1.0)


def test_buckets_by_closing_probability_and_reads_hit_rate_against_it():
    rows = [_g(0.55, -120, i % 2 == 0) for i in range(40)]  # pick: 50% hit vs 55% close
    rows += [_g(0.15, +600, i < 2) for i in range(20)]  # long shots: 10% hit vs 15% close
    out = closecheck.read(rows)
    assert [b.label.split()[0] for b in out] == ["long", "pick"]
    pick = out[1]
    assert (pick.n, pick.close_prob, pick.hit, pick.gap) == (40, 0.55, 0.5, -0.05)
    assert pick.close_roi == round(0.55 * (1 + 100 / 120) - 1, 4)
    assert pick.verdict == "fits"  # 5 pts on 40 games is inside 2 SE
    assert out[0].n == 20 and closecheck.row(out[0])[1] == "20"


def test_a_two_se_gap_is_called_and_singletons_get_no_verdict():
    chalk = [_g(0.80, -400, i < 60) for i in range(100)]  # 60% hit vs 80% close
    (b,) = closecheck.read(chalk)
    assert b.verdict == "close too high" and b.gap == -0.2
    (one,) = closecheck.read([_g(0.5, +100, True)])
    assert one.verdict == "-" and one.gap_se == 0.0
    assert closecheck.read([]) == []
