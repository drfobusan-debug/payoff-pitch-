from lineshop import distribution as dist


def test_key_numbers_are_lumpier_than_their_neighbours():
    # The whole reason half a point is worth buying: 3 and 7 carry several times
    # the mass of the numbers either side of them, in both sports.
    for sport in dist.SPORTS:
        factors = dist.load(sport).key_factors["margin"]
        assert factors[3.0] > 2.0
        assert factors[7.0] > 2.0
        assert factors[4.0] < 1.0
        assert factors[5.0] < 1.0


def test_the_lump_is_conditional_on_the_line():
    # A 3-point margin is common when the game is priced at 3 and rare when it
    # is priced at 21 -- pricing a crossing off the unconditional rate is the
    # error this module exists to avoid.
    near = dist.p_at("cfb", dist.MARGIN, 3.0, 3.0).p
    far = dist.p_at("cfb", dist.MARGIN, 21.0, 3.0).p
    assert near > 0.04
    assert far < 0.6 * near


def test_distribution_sums_to_one_and_centres_on_the_line():
    pmf, n, _ = dist.pmf("nfl", dist.MARGIN, 6.5)
    assert n > 400
    assert abs(sum(pmf.values()) - 1.0) < 1e-9
    mean = sum(v * p for v, p in pmf.items())
    assert 3.0 < mean < 10.0


def test_a_window_is_the_sum_of_the_numbers_inside_it():
    line = 4.0
    window = dist.p_between("nfl", dist.MARGIN, line, 1.0, 4.0)
    inside = sum(dist.p_at("nfl", dist.MARGIN, line, v).p for v in (2.0, 3.0))
    assert abs(window.p - inside) < 1e-9


def test_thin_samples_are_pulled_toward_the_fitted_shape():
    # At a 45-point line a few hundred games leave 2-3% of noise on every
    # integer, which is the size of the thing being measured. The shrink has to
    # bite there and barely bite at a 3-point line, where the sample is real.
    for market_line, ceiling in ((45.5, 0.045), (3.0, 0.20)):
        counts, _ = dist._raw_sample("cfb", dist.MARGIN, market_line)
        n = sum(counts.values())
        blended, _, _ = dist.pmf("cfb", dist.MARGIN, market_line)
        worst = max(
            abs(blended[value] - count / n) for value, count in counts.items() if value in blended
        )
        assert worst < ceiling


def test_unknown_line_falls_back_to_the_widest_band_rather_than_raising():
    estimate = dist.p_at("cfb", dist.MARGIN, 90.0, 90.0)
    assert estimate.p == 0.0
    assert estimate.n == 0
