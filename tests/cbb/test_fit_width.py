"""fit_width key-number test: a real excess at one total is found; a plain normal sample yields none."""

import random

from scripts.cbb import fit_width


def _rows(finals, close=145.0):
    return [
        {"close_total": str(close), "final_total": str(f), "paired_books": "6", "spread_abs": "5"}
        for f in finals
    ]


def test_key_mass_fires_on_spike_only():
    rng = random.Random(1)
    base = [round(rng.gauss(145, 16)) for _ in range(3000)]
    assert fit_width.fit(_rows(base))["key_mass"] == {}
    spiked = base + [145] * 150
    km = fit_width.fit(_rows(spiked))["key_mass"]
    assert isinstance(km, dict) and "145" in km
