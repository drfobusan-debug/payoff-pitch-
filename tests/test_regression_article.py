"""The regression article must keep luck and level as separate claims."""

from __future__ import annotations

from datetime import date as Date

import pytest

art = pytest.importorskip("scripts.regression_article")


def _pitcher(**over) -> dict:
    p = {
        "name": "Zack Wheeler",
        "siera": 3.10,
        "xk": 0.30,
        "vfa": 95.2,
        "babip": 0.340,
        "woba": 0.360,
        "xwoba": 0.300,
        "dxwoba": -0.060,
        "unlucky_babip": 0.050,
        "d_siera": 0.00,
        "d_xk": 0.00,
        "d_vfa": 0.00,
    }
    p.update(over)
    return p


def _batter(**over) -> dict:
    b = {
        "name": "Gunnar Henderson",
        "woba": 0.252,
        "xwoba": 0.359,
        "dxwoba": 0.107,
        "xslg": 0.484,
        "barrel": 0.09,
        "woba6": 0.252,
        "woba3": 0.250,
    }
    b.update(over)
    return b


def test_declining_arm_is_not_sold_as_a_clean_buy_low() -> None:
    """High BABIP plus a worsening arm must warn, not celebrate."""
    text = art._pitcher_verdict(_pitcher(d_siera=0.60, d_vfa=-1.2), positive=True)
    assert "Read this one carefully" in text
    assert "may not be the one he" in text
    assert "clean version" not in text


def test_stable_arm_with_bad_luck_is_the_clean_case() -> None:
    text = art._pitcher_verdict(_pitcher(d_siera=-0.40, d_vfa=0.3), positive=True)
    assert "clean version" in text


def test_luck_sentence_names_both_luck_terms() -> None:
    text = art._luck_sentence(_pitcher(), positive=True)
    assert ".340" in text and ".290" in text  # BABIP against the norm
    assert "60 points worse" in text  # the wOBA - xwOBA gap


def test_negative_regression_reads_as_borrowed_results() -> None:
    p = _pitcher(babip=0.250, unlucky_babip=-0.040, woba=0.270,
                 xwoba=0.330, dxwoba=0.060)
    text = art._luck_sentence(p, positive=False)
    assert "finding gloves" in text
    assert "flatters" in text


def test_batter_entry_states_the_gap_and_the_power() -> None:
    html = art._batter_entry(_batter(), {"matchup": "BAL @ MIN"}, None, True)
    assert "107-point shortfall" in html
    assert "real power" in html
    assert "passes his props" in html


def test_bet_sentence_lists_only_buys() -> None:
    bets = [
        {"selection": "Wheeler Ks o6.5", "tier": "Strong buy",
         "model_prob": 0.61, "market_american": -115},
        {"selection": "Wheeler Hits u5.5", "tier": "Pass",
         "model_prob": 0.40, "market_american": 100},
    ]
    text = art._bet_sentence(bets, "Wheeler&rsquo;s")
    assert "Ks o6.5" in text
    assert "Hits u5.5" not in text


def test_article_flags_that_the_trend_arrows_are_unproven() -> None:
    html = art.build_html(
        Date(2026, 8, 12), [_pitcher()], [], {}, [_batter()], [], {}, []
    )
    assert "three-week direction</i> of those same" in html
    assert "does not predict the next start" in html
    assert "Part one" in html and "Part two" in html
