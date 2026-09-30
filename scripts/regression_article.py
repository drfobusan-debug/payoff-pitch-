"""Compatibility shim: the regression article now lives in the engine.

It is built by ``mlb-engine run`` (and by ``scripts/regen_regression.py``), so
the writer moved to :mod:`mlb_engine.output.regression_article`. This module
keeps the old import path working for the study scripts.
"""

from __future__ import annotations

from mlb_engine.output.regression_article import *  # noqa: F401,F403
from mlb_engine.output.regression_article import (  # noqa: F401
    _air_sentence,
    _bat_air_sentence,
    _batter_entry,
    _bet_sentence,
    _luck_sentence,
    _pitcher_entry,
    _pitcher_verdict,
    _swing_line,
    _swing_sentence,
    build_article,
    build_article_pdf,
    build_html,
)
