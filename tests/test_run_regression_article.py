"""The daily run builds the prose regression article, and never at the card's cost."""

from __future__ import annotations

import dataclasses
import json
from datetime import date as Date

import pandas as pd

import mlb_engine.cli as cli
import mlb_engine.output.regression_article as art


class _Pipe:
    """Just the two attributes the article builder reads off a finished run."""

    def __init__(self, statcast: pd.DataFrame | None) -> None:
        self.statcast = statcast


def _cfg(tmp_path):
    cfg = dataclasses.replace(cli.load_config(), data_dir=tmp_path)
    cfg.audit_dir.mkdir(parents=True)
    cfg.output_dir.mkdir(parents=True)
    return cfg


def _write_inputs(cfg, day: Date) -> None:
    iso = day.isoformat()
    (cfg.audit_dir / f"previews_{iso}.json").write_text(json.dumps([]))
    (cfg.audit_dir / f"predictions_{iso}.json").write_text(json.dumps([]))


def test_the_article_is_written_beside_the_card_and_returned_for_the_email(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    day = Date(2026, 8, 18)
    _write_inputs(cfg, day)
    monkeypatch.setattr(
        cli, "build_article_pdf", lambda *a, **k: (b"%PDF-1.7 article", "<html>a</html>")
    )

    pdf = cli._build_regression_article(_Pipe(pd.DataFrame({"pitcher": [1]})), day, cfg)

    assert pdf == b"%PDF-1.7 article"
    stem = cfg.output_dir / f"PayoffPitch_Regression_{day.isoformat()}"
    assert stem.with_suffix(".pdf").read_bytes() == b"%PDF-1.7 article"
    assert stem.with_suffix(".html").read_text() == "<html>a</html>"


def test_a_slate_with_nothing_rankable_writes_no_article(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    day = Date(2026, 8, 18)
    _write_inputs(cfg, day)
    monkeypatch.setattr(cli, "build_article_pdf", lambda *a, **k: None)

    assert cli._build_regression_article(_Pipe(pd.DataFrame({"pitcher": [1]})), day, cfg) is None
    assert not list(cfg.output_dir.glob("PayoffPitch_Regression_*"))


def test_a_failed_render_costs_the_email_its_article_and_nothing_else(tmp_path, monkeypatch):
    """The bet sheet ships even when WeasyPrint or a profile blows up."""
    cfg = _cfg(tmp_path)
    day = Date(2026, 8, 18)
    _write_inputs(cfg, day)

    def boom(*a, **k):
        raise RuntimeError("no fonts")

    monkeypatch.setattr(cli, "build_article_pdf", boom)

    assert cli._build_regression_article(_Pipe(pd.DataFrame({"pitcher": [1]})), day, cfg) is None


def test_no_statcast_frame_means_no_article(tmp_path):
    cfg = _cfg(tmp_path)
    day = Date(2026, 8, 18)
    _write_inputs(cfg, day)

    assert cli._build_regression_article(_Pipe(None), day, cfg) is None


def test_missing_previews_do_not_raise_out_of_the_run(tmp_path):
    cfg = _cfg(tmp_path)

    assert (
        cli._build_regression_article(
            _Pipe(pd.DataFrame({"pitcher": [1]})), Date(2026, 8, 18), cfg
        )
        is None
    )


def test_an_empty_slate_ranks_nobody(monkeypatch):
    """``build_article`` reports "nothing to write" rather than an empty document."""
    monkeypatch.setattr(art, "build_profiles", lambda *a: ([], [], {}))
    monkeypatch.setattr(art, "build_batter_profiles", lambda *a: ([], []))

    assert art.build_article(Date(2026, 8, 18), [], [], pd.DataFrame()) is None


def test_a_ranked_arm_reaches_the_document(monkeypatch):
    prof = {
        "name": "Cristian Javier",
        "siera": 3.9,
        "xk": 0.22,
        "vfa": 95.2,
        "fb": 0.44,
        "gb": 0.34,
        "babip": 0.330,
        "dxwoba": -0.020,
        "xwoba": 0.300,
        "woba": 0.320,
        "csw": 0.29,
        "k_pct": 0.22,
        "bb_pct": 0.08,
        "barrel": 0.07,
        "biomech": {"ext": 6.4, "ivb": 15.0, "spin": 2300.0, "scatter": 2.0},
        "d_siera": 0.0,
        "d_xk": 0.0,
        "d_vfa": 0.0,
        "reg_index": 1.4,
        "unlucky_babip": 0.040,
        "unlucky_xwoba": 0.020,
        "pitches": 900,
        "siera_pa": 300,
    }
    monkeypatch.setattr(art, "build_profiles", lambda *a: ([prof], [], {}))
    monkeypatch.setattr(art, "build_batter_profiles", lambda *a: ([], []))

    html = art.build_article(Date(2026, 8, 18), [], [], pd.DataFrame())

    assert html is not None
    assert "Cristian Javier" in html
    assert "fastball at 95.2" in html
    assert "fly-ball arm" in html
