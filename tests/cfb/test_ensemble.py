"""Ensemble model parsers and the SP+ blend."""

from __future__ import annotations

import json
import math
from pathlib import Path

from cfb_engine.data.cfbd import RatingBook, TeamRating
from cfb_engine.data.ensemble import (
    EnsembleProvider,
    ModelRatings,
    blend_ensemble,
    consensus_net,
    parse_fpi,
    parse_ratings_csv,
    parse_sagarin,
    parse_teamrankings,
)
from cfb_engine.data.preseason import MAKINEN, TEAMRANKINGS
from cfb_engine.data.teamnames import school_key

SAGARIN_TEXT = """
Jeff Sagarin Ratings

HOME ADVANTAGE=2.30

   1  Georgia              A =   93.72   1  ...
   2  Ohio State           A =   91.10   2  ...
   3  Alabama              A =   90.05   3  ...
   4  Michigan             A =   85.40   4  ...
"""


def test_parse_sagarin_ratings_and_hfa():
    m = parse_sagarin(SAGARIN_TEXT)
    assert m.source == "sagarin"
    assert math.isclose(m.hfa or 0.0, 2.30, rel_tol=1e-6)
    assert math.isclose(m.net["georgia"], 93.72, rel_tol=1e-6)
    assert m.net["ohio state"] == 91.10
    assert "alabama" in m.net and "michigan" in m.net


def test_parse_fpi_uses_location():
    payload = {
        "teams": [
            {
                "team": {"location": "Ohio State", "displayName": "Ohio State Buckeyes"},
                "categories": [{"names": ["fpi"], "values": [22.5]}],
            },
            {
                "team": {"location": "Alabama"},
                "categories": [{"names": ["fpi"], "values": [19.1]}],
            },
        ]
    }
    m = parse_fpi(json.loads(json.dumps(payload)))
    assert m.net["ohio state"] == 22.5
    assert m.net["alabama"] == 19.1


def test_parse_ratings_csv_net_and_spread():
    net_csv = "team,net\nGeorgia,20.5\nAlabama,17.0\n"
    m = parse_ratings_csv(net_csv, "cfbgraphs")
    assert m.net["georgia"] == 20.5

    # A projected spread is home-negative, so it flips sign into a net rating.
    spread_csv = "team,spread\nGeorgia,-14\nAlabama,-3\n"
    s = parse_ratings_csv(spread_csv, "tsi")
    assert s.net["georgia"] == 14.0
    assert s.net["alabama"] == 3.0


def test_consensus_standardizes_scales():
    # Two models on wildly different scales should agree on the ordering.
    a = ModelRatings("a", {"x": 30.0, "y": 20.0, "z": 10.0})
    b = ModelRatings("b", {"x": 3.0, "y": 2.0, "z": 1.0})
    cons = consensus_net([a, b], {"a": 1.0, "b": 1.0}, target_sd=10.0)
    assert cons["x"] > cons["y"] > cons["z"]


def test_blend_moves_net_preserves_total():
    base = RatingBook(
        ratings={
            "georgia": TeamRating("Georgia", 35.0, 15.0),  # net +20
            "alabama": TeamRating("Alabama", 30.0, 20.0),  # net +10
            "vanderbilt": TeamRating("Vanderbilt", 22.0, 30.0),  # net -8
        },
        league_avg=27.5,
    )
    # Consensus disagrees: it rates Alabama above Georgia.
    models = [ModelRatings("m", {"georgia": 5.0, "alabama": 25.0, "vanderbilt": -30.0})]
    blended = blend_ensemble(base, models, blend=0.5, weights={"m": 1.0}, target_sd=14.0)
    assert blended is not None
    g = blended.ratings["georgia"]
    # Total (offense + defense) is preserved; only the net (margin) moves.
    assert math.isclose(g.offense + g.defense, 50.0, rel_tol=1e-9)
    # Georgia's net is pulled down toward the consensus that rates it lower.
    assert (g.offense - g.defense) < 20.0
    assert blended.ratings["vanderbilt"].offense - blended.ratings["vanderbilt"].defense < 0


def test_blend_with_no_base_builds_book():
    models = [ModelRatings("m", {"georgia": 20.0, "alabama": 10.0, "auburn": -30.0})]
    book = blend_ensemble(None, models, blend=0.5, weights={"m": 1.0}, target_sd=14.0)
    assert book is not None
    assert "georgia" in book.ratings


def test_empty_consensus_returns_base_unchanged():
    base = RatingBook(ratings={"x": TeamRating("X", 28.0, 27.0)}, league_avg=27.5)
    assert blend_ensemble(base, [], blend=0.5) is base


TEAMRANKINGS_HTML = """
<table>
  <thead><tr><th>Rank</th><th>Team</th><th>Rating</th></tr></thead>
  <tbody>
    <tr><td>1</td><td>Ohio St (12-1)</td><td>32.3</td></tr>
    <tr><td>2</td><td>Georgia (11-2)</td><td>28.4</td></tr>
    <tr><td>3</td><td>UMass (2-10)</td><td>-28.5</td></tr>
    <tr><td>4</td><td>Fresno St (0-0)</td><td>--</td></tr>
  </tbody>
</table>
"""


def test_parse_teamrankings_strips_record_and_skips_placeholder():
    m = parse_teamrankings(TEAMRANKINGS_HTML, "teamrankings")
    assert m.source == "teamrankings"
    assert m.net[school_key("Ohio State")] == 32.3
    assert m.net[school_key("Georgia")] == 28.4
    assert m.net[school_key("UMass")] == -28.5
    # "--" placeholder (no games yet) is dropped, not coerced to 0.
    assert school_key("Fresno State") not in m.net


def test_provider_collects_preseason_sources(tmp_path: Path, monkeypatch):
    # No network sources; live TeamRankings fetch stubbed to empty so the
    # baked-in preseason snapshot is used (keeps the test hermetic/offline).
    for src in ("sagarin", "fpi", "fei"):
        monkeypatch.setenv(f"CFBE_{src.upper()}", "0")
    provider = EnsembleProvider(tmp_path / "cache", tmp_path / "models")
    monkeypatch.setattr(provider, "_get_text", lambda *a, **k: "")
    sources = {m.source for m in provider.collect(2026)}
    assert {"makinen", "teamrankings"} <= sources
    assert set(provider.weights()) >= {
        "makinen", "teamrankings", "teamrankings_l5", "teamrankings_l10"
    }


def test_preseason_only_ensemble_builds_full_book():
    # With no CFBD base, Makinen + TeamRankings alone yield a real rating book
    # with a spread of net ratings (so the engine produces edges preseason).
    models = [
        ModelRatings("makinen", dict(MAKINEN)),
        ModelRatings("teamrankings", dict(TEAMRANKINGS)),
    ]
    book = blend_ensemble(
        None, models, blend=0.35, weights={"makinen": 1.0, "teamrankings": 1.0}
    )
    assert book is not None and len(book.ratings) == 138
    osu = book.ratings[school_key("Ohio State")]
    umass = book.ratings[school_key("Massachusetts")]
    assert (osu.offense - osu.defense) > (umass.offense - umass.defense)
