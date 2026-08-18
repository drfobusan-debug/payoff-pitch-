"""A pitch-shape grade: how good his pitches are, before any of them is thrown.

Every strikeout read the engine already trusts is a *result* -- CSW%, SwStr%,
observed K% -- and #195 showed those add nothing on top of one another because
they are the same events counted three ways. This one is not a result. Each
pitch is scored on its own physics (velocity, ride, spin, extension, release
point, and how far it separates from the arm's own fastball) against how often a
pitch shaped like that is missed, and the grade is the usage-weighted average of
how much better that is than a *league-average pitch of the same type*.

That last clause is the whole measurement. Grading against the league as a whole
would make "throws sliders" look like stuff, and the study says arsenal
composition is worth nothing: mixed into the shipped strikeout prior, pitch mix
alone measured a coefficient of +0.002 and left held-out deviance unchanged
(1.03924 vs 1.03922), while within-type shape carried +0.092 and improved it to
1.03731. The composition term also made the pair *worse* than shape alone
(walk-forward 1.04140 vs 1.04095), so it is subtracted out rather than kept.

Fitted per pitch type on 2026 pitch-level Statcast (``scripts/stuff_fit.py`` ->
``data/stuff_shape.json``), on swings only: the model answers "was this pitch
missed, given a swing", which is the part of a strikeout that shape controls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_MODEL_FILE = Path(__file__).parent.parent / "data" / "stuff_shape.json"

# Statcast pitch codes -> the type the model was fitted per. Codes not listed
# (eephus, screwball, unknown) are dropped: too rare to fit and too rare to matter.
PITCH_GROUP = {
    "FF": "fastball",
    "FA": "fastball",
    "SI": "sinker",
    "FT": "sinker",
    "FC": "cutter",
    "SL": "slider",
    "ST": "sweeper",
    "SV": "slurve",
    "CU": "curve",
    "KC": "curve",
    "CS": "curve",
    "CH": "change",
    "FS": "splitter",
    "FO": "splitter",
}

FASTBALL_GROUPS = ("fastball", "sinker")

# The grade needs enough pitches to be a measurement rather than a rounding of
# one outing. Shape stabilizes fast -- start-to-start r=+0.99 against +0.90 for
# the observed K rate it helps predict -- so this floor is low on purpose.
MIN_PITCHES = 100

# Clipped before use: the fitted line is measured over the middle of the
# distribution and 0.05 sits outside the 1st/99th percentile of starter windows
# (-0.060 / +0.055), so this bounds the tails without touching anyone real.
GRADE_CLIP = 0.05


@dataclass(frozen=True)
class _GroupModel:
    mean: tuple[float, ...]
    sd: tuple[float, ...]
    coef: tuple[float, ...]
    intercept: float
    base_whiff: float


def _load() -> tuple[tuple[str, ...], dict[str, _GroupModel]]:
    if not _MODEL_FILE.exists():
        return (), {}
    payload = json.loads(_MODEL_FILE.read_text())
    features = tuple(str(f) for f in payload["features"])
    groups = {
        str(name): _GroupModel(
            mean=tuple(float(v) for v in spec["mean"]),
            sd=tuple(float(v) for v in spec["sd"]),
            coef=tuple(float(v) for v in spec["coef"]),
            intercept=float(spec["intercept"]),
            base_whiff=float(spec["base_whiff"]),
        )
        for name, spec in payload["groups"].items()
    }
    return features, groups


_FEATURES, _GROUPS = _load()

_NEEDED = {
    "release_speed",
    "pfx_z",
    "release_spin_rate",
    "release_extension",
    "release_pos_z",
    "release_pos_x",
    "pitch_type",
    "p_throws",
}


def _shape_frame(pdf: pd.DataFrame) -> pd.DataFrame | None:
    """The model's inputs, one row per pitch with a complete shape measurement."""
    if not _GROUPS or not _NEEDED.issubset(pdf.columns):
        return None
    df = pd.DataFrame(
        {
            "group": pdf["pitch_type"].map(PITCH_GROUP),
            "velo": pd.to_numeric(pdf["release_speed"], errors="coerce"),
            "ivb": pd.to_numeric(pdf["pfx_z"], errors="coerce") * 12.0,
            "spin": pd.to_numeric(pdf["release_spin_rate"], errors="coerce"),
            "ext": pd.to_numeric(pdf["release_extension"], errors="coerce"),
            "rel_z": pd.to_numeric(pdf["release_pos_z"], errors="coerce"),
            # mirrored so release side means the same thing for both hands
            "rel_x": pd.to_numeric(pdf["release_pos_x"], errors="coerce")
            * pdf["p_throws"].map({"L": -1.0, "R": 1.0}).fillna(1.0),
        }
    )
    df = df[df["group"].isin(_GROUPS)].dropna()
    if df.empty:
        return None

    # Separation from his own fastball, not from the league's: a 90 mph changeup
    # is a different pitch behind a 100 mph fastball than behind a 91 mph one.
    fastball = df[df["group"].isin(FASTBALL_GROUPS)]
    if len(fastball):
        df["velo_diff"] = df["velo"] - float(fastball["velo"].mean())
        df["ivb_diff"] = df["ivb"] - float(fastball["ivb"].mean())
    else:
        df["velo_diff"] = 0.0
        df["ivb_diff"] = 0.0
    return df


def _whiff_rates(model: _GroupModel, rows: pd.DataFrame) -> np.ndarray:
    x = rows[list(_FEATURES)].to_numpy(float)
    sd = np.asarray(model.sd, dtype=float)
    sd[sd <= 0.0] = np.inf  # a feature with no spread cannot carry a slope
    z = model.intercept + ((x - np.asarray(model.mean)) / sd) @ np.asarray(model.coef)
    return 1.0 / (1.0 + np.exp(-z))


def shape_plus(pdf: pd.DataFrame) -> float:
    """Usage-weighted whiff rate his shapes earn, above the same pitch types'.

    Returns 0.0 -- no opinion -- when the model file is missing, the frame has no
    shape columns, or the window is too thin to grade.
    """
    df = _shape_frame(pdf)
    if df is None or len(df) < MIN_PITCHES:
        return 0.0
    total = 0.0
    for group, rows in df.groupby("group", sort=False):
        model = _GROUPS[str(group)]
        total += float((_whiff_rates(model, rows) - model.base_whiff).sum())
    return total / len(df)
