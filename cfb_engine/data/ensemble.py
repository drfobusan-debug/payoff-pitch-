"""Ensemble of public college-football power models, blended with CFBD SP+.

Each source is reduced to a single **net** power rating per team (points better
than an average team on a neutral field). Because the sources live on different
scales (Sagarin/FPI in points, FEI per-possession), every model is standardized
to a z-score across its own teams and then rescaled to a common points spread
before the weighted consensus is formed -- so no single scale dominates and a
missing source simply drops out of the average.

Adapters, by how reliably they can be pulled:

* **Sagarin** -- USA Today plain-text block (regex parse).
* **ESPN FPI** -- hidden JSON endpoint.
* **FEI / BCFToys** -- static HTML table (``pandas.read_html``).
* **TSI**, **CFB Graphs EPA** -- weekly CSV/JSON drop-ins under
  ``~/.cfb_engine/models/`` (TSI is proprietary and only published as prose, so
  it is pasted in rather than scraped).

Every network adapter caches its raw payload and fails soft: a site change or a
blocked request skips that model without breaking the slate. The parsers are
pure functions so they can be unit-tested against saved fixtures.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from cfb_engine.data.cfbd import RatingBook, TeamRating
from cfb_engine.data.preseason import MAKINEN, TEAMRANKINGS
from cfb_engine.data.teamnames import school_key

log = logging.getLogger(__name__)

_DEFAULT_LEAGUE_AVG = 27.5
_DEFAULT_TARGET_SD = 14.0

SAGARIN_URL = "https://sagarin.com/sports/cfsend.htm"
FPI_URL = (
    "https://site.web.api.espn.com/apis/fitt/v3/sports/football/"
    "college-football/powerindex?region=us&lang=en&limit=200&season={season}"
)
FEI_URL = "https://www.bcftoys.com/{season}-fei/"


@dataclass(frozen=True)
class ModelRatings:
    """One model's net power ratings, keyed by :func:`school_key`."""

    source: str
    net: dict[str, float] = field(default_factory=dict)
    hfa: float | None = None

    def standardized(self) -> dict[str, float]:
        """Z-scored net ratings (mean 0, sd 1); empty if too few teams."""
        vals = list(self.net.values())
        if len(vals) < 3:
            return {}
        mean = statistics.fmean(vals)
        sd = statistics.pstdev(vals)
        if sd <= 0:
            return {}
        return {team: (v - mean) / sd for team, v in self.net.items()}


# --------------------------------------------------------------------------- #
# Pure parsers (unit-tested against fixtures)
# --------------------------------------------------------------------------- #
_SAGARIN_LINE = re.compile(
    r"^\s*\d+\s+([A-Za-z&.'()\- ]+?)\s+[A-Z]?\s*=?\s*(-?\d+\.\d+)"
)
_SAGARIN_HFA = re.compile(r"HOME\s+(?:ADVANTAGE|EDGE)[^0-9\-]*(-?\d+\.\d+)", re.I)


def parse_sagarin(text: str) -> ModelRatings:
    """Parse the Sagarin plain-text ratings block.

    Lines look like ``  1  Georgia            A =  93.72 ...``; we take the
    first float after the team name as the rating and read the home-edge line
    when present.
    """
    net: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or not line[0].isdigit():
            continue
        m = _SAGARIN_LINE.match(line)
        if not m:
            continue
        team = m.group(1).strip()
        if not team or team.lower().startswith(("home", "team", "rank")):
            continue
        net[school_key(team)] = float(m.group(2))
    hfa_m = _SAGARIN_HFA.search(text)
    hfa = float(hfa_m.group(1)) if hfa_m else None
    return ModelRatings("sagarin", net, hfa)


def parse_fpi(payload: object) -> ModelRatings:
    """Parse ESPN's FPI power-index JSON into net ratings.

    ESPN nests ratings under ``teams[].categories[].values`` with a parallel
    ``names`` list; we pull the ``fpi`` value and the team's display name.
    """
    net: dict[str, float] = {}
    if not isinstance(payload, dict):
        return ModelRatings("fpi", net)
    teams = payload.get("teams")
    if not isinstance(teams, list):
        return ModelRatings("fpi", net)
    for entry in teams:
        if not isinstance(entry, dict):
            continue
        team_obj = entry.get("team")
        # Prefer ESPN's school-only "location" ("Ohio State") over "displayName"
        # ("Ohio State Buckeyes") so matching does not depend on mascot stripping.
        name = None
        if isinstance(team_obj, dict):
            name = team_obj.get("location") or team_obj.get("displayName")
        if not name:
            continue
        value = _fpi_value(entry.get("categories"))
        if value is not None:
            net[school_key(str(name))] = value
    return ModelRatings("fpi", net)


def _fpi_value(categories: object) -> float | None:
    if not isinstance(categories, list):
        return None
    for cat in categories:
        if not isinstance(cat, dict):
            continue
        names = cat.get("names")
        values = cat.get("values")
        if not isinstance(names, list) or not isinstance(values, list):
            continue
        for i, nm in enumerate(names):
            if str(nm).lower() == "fpi" and i < len(values):
                val = values[i]
                if isinstance(val, (int, float)):
                    return float(val)
    return None


def parse_fei(html: str) -> ModelRatings:
    """Parse the BCFToys FEI HTML table via ``pandas.read_html``.

    Returns an empty model if pandas/lxml is unavailable or no ``FEI`` column is
    found, so a missing optional dependency never breaks the run.
    """
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a project dep
        log.warning("pandas unavailable; skipping FEI parse")
        return ModelRatings("fei", {})
    try:
        tables = pd.read_html(io.StringIO(html))
    except (ValueError, ImportError) as exc:
        log.warning("could not parse FEI table: %s", exc)
        return ModelRatings("fei", {})
    net: dict[str, float] = {}
    for table in tables:
        cols = {str(c).strip().lower(): c for c in table.columns}
        team_col = next((cols[c] for c in cols if c in ("team", "teams")), None)
        fei_col = next((cols[c] for c in cols if c == "fei" or c.startswith("fei")), None)
        if team_col is None or fei_col is None:
            continue
        for _, row in table.iterrows():
            team = str(row[team_col]).strip()
            try:
                val = float(row[fei_col])
            except (TypeError, ValueError):
                continue
            if team and team.lower() != "nan":
                net[school_key(team)] = val
        if net:
            break
    return ModelRatings("fei", net)


def parse_ratings_csv(text: str, source: str) -> ModelRatings:
    """Parse a drop-in CSV/tab file into net ratings.

    Accepts a ``team`` column plus one of ``net`` / ``rating`` / ``power``, or a
    projected ``spread`` (home-negative), for TSI and CFB Graphs exports.
    """
    net: dict[str, float] = {}
    reader = csv.DictReader(io.StringIO(text))
    headers = [h.lower().strip() for h in (reader.fieldnames or [])]

    def col(*needles: str) -> str | None:
        for field_name in reader.fieldnames or []:
            low = field_name.lower().strip()
            if any(n == low or n in low for n in needles):
                return field_name
        return None

    team_col = col("team", "school")
    val_col = col("net", "rating", "power", "spread")
    if team_col is None or val_col is None:
        log.warning("%s drop-in missing team/value columns (have %s)", source, headers)
        return ModelRatings(source, net)
    sign = -1.0 if "spread" in (val_col.lower()) else 1.0
    for row in reader:
        team = (row.get(team_col) or "").strip()
        try:
            val = float(row[val_col])
        except (TypeError, ValueError, KeyError):
            continue
        if team:
            net[school_key(team)] = sign * val
    return ModelRatings(source, net)


# --------------------------------------------------------------------------- #
# Consensus + blend
# --------------------------------------------------------------------------- #
def consensus_net(
    models: list[ModelRatings], weights: dict[str, float], target_sd: float
) -> dict[str, float]:
    """Weighted mean of standardized model nets, rescaled to ``target_sd``."""
    acc: dict[str, float] = {}
    wsum: dict[str, float] = {}
    for model in models:
        w = weights.get(model.source, 1.0)
        if w <= 0:
            continue
        for team, z in model.standardized().items():
            acc[team] = acc.get(team, 0.0) + w * z
            wsum[team] = wsum.get(team, 0.0) + w
    return {team: (acc[team] / wsum[team]) * target_sd for team in acc if wsum[team] > 0}


def blend_ensemble(
    base: RatingBook | None,
    models: list[ModelRatings],
    *,
    blend: float,
    weights: dict[str, float] | None = None,
    target_sd: float = 0.0,
) -> RatingBook | None:
    """Pull each team's net rating toward the ensemble consensus by ``blend``.

    Totals (offense + defense) are preserved; only the net (offense - defense)
    moves, since power-model ratings inform margin, not scoring pace. With no
    CFBD base the consensus becomes the book on its own, centered on the league
    scoring average.
    """
    weights = weights or {}
    if target_sd <= 0:
        target_sd = _net_spread(base)
    consensus = consensus_net(models, weights, target_sd)
    if not consensus:
        return base

    if base is None:
        ratings = {
            school_key(team): TeamRating(team, _DEFAULT_LEAGUE_AVG + n / 2, _DEFAULT_LEAGUE_AVG - n / 2)
            for team, n in consensus.items()
        }
        return RatingBook(ratings=ratings, league_avg=_DEFAULT_LEAGUE_AVG) if ratings else None

    b = min(max(blend, 0.0), 1.0)
    out: dict[str, TeamRating] = {}
    for key, rating in base.ratings.items():
        ens = consensus.get(school_key(rating.team))
        if ens is None:
            out[key] = rating
            continue
        total = rating.offense + rating.defense
        new_net = (1 - b) * (rating.offense - rating.defense) + b * ens
        out[key] = TeamRating(rating.team, (total + new_net) / 2, (total - new_net) / 2)
    return RatingBook(ratings=out, league_avg=base.league_avg)


def _net_spread(base: RatingBook | None) -> float:
    if base is None or len(base.ratings) < 3:
        return _DEFAULT_TARGET_SD
    nets = [r.offense - r.defense for r in base.ratings.values()]
    sd = statistics.pstdev(nets)
    return sd if sd > 0 else _DEFAULT_TARGET_SD


# --------------------------------------------------------------------------- #
# Fetchers (cached, fail-soft)
# --------------------------------------------------------------------------- #
def _weight_env(source: str) -> float:
    raw = os.getenv(f"CFBE_W_{source.upper()}")
    return float(raw) if raw not in (None, "") else 1.0


def _enabled(source: str) -> bool:
    raw = os.getenv(f"CFBE_{source.upper()}")
    if raw in (None, ""):
        return True
    return raw not in ("0", "false", "False")


class EnsembleProvider:
    """Collect the enabled ensemble sources for a season."""

    def __init__(self, cache_dir: Path, models_dir: Path, *, ttl: int = 21600) -> None:
        self.cache_dir = cache_dir
        self.models_dir = models_dir
        self.ttl = ttl

    def weights(self) -> dict[str, float]:
        sources = ("sagarin", "fpi", "fei", "tsi", "cfbgraphs", "makinen", "teamrankings")
        return {s: _weight_env(s) for s in sources}

    def collect(self, season: int) -> list[ModelRatings]:
        models: list[ModelRatings] = []
        if _enabled("makinen"):
            models.append(ModelRatings("makinen", dict(MAKINEN)))
        if _enabled("teamrankings"):
            models.append(ModelRatings("teamrankings", dict(TEAMRANKINGS)))
        if _enabled("sagarin"):
            models.append(self._sagarin())
        if _enabled("fpi"):
            models.append(self._fpi(season))
        if _enabled("fei"):
            models.append(self._fei(season))
        models.extend(self._drop_ins())
        return [m for m in models if m.net]

    # -- individual sources ----------------------------------------------
    def _sagarin(self) -> ModelRatings:
        text = self._get_text("sagarin.html", SAGARIN_URL)
        return parse_sagarin(text) if text else ModelRatings("sagarin", {})

    def _fpi(self, season: int) -> ModelRatings:
        url = os.getenv("CFBE_FPI_URL") or FPI_URL.format(season=season)
        text = self._get_text("fpi.json", url)
        if not text:
            return ModelRatings("fpi", {})
        try:
            return parse_fpi(json.loads(text))
        except json.JSONDecodeError as exc:
            log.warning("FPI JSON decode failed: %s", exc)
            return ModelRatings("fpi", {})

    def _fei(self, season: int) -> ModelRatings:
        url = os.getenv("CFBE_FEI_URL") or FEI_URL.format(season=season)
        text = self._get_text("fei.html", url)
        return parse_fei(text) if text else ModelRatings("fei", {})

    def _drop_ins(self) -> list[ModelRatings]:
        out: list[ModelRatings] = []
        if not self.models_dir.exists():
            return out
        for path in sorted(self.models_dir.glob("*.csv")):
            source = path.stem.lower()
            if not _enabled(source):
                continue
            try:
                out.append(parse_ratings_csv(path.read_text(), source))
            except OSError as exc:
                log.warning("could not read model drop-in %s: %s", path, exc)
        return out

    # -- cached HTTP -----------------------------------------------------
    def _get_text(self, cache_name: str, url: str) -> str | None:
        cached = self._cache_read(cache_name)
        if cached is not None:
            return cached
        try:
            resp = requests.get(url, timeout=25, headers={"User-Agent": "payoff-pitch-cfb/1.0"})
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("ensemble fetch failed (%s): %s", url, exc)
            return None
        self._cache_write(cache_name, resp.text)
        return resp.text

    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / "models" / name

    def _cache_read(self, name: str) -> str | None:
        path = self._cache_path(name)
        if not path.exists():
            return None
        if self.ttl > 0 and time.time() - path.stat().st_mtime > self.ttl:
            return None
        try:
            return path.read_text()
        except OSError:
            return None

    def _cache_write(self, name: str, text: str) -> None:
        path = self._cache_path(name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        except OSError as exc:
            log.warning("could not cache %s: %s", name, exc)
