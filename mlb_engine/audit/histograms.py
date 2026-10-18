"""The simulator's raw joint distributions, kept beside the card for the shadow.

Built from the same ``GameSimResult`` arrays the card prices from, before any
calibration, market anchor, outs bias or run-environment shift. Three shapes
per game: full-game home x away runs, first-five home x away runs, and each
priced prop's count x full-game total (so a prop can be read conditional on a
pinned total). Stored sparse, as ``[i, j, n]`` path counts.

Identity is per game: ``pred_hash`` is a hash of that game's own prediction
rows. A late pass re-prices only the games near first pitch, so it replaces
only those games' histograms, and the shadow skips a game whose hash no longer
matches the predictions file. The live card never reads this file.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from datetime import date as Date
from pathlib import Path
from typing import Any

import numpy as np

from mlb_engine.audit.tape import game_key as _game_key
from mlb_engine.models.montecarlo import GameSimResult
from mlb_engine.recommendations import Recommendation
from mlb_engine.schemas import Game

HIST_VERSION = 1
BATTER_STATS = ("H", "1B", "2B", "HR", "R", "RBI", "BB", "K")
PITCHER_STATS = ("K", "outs", "H", "BB", "ER")


def hist_path(audit_dir: Path, slate_date: Date) -> Path:
    return audit_dir / f"hist_{slate_date.isoformat()}.json.gz"


def sparse2d(x: np.ndarray, y: np.ndarray) -> list[list[int]]:
    """Joint path counts of two integer arrays as ``[[x, y, n], ...]``."""
    pairs = np.stack([x.astype(np.int64), y.astype(np.int64)], axis=1)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    return [[int(a), int(b), int(n)] for (a, b), n in zip(uniq, counts, strict=True)]


def dense(cells: Iterable[Iterable[int]]) -> np.ndarray:
    """A sparse ``[[i, j, n]]`` histogram as a dense count array indexed ``[i, j]``."""
    rows = np.asarray(list(cells), dtype=np.int64).reshape(-1, 3)
    if rows.size == 0:
        return np.zeros((1, 1))
    out = np.zeros((int(rows[:, 0].max()) + 1, int(rows[:, 1].max()) + 1))
    np.add.at(out, (rows[:, 0], rows[:, 1]), rows[:, 2])
    return out


def prop_key(kind: str, player_id: int, stat: str) -> str:
    return f"{kind}|{player_id}|{stat}"


def build_game_hists(res: GameSimResult, game: Game) -> dict[str, Any]:
    """Every raw histogram the shadow reads for one game."""
    h, a = res.home_runs_full, res.away_runs_full
    total = h.astype(np.int64) + a.astype(np.int64)
    props: dict[str, list[list[int]]] = {}
    for team_key, team in (("home", game.home), ("away", game.away)):
        bat = res.bat[team_key]
        for i, slot in enumerate(team.lineup[:9]):
            pid = slot.player.mlbam_id
            for stat in BATTER_STATS:
                props[prop_key("batter", pid, stat)] = sparse2d(bat[stat][:, i], total)
            hrr = bat["H"][:, i] + bat["R"][:, i] + bat["RBI"][:, i]
            props[prop_key("batter", pid, "HRR")] = sparse2d(hrr, total)
            tb = bat["1B"][:, i] + 2 * bat["2B"][:, i] + 3 * bat["3B"][:, i] + 4 * bat["HR"][:, i]
            props[prop_key("batter", pid, "TB")] = sparse2d(tb, total)
        sp = team.probable_pitcher
        if sp is not None:
            for stat in PITCHER_STATS:
                props[prop_key("pitcher", sp.mlbam_id, stat)] = sparse2d(
                    res.pit[team_key][stat], total
                )
    return {
        "n_sims": int(res.n_sims),
        "full": sparse2d(h, a),
        "f5": sparse2d(res.home_runs_f5, res.away_runs_f5),
        "props": props,
    }


def _canonical_row(r: Recommendation) -> dict[str, Any]:
    d = asdict(r)
    d["game_date"] = r.game_date.isoformat()
    d["tier"] = r.tier.value
    return d


def pred_hash(recs: Iterable[Recommendation]) -> str:
    """sha256 of one game's prediction rows, canonicalized and sorted."""
    rows = sorted(
        json.dumps(_canonical_row(r), sort_keys=True, default=str) for r in recs
    )
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def pred_hashes(recs: Iterable[Recommendation]) -> dict[int, str]:
    by_game: dict[int, list[Recommendation]] = {}
    for r in recs:
        by_game.setdefault(r.game_pk, []).append(r)
    return {pk: pred_hash(rs) for pk, rs in by_game.items()}


def load_histograms(path: Path) -> dict[str, dict[str, Any]]:
    """``{game_key: entry}``; empty when the file is missing or of another version."""
    if not path.exists():
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        raw = json.load(f)
    if raw.get("hist_version") != HIST_VERSION:
        return {}
    games: dict[str, dict[str, Any]] = raw.get("games", {})
    return games


def _write(path: Path, games: Mapping[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump({"hist_version": HIST_VERSION, "games": dict(games)}, f, sort_keys=True)
    tmp.replace(path)


def save_histograms(
    audit_dir: Path,
    slate_date: Date,
    hists: Mapping[int, dict[str, Any]],
    recs: Iterable[Recommendation],
    games: Iterable[Game],
    event_ids: Mapping[int, str] | None = None,
) -> int:
    """Write this run's games into the slate's histogram file; returns games written.

    Only the games in ``hists`` -- the ones this run simulated -- are replaced;
    every other game already on file is carried forward as it was.
    """
    if not hists:
        return 0
    hashes = pred_hashes(recs)
    by_pk = {g.game_pk: g for g in games}
    ids = event_ids or {}
    path = hist_path(audit_dir, slate_date)
    on_file = load_histograms(path)
    written = 0
    for pk, h in hists.items():
        g = by_pk.get(pk)
        if g is None or pk not in hashes:
            continue
        eid = ids.get(pk)
        key = _game_key(eid, g.matchup(), g.game_datetime_utc)
        on_file = {k: v for k, v in on_file.items() if v.get("game_pk") != pk}
        on_file[key] = {
            **h,
            "game_pk": pk,
            "event_id": eid,
            "matchup": g.matchup(),
            "commence_utc": g.game_datetime_utc,
            "pred_hash": hashes[pk],
        }
        written += 1
    _write(path, on_file)
    return written


def merge_hist_files(remote: Path, local: Path) -> bool:
    """Union two machines' histogram files by game; the local copy of a game wins."""
    if not remote.exists():
        return False
    merged = {**load_histograms(remote), **load_histograms(local)}
    _write(local, merged)
    return True
