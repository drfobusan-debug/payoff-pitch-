"""Versioned fitted values: every number the engine prices with, with its receipts.

``<data>/params/<name>/<version>.json``. A refit writes a new version and never
edits an old one, so a ledger row can name the version that priced it and the
audit can replay it. ``version`` is the fit's UTC time plus a content hash.
Nothing fitted lives in ``config.py`` (plan §13, "Config vs fitted values").
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def params_dir(data_dir: Path, name: str) -> Path:
    return data_dir / "params" / name


def write(data_dir: Path, name: str, payload: dict, *, now: datetime | None = None) -> Path:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    version = f"{stamp}_{hashlib.sha256(body.encode()).hexdigest()[:8]}"
    path = params_dir(data_dir, name) / f"{version}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": name, "version": version, **payload}, indent=1))
    return path


def latest(data_dir: Path, name: str) -> dict | None:
    """The newest version of ``name``, or ``None`` if it was never fitted."""
    directory = params_dir(data_dir, name)
    for path in sorted(directory.glob("*.json"), reverse=True) if directory.is_dir() else []:
        try:
            payload = json.loads(path.read_text())
        except ValueError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


__all__ = ["latest", "params_dir", "write"]
