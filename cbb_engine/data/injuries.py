"""RotoWire's college basketball injury list, archived per fetch (display only).

``/cbasketball/tables/injury-report.php`` returns every listed player as JSON
(player, school, position, injury, status). Each fetch is saved under
``<data>/injuries/<day>/rotowire_<HHMMSS>.json`` so a card can be replayed with
the list it was built from.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

from mlb_engine.data import http

log = logging.getLogger(__name__)

URL = "https://www.rotowire.com/cbasketball/tables/injury-report.php?team=ALL&pos=ALL"


@dataclass(frozen=True)
class Injury:
    player: str
    team: str
    position: str
    injury: str
    status: str


def fetch(timeout: int = 25) -> list[dict[str, object]] | None:
    try:
        resp = http.session(user_agent="cbb-engine/0.1", timeout=timeout).get(URL)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        log.warning("RotoWire CBB injury list unavailable", exc_info=True)
        return None
    return payload if isinstance(payload, list) else None


def archive(root: Path, day: Date, payload: list[dict[str, object]], *, at: str) -> Path:
    d = root / "injuries" / day.isoformat()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"rotowire_{at[11:19].replace(':', '')}.json"
    path.write_text(json.dumps(payload))
    return path


def latest(root: Path, day: Date) -> tuple[Path | None, list[Injury]]:
    files = sorted((root / "injuries" / day.isoformat()).glob("rotowire_*.json"))
    if not files:
        return None, []
    return files[-1], parse(json.loads(files[-1].read_text()))


def parse(payload: Iterable[object]) -> list[Injury]:
    out = []
    for p in payload:
        if not isinstance(p, dict) or not p.get("player") or not p.get("team"):
            continue
        out.append(
            Injury(
                player=str(p["player"]),
                team=str(p["team"]),
                position=str(p.get("position", "")),
                injury=str(p.get("injury", "")),
                status=str(p.get("status", "")),
            )
        )
    return out


__all__ = ["URL", "Injury", "archive", "fetch", "latest", "parse"]
