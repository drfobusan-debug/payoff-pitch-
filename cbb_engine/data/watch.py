"""The 5-minute watch: ESPN scoreboard + RotoWire lineups page, archived and diffed.

Spends no Odds API credits and never pulls a board. A hit (a game's status or
the lineups page changed since the last tick) is written to ``watch/<day>/hits.jsonl``
and ``featured_pull_forward`` tells the next capture tick to run the featured
pass now instead of waiting for its clock.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date as Date
from pathlib import Path

import requests

from cbb_engine.data.espn import Game
from mlb_engine.data import http

log = logging.getLogger(__name__)
ROTOWIRE_LINEUPS = "https://www.rotowire.com/basketball/college-lineups.php"


def _dir(root: Path, day: Date) -> Path:
    d = root / "watch" / day.isoformat()
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_rotowire(timeout: int = 25) -> str | None:
    try:
        resp = http.session(user_agent="cbb-engine/0.1", timeout=timeout).get(ROTOWIRE_LINEUPS)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("RotoWire lineups failed: %s", exc)
        return None
    return resp.text


def tick(
    root: Path, day: Date, games: list[Game], rotowire_html: str | None, *, at: str
) -> list[dict[str, object]]:
    """Archive this look, compare with the last one, return the hits."""
    d = _dir(root, day)
    hits: list[dict[str, object]] = []
    state_path = d / "last_state.json"
    prev: dict[str, str] = json.loads(state_path.read_text()) if state_path.exists() else {}
    cur: dict[str, str] = {g.espn_id: g.state for g in games}
    for eid, st in cur.items():
        if eid in prev and prev[eid] != st:
            hits.append(
                {"at": at, "kind": "espn_status", "espn_id": eid, "from": prev[eid], "to": st}
            )
    if rotowire_html is not None:
        digest = hashlib.sha256(rotowire_html.encode()).hexdigest()
        (d / f"rotowire_{at.replace(':', '')}.html").write_text(rotowire_html)
        if prev.get("_rotowire") not in (None, digest):
            hits.append({"at": at, "kind": "rotowire_change", "sha": digest})
        cur["_rotowire"] = digest
    elif "_rotowire" in prev:
        cur["_rotowire"] = prev["_rotowire"]
    state_path.write_text(json.dumps(cur, sort_keys=True))
    if hits:
        with (d / "hits.jsonl").open("a") as fh:
            for h in hits:
                fh.write(json.dumps(h) + "\n")
        (d / "featured_pull_forward").write_text(at)
    return hits


def pull_forward(root: Path, day: Date) -> bool:
    """True once, if a watch hit asked for the next featured pass early."""
    flag = root / "watch" / day.isoformat() / "featured_pull_forward"
    if flag.exists():
        flag.unlink()
        return True
    return False


__all__ = ["ROTOWIRE_LINEUPS", "fetch_rotowire", "pull_forward", "tick"]
