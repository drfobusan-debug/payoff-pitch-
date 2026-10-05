"""ESPN's league-wide NBA injury feed: the fastest free availability signal.

The official report posts on the hour; ESPN's feed carries each status change
with its own timestamp as teams announce it. Each distinct state of the feed is
archived content-addressed under ``<data>/injuries/<YYYY-MM-DD>/espn_<stamp>_<fp>.csv``
(the fingerprint covers team, player and status only, so a reworded comment is
not a new state), and the alarm compares consecutive states.
"""

from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import asdict, dataclass, fields
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import requests

from mlb_engine.data import http
from nba_engine.data import teamnames
from nba_engine.data.injuries import injuries_dir

log = logging.getLogger(__name__)

URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"


@dataclass(frozen=True)
class FeedRow:
    team: str
    player: str
    espn_id: str
    status: str
    updated: str
    comment: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.team, self.espn_id or self.player)


FIELDS = tuple(f.name for f in fields(FeedRow))


def parse_feed(payload: dict) -> list[FeedRow]:
    out: list[FeedRow] = []
    for team in payload.get("injuries", []):
        code = teamnames.code_for(str(team.get("displayName", ""))) or ""
        for item in team.get("injuries", []):
            ath = item.get("athlete", {})
            link = next(
                (
                    lk.get("href", "")
                    for lk in ath.get("links", [])
                    if "playercard" in lk.get("rel", [])
                ),
                "",
            )
            pid = link.split("/id/")[1].split("/")[0] if "/id/" in link else ""
            out.append(
                FeedRow(
                    team=code,
                    player=str(ath.get("displayName", "")),
                    espn_id=pid,
                    status=str(item.get("status", "")),
                    updated=str(item.get("date", "")),
                    comment=str(item.get("shortComment", "")).replace("\n", " "),
                )
            )
    return sorted(out, key=lambda r: (r.team, r.player))


def _fp(rows: list[FeedRow]) -> str:
    body = "\n".join(f"{r.team}|{r.espn_id or r.player}|{r.status}" for r in rows)
    return hashlib.sha256(body.encode()).hexdigest()[:10]


def feed_paths(data_dir: Path, day: Date) -> list[Path]:
    directory = injuries_dir(data_dir, day)
    return sorted(directory.glob("espn_*.csv")) if directory.is_dir() else []


def latest_feed_path(data_dir: Path, day: Date, lookback: int = 3) -> Path | None:
    """The newest archived state on or before ``day`` (the overnight state is the
    baseline the first look of a day is compared against)."""
    for back in range(lookback + 1):
        held = feed_paths(data_dir, day - timedelta(days=back))
        if held:
            return held[-1]
    return None


def read_feed(path: Path) -> list[FeedRow]:
    with path.open(newline="") as fh:
        return [FeedRow(**{k: r.get(k, "") for k in FIELDS}) for r in csv.DictReader(fh)]


def write_feed(rows: list[FeedRow], data_dir: Path, day: Date, stamp: str) -> Path | None:
    """Archive this state unless it matches the day's latest one."""
    if not rows:
        return None
    latest = latest_feed_path(data_dir, day)
    fp = _fp(rows)
    if latest is not None and latest.stem.endswith(fp):
        return None
    directory = injuries_dir(data_dir, day)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"espn_{stamp.replace(':', '').replace('-', '')}_{fp}.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return path


def fetch_feed(timeout: int = 20) -> list[FeedRow] | None:
    session = http.session(user_agent="nba-engine/0.1", timeout=timeout)
    try:
        resp = session.get(URL)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        log.warning("ESPN injury feed failed: %s", exc)
        return None
    return parse_feed(payload) if isinstance(payload, dict) else None


__all__ = [
    "FIELDS",
    "FeedRow",
    "feed_paths",
    "fetch_feed",
    "latest_feed_path",
    "parse_feed",
    "read_feed",
    "write_feed",
]
