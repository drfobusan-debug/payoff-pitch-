"""Daily Faceoff starting goalies page: per game, each team's goalie with a news strength.

``https://www.dailyfaceoff.com/starting-goalies/`` (``/<YYYY-MM-DD>`` for another
slate) is a Next.js page; the goalie board sits in the ``__NEXT_DATA__`` JSON as
``props.pageProps.data`` (one record per game, ``home*``/``away*`` fields). The raw
page is archived under ``<data_dir>/dailyfaceoff/<slate>/<stamp>.html`` before
parsing, like RotoWire.

Status mapping into the starter ladder (``features/starters.py``):

* ``Confirmed``            -> ``confirmed``
* ``Likely`` / ``Expected`` -> ``probable``
* ``Unconfirmed`` / none   -> nothing written (the projected leader stands)

The parsed slate reuses the RotoWire dataclasses (goalie only, no PP units or
injuries) so ``lineup_feed.apply_slate`` applies it with the same precedence rule:
a stronger status wins, an equal one is replaced only by the newer read.
"""

from __future__ import annotations

import json
import re
from datetime import date as Date
from datetime import datetime, timezone
from pathlib import Path

import requests

from nhl_engine.data.rotowire import USER_AGENT, RotoGame, RotoSlate, RotoTeam
from nhl_engine.data.teamnames import canonical

URL = "https://www.dailyfaceoff.com/starting-goalies/"
SOURCE = "dailyfaceoff"

STATUS_MAP = {"confirmed": "confirmed", "likely": "probable", "expected": "probable"}

_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S
)


def _status(raw: object) -> str:
    return STATUS_MAP.get(str(raw or "").strip().lower(), "")


def _team(rec: dict, side: str) -> RotoTeam:
    return RotoTeam(
        team=canonical(str(rec.get(f"{side}TeamAbbreviation", ""))),
        goalie=str(rec.get(f"{side}GoalieName") or "").strip(),
        goalie_status=_status(rec.get(f"{side}NewsStrengthName")),
    )


def _started(rec: dict, fetched_at: str) -> bool:
    gmt = str(rec.get("dateGmt") or "")
    if not gmt or not fetched_at:
        return False
    try:
        tip = datetime.fromisoformat(gmt.replace("Z", "+00:00"))
        now = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    return now >= tip


def parse_page(page: str, slate: Date, fetched_at: str = "") -> RotoSlate:
    """Parse the goalie board; a page without the expected JSON yields an empty slate."""
    out = RotoSlate(slate=slate, fetched_at=fetched_at)
    m = _NEXT_DATA_RE.search(page)
    if not m:
        return out
    try:
        data = json.loads(m.group(1))["props"]["pageProps"]["data"]
    except (ValueError, KeyError, TypeError):
        return out
    for rec in data if isinstance(data, list) else []:
        if (
            not isinstance(rec, dict)
            or str(rec.get("date", slate.isoformat())) != slate.isoformat()
        ):
            continue
        away, home = _team(rec, "away"), _team(rec, "home")
        if not away.team or not home.team:
            continue
        out.games.append(
            RotoGame(
                away=away,
                home=home,
                time_et=str(rec.get("time") or ""),
                started=_started(rec, fetched_at),
            )
        )
    return out


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def raw_dir(data_dir: Path, slate: Date) -> Path:
    return data_dir / "dailyfaceoff" / slate.isoformat()


def url_for(slate: Date, today: Date) -> str:
    return URL if slate == today else URL + slate.isoformat()


def fetch(
    slate: Date,
    *,
    today: Date,
    data_dir: Path | None = None,
    session: requests.Session | None = None,
    timeout: float = 20.0,
) -> RotoSlate:
    """Download the goalie board for ``slate``, archive the raw page, parse it."""
    sess = session or requests.Session()
    resp = sess.get(url_for(slate, today), headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    fetched = _now()
    raw_path: Path | None = None
    if data_dir is not None:
        raw_path = raw_dir(data_dir, slate) / f"{fetched.replace(':', '').replace('-', '')}.html"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(resp.text, encoding="utf-8")
    parsed = parse_page(resp.text, slate, fetched)
    parsed.raw_path = raw_path
    return parsed


__all__ = ["SOURCE", "STATUS_MAP", "URL", "fetch", "parse_page", "raw_dir", "url_for"]
