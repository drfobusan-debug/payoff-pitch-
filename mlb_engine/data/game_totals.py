"""The posted game total for every game on a slate, from whichever public board has it.

VSIN lists a total only once its splits are moving, and the card is only on disk
where the slate was priced -- so a sheet built on another machine, or before the
first pass, had fourteen blank totals. Every book posts the number, so ask them
in turn: the Odds API board (one credit, already keyed), ESPN's scoreboard (free,
DraftKings' line on every unplayed game) and TeamRankings' picks grid (their
own line on each total they call). Each source fills only what the ones before
it left blank.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date as Date

import requests

from mlb_engine.config import Config
from mlb_engine.data import http
from mlb_engine.data.oddsapi import OddsAPIClient, Quotes
from mlb_engine.data.teamrankings import TEAM_CODES, TeamRankingsClient
from mlb_engine.schemas import Slate

log = logging.getLogger(__name__)

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
# ESPN answers 403 to an unfamiliar agent string and 200 to curl's.
_ESPN_UA = "curl/8.5.0"

Totals = dict[str, float]
Source = Callable[[Config, Slate], Totals]


def oddsapi_totals(cfg: Config, slate: Slate) -> Totals:
    """Matchup -> the line the Odds API board quotes nearest even money."""
    client = OddsAPIClient(cfg.creds.odds_api_key, cache_dir=cfg.odds_cache_dir)
    if not client.available():
        return {}
    # in-play a total is the runs left, not the number posted, so started games are skipped
    return board_totals(client.fetch(slate, include_props=False, pregame_only=True))


def board_totals(quotes: Quotes) -> Totals:
    best: dict[str, tuple[float, float]] = {}
    for (matchup, market, selection), qs in quotes.items():
        if market != "game_total" or not selection.startswith("Over ") or not qs:
            continue
        try:
            line = float(selection.split()[1])
        except (IndexError, ValueError):
            continue
        # the over closest to -110 is the book's own number; alternates sit far from it
        d = min(abs(abs(q.american) - 110) for q in qs)
        if matchup not in best or d < best[matchup][0]:
            best[matchup] = (d, line)
    return {m: line for m, (_, line) in best.items()}


def espn_totals(cfg: Config, slate: Slate) -> Totals:
    """Matchup -> the over/under ESPN's scoreboard carries for the slate's date."""
    day: Date = slate.slate_date
    resp = http.get(
        ESPN_SCOREBOARD, params={"dates": day.strftime("%Y%m%d")}, timeout=20, user_agent=_ESPN_UA
    )
    resp.raise_for_status()
    return parse_espn(resp.json())


def parse_espn(payload: object) -> Totals:
    out: Totals = {}
    if not isinstance(payload, dict):
        return out
    for event in payload.get("events", []) or []:
        for comp in event.get("competitions", []) or []:
            teams = {
                t.get("homeAway"): TEAM_CODES.get(ab, ab)
                for t in comp.get("competitors", []) or []
                for ab in [str(t.get("team", {}).get("abbreviation", ""))]
            }
            if not teams.get("away") or not teams.get("home"):
                continue
            for odds in comp.get("odds", []) or []:
                ou = odds.get("overUnder")
                if isinstance(ou, (int, float)):
                    out[f"{teams['away']} @ {teams['home']}"] = float(ou)
                    break
    return out


def teamrankings_totals(cfg: Config, slate: Slate) -> Totals:
    """Matchup -> the total TeamRankings' grid prices, on the games it calls."""
    picks = TeamRankingsClient(cfg.creds).fetch(slate.slate_date.isoformat())
    return {p.matchup: p.line for p in picks if p.market == "game_total" and p.line is not None}


SOURCES: tuple[tuple[str, Source], ...] = (
    ("Odds API", oddsapi_totals),
    ("ESPN", espn_totals),
    ("TeamRankings", teamrankings_totals),
)


def posted_totals(
    cfg: Config,
    slate: Slate,
    missing: set[str],
    sources: tuple[tuple[str, Source], ...] = SOURCES,
) -> Totals:
    """Fill ``missing`` matchups from each source in turn, stopping once none remain.

    A source that fails is logged and skipped: a blank total is what we are
    trying to avoid, so nobody's outage gets to blank the sheet.
    """
    found: Totals = {}
    for name, fetch in sources:
        left = missing - found.keys()
        if not left:
            break
        try:
            got = fetch(cfg, slate)
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            log.warning("totals: %s board unavailable: %s", name, exc)
            continue
        hits = {m: line for m, line in got.items() if m in left}
        if hits:
            log.info("totals: %s posted %d of %d missing total(s)", name, len(hits), len(left))
        found.update(hits)
    return found
