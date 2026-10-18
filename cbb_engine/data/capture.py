"""Odds API payload -> ``engine_runtime.quotes.Quote`` rows for the CBB market map.

One side per row; the opposite side's price is attached so de-vig needs no
join. Unknown provider markets are dropped, not guessed.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta, timezone

from cbb_engine.markets import BY_PROVIDER
from engine_runtime.quotes import Quote

ET = timezone(timedelta(hours=-5))


def slate_date_of(start_utc: str) -> str:
    """Game date on the US clock (a 00:30Z tip is the previous evening's slate)."""
    try:
        dt = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
    except ValueError:
        return ""
    return dt.astimezone(ET).date().isoformat()


def _side_key(market_key: str, outcome: dict[str, object], home: str, away: str) -> tuple[str, str]:
    name = str(outcome.get("name", ""))
    entity = str(outcome.get("description", ""))
    if market_key.startswith("pl_") or market_key == "team_total":
        return name.lower(), entity or name
    if market_key.endswith("total") or market_key.endswith("total_alt"):
        return name.lower(), ""
    if name == home:
        return "home", home
    if name == away:
        return "away", away
    return name.lower(), name


def teams(events: Iterable[dict[str, object]]) -> dict[str, list[str]]:
    """``event_id -> [home, away]`` for the shard manifest; the audit matches ESPN finals on these."""
    return {str(ev.get("id", "")): [str(ev.get("home_team", "")), str(ev.get("away_team", ""))] for ev in events}


def rows_from_events(
    events: Iterable[dict[str, object]], *, captured_at: str, pass_id: str, pass_kind: str
) -> list[Quote]:
    out: list[Quote] = []
    for ev in events:
        event_id = str(ev.get("id", ""))
        tip = str(ev.get("commence_time", ""))
        home, away = str(ev.get("home_team", "")), str(ev.get("away_team", ""))
        game_date = slate_date_of(tip)
        books = ev.get("bookmakers")
        if not isinstance(books, list):
            continue
        for bk in books:
            book = str(bk.get("key", ""))
            markets = bk.get("markets")
            if not isinstance(markets, list):
                continue
            for mk in markets:
                provider = str(mk.get("key", ""))
                market = BY_PROVIDER.get(provider)
                outcomes = mk.get("outcomes")
                if market is None or not isinstance(outcomes, list):
                    continue
                by_line: dict[tuple[float | None, str], list[dict[str, object]]] = {}
                for oc in outcomes:
                    pt = oc.get("point")
                    line = float(pt) if isinstance(pt, (int, float)) else None
                    group = str(oc.get("description", ""))
                    by_line.setdefault(
                        (
                            abs(line) if line is not None and market.key.endswith("ats") else line,
                            group,
                        ),
                        [],
                    ).append(oc)
                for (_, _), ocs in by_line.items():
                    for oc in ocs:
                        others = [o for o in ocs if o is not oc]
                        opp = others[0].get("price") if len(others) == 1 else None
                        price = oc.get("price")
                        if not isinstance(price, (int, float)):
                            continue
                        pt = oc.get("point")
                        side, entity = _side_key(market.key, oc, home, away)
                        out.append(
                            Quote(
                                captured_at=captured_at,
                                game_date=game_date,
                                event_id=event_id,
                                tip_utc=tip,
                                market=market.key,
                                side=side,
                                entity=entity,
                                line=float(pt) if isinstance(pt, (int, float)) else None,
                                book=book,
                                american=float(price),
                                opposite_american=float(opp)
                                if isinstance(opp, (int, float))
                                else None,
                                pass_id=pass_id,
                                pass_kind=pass_kind,
                            )
                        )
    return out


def tips_of(events: Iterable[dict[str, object]]) -> dict[str, str]:
    return {str(e.get("id", "")): str(e.get("commence_time", "")) for e in events}


__all__ = ["rows_from_events", "slate_date_of", "tips_of"]
