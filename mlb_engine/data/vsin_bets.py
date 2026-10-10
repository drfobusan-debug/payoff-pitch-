"""VSiN's written MLB best bets: captured with the card, printed on it, graded W/L.

The articles are read by :mod:`engine_common.podcasts.vsin` (``Pick: White Sox
+114`` under a ``White Sox vs. Guardians (-137, 7) Prediction`` heading). Each
game bet is placed on the day's MLB slate and settled at the writer's own number
against the official final; the writer's record comes from that ledger, the
same way the podcast records do. Props and parlays are printed as written and
never graded. Nothing here feeds a price, a gate, a tier or a stake.
"""

from __future__ import annotations

import html
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict
from datetime import date as Date
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from engine_common.podcasts import picks as core
from engine_common.podcasts import vsin
from engine_common.podcasts.extract import Pick
from engine_common.podcasts.shows import MLB
from mlb_engine.data.results import GameResult
from mlb_engine.schemas import Slate

LEDGER = "vsin_bets_ledger.csv"
_ET = ZoneInfo("America/New_York")
_KEY = ("date", "pick_id")
PRICE_SOURCE = "writer"


def picks_path(audit_dir: Path, day: Date) -> Path:
    return audit_dir / f"vsin_bets_{day.isoformat()}.json"


def load(path: Path) -> list[Pick]:
    if not path.exists():
        return []
    return [Pick(**d) for d in json.loads(path.read_text())]


def save(path: Path, picks: Iterable[Pick]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted({p.pick_id: p for p in picks}.values(), key=lambda p: (p.published, p.pick_id))
    path.write_text(json.dumps([asdict(p) for p in rows], indent=1))


def merge_files(remote: Path, local: Path) -> bool:
    """Union two captures of a day's articles; a pick is the same pick on both machines."""
    if not remote.exists():
        return False
    save(local, [*load(remote), *load(local)])
    return True


def window(day: Date) -> tuple[datetime, datetime]:
    """From the morning before (previews post the night before) to the end of ``day`` ET."""
    end = datetime.combine(day + timedelta(days=1), time(0), _ET).astimezone(timezone.utc)
    return end - timedelta(days=2), end


def capture(audit_dir: Path, day: Date) -> list[Pick]:
    """Read the day's articles and keep every bet in them, merged into the day's file."""
    since, until = window(day)
    path = picks_path(audit_dir, day)
    found = [p for post in vsin.fetch_posts(MLB, since, until) for p in vsin.picks_in(post)]
    save(path, [*load(path), *found])
    return load(path)


def _nick(name: str) -> str:
    words = name.lower().replace(".", "").split()
    return " ".join(words[-2:]) if words[-2:-1] in (["red"], ["white"], ["blue"]) else words[-1]


def same_team(card_name: str, said: str) -> bool:
    """``Chicago White Sox`` is ``White Sox``, ``Chicago``-alone is not (two Chicago teams)."""
    s = f" {said.lower().replace('.', '')} "
    return f" {_nick(card_name)} " in s or f" {card_name.lower()} " in s


# A series puts the same two teams on the board every day: a pick only belongs to
# a first pitch within a day of the article, not to the next game of the series.
LEAGUE: core.League[None] = core.League(MLB, same_team, horizon=timedelta(hours=24))


def games(slate: Slate) -> list[core.Game[None]]:
    return [
        core.Game(
            matchup=g.matchup(),
            home=g.home.name,
            away=g.away.name,
            kickoff=(
                datetime.fromisoformat(g.game_datetime_utc.replace("Z", "+00:00"))
                if g.game_datetime_utc
                else None
            ),
            day=g.game_date,
            rows=None,
        )
        for g in slate.games
    ]


def place_all(picks: Iterable[Pick], slate: Slate) -> list[core.Placed[None]]:
    card = games(slate)
    return [pl for p in picks if (pl := core.place(p, card, LEAGUE)) is not None]


def grade(
    day: Date,
    picks: Iterable[Pick],
    slate: Slate,
    result_of: Callable[[int], GameResult | None],
) -> list[dict[str, str]]:
    """One ledger row per placed game bet; ungraded until its game is final."""
    pk = {g.matchup(): g.game_pk for g in slate.games}
    rows: list[dict[str, str]] = []
    for pl in place_all(picks, slate):
        res = result_of(pk[pl.matchup])
        outcome = (
            core.settle(pl, res.home_runs, res.away_runs) if res is not None and res.final else None
        )
        rows.append(
            core.ledger_row(
                pl,
                league=MLB,
                day=day,
                outcome=outcome,
                price=pl.pick.price,
                price_source=PRICE_SOURCE,
                close_line=None,
                clv_pts=None,
                engine="",
            )
        )
    return rows


def update(audit_dir: Path, day: Date, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return core.update_ledger(audit_dir / LEDGER, rows, [day])


def writer_records(ledger: list[dict[str, str]]) -> list[core.Record]:
    graded = [r for r in ledger if r.get("result")]
    by: dict[str, list[dict[str, str]]] = {}
    for r in graded:
        by.setdefault(r.get("host") or vsin.SHOW_NAME, []).append(r)
    recs = [core.record(name, rows) for name, rows in by.items()]
    return sorted(recs, key=lambda r: (-r.n, r.label))


def _record_text(r: core.Record) -> str:
    out = f"{r.wins}-{r.losses}" + (f"-{r.pushes}" if r.pushes else "")
    if r.roi is not None:
        out += f", {r.units:+.2f}u on {r.priced} priced ({r.roi:+.1%})"
    return out + (", underpowered" if r.underpowered else "")


def render_text(ledger: list[dict[str, str]]) -> str:
    total = core.record("All VSiN best bets", [r for r in ledger if r.get("result")])
    lines = [f"VSiN best bets (graded at the writer's own number): {_record_text(total)}"]
    lines += [f"  {r.label:<18} {_record_text(r)}" for r in writer_records(ledger)]
    return "\n".join(lines) + "\n"


def html_block(
    day: Date, day_picks: list[Pick], slate: Slate | None, ledger: list[dict[str, str]]
) -> str:
    """The card's VSiN section: each bet with its writer's graded record."""
    if not day_picks:
        return ""
    recs = {r.label: r for r in writer_records(ledger)}
    placed = {pl.pick.pick_id: pl for pl in place_all(day_picks, slate)} if slate else {}
    since, _ = window(day)
    today = since + timedelta(days=1)
    shown = [
        p for p in day_picks if p.pick_id in placed or datetime.fromisoformat(p.published) >= today
    ]
    if not shown:
        return ""
    items = []
    for p in shown:
        writer = p.host or vsin.SHOW_NAME
        rec = recs.get(writer)
        pl = placed.get(p.pick_id)
        where = f"{pl.matchup}: " if pl is not None else ""
        note = "" if pl is not None or p.market == "other" else " (no game on today's slate)"
        items.append(
            f"<li><b>{html.escape(writer)}</b> "
            f"({html.escape(_record_text(rec) if rec else 'no graded picks yet')}). "
            f"{html.escape(where)}<b>{html.escape(p.description)}</b>"
            f"{' (not graded)' if p.market == 'other' else html.escape(note)}</li>"
        )
    return (
        "<h2>VSiN best bets</h2><p><em>As written in VSiN's articles, with each writer's record "
        "graded by us at the writer's own number. Not a model input and not a PayoffPitch "
        "play; a record says nothing about an edge until 100+ graded bets.</em></p>"
        f"<ul>{''.join(items)}</ul>"
    )


__all__ = [
    "LEAGUE",
    "LEDGER",
    "capture",
    "grade",
    "html_block",
    "load",
    "merge_files",
    "picks_path",
    "render_text",
    "same_team",
    "save",
    "update",
    "window",
    "writer_records",
]
