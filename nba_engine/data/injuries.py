"""Archive and parse the NBA's official injury report PDFs.

The league posts the report hourly on game days (11 AM to 9 PM ET checked from
here for the files named ``..._11AM.pdf`` through ``..._09PM.pdf``) at
``ak-static.cms.nba.com``. Each PDF is archived as published, content-addressed,
under ``<data>/injuries/<YYYY-MM-DD>/<HHMM>_<fp>.pdf`` with the parsed rows
beside it, so the availability state at any send time can be replayed later
(plan §5, §10).

Parsing reads the text items' x positions against the header columns, which
survives names and reasons that wrap onto several lines.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
from dataclasses import asdict, dataclass, fields
from datetime import date as Date
from datetime import datetime
from pathlib import Path

import requests
from pypdf import PdfReader

from mlb_engine.data import http
from nba_engine.data.oddsapi import SLATE_TZ

log = logging.getLogger(__name__)

URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{day}_{hour}.pdf"
HOURS: tuple[int, ...] = tuple(range(11, 22))  # 11 AM .. 9 PM ET
STATUSES: frozenset[str] = frozenset({"Out", "Doubtful", "Questionable", "Probable", "Available"})
_COLUMNS: tuple[tuple[str, str], ...] = (
    ("date", "Date"),
    ("time", "Time"),
    ("matchup", "Matchup"),
    ("team", "Team"),
    ("player", "Player"),
    ("status", "Current"),
    ("reason", "Reason"),
)
_DEFAULT_X: dict[str, float] = {
    "date": 23.0,
    "time": 119.0,
    "matchup": 200.0,
    "team": 264.0,
    "player": 425.0,
    "status": 585.0,
    "reason": 666.0,
}
_MATCHUP = re.compile(r"^[A-Z]{2,3}@[A-Z]{2,3}$")
_DATE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
# A team that has not filed prints "NOT YET SUBMITTED" in the reason column.
_TEAM_NOTES: frozenset[str] = frozenset({"NOT", "YET", "SUBMITTED"})


@dataclass(frozen=True)
class InjuryRow:
    report: str  # report stamp, e.g. 2025-12-10T17:00
    game_date: str
    tip_et: str
    matchup: str
    team: str
    player: str  # "Last, First" as printed
    status: str
    reason: str


FIELDS = tuple(f.name for f in fields(InjuryRow))


def hour_label(hour: int) -> str:
    """24h hour -> the file's label: 17 -> ``05PM``."""
    suffix = "AM" if hour < 12 else "PM"
    return f"{(hour - 1) % 12 + 1:02d}{suffix}"


def url_for(day: Date, hour: int) -> str:
    return URL.format(day=day.isoformat(), hour=hour_label(hour))


@dataclass(frozen=True)
class _Item:
    x: float
    y: float
    text: str


def _page_items(page: object) -> list[_Item]:
    items: list[_Item] = []

    def visit(text: str, cm: list[float], tm: list[float], _font: object, _size: object) -> None:
        if text.strip():
            items.append(_Item(x=tm[4] * cm[0] + cm[4], y=tm[5] * cm[3] + cm[5], text=text.strip()))

    page.extract_text(visitor_text=visit)  # type: ignore[attr-defined]
    return items


def _columns(items: list[_Item]) -> tuple[dict[str, float], float]:
    """Column left edges from the header row, and the header's y."""
    header_y = next((i.y for i in items if i.text == "Matchup"), None)
    if header_y is None:
        return dict(_DEFAULT_X), float("inf")
    row = [i for i in items if abs(i.y - header_y) < 2]
    cols = dict(_DEFAULT_X)
    for key, word in _COLUMNS:
        hits = [i.x for i in row if i.text == word]
        if hits:
            # "Game Date"/"Game Time"/"Player Name"/"Current Status": the label's first word.
            first = [i.x for i in row if i.x < hits[0] and hits[0] - i.x < 45]
            cols[key] = min(first + hits) - 1.5
    return cols, header_y


def _column(x: float, cols: dict[str, float]) -> str:
    name = "date"
    for key, left in sorted(cols.items(), key=lambda kv: kv[1]):
        if x >= left:
            name = key
    return name


def parse_pdf(data: bytes, report: str = "") -> list[InjuryRow]:
    """Every player row in the report, carrying the game and team it sits under."""
    rows: list[InjuryRow] = []
    carry = {"date": "", "time": "", "matchup": "", "team": ""}
    for page in PdfReader(io.BytesIO(data)).pages:
        items = _page_items(page)
        cols, header_y = _columns(items)
        footer_y = next((i.y for i in items if i.text == "Page"), None)
        body = [
            i
            for i in items
            if i.y < header_y - 2
            and (footer_y is None or abs(i.y - footer_y) > 2)
            and i.text not in _TEAM_NOTES
        ]
        by_col: dict[str, list[_Item]] = {}
        for item in body:
            by_col.setdefault(_column(item.x, cols), []).append(item)
        anchors = sorted(
            (i for i in by_col.get("status", []) if i.text in STATUSES), key=lambda i: -i.y
        )
        if not anchors:
            continue

        ys = [a.y for a in anchors]

        names: dict[int, list[_Item]] = {}
        reasons: dict[int, list[_Item]] = {}
        for item in by_col.get("player", []):
            names.setdefault(_nearest(ys, item.y), []).append(item)
        for item in by_col.get("reason", []):
            reasons.setdefault(_nearest(ys, item.y), []).append(item)

        context = sorted(
            ((key, i) for key in ("date", "time", "matchup", "team") for i in by_col.get(key, [])),
            key=lambda kv: (-kv[1].y, kv[1].x),
        )
        lines: dict[tuple[str, float], list[_Item]] = {}
        for key, item in context:
            lines.setdefault((key, round(item.y, 0)), []).append(item)
        stamped = sorted(
            (
                (key, y, " ".join(i.text for i in sorted(v, key=lambda i: i.x)))
                for (key, y), v in lines.items()
            ),
            key=lambda t: -t[1],
        )
        cursor = 0
        for k, anchor in enumerate(anchors):
            while cursor < len(stamped) and stamped[cursor][1] >= anchor.y - 1:
                key, _, text = stamped[cursor]
                if key == "time":
                    text = text.replace("(ET)", "").strip()
                if key == "date" and not _DATE.match(text):
                    text = carry["date"]
                if key == "matchup" and not _MATCHUP.match(text):
                    text = carry["matchup"]
                carry[key] = text
                if key == "matchup":
                    carry["team"] = ""
                cursor += 1
            rows.append(
                InjuryRow(
                    report=report,
                    game_date=carry["date"],
                    tip_et=carry["time"],
                    matchup=carry["matchup"],
                    team=carry["team"],
                    player=_join(names.get(k, [])),
                    status=anchor.text,
                    reason=_join(reasons.get(k, [])),
                )
            )
    return rows


def _nearest(ys: list[float], y: float) -> int:
    return min(range(len(ys)), key=lambda k: abs(ys[k] - y))


def _join(items: list[_Item]) -> str:
    ordered = sorted(items, key=lambda i: (-round(i.y, 0), i.x))
    text = re.sub(r"\s+", " ", " ".join(i.text for i in ordered)).replace(" ,", ",")
    return re.sub(r"(\w)- (\w)", r"\1-\2", text).strip()


def injuries_dir(data_dir: Path, day: Date) -> Path:
    return data_dir / "injuries" / day.isoformat()


def _fp(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:10]


def archive(data_dir: Path, day: Date, hour: int, data: bytes) -> Path | None:
    """Store one report (and its parsed rows) unless an identical one is held."""
    directory = injuries_dir(data_dir, day)
    fp = _fp(data)
    if directory.is_dir() and any(directory.glob(f"*_{fp}.pdf")):
        return None
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{hour:02d}00_{fp}"
    pdf = directory / f"{stem}.pdf"
    pdf.write_bytes(data)
    try:
        rows = parse_pdf(data, report=f"{day.isoformat()}T{hour:02d}:00")
    except Exception as exc:  # noqa: BLE001 - a layout change must not lose the PDF
        log.warning("injury report %s did not parse: %s", pdf.name, exc)
        return pdf
    with (directory / f"{stem}.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return pdf


def report_csvs(data_dir: Path, day: Date) -> list[Path]:
    """Parsed official reports held for a day, oldest first."""
    directory = injuries_dir(data_dir, day)
    return sorted(directory.glob("[0-9]*.csv")) if directory.is_dir() else []


def read_report(path: Path) -> list[InjuryRow]:
    with path.open(newline="") as fh:
        return [InjuryRow(**{k: r.get(k, "") for k in FIELDS}) for r in csv.DictReader(fh)]


def read_latest(data_dir: Path, day: Date) -> list[InjuryRow]:
    """Rows of the newest parsed report held for a day."""
    files = report_csvs(data_dir, day)
    return read_report(files[-1]) if files else []


class InjuryClient:
    def __init__(self, timeout: int = 25) -> None:
        self._session = http.session(user_agent="nba-engine/0.1", timeout=timeout)

    def fetch(self, day: Date, hour: int) -> bytes | None:
        try:
            resp = self._session.get(url_for(day, hour))
        except requests.RequestException as exc:
            log.warning("injury report %s %s failed: %s", day, hour_label(hour), exc)
            return None
        if resp.status_code != 200 or not resp.content.startswith(b"%PDF"):
            return None
        return resp.content

    def capture(self, data_dir: Path, day: Date, now: datetime | None = None) -> list[Path]:
        """Archive every hourly report for ``day`` published by ``now`` not yet held."""
        moment = (now or datetime.now(SLATE_TZ)).astimezone(SLATE_TZ)
        held = {p.name[:2] for p in injuries_dir(data_dir, day).glob("[0-9]*.pdf")}
        out: list[Path] = []
        for hour in HOURS:
            if day == moment.date() and hour > moment.hour:
                break
            if f"{hour:02d}" in held and hour < moment.hour:
                continue
            data = self.fetch(day, hour)
            if data is None:
                continue
            path = archive(data_dir, day, hour, data)
            if path is not None:
                out.append(path)
        return out


__all__ = [
    "FIELDS",
    "HOURS",
    "STATUSES",
    "InjuryClient",
    "InjuryRow",
    "archive",
    "hour_label",
    "injuries_dir",
    "parse_pdf",
    "read_latest",
    "read_report",
    "report_csvs",
    "url_for",
]
