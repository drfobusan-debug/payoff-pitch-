"""``cbb-engine``: commands — capture | watch | price | audit | results | torvik | budget | gaps.

``capture`` is one tick of the §5 cadence: it decides from the slate's tips which
of board / featured / close passes are due, pulls only those, and writes each as
its own Parquet shard through the runtime. Pricing, the card and the audit come
with later phases; nothing here writes a ledger row.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date as Date
from datetime import datetime, timezone

from cbb_engine import audit, cadence, card, email
from cbb_engine.config import SPORT, SPORT_KEY, data_dir, load_config, runtime_root
from cbb_engine.data import capture, espn, torvik, watch
from cbb_engine.markets import EVENT_MARKETS, FEATURED
from cbb_engine.models import distribution as dist
from engine_runtime import ledger, quotes, runner
from engine_runtime.odds_client import OddsClient


def _parse_date(raw: str | None) -> Date:
    return Date.fromisoformat(raw) if raw else datetime.now(timezone.utc).date()


def _client(cache_ttl: int | None = None) -> OddsClient | None:
    cfg = load_config()
    client = OddsClient(
        SPORT,
        SPORT_KEY,
        cfg.creds.odds_api_key,
        regions=cfg.regions,
        cache_dir=data_dir() / "cache" / "oddsapi",
        cache_ttl=cfg.cache_ttl if cache_ttl is None else cache_ttl,
        runtime_root=runtime_root(),
    )
    if not client.available():
        print("no Odds API key (THE_ODDS_API_KEY / ODDS_API_KEY)", file=sys.stderr)
        return None
    return client


def _last_pass(manifest: list[dict[str, object]], kind: str) -> datetime | None:
    stamps = [
        cadence.parse_utc(str(m["captured_at"])) for m in manifest if m.get("pass_kind") == kind
    ]
    held = [s for s in stamps if s is not None]
    return max(held) if held else None


def _closes_taken(manifest: list[dict[str, object]]) -> set[str]:
    return {str(m.get("event_id", "")) for m in manifest if m.get("pass_kind") == "close"}


def cmd_capture(args: argparse.Namespace) -> int:
    cfg = load_config()
    day = _parse_date(args.date)
    root = runtime_root()
    client = _client()
    if client is None:
        return 2
    now = datetime.now(timezone.utc)
    at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    events = client.events(commenceTimeFrom=now.strftime("%Y-%m-%dT%H:%M:%SZ"))
    slate = [
        e
        for e in events
        if capture.slate_date_of(str(e.get("commence_time", ""))) == day.isoformat()
    ]
    tips = {
        eid: t
        for eid, raw in capture.tips_of(slate).items()
        if (t := cadence.parse_utc(raw)) is not None
    }
    if not tips:
        print(f"{day}: no posted games")
        return 0
    manifest = quotes.read_manifest(root, SPORT, day.isoformat())
    written: list[str] = []

    def shard(kind: str, rows: list[quotes.Quote], pass_id: str, **extra: object) -> None:
        p = quotes.write_shard(
            root, SPORT, day.isoformat(), rows, pass_id=pass_id, pass_kind=kind, extra=extra
        )
        if p is not None:
            written.append(p.name)

    interval = cadence.slate_board_interval(tips.values(), now, cfg.cadence)
    if args.force or cadence.due(_last_pass(manifest, "board"), interval, now):
        ps = runner.start(SPORT, day.isoformat(), "board", root, started_at=at)

        def board(p: runner.Pass) -> None:
            evs = client.board(FEATURED, pass_id=p.pass_id)
            p.spend(client.spent - p.credits)
            shard(
                "board",
                capture.rows_from_events(
                    evs, captured_at=p.started_at, pass_id=p.pass_id, pass_kind="board"
                ),
                p.pass_id,
                teams=capture.teams(evs),
                interval_min=interval,
            )
            if cfg.event_markets_on:
                for eid in sorted(tips)[: cfg.max_events]:
                    ev = client.event(eid, EVENT_MARKETS, pass_id=p.pass_id)
                    if ev is not None:
                        shard(
                            "board",
                            capture.rows_from_events(
                                [ev],
                                captured_at=p.started_at,
                                pass_id=f"{p.pass_id}_{eid[:8]}",
                                pass_kind="board",
                            ),
                            f"{p.pass_id}_{eid[:8]}",
                            event_id=eid,
                        )

        ps.run("board", board)

    pulled = watch.pull_forward(data_dir(), day)
    if cadence.featured_due(tips.values(), now, cfg.cadence) and (
        args.force
        or pulled
        or cadence.due(_last_pass(manifest, "featured"), cfg.cadence.featured, now)
    ):
        ps = runner.start(SPORT, day.isoformat(), "featured", root, started_at=at)

        def featured(p: runner.Pass) -> None:
            evs = client.board(FEATURED, pass_id=p.pass_id)
            p.spend(client.spent - p.credits)
            shard(
                "featured",
                capture.rows_from_events(
                    evs, captured_at=p.started_at, pass_id=p.pass_id, pass_kind="featured"
                ),
                p.pass_id,
                teams=capture.teams(evs),
                pulled_forward=pulled,
            )

        ps.run("featured", featured)

    taken = _closes_taken(manifest)
    for eid in cadence.close_due(tips, now, cfg.cadence):
        if eid in taken and not args.force:
            continue
        ps = runner.start(SPORT, day.isoformat(), "close", root, started_at=at)

        def close(p: runner.Pass, eid: str = eid) -> None:
            ev = client.event(eid, FEATURED, pass_id=p.pass_id)
            p.spend(client.spent - p.credits)
            if ev is not None:
                shard(
                    "close",
                    capture.rows_from_events(
                        [ev], captured_at=p.started_at, pass_id=p.pass_id, pass_kind="close"
                    ),
                    p.pass_id,
                    event_id=eid,
                )

        ps.run("close", close)

    print(
        f"{day}: {len(tips)} game(s); wrote {written}; spent {client.spent}; remaining {client.credits_remaining}"
    )
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    day = _parse_date(args.date)
    root = data_dir()
    at = quotes.now_utc()
    games = espn.ESPNClient().scoreboard(day, archive=root)
    html = None if args.no_rotowire else watch.fetch_rotowire()
    hits = watch.tick(root, day, games, html, at=at)
    for h in hits:
        print("HIT", json.dumps(h))
    print(f"{day}: {len(games)} game(s) on ESPN, {len(hits)} hit(s); no credits spent")
    return 0


def cmd_results(args: argparse.Namespace) -> int:
    day = _parse_date(args.date)
    games = espn.ESPNClient().scoreboard(day, archive=data_dir())
    for g in games:
        f = g.final
        score = f"{f[0]}-{f[1]} halves {g.away_halves}/{g.home_halves}" if f else g.state
        print(f"{g.away} @ {g.home}{' (N)' if g.neutral else ''}: {score}")
    return 0


def cmd_torvik(args: argparse.Namespace) -> int:
    day = _parse_date(args.date)
    client = torvik.TorvikClient()
    for name in args.files:
        p = client.archive(data_dir(), name, day)
        print(f"{name}: {'wrote ' + str(p) if p else 'FAILED'}")
    return 0


def cmd_price(args: argparse.Namespace) -> int:
    """Write the card pass (bet of record) from the latest board/featured shard."""
    day = _parse_date(args.date)
    root = runtime_root()
    ps = runner.start(SPORT, day.isoformat(), "featured", root)
    hits_path = data_dir() / "watch" / day.isoformat() / "hits.jsonl"
    pending = set()
    if hits_path.exists():
        for line in hits_path.read_text().splitlines():
            h = json.loads(line)
            if "espn_id" in h:
                pending.add(str(h["espn_id"]))
    cfg = load_config()
    priced = card.price(root, day.isoformat(), pass_id=ps.pass_id, pass_kind="featured", code_sha=ps.code_sha, env_lock_hash=ps.env_lock_hash, pending_events=pending, width=dist.load_width(), max_live_buys=cfg.exposure.max_live_buys, max_per_book=cfg.exposure.max_per_book)
    if priced is None:
        print(f"{day}: no board shard to price")
        return 1
    if not args.dry_run:
        ledger.write_once(root, SPORT, day.isoformat(), ps.pass_id, priced.rows)
    text = card.render(priced.rows)
    out = data_dir() / "cards" / f"{day.isoformat()}_{ps.pass_id}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    if args.email and not args.dry_run:
        print(f"emailed {email.send(cfg, subject=f'CBB card {day} ({ps.pass_id})', text=text, paths={'card': out}, to=args.to)}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Grade yesterday's bet-of-record rows against close shards + ESPN finals; print the views."""
    day = _parse_date(args.date)
    root = runtime_root()
    games = espn.ESPNClient().scoreboard(day, archive=data_dir()) if not args.offline else []
    teams: dict[str, tuple[str, str]] = {}
    for m in quotes.read_manifest(root, SPORT, day.isoformat()):
        extra = m.get("extra")
        if isinstance(extra, dict) and isinstance(extra.get("teams"), dict):
            teams.update({str(k): (str(v[0]), str(v[1])) for k, v in extra["teams"].items()})
    ps = runner.start(SPORT, day.isoformat(), "board", root)
    written = audit.grade(root, day.isoformat(), games, teams, pass_id=ps.pass_id)
    rows = list(ledger.record(root, SPORT, day.isoformat()).values())
    rungs = ["v0_shape", "foul_redistribution", "half_correlation", "blend_shift", "availability_residual"]
    text = audit.render(audit.report(rows, rungs, quotes.read_day(root, SPORT, day.isoformat())))
    out = data_dir() / "audit" / f"{day.isoformat()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"graded {len(written)} row(s)")
    if args.email:
        print(f"emailed {email.send(load_config(), subject=f'CBB audit {day}', text=text, paths={'audit': out}, to=args.to)}")
    return 0


def cmd_budget(args: argparse.Namespace) -> int:
    cfg = load_config()
    out = cadence.daily_budget(
        args.games,
        cfg.cadence,
        event_cost=len(EVENT_MARKETS) if cfg.event_markets_on else 0,
        regions=len(cfg.regions.split(",")),
    )
    print(json.dumps({"games": args.games, "regions": cfg.regions, **out}, indent=2))
    return 0


def cmd_gaps(args: argparse.Namespace) -> int:
    day = _parse_date(args.date)
    man = quotes.read_manifest(runtime_root(), SPORT, day.isoformat())
    tips: dict[str, str] = {}
    for m in man:
        extra = m.get("extra")
        if isinstance(extra, dict) and "tips" in extra and isinstance(extra["tips"], dict):
            tips.update({str(k): str(v) for k, v in extra["tips"].items()})
    print(json.dumps(quotes.expected_passes(tips, man), indent=2))
    print(f"bad shards: {quotes.verify(runtime_root(), SPORT, day.isoformat())}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cbb-engine", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser(
        "capture", help="one cadence tick: board / featured / close passes that are due"
    )
    cap.add_argument("--date")
    cap.add_argument("--force", action="store_true", help="ignore the clocks (dry runs)")
    cap.set_defaults(func=cmd_capture)
    wa = sub.add_parser("watch", help="ESPN + RotoWire look; no credits")
    wa.add_argument("--date")
    wa.add_argument("--no-rotowire", action="store_true")
    wa.set_defaults(func=cmd_watch)
    re_ = sub.add_parser("results", help="ESPN finals with half scores")
    re_.add_argument("--date")
    re_.set_defaults(func=cmd_results)
    tv = sub.add_parser("torvik", help="archive today's Torvik files")
    tv.add_argument("--date")
    tv.add_argument("files", nargs="*", default=list(torvik.FILES))
    tv.set_defaults(func=cmd_torvik)
    pr = sub.add_parser("price", help="price the slate from the latest shard and write the card pass")
    pr.add_argument("--date")
    pr.add_argument("--dry-run", action="store_true", help="print the card, write no ledger rows, send nothing")
    pr.add_argument("--email", action="store_true")
    pr.add_argument("--to")
    pr.set_defaults(func=cmd_price)
    au = sub.add_parser("audit", help="grade bet-of-record rows (close shards + ESPN finals), print the views")
    au.add_argument("--date")
    au.add_argument("--offline", action="store_true", help="no ESPN call; CLV only")
    au.add_argument("--email", action="store_true")
    au.add_argument("--to")
    au.set_defaults(func=cmd_audit)
    bu = sub.add_parser("budget", help="credits one slate day costs under the cadence table")
    bu.add_argument("--games", type=int, default=60)
    bu.set_defaults(func=cmd_budget)
    ga = sub.add_parser("gaps", help="capture gaps per pass_kind + shard integrity")
    ga.add_argument("--date")
    ga.set_defaults(func=cmd_gaps)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
