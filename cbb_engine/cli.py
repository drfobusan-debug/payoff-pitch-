"""``cbb-engine``: commands — capture | watch | price | audit | results | torvik | budget | gaps.

``capture`` is one tick of the §5 cadence: it decides from the slate's tips which
of board / featured / close passes are due, pulls only those, and writes each as
its own Parquet shard through the runtime. Pricing, the card and the audit come
with later phases; nothing here writes a ledger row.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import date as Date
from datetime import datetime, timezone
from pathlib import Path

from cbb_engine import audit, cadence, card, checklist, email, history, pilot, state, study
from cbb_engine.config import SPORT, SPORT_KEY, data_dir, dry_run, load_config, runtime_root
from cbb_engine.data import capture, espn, injuries, torvik, watch
from cbb_engine.markets import EVENT_MARKETS, FEATURED
from cbb_engine.models import distribution as dist
from cbb_engine.output import movement, pdf, teamstats
from engine_runtime import ledger, quotes, runner, views
from engine_runtime.odds_client import OddsClient

log = logging.getLogger(__name__)


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
    out: set[str] = set()
    for m in manifest:
        if m.get("pass_kind") != "close":
            continue
        ids = m.get("event_ids")
        out.update(str(e) for e in ids) if isinstance(ids, list) else out.add(str(m.get("event_id", "")))
    return out


def cmd_capture(args: argparse.Namespace) -> int:
    cfg = load_config()
    day = _parse_date(args.date)
    root = runtime_root()
    client = _client()
    if client is None:
        return 2
    state.auto_pull(trees=("quotes",))
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
    closing = {e for e in cadence.close_due(tips, now, cfg.cadence) if args.force or e not in taken}
    if closing:
        ps = runner.start(SPORT, day.isoformat(), "close", root, started_at=at)

        def close(p: runner.Pass) -> None:
            evs = [e for e in client.board(FEATURED, pass_id=p.pass_id) if str(e.get("id", "")) in closing]
            p.spend(client.spent - p.credits)
            rows = capture.rows_from_events(evs, captured_at=p.started_at, pass_id=p.pass_id, pass_kind="close")
            shard(
                "close", rows, p.pass_id,
                event_ids=sorted({q.event_id for q in rows}), teams=capture.teams(evs),
            )

        ps.run("close", close)

    if written:
        state.auto_push(f"cbb capture {day} {len(written)} shard(s)", trees=("quotes",))
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
    state.auto_pull(trees=("quotes", "ledger"))
    ps = runner.start(SPORT, day.isoformat(), "featured", root)
    hits_path = data_dir() / "watch" / day.isoformat() / "hits.jsonl"
    pending = set()
    if hits_path.exists():
        for line in hits_path.read_text().splitlines():
            h = json.loads(line)
            if "espn_id" in h:
                pending.add(str(h["espn_id"]))
    cfg = load_config()
    priced = card.price(
        root,
        day.isoformat(),
        pass_id=ps.pass_id,
        pass_kind="featured",
        code_sha=ps.code_sha,
        env_lock_hash=ps.env_lock_hash,
        pending_events=pending,
        width=dist.load_width(),
        max_live_buys=cfg.exposure.max_live_buys,
        max_per_book=cfg.exposure.max_per_book,
    )
    if priced is None:
        print(f"{day}: no board shard to price")
        return 1
    if dry_run():
        args.dry_run = True
        print("CBBE_DRY_RUN=1: no ledger rows, no email")
    if not args.dry_run:
        ledger.write_once(root, SPORT, day.isoformat(), ps.pass_id, priced.rows)
    text = card.render(priced.rows)
    out = data_dir() / "cards" / f"{day.isoformat()}_{ps.pass_id}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    paths = {"card": out}
    card_pdf = _card_pdf(root, day, ps.pass_id, priced.rows, out.with_suffix(".pdf"))
    if card_pdf is not None:
        paths["card_pdf"] = card_pdf
    if not args.dry_run:
        state.auto_push(f"cbb card {day} {ps.pass_id}", trees=("ledger", "cards"))
    if args.email and not args.dry_run:
        print(
            f"emailed {email.send(cfg, subject=f'CBB card {day} ({ps.pass_id})', text=text, paths=paths, to=args.to)}"
        )
    return 0


def _manifest_teams(root: Path, day: Date) -> dict[str, tuple[str, str]]:
    teams: dict[str, tuple[str, str]] = {}
    for m in quotes.read_manifest(root, SPORT, day.isoformat()):
        tm = m.get("teams")
        if isinstance(tm, dict):
            teams.update({str(k): (str(v[0]), str(v[1])) for k, v in tm.items()})
    return teams


def _card_pdf(root: Path, day: Date, pass_id: str, rows: list[ledger.Row], out: Path) -> Path | None:
    """Display-only PDF beside the text card; any failure leaves the text card as the card."""
    try:
        logs = torvik.as_of(data_dir(), "game_logs", day).path
        adj = torvik.as_of(data_dir(), "team_results", day).path
        stats = teamstats.season(teamstats.read_logs(logs), day, torvik.read_csv(adj) if adj else ())
        payload = injuries.fetch()
        if payload is not None:
            injuries.archive(data_dir(), day, payload, at=quotes.now_utc())
        src, listed = injuries.latest(data_dir(), day)
        body = pdf.build_html(
            day,
            rows,
            _manifest_teams(root, day),
            stats,
            title=f"pass {pass_id} ·",
            injuries=listed,
            injuries_source=f"(RotoWire {src.stem[-6:-2]} UTC) " if src else "(RotoWire: none fetched) ",
            moves=movement.moves(quotes.read_day(root, SPORT, day.isoformat()), until=quotes.now_utc()),
        )
        out.write_bytes(pdf.render_pdf(body))
    except Exception:
        log.warning("CBB card PDF failed; sending the text card only", exc_info=True)
        return None
    return out


def cmd_audit(args: argparse.Namespace) -> int:
    """Grade yesterday's bet-of-record rows against close shards + ESPN finals; print the views."""
    day = _parse_date(args.date)
    root = runtime_root()
    state.auto_pull(trees=("quotes", "ledger"))
    games = espn.ESPNClient().scoreboard(day, archive=data_dir()) if not args.offline else []
    teams = _manifest_teams(root, day)
    ps = runner.start(SPORT, day.isoformat(), "board", root)
    written = audit.grade(root, day.isoformat(), games, teams, pass_id=ps.pass_id, width=dist.load_width())
    if written:
        state.auto_push(f"cbb grade {day}", trees=("ledger",))
    rows = list(ledger.record(root, SPORT, day.isoformat()).values())
    rungs = [
        "v0_shape",
        "foul_redistribution",
        "half_correlation",
        "blend_shift",
        "availability_residual",
    ]
    shards = quotes.read_day(root, SPORT, day.isoformat())
    rep = audit.report(rows, rungs, shards)
    # The stake bar is season-to-date: eight live buys a day never reach 100 in one slate.
    season = [
        r
        for d in sorted(p.name for p in (root / "ledger" / SPORT).glob("*") if p.is_dir())
        if d <= day.isoformat()
        for r in ledger.record(root, SPORT, d).values()
    ]
    rep["v_promotion"] = views.v_promotion(season)
    if args.time_travel:
        stamps = [("quote", q.captured_at) for q in shards] + [("row", r.tip_utc) for r in rows]
        rep["time_travel_violations"] = study.time_travel(day.isoformat(), stamps)
    text = (
        checklist.render(checklist.run(load_config(), root, day.isoformat()))
        + "\n\n"
        + audit.render(rep)
    )
    out = data_dir() / "audit" / f"{day.isoformat()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"graded {len(written)} row(s)")
    if args.email:
        print(
            f"emailed {email.send(load_config(), subject=f'CBB audit {day}', text=text, paths={'audit': out}, to=args.to)}"
        )
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


def cmd_study(args: argparse.Namespace) -> int:
    """Tape-only studies: `funnel` (gate survival by market x bucket) and `featured` (cut 10)."""
    day = _parse_date(args.date)
    root = runtime_root()
    rows = list(ledger.record(root, SPORT, day.isoformat()).values())
    if args.which == "funnel":
        print(json.dumps(study.funnel(rows), indent=2))
    else:
        print(
            json.dumps(
                study.featured_value(rows, quotes.read_day(root, SPORT, day.isoformat())), indent=2
            )
        )
    return 0


def cmd_checklist(args: argparse.Namespace) -> int:
    day = _parse_date(args.date)
    items = checklist.run(
        load_config(), runtime_root(), day.isoformat(), daily_credits=args.daily_credits
    )
    print(checklist.render(items))
    return 0 if all(i.ok for i in items) else 1


def cmd_history(args: argparse.Namespace) -> int:
    """§4.e buy: plan (free, ESPN) → print cost → pull only with --yes; `export` is offline."""
    root, ddir = runtime_root(), data_dir()
    start, end = Date.fromisoformat(args.start), Date.fromisoformat(args.end)
    days = history.days_between(start, end)
    if args.action == "pilot":
        return cmd_pilot(args, root, days)
    if args.action == "export":
        rows: list[dict[str, object]] = []
        for d in days:
            rows += history.export_day(root, ddir, d)
        n = history.write_csv(rows, Path(args.out))
        graded = sum(1 for r in rows if r["final_total"] != "")
        print(f"{n} games with a close, {graded} with an ESPN final -> {args.out}")
        return 0
    esp = espn.ESPNClient()
    tips_by_day: dict[str, list[str]] = {}
    for d in days:
        games = history.latest_scoreboard(ddir, d)
        if not games:
            games = esp.scoreboard(Date.fromisoformat(d), archive=ddir)
        tips = history.tips_from_games(games, d)
        if tips:
            tips_by_day[d] = tips
    cfg = load_config()
    dense = tuple(args.dense or ())
    pl = history.plan(tips_by_day, dense_days=dense, regions=len(cfg.regions.split(",")))
    print(json.dumps({k: v for k, v in pl.items() if k != "days"}, indent=2))
    print(f"days with games: {len(tips_by_day)}; dense slates: {list(dense)}")
    if args.action == "plan" or not args.yes:
        if args.action == "pull":
            print("nothing bought: re-run with --yes to spend the credits above", file=sys.stderr)
        return 0
    client = _client(cache_ttl=0)
    if client is None:
        return 1
    totals = {"snapshots": 0, "skipped": 0, "rows": 0, "events": 0}
    for d, tips in sorted(tips_by_day.items()):
        c = history.pull_day(client, root, d, tips, dense=d in dense)
        for k in totals:
            totals[k] += c[k]
        print(f"{d}: {c} remaining={client.credits_remaining}")
    print(json.dumps({**totals, "spent": client.spent}, indent=2))
    return 0


def cmd_pilot(args: argparse.Namespace, root: Path, days: list[str]) -> int:
    """§4.e pilot over the dense slates already on disk; --write fills the TODO-PILOT slots."""
    rows: list[dict[str, object]] = []
    deltas: list[float] = []
    width = dist.load_width()
    if width is None:
        print("width.json not fitted; run scripts.cbb.fit_width first")
        return 2
    sd = width.sd_by_bucket["all"]
    for d in days:
        shards = quotes.read_day(root, SPORT, d)
        if not any(
            q.pass_kind == "featured" and q.pass_id.startswith(history.HIST_PREFIX) for q in shards
        ):
            continue
        rows += pilot.clv_rows(shards, sd)
        deltas += pilot.featured_deltas(shards, sd)
    scores: list[float] = []
    if args.closes:
        with open(args.closes, newline="") as fh:
            scores = pilot.log_scores(csv.DictReader(fh), sd, start=args.start)
    rep = pilot.summarize(
        rows, deltas, scores, cluster_inflation=card.prereg_scalar("cluster_inflation", 1.4)
    )
    print(json.dumps(rep, indent=2))
    if args.write:
        prereg = rep["prereg"]
        assert isinstance(prereg, dict)
        done = pilot.write_prereg(
            card.PREREG,
            prereg,
            note=f"{Date.today().isoformat()}, {len(rows)} quotes on {len(rep['days'])} dense slates",
        )  # type: ignore[arg-type]
        print(f"wrote {done} to {card.PREREG}")
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
    pr = sub.add_parser(
        "price", help="price the slate from the latest shard and write the card pass"
    )
    pr.add_argument("--date")
    pr.add_argument(
        "--dry-run", action="store_true", help="print the card, write no ledger rows, send nothing"
    )
    pr.add_argument("--email", action="store_true")
    pr.add_argument("--to")
    pr.set_defaults(func=cmd_price)
    au = sub.add_parser(
        "audit", help="grade bet-of-record rows (close shards + ESPN finals), print the views"
    )
    au.add_argument("--date")
    au.add_argument("--offline", action="store_true", help="no ESPN call; CLV only")
    au.add_argument(
        "--time-travel",
        action="store_true",
        help="§4.y: flag any input stamped after the slate day",
    )
    au.add_argument("--email", action="store_true")
    au.add_argument("--to")
    au.set_defaults(func=cmd_audit)
    st = sub.add_parser("study", help="gate funnel / featured-pass value (cut 10) from the tape")
    st.add_argument("which", choices=["funnel", "featured"])
    st.add_argument("--date")
    st.set_defaults(func=cmd_study)
    ck = sub.add_parser("checklist", help="Phase 0b go/no-go items, pass/fail; exit 1 on any fail")
    ck.add_argument("--date")
    ck.add_argument("--daily-credits", type=int, help="credit limit the budget must fit under")
    ck.set_defaults(func=cmd_checklist)
    bu = sub.add_parser("budget", help="credits one slate day costs under the cadence table")
    bu.add_argument("--games", type=int, default=60)
    bu.set_defaults(func=cmd_budget)
    hi = sub.add_parser(
        "history",
        help="§4.e: buy 2024-25 closes from the historical endpoint; export fit_width CSV",
    )
    hi.add_argument("action", choices=["plan", "pull", "export", "pilot"])
    hi.add_argument("--start", default="2024-11-04")
    hi.add_argument("--end", default="2025-04-07")
    hi.add_argument(
        "--dense", nargs="*", help="slate dates that also get the 10-min featured grid from T-3h"
    )
    hi.add_argument("--yes", action="store_true", help="actually spend credits (pull)")
    hi.add_argument("--out", default="closes_2024_25.csv")
    hi.add_argument("--closes", help="pilot: fit_width CSV for the week-5 log-score sample")
    hi.add_argument(
        "--write", action="store_true", help="pilot: fill TODO-PILOT slots in preregistered.yaml"
    )
    hi.set_defaults(func=cmd_history)
    ga = sub.add_parser("gaps", help="capture gaps per pass_kind + shard integrity")
    ga.add_argument("--date")
    ga.set_defaults(func=cmd_gaps)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
