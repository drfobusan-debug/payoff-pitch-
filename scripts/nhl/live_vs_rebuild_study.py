"""Price the same NHL games two ways: the live ledger vs a from-scratch rebuild.

    python scripts/nhl/live_vs_rebuild_study.py <ledger_dir> <out.csv> [--start D] [--end D]

Reads ``graded_*.json`` from an engine-state ``nhl/ledger`` copy and keeps one
``game_total`` Over row per game (``early``/``initial`` card, the line whose
consensus is closest to 50%). For each game it re-prices that exact line with
``pipeline.team_inputs`` (``lineups={}``, as ``decompose`` does) twice:

* ``reb``: official boxscore starters (``closetest.goalie_overrides``);
* ``reb_livegoalie``: the goalies the live card named, matched to boxscore
  ids by team, first initial and last name (blank when either is unmatched).

If ``live - reb`` is ~0, the live gap to the market is the model's own opinion,
not a live-only input. If ``reb_livegoalie - reb`` is large, the goalie guess
is the cause. Research only: nothing here prices or gates a card.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date as Date
from pathlib import Path
from typing import Any

from nhl_engine import closetest, pipeline
from nhl_engine.calibration import Calibrator
from nhl_engine.config import cache_dir, load_config
from nhl_engine.data import preseason
from nhl_engine.data.moneypuck import MoneyPuckClient, season_of
from nhl_engine.data.nhlapi import NHLAPIClient
from nhl_engine.models import markets
from nhl_engine.models.goals import game_rates
from nhl_engine.models.periods import simulate


def live_rows(
    ledger: Path, start: Date | None, end: Date | None
) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for f in sorted(ledger.glob("graded_*.json")):
        for r in json.loads(f.read_text()):
            if r["market"] != "game_total" or r["side"] != "over" or not r.get("line"):
                continue
            if r.get("card_tag") not in ("early", "initial"):
                continue
            day = Date.fromisoformat(r["slate_date"])
            if (start and day < start) or (end and day > end):
                continue
            k = (r["slate_date"], r["matchup"])
            if k not in out or abs(r["consensus"] - 0.5) < abs(out[k]["consensus"] - 0.5):
                out[k] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ledger", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--start", type=Date.fromisoformat)
    ap.add_argument("--end", type=Date.fromisoformat)
    args = ap.parse_args()

    cfg = load_config()
    nhl = NHLAPIClient(cache_dir=cache_dir() / "nhlapi")
    mp = MoneyPuckClient(cache_dir=cache_dir())
    cal = Calibrator()
    ot = pipeline.DEFAULT_OT_RULE["game_total"]
    live = live_rows(args.ledger, args.start, args.end)
    priors: dict[int, preseason.PreseasonPrior] = {}
    rows: list[dict[str, Any]] = []
    for day in sorted({Date.fromisoformat(d) for d, _ in live}):
        season = season_of(day)
        if season not in priors:
            priors[season] = preseason.build(
                mp, season, r_yy=cfg.prior.r_yy, today=Date(season, 10, 1)
            )
        prior = priors[season]
        games = [g for g in nhl.schedule(day) if g.start_utc]
        results = {g.matchup: r for g in games if (r := nhl.result(g)) is not None and r.decided}
        league = pipeline.league_for(mp, day, season, prior, cfg.shrink.league_k_games)
        official = closetest.goalie_overrides(games, results)
        ids = {
            (p.team, p.name[0], p.name.split(". ", 1)[-1].lower()): p.player_id
            for res in results.values()
            for p in res.players
            if p.position == "G" and p.player_id
        }
        for i, g in enumerate(sorted(games, key=lambda g: g.matchup)):
            lr = live.get((day.isoformat(), g.matchup))
            res = results.get(g.matchup)
            if lr is None or res is None:
                continue
            named: dict[str, dict[str, object]] | None = dict(official)
            for team, name in ((g.home, lr["home_goalie"]), (g.away, lr["away_goalie"])):
                pid = ids.get((team, name[:1], name.split(" ", 1)[-1].lower()))
                if pid is None or named is None:
                    named = None
                    continue
                named[team] = {
                    "player_id": pid,
                    "name": name,
                    "status": "confirmed",
                    "source": "ledger",
                }
            out: dict[str, Any] = {
                "date": day.isoformat(),
                "matchup": g.matchup,
                "goals": res.final_home + res.final_away,
                "line": lr["line"],
                "mkt": lr["consensus"],
                "live": lr["model_prob"],
                "goalie_status": lr["goalie_status"],
                "team_rate_sides": lr["lineup_source"].count("team_rate"),
            }
            for tag, ov in (("reb", official), ("reb_livegoalie", named)):
                if ov is None:
                    out[tag] = out[tag + "_exp"] = out[tag + "_gsax"] = ""
                    continue
                h, a = (
                    pipeline.team_inputs(
                        mp,
                        c,
                        day,
                        season=season,
                        cfg=cfg,
                        prior=prior,
                        league=league,
                        lineups={},
                        overrides=ov,
                    )
                    for c in (g.home, g.away)
                )
                rates = game_rates(
                    h.rates,
                    a.rates,
                    league,
                    home_goalie_gsax60=h.starter.gsax60,
                    away_goalie_gsax60=a.starter.gsax60,
                    params=cfg.sim,
                )
                sim = simulate(rates, cfg.sim, seed=int(day.strftime("%Y%m%d")) * 100 + i)
                p = markets.price(
                    sim,
                    market="game_total",
                    side="over",
                    entity="",
                    line=lr["line"],
                    ot_rule=ot,
                    home=g.home,
                    away=g.away,
                )
                out[tag] = round(cal.apply("game_total", p)[0].win, 4)
                out[tag + "_exp"] = round(markets.summary(sim)["exp_total"], 3)
                out[tag + "_gsax"] = round(h.starter.gsax60 + a.starter.gsax60, 3)
            rows.append(out)
        print(day, len(rows), flush=True)
    if not rows:
        print("no ledger total rows in range")
        return 1
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def gap(col: str) -> str:
        d = [float(r[col]) - float(r["mkt"]) for r in rows if r[col] != ""]
        return f"{100 * sum(d) / len(d):+.2f} pts (n={len(d)})"

    print(f"{len(rows)} games -> {args.out}")
    for col in ("live", "reb", "reb_livegoalie"):
        print(f"{col:15} Over prob - market {gap(col)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
