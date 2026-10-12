"""Walk-forward sweep of ``market_blend`` (and rest/bye/travel on/off) on the stored card.

Read-only analysis; nothing here feeds the live price. For every game the engine
priced (``predictions_<day>.json`` on engine-state) it recovers the engine's own
pre-game ratings margin from the stored mean:

    exp_margin = (1 - w) * rating + w * market + rest/bye/travel
    rating     = (exp_margin - adj - w * market) / (1 - w),  w = the live blend

so the ratings are the ones the card actually used, fixed before kickoff. Each
weight is then re-priced and scored on the final:

* spread Brier at the bet-time home line (normal, ``margin_sd``), pushes dropped;
* moneyline Brier with the engine's market-SD win probability;
* margin squared error.

The market is weight 1 with the situational terms off. A weight ships only if the
walk-forward pick (chosen on the slates before each slate, scored on that slate)
beats the market; the 95% range resamples whole slates.

    python -m scripts.cfb.blend_sweep --state-dir <engine-state>/cfb --live-blend 0.35
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

from cfb_engine.audit.grade import build_result_index, result_for
from cfb_engine.config import Config
from cfb_engine.data.cfbd import CFBDClient
from cfb_engine.data.teamnames import label_matches
from cfb_engine.models.montecarlo import market_win_prob
from cfb_engine.recommendations import Recommendation, load_json

WEIGHTS = tuple(round(0.05 * i, 2) for i in range(21))
MIN_TRAIN = 3
_REST = re.compile(r"^Rest edge (.+) \((\d+)d vs (\d+)d\)$")
_BYE = re.compile(r"^(.+) off a bye$")
_TRAVEL = re.compile(r"^(.+) travels (\d+) mi$")


@dataclass(frozen=True)
class Game:
    day: str
    matchup: str
    rating: float
    market: float  # bet-time home margin (minus the home spread)
    line: float  # bet-time home spread
    adj: float  # rest + bye + travel, home points
    ml_market: float | None  # bet-time no-vig home win probability
    ats_market: float | None  # bet-time no-vig home cover probability
    margin: int


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def situational(reasons: list[str], home: str, cfg: Config) -> float:
    """The rest/bye/travel points the card added, rebuilt from its reasons."""
    f = cfg.features
    pts = 0.0
    for r in reasons:
        if m := _REST.match(r):
            net = int(m.group(2)) - int(m.group(3))
            edge = max(-f.rest_max_pts, min(f.rest_max_pts, net * f.rest_pts_per_day))
            pts += edge if m.group(1) == home else -edge
        elif m := _BYE.match(r):
            pts += f.bye_bonus_pts if m.group(1) == home else -f.bye_bonus_pts
        elif m := _TRAVEL.match(r):
            miles = float(m.group(2))
            if miles >= f.travel_min_miles:
                pts += min(miles / 1000.0 * f.travel_pts_per_1000mi, f.travel_max_pts)
    return pts


def games_for_day(
    recs: list[Recommendation], raw: list[dict[str, object]], cfg: Config, w: float, index: object
) -> list[Game]:
    by_game: dict[str, list[tuple[Recommendation, dict[str, object]]]] = {}
    for rec, row in zip(recs, raw, strict=True):
        by_game.setdefault(str(row.get("game_id")), []).append((rec, row))
    out: list[Game] = []
    for rows in by_game.values():
        ats = next(
            (
                (rec, row)
                for rec, row in rows
                if rec.market == "game_ats" and row.get("team_side") == "home"
            ),
            None,
        )
        if ats is None:
            continue
        rec, row = ats
        exp_margin = row.get("exp_margin")
        if rec.line is None or not isinstance(exp_margin, (int, float)):
            continue
        final = result_for(rec, index)  # type: ignore[arg-type]
        if final is None:
            continue
        home = str(row.get("home_abbrev"))
        margin = final.home_points - final.away_points
        if not label_matches(home, final.home):
            margin = -margin
        line = rec.line
        market = -line
        reasons = row.get("reasons")
        adj = situational([str(r) for r in reasons] if isinstance(reasons, list) else [], home, cfg)
        rating = (float(exp_margin) - adj - w * market) / (1.0 - w)
        ml = next(
            (
                r.fair_prob
                for r, x in rows
                if r.market == "game_ml" and x.get("team_side") == "home"
            ),
            None,
        )
        out.append(
            Game(
                rec.game_date.isoformat(),
                rec.matchup,
                rating,
                market,
                line,
                adj,
                ml,
                rec.fair_prob,
                margin,
            )
        )
    return out


@dataclass
class Score:
    n_ats: int = 0
    ats: float = 0.0
    n_ml: int = 0
    ml: float = 0.0
    n: int = 0
    se: float = 0.0

    def add(self, o: Score) -> None:
        self.n_ats += o.n_ats
        self.ats += o.ats
        self.n_ml += o.n_ml
        self.ml += o.ml
        self.n += o.n
        self.se += o.se


def score(games: list[Game], w: float | None, adj: bool, cfg: Config) -> Score:
    """``w=None`` scores the bet-time market's own no-vig probabilities."""
    s = Score()
    for g in games:
        mean = g.market if w is None else (1 - w) * g.rating + w * g.market
        mean += g.adj if adj and w is not None else 0.0
        cover = g.margin + g.line
        if cover != 0:
            p = (
                g.ats_market
                if w is None and g.ats_market is not None
                else _phi((mean + g.line) / cfg.model.margin_sd)
            )
            s.n_ats += 1
            s.ats += (p - (1.0 if cover > 0 else 0.0)) ** 2
        if g.margin != 0:
            p = (
                g.ml_market
                if w is None and g.ml_market is not None
                else market_win_prob(mean, cfg.model)
            )
            s.n_ml += 1
            s.ml += (p - (1.0 if g.margin > 0 else 0.0)) ** 2
        s.n += 1
        s.se += (mean - g.margin) ** 2
    return s


def _fmt(s: Score) -> str:
    return (
        f"ATS Brier {s.ats / max(s.n_ats, 1):.4f} ({s.n_ats})  "
        f"ML Brier {s.ml / max(s.n_ml, 1):.4f} ({s.n_ml})  "
        f"margin RMSE {math.sqrt(s.se / max(s.n, 1)):.2f} ({s.n})"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state-dir", type=Path, required=True)
    ap.add_argument("--live-blend", type=float, default=0.35)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--through", type=str, default=None)
    ap.add_argument("--boot", type=int, default=4000)
    args = ap.parse_args(argv)
    cfg = Config()
    cfbd = CFBDClient(cfg.creds.cfbd_api_key)
    slates: dict[str, list[Game]] = {}
    for path in sorted((args.state_dir / "predictions").glob("predictions_*.json.gz")):
        day = path.name[len("predictions_") : -len(".json.gz")]
        if args.through and day > args.through:
            continue
        raw = json.loads(gzip.decompress(path.read_bytes()))
        tmp = path.with_suffix("")
        tmp = Path("/tmp") / tmp.name
        tmp.write_text(json.dumps(raw))
        recs = load_json(tmp)
        index = build_result_index(cfbd.fetch_results(args.season, Date.fromisoformat(day)))
        games = games_for_day(recs, raw, cfg, args.live_blend, index)
        if games:
            slates[day] = games
    days = sorted(slates)
    every = [g for d in days for g in slates[d]]
    print(f"{len(every)} games on {len(days)} slates ({days[0]} to {days[-1]})")
    print(f"market (bet-time no-vig)  {_fmt(score(every, None, False, cfg))}")
    for adj in (False, True):
        for w in (0.0, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0):
            print(f"w={w:.2f} adj={'on ' if adj else 'off'}  {_fmt(score(every, w, adj, cfg))}")

    # Walk-forward: pick (w, adj) on the slates before, score on the slate.
    grid = [(w, a) for w in WEIGHTS for a in (False, True)]
    picks: list[tuple[str, float, bool]] = []
    wf_per: dict[str, tuple[Score, Score, Score]] = {}
    for k, day in enumerate(days):
        if k < MIN_TRAIN:
            continue
        train = [g for d in days[:k] for g in slates[d]]

        def loss(c: tuple[float, bool], train: list[Game] = train) -> float:
            s = score(train, c[0], c[1], cfg)
            return s.ats / max(s.n_ats, 1)

        best = min(grid, key=loss)
        picks.append((day, best[0], best[1]))
        wf_per[day] = (
            score(slates[day], best[0], best[1], cfg),
            score(slates[day], None, False, cfg),
            score(slates[day], args.live_blend, True, cfg),
        )
    tot = [Score(), Score(), Score()]
    for trio in wf_per.values():
        for t, s in zip(tot, trio, strict=True):
            t.add(s)
    print("\nWalk-forward (choose on earlier slates by ATS Brier, score the next):")
    print("  picks: " + ", ".join(f"{d[5:]} w={w:.2f}{'+adj' if a else ''}" for d, w, a in picks))
    for label, s in zip(
        ("walk-forward pick", "market", f"live w={args.live_blend}+adj"), tot, strict=True
    ):
        print(f"  {label:<22}{_fmt(s)}")

    rng = random.Random(7)
    keys = list(wf_per)
    for mi, name in ((0, "ATS"), (1, "ML")):
        diffs = []
        for _ in range(args.boot):
            a = b = c = na = 0.0
            for d in (rng.choice(keys) for _ in keys):
                pick, mkt, live = wf_per[d]
                if mi == 0:
                    a += pick.ats
                    b += mkt.ats
                    c += live.ats
                    na += pick.n_ats
                else:
                    a += pick.ml
                    b += mkt.ml
                    c += live.ml
                    na += pick.n_ml
            diffs.append(((a - b) / na, (c - b) / na))
        for j, label in ((0, "walk-forward pick"), (1, "live")):
            v = sorted(x[j] for x in diffs)
            lo, hi = v[int(0.025 * len(v))], v[int(0.975 * len(v))]
            print(
                f"  {name} Brier, {label} minus market: 95% range {lo:+.4f} to {hi:+.4f} (below 0 = beats market)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
