"""Ad-hoc: pull per-starter regression signals for a slate (no pricing/MC)."""
from __future__ import annotations

import json
import sys
from datetime import date as Date

from mlb_engine.config import load_config
from mlb_engine.data.fangraphs import FanGraphsClient
from mlb_engine.data.mlb_statsapi import MLBStatsClient
from mlb_engine.data.oddsapi import OddsAPIClient
from mlb_engine.data.rotowire import RotowireClient
from mlb_engine.data.statcast import StatcastRepository
from mlb_engine.data.vsin import VSINClient
from mlb_engine.features.regression import (
    BL_BABIP,
    BL_BARREL_ALLOWED,
    BL_BB_PCT,
    BL_CSW,
    BL_K_PCT,
    BL_SWSTR,
    build_pitcher_regression,
)
from mlb_engine.filters.weather import WeatherProvider
from mlb_engine.pipeline import Pipeline, PipelineDeps

day = Date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else Date.today()
cfg = load_config()
deps = PipelineDeps(
    stats=MLBStatsClient(),
    statcast=StatcastRepository(cfg.cache_dir),
    weather=WeatherProvider(
        cache_dir=cfg.weather_cache_dir, cache_ttl=cfg.weather_cache_ttl
    ),
    vsin=VSINClient(cfg.creds),
    oddsapi=OddsAPIClient(cfg.creds.odds_api_key),
    rotowire=RotowireClient(cfg.creds),
    fangraphs=FanGraphsClient(cfg.creds),
)
pipe = Pipeline(cfg, deps)
slate = deps.stats.get_slate(day)
if deps.rotowire is not None:
    pipe._enrich_expected_lineups(slate, day)

w = cfg.windows
statcast = deps.statcast.max_window(
    day,
    [w.pitcher_form_days, w.batter_home_away_days, w.batter_vs_rhp_days, w.batter_vs_lhp_days],
)

out = []
for g in slate.games:
    for team in (g.away, g.home):
        p = team.probable_pitcher
        if not p or not p.mlbam_id:
            continue
        rows = statcast[statcast["pitcher"] == p.mlbam_id]
        reg = build_pitcher_regression(rows)
        out.append({
            "matchup": g.matchup(),
            "team": team.abbrev,
            "pitcher": p.name,
            "throws": p.throws.value if p.throws else None,
            "pitches": reg.pitches,
            "bbe": reg.bbe,
            "k_pct": round(reg.k_pct, 3),
            "xk_pct": round(reg.expected_k_pct(), 3),
            "bb_pct": round(reg.bb_pct, 3),
            "xbb_pct": round(reg.expected_bb_pct(), 3),
            "csw": round(reg.csw, 3),
            "swstr": round(reg.swstr, 3),
            "two_strike_whiff": round(reg.two_strike_whiff, 3),
            "babip_allowed": round(reg.babip_allowed, 3),
            "woba_allowed": round(reg.woba_allowed, 3),
            "xwoba_allowed": round(reg.xwoba_allowed, 3),
            "dxwoba_allowed": round(reg.dxwoba, 3),
            "hard_hit_allowed": round(reg.hard_hit_allowed, 3),
            "barrel_allowed": round(reg.barrel_allowed, 3),
            "k_mult": round(reg.k_multiplier(), 3),
            "allowed_mult": {k: round(v, 3) for k, v in reg.allowed_multipliers().items()},
        })

print(json.dumps({
    "date": day.isoformat(),
    "baselines": {
        "k_pct": BL_K_PCT, "bb_pct": BL_BB_PCT, "csw": BL_CSW,
        "swstr": BL_SWSTR, "babip": BL_BABIP, "barrel_allowed": BL_BARREL_ALLOWED,
    },
    "starters": out,
}, indent=2))
