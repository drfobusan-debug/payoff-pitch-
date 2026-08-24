"""Ad-hoc: per-batter regression signals for a slate (no pricing/MC)."""
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
from mlb_engine.features.regression import build_batter_regression
from mlb_engine.features.swing import build_swing_profile, stage_two
from mlb_engine.filters.weather import WeatherProvider
from mlb_engine.pipeline import Pipeline, PipelineDeps, load_sprint_speeds

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
try:
    sprint = load_sprint_speeds(day.year)
except Exception:
    sprint = {}

out = []
seen = set()
for g in slate.games:
    for team in (g.away, g.home):
        for slot in team.lineup:
            p = slot.player
            if not p.mlbam_id or p.mlbam_id in seen:
                continue
            seen.add(p.mlbam_id)
            rows = statcast[statcast["batter"] == p.mlbam_id]
            reg = build_batter_regression(rows, sprint.get(p.mlbam_id, 27.0))
            if reg.bbe < 20:
                continue
            prof = build_swing_profile(rows)
            out.append({
                "matchup": g.matchup(),
                "team": team.abbrev,
                "batter": p.name,
                "order": slot.order,
                "bats": p.bats.value if p.bats else None,
                "bbe": reg.bbe,
                "woba": round(reg.woba, 3),
                "xwoba": round(reg.xwoba, 3),
                "dxwoba": round(reg.dxwoba, 3),
                "babip": round(reg.babip, 3),
                "barrel": round(reg.barrel_rate, 3),
                "hard_hit": round(reg.hard_hit, 3),
                "xba": round(reg.xba, 3),
                "xslg": round(reg.xslg, 3),
                "sweet_spot": round(reg.sweet_spot, 3),
                "whiff": round(reg.whiff, 3),
                "mult": {k: round(v, 3) for k, v in reg.multipliers().items()},
                "swings": prof.swings,
                "bat_speed": round(prof.bat_speed, 1),
                "fast": round(prof.fast, 3),
                "squared_up": round(prof.squared_up, 3),
                "blast": round(prof.blast, 3),
                "swing_length": round(prof.swing_length, 2),
                "power_z": round(prof.power_z, 2),
                "contact_z": round(prof.contact_z, 2),
                "stage2": stage_two(-reg.dxwoba, prof),
            })

print(json.dumps({"date": day.isoformat(), "batters": out}, indent=2))
