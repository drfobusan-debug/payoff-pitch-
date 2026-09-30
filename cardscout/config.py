"""Paths and environment for cardscout."""

from __future__ import annotations

import os
from pathlib import Path

USER_AGENT = "cardscout/0.1 (+https://github.com/drfobusan-debug/payoff-pitch-)"
BROWSER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)

# tcgcsv.com category id for Pokemon (TCGplayer's categoryId).
POKEMON_CATEGORY = 3
TCGCSV_BASE = "https://tcgcsv.com/tcgplayer"

HOME_ZIP = "23229"
HOME_AREA_CODE = "804"
HOME_TZ = "America/New_York"


def data_dir() -> Path:
    root = Path(os.environ.get("CARDSCOUT_DATA_DIR", Path.home() / ".cardscout"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def db_path() -> Path:
    return data_dir() / "cardscout.sqlite"


def cache_dir() -> Path:
    d = data_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d
