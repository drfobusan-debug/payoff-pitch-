"""The shows read for CFB picks: the shared registry, narrowed to college football."""

from __future__ import annotations

from engine_common.podcasts.shows import CFB, LEAGUE_WORDS, Show, for_league

CFB_WORDS = LEAGUE_WORDS[CFB]
SHOWS: tuple[Show, ...] = for_league(CFB)
BY_KEY = {s.key: s for s in SHOWS}

__all__ = ["BY_KEY", "CFB_WORDS", "SHOWS", "Show"]
