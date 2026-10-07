"""The podcast picks in the CFB slate PDF: under each game, and their records at the back.

The HTML is shared (:mod:`engine_common.podcasts.render`); this binds it to the
CFB card. Context only: nothing here reads a model probability into a decision.
"""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

from cfb_engine.podcast.picks import Rows, load_ledger, records, slate_picks
from cfb_engine.recommendations import Recommendation
from engine_common.podcasts import render
from engine_common.podcasts.extract import PICKS_DIR, load_extractions
from engine_common.podcasts.render import CSS, game_block

PodcastView = render.PodcastView[Rows]


def podcast_view(
    store: Path, ledger_file: Path, recs: list[Recommendation], day: Date
) -> PodcastView | None:
    """Everything the PDF prints about podcasts for ``day``; ``None`` if none were read."""
    extractions = load_extractions(store) if (store / PICKS_DIR).exists() else []
    if not extractions:
        return None
    return render.PodcastView(
        slate_picks(extractions, recs, day), records(load_ledger(ledger_file))
    )


def records_block(view: PodcastView | None, ordered: list[list[Recommendation]]) -> str:
    return render.records_block(view, [g[0].matchup for g in ordered if g])


__all__ = ["CSS", "PodcastView", "game_block", "podcast_view", "records_block"]
