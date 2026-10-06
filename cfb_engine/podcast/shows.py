"""The podcasts Franz listens to, as RSS feeds.

Spotify does not serve the audio, so each show is registered by the RSS feed
behind it (found through Apple's directory and checked against the Spotify
show's newest episode). ``cfb_only`` shows are read in full; the rest cover every
sport and only episodes whose title or description mention college football are
read -- except where the titles never say (``read_all``), in which case every
episode is transcribed and the extractor keeps only the college picks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CFB_WORDS = re.compile(
    r"college football|\bcfb\b|ncaaf|\bfcs\b|heisman|college gameday|maction|"
    r"\bsec\b|big ten|big 12|\bacc\b|pac[- ]12|mountain west|sun belt|group of five",
    re.I,
)


@dataclass(frozen=True)
class Show:
    key: str
    name: str
    feed_url: str
    hosts: tuple[str, ...]
    cfb_only: bool = False
    read_all: bool = False
    skip: str = ""  # title regex for episodes with no picks in them

    def wants(self, title: str, description: str) -> bool:
        if self.skip and re.search(self.skip, title, re.I):
            return False
        if self.cfb_only or self.read_all:
            return True
        return bool(CFB_WORDS.search(title) or CFB_WORDS.search(description))


SHOWS: tuple[Show, ...] = (
    Show(
        "cfe",
        "The College Football Experience",
        "https://feeds.simplecast.com/JJ88ZXKg",
        ("Colby Dant", "Ryan McIntyre", "Troy Chewning", "Pick Dundee"),
        cfb_only=True,
        skip=r"reaction|recap",
    ),
    Show(
        "vsin_cfb",
        "VSiN College Football Betting Podcast",
        "https://feeds.simplecast.com/_iUqgztL",
        ("Tim Murray", "Wes Reynolds", "Brett Ciancia", "Parker Fleming"),
        cfb_only=True,
        skip=r"recap|reaction",
    ),
    Show(
        "vsin_morning",
        "VSiN Daily Morning Bets",
        "https://feeds.simplecast.com/8yj4Zsok",
        ("Dustin Swedelson",),
        read_all=True,
    ),
    Show(
        "ftm",
        "Follow the Money",
        "https://feeds.simplecast.com/AosjnJ4d",
        ("Mitch Moss", "Pauly Howard"),
        read_all=True,
        skip=r"^best of",
    ),
    Show(
        "wagertalk",
        "WagerTalk",
        "https://www.spreaker.com/show/2834270/episodes/feed",
        ("Andy Lang", "Teddy Covers"),
    ),
    Show(
        "propcast",
        "The Propcast",
        "https://feeds.simplecast.com/mc2hMWfs",
        ("Adam", "Scott"),
    ),
    Show(
        "early_edge",
        "The Early Edge (SportsLine)",
        "https://rss.amperwave.net/v2/feed/audacynetwork/earlyedge",
        ("Sia Nejad", "Larry Hartstein", "Mike McClure", "Brady Kannon"),
    ),
    Show(
        "btb",
        "Beating The Book with Gill Alexander",
        "https://feeds.simplecast.com/yHrCf9a8",
        ("Gill Alexander", "Kelley Bydlon"),
    ),
    Show(
        "htb",
        "Hit The Books",
        "https://feeds.megaphone.fm/HAMMR1605208379",
        ("Brad Powers", "Joey Knish"),
        cfb_only=True,
    ),
    Show(
        "bboc",
        "Big Bets On Campus (Action Network)",
        "https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/"
        "61403825-97cd-4547-b4f2-b3ec011d3f83/b96a4573-5dc1-4181-add2-b3ec011d3f8a/podcast.rss",
        ("Stuckey", "Collin Wilson", "Michael Calabrese", "Josh Nunn"),
        cfb_only=True,
        skip=r"betting recap",
    ),
    Show(
        "bear_bets",
        "Bear Bets (FOX Sports)",
        "https://feeds.megaphone.fm/bearbets",
        ("Chris Fallica", "Geoff Schwartz", "Sammy Panayotovich", "Bruce Feldman"),
        skip=r"\bNFL\b",
    ),
)

BY_KEY = {s.key: s for s in SHOWS}

__all__ = ["BY_KEY", "CFB_WORDS", "SHOWS", "Show"]
