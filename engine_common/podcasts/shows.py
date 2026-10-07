"""Every podcast read for picks, and which leagues each is read for.

One registry for all engines. A show lists the leagues its picks are kept for;
an episode is read once, and each pick it yields is filed under its own league,
so a multi-sport show feeds every slate it talks about. A dedicated show is read
in full; a multi-sport one only when the episode's title or description names a
league it is read for, unless ``read_all`` (titles that say nothing, like "Hour 2").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

CFB, NFL, MLB, NBA, CBB, NHL = "cfb", "nfl", "mlb", "nba", "cbb", "nhl"
LEAGUES: tuple[str, ...] = (CFB, NFL, MLB, NBA, CBB, NHL)
LEAGUE_NAME = {
    CFB: "college football",
    NFL: "NFL",
    MLB: "MLB",
    NBA: "NBA",
    CBB: "college basketball",
    NHL: "NHL",
}

LEAGUE_WORDS: dict[str, re.Pattern[str]] = {
    CFB: re.compile(
        r"college football|\bcfb\b|ncaaf|\bfcs\b|heisman|college gameday|maction|"
        r"\bsec\b|big ten|big 12|\bacc\b|pac[- ]12|mountain west|sun belt|group of five",
        re.I,
    ),
    NFL: re.compile(
        r"\bnfl\b|super bowl|thursday night football|sunday night football|monday night football|"
        r"\b(tnf|snf|mnf)\b|\bafc\b|\bnfc\b",
        re.I,
    ),
    MLB: re.compile(r"\bmlb\b|baseball|world series|\balcs\b|\bnlcs\b|\balds\b|\bnlds\b", re.I),
    NBA: re.compile(r"\bnba\b|nba finals|in-season tournament|nba cup", re.I),
    CBB: re.compile(
        r"college basketball|\bcbb\b|ncaab|march madness|final four|big dance|bracketology", re.I
    ),
    NHL: re.compile(r"\bnhl\b|hockey|stanley cup", re.I),
}


@dataclass(frozen=True)
class Show:
    key: str
    name: str
    feed_url: str
    hosts: tuple[str, ...]
    leagues: tuple[str, ...] = LEAGUES
    dedicated: bool = False  # every episode is about its league(s): read in full
    read_all: bool = False  # multi-sport, but titles don't say which: read in full
    skip: str = ""  # title regex for episodes with no picks in them

    def wants(self, title: str, description: str) -> bool:
        if self.skip and re.search(self.skip, title, re.I):
            return False
        if self.dedicated or self.read_all:
            return True
        return any(
            LEAGUE_WORDS[lg].search(title) or LEAGUE_WORDS[lg].search(description)
            for lg in self.leagues
        )

    def for_league(self, league: str) -> Show:
        """The show as one league's engine sees it: read only for that league."""
        return replace(self, leagues=(league,))


SHOWS: tuple[Show, ...] = (
    # ---- college football
    Show(
        "cfe",
        "The College Football Experience",
        "https://feeds.simplecast.com/JJ88ZXKg",
        ("Colby Dant", "Ryan McIntyre", "Troy Chewning", "Pick Dundee"),
        leagues=(CFB,),
        dedicated=True,
        skip=r"reaction|recap",
    ),
    Show(
        "vsin_cfb",
        "VSiN College Football Betting Podcast",
        "https://feeds.simplecast.com/_iUqgztL",
        ("Tim Murray", "Wes Reynolds", "Brett Ciancia", "Parker Fleming"),
        leagues=(CFB,),
        dedicated=True,
        skip=r"recap|reaction",
    ),
    Show(
        "htb",
        "Hit The Books",
        "https://feeds.megaphone.fm/HAMMR1605208379",
        ("Brad Powers", "Joey Knish"),
        leagues=(CFB,),
        dedicated=True,
    ),
    Show(
        "bboc",
        "Big Bets On Campus (Action Network)",
        "https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/"
        "61403825-97cd-4547-b4f2-b3ec011d3f83/b96a4573-5dc1-4181-add2-b3ec011d3f8a/podcast.rss",
        ("Stuckey", "Collin Wilson", "Michael Calabrese", "Josh Nunn"),
        leagues=(CFB,),
        dedicated=True,
        skip=r"betting recap",
    ),
    Show(
        "bear_bets",
        "Bear Bets (FOX Sports)",
        "https://feeds.megaphone.fm/bearbets",
        ("Chris Fallica", "Geoff Schwartz", "Sammy Panayotovich", "Bruce Feldman"),
        leagues=(CFB,),
        skip=r"\bNFL\b",
    ),
    # ---- NFL
    Show(
        "schwartz",
        "Schwartz Football Show",
        "https://rss.amperwave.net/v2/feed/audacynetwork/3d8c42b0b99a7f80222583735fcbbbe8",
        ("Geoff Schwartz", "Mitchell Schwartz"),
        leagues=(NFL, CFB),
        dedicated=True,
    ),
    Show(
        "action_nfl",
        "The Action Network Sports Betting Podcast",
        "https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/"
        "390f52fb-437c-4290-bdaa-b3ec011d3fc8/2d9ae039-d4b2-4a37-8a77-b3ec011d3fce/podcast.rss",
        (
            "Stuckey",
            "Doug Kezirian",
            "Brandon Anderson",
            "Matt Moore",
            "Sean Koerner",
            "Grant Neiffer",
            "Gilles Gallant",
            "Evan Abrams",
        ),
        skip=r"futures watch|touchdown show",
    ),
    # ---- MLB
    Show(
        "action_mlb",
        "Payoff Pitch (Action Network)",
        "https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/"
        "125660fe-ff74-4a0e-8b0e-b3ec011d3ff3/a354ddb2-8b6c-4926-a060-b3ec011d3ffa/podcast.rss",
        ("Sean Zerillo", "Collin Whitchurch", "Derek Carty", "Tanner McGrath", "Anthony Dabbundo"),
        leagues=(MLB,),
        dedicated=True,
    ),
    Show(
        "daily_diamond",
        "The VSiN Daily Diamond",
        "https://feeds.simplecast.com/KVyLqaaF",
        ("Jensen Lewis", "Dave Ross"),
        leagues=(MLB,),
        dedicated=True,
    ),
    # ---- NBA
    Show(
        "hardwood",
        "Hardwood Handicappers (VSiN)",
        "https://feeds.simplecast.com/2GlVqHyg",
        ("Zachary Cohen", "Jonathan Von Tobel", "Kelley Bydlon"),
        leagues=(NBA, CBB),
        dedicated=True,
    ),
    Show(
        "buckets",
        "BUCKETS (Action Network)",
        "https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/"
        "8abd9bed-7526-4d83-8dbf-b3ec011d3fb0/083413d6-9987-45fa-9a36-b3ec011d3fb9/podcast.rss",
        ("Matt Moore", "Brandon Anderson", "Joe Dellera", "Brandon Kravitz", "Kyle Murray"),
        leagues=(NBA,),
        dedicated=True,
    ),
    Show(
        "nba_gambling",
        "NBA Gambling Podcast (SGPN)",
        "https://feeds.simplecast.com/uxdOBkrq",
        ("Lonte Smith", "Scott Reichel", "Terrell Furman Jr"),
        leagues=(NBA,),
        dedicated=True,
    ),
    Show(
        "duncd_on",
        "Dunc'd On Basketball",
        "https://feeds.simplecast.com/JGGpY1xu",
        ("Nate Duncan", "Danny Leroux"),
        leagues=(NBA,),
        dedicated=True,
    ),
    # ---- multi-sport
    Show(
        "wisekracks",
        "WiseKracks (WagerTalk)",
        "https://www.spreaker.com/show/6668157/episodes/feed",
        ("Bill Krackomberger", "Jon Orlando"),
    ),
    Show(
        "btp",
        "Bet The Process",
        "https://feeds.soundcloud.com/users/soundcloud:users:330500902/sounds.rss",
        ("Jeff Ma", "Rufus Peabody"),
    ),
    Show(
        "numbers_game",
        "A Numbers Game (VSiN)",
        "https://feeds.simplecast.com/dQdfx569",
        ("Gill Alexander",),
        read_all=True,
        skip=r"^best of",
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
        leagues=(NFL, CFB),
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
        leagues=(NFL, CFB),
    ),
    Show(
        "bet_the_board",
        "Bet The Board (Action Network)",
        "https://www.omnycontent.com/d/playlist/e73c998e-6e60-432f-8610-ae210140c5b1/"
        "f976ef47-3c13-49dc-bb19-b4c100fc71b8/68381fd1-b072-4d63-9a06-b4c100fc75eb/podcast.rss",
        ("Todd Fuhrman", "Billy"),
        leagues=(NFL, CFB, NBA, CBB),
        dedicated=True,
    ),
)

BY_KEY = {s.key: s for s in SHOWS}


def for_league(league: str) -> tuple[Show, ...]:
    """The shows read for ``league``, each narrowed to it."""
    return tuple(s.for_league(league) for s in SHOWS if league in s.leagues)


__all__ = [
    "BY_KEY",
    "CBB",
    "CFB",
    "LEAGUES",
    "LEAGUE_NAME",
    "LEAGUE_WORDS",
    "MLB",
    "NBA",
    "NFL",
    "NHL",
    "SHOWS",
    "Show",
    "for_league",
]
