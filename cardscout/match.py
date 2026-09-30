"""Match a shop listing title to a TCGplayer catalog product.

Cards: the title almost always carries the collector number ("Charmander
038/131", "(048/086)", "#38"), so the number narrows the field to a few dozen
cards across sets and the name plus any set words in the title settle it.

Sealed: token overlap between the listing title and the product name, after
expanding shop shorthand (ETB, BB) and dropping filler ("Pokemon TCG:",
"Scarlet & Violet -", "Factory Sealed"). A match needs most of the product's
tokens present and no contradicting product-type word (a "Booster Bundle" is
not a "Booster Box").
"""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from cardscout.store import Listing

# "038/132", "038/MEP" (promo set code as the denominator) or "#38"
NUMBER_RE = re.compile(
    r"(?<![\d/])(\d{1,3})\s*/\s*(\d{1,3}|[A-Za-z]{2,5})(?![\d\w])|#\s*(\d{1,3})\b"
)
BRACKET_RE = re.compile(r"[\[\(].*?[\]\)]")
FILLER = {
    "pokemon",
    "pokémon",
    "tcg",
    "the",
    "and",
    "of",
    "factory",
    "sealed",
    "new",
    "english",
    "en",
    "card",
    "cards",
    "trading",
    "game",
    "scarlet",
    "violet",
    "sv",
    "sword",
    "shield",
    "swsh",
    "me",
    "series",
    "presale",
    "preorder",
    "pre-order",
    "in",
    "stock",
    "ships",
}
SYNONYMS = {
    "etb": ["elite", "trainer", "box"],
    "bb": ["booster", "box"],
    "upc": ["ultra", "premium", "collection"],
    "spc": ["super", "premium", "collection"],
    "pc": ["pokemon", "center"],
    "&": ["and"],
}
# Product-type words that must agree between title and product for a sealed match.
TYPE_WORDS = {
    "box",
    "bundle",
    "pack",
    "blister",
    "tin",
    "collection",
    "binder",
    "deck",
    "case",
    "display",
    "sleeved",
    "checklane",
    "poster",
    "sticker",
    "mini",
    "premium",
    "ultra",
    "super",
    "elite",
    "build",
    "battle",
    "stadium",
    "surprise",
    "tech",
    "japanese",
    "korean",
    "chinese",
    "french",
    "german",
    "italian",
    "spanish",
    "portuguese",
}

# Words on a single's title that describe the printing or grade, not the card.
DESCRIPTORS = {
    "holo",
    "holofoil",
    "reverse",
    "foil",
    "promo",
    "rare",
    "common",
    "uncommon",
    "single",
    "nm",
    "lp",
    "mint",
    "near",
    "played",
    "ultra",
    "secret",
    "illustration",
    "special",
    "full",
    "art",
    "cosmos",
    "1st",
    "edition",
    "unlimited",
    "shadowless",
    "psa",
    "cgc",
    "graded",
}

_WORD = re.compile(r"[a-z0-9é&']+")
# Graded slabs trade on the grade, not the raw-card market.
GRADED_RE = re.compile(
    r"\b(psa|cgc|bgs|beckett|sgc|ace)\s*-?\s*(10|[1-9](?:\.5)?)\b|\bgraded\b", re.I
)
# Series a title or a set name belongs to; two products from different
# series never match ("Scarlet & Violet Booster Pack" is not a SWSH one).
SERIES_RE = {
    "sv": re.compile(r"\bsv\d*\b|scarlet", re.I),
    "swsh": re.compile(r"\bswsh\d*\b|sword", re.I),
    "me": re.compile(r"\bme\d*\b|mega evolution", re.I),
    "sm": re.compile(r"\bsm\d*\b|sun\s*(?:&|and)\s*moon", re.I),
    "xy": re.compile(r"\bxy\d*\b", re.I),
}


def series_of(text: str) -> str | None:
    hits = [k for k, rx in SERIES_RE.items() if rx.search(text)]
    return hits[0] if len(hits) == 1 else None


def tokens(text: str) -> list[str]:
    out: list[str] = []
    for w in _WORD.findall(text.lower().replace("—", " ").replace("-", " ")):
        w = w.replace("é", "e").strip("'")
        if w in SYNONYMS:
            out.extend(SYNONYMS[w])
        elif w and w not in FILLER:
            out.append(w)
    return out


def printing_of(text: str) -> str | None:
    t = text.lower()
    if "reverse" in t:
        return "Reverse Holofoil"
    if "1st edition" in t or "first edition" in t:
        return "1st Edition Holofoil" if "holo" in t else "1st Edition"
    if "holo" in t or "foil" in t:
        return "Holofoil"
    return None


def pick_printing(available: tuple[str, ...], text: str) -> str:
    """Choose the market printing a listing refers to.

    A listing never means "1st Edition" unless it says so, so when the wanted
    printing is not offered the fallback is the cheapest-tier equivalent
    (Unlimited before 1st Edition, non-holo before holo).
    """
    wanted = printing_of(text) or "Normal"
    if not available or wanted in available:
        return wanted
    t = text.lower()
    first = "1st edition" in t or "first edition" in t
    holo = "holo" in t or "foil" in t
    order = [
        p for p in available if first == p.startswith("1st") and holo == ("holo" in p.lower())
    ] + [p for p in available if first == p.startswith("1st")]
    order += [p for p in available if not p.startswith("1st")] + list(available)
    return order[0]


@dataclass
class Candidate:
    product_id: int
    name: str
    group_id: int
    set_name: str
    number: str | None
    is_sealed: bool
    printings: tuple[str, ...]


class Matcher:
    def __init__(self, products: Iterable[sqlite3.Row], group_names: dict[int, str]):
        self.by_number: dict[str, list[Candidate]] = defaultdict(list)
        self.sealed: list[tuple[Candidate, frozenset[str], frozenset[str]]] = []
        self.set_tokens: dict[int, frozenset[str]] = {}
        self.set_series: dict[int, str | None] = {
            gid: series_of(name.split(":", 1)[0]) for gid, name in group_names.items()
        }
        # Every word that names a set, so a title carrying another set's name
        # ("Pitch Black ETB") cannot match a product from this one.
        self.set_vocab: set[str] = set()
        for gid, name in group_names.items():
            toks = frozenset(tokens(name.split(":", 1)[-1]))
            self.set_tokens[gid] = toks
            self.set_vocab |= toks
        for p in products:
            raw = p["printings"] if "printings" in p.keys() else None
            printings = tuple(raw.split(",")) if raw else ()
            c = Candidate(
                product_id=p["product_id"],
                name=p["name"],
                group_id=p["group_id"],
                set_name=group_names.get(p["group_id"], ""),
                number=p["number"],
                is_sealed=bool(p["is_sealed"]),
                printings=printings,
            )
            if c.is_sealed:
                own = frozenset(tokens(c.name)) | self.set_tokens.get(c.group_id, frozenset())
                self.sealed.append((c, frozenset(tokens(c.name)), own))
            elif c.number:
                short = c.number.split("/")[0].lstrip("0") or "0"
                self.by_number[short].append(c)

    def match(self, title: str, variant: str = "") -> tuple[Candidate | None, str | None, float]:
        """Return (candidate, printing, confidence in [0, 1])."""
        if GRADED_RE.search(f"{title} {variant}"):
            return None, None, 0.0
        m = NUMBER_RE.search(title)
        if m:
            cand, conf = self._match_card(title, m)
            if cand is not None:
                printing = pick_printing(cand.printings, f"{title} {variant}")
                return cand, printing, conf
        cand, conf = self._match_sealed(title)
        return cand, None, conf

    def _match_card(self, title: str, m: re.Match) -> tuple[Candidate | None, float]:
        number = (m.group(1) or m.group(3)).lstrip("0") or "0"
        total = m.group(2).lstrip("0") if m.group(2) and m.group(2).isdigit() else None
        name_part = set(tokens(BRACKET_RE.sub(" ", title[: m.start()]))) - DESCRIPTORS
        rest = set(tokens(title))
        best, best_score = None, 0.0
        for c in self.by_number.get(number, []):
            name_tokens = set(tokens(BRACKET_RE.sub(" ", c.name.split(" - ")[0]))) - DESCRIPTORS
            if not name_tokens or not name_tokens & name_part:
                continue
            score = len(name_tokens & name_part) / len(name_tokens | name_part)
            if total and c.number and "/" in c.number:
                score += 0.5 if c.number.split("/")[-1].lstrip("0") == total else -0.5
            set_toks = self.set_tokens.get(c.group_id, frozenset())
            hits = set_toks & rest
            foreign = (rest & self.set_vocab) - set_toks - name_tokens
            if foreign and not hits:
                continue  # the title names a different set
            score += 0.25 * len(hits) - 0.3 * len(foreign)
            if score > best_score:
                best, best_score = c, score
        if best is None:
            return None, 0.0
        return best, min(1.0, best_score / 1.5)

    def _match_sealed(self, title: str) -> tuple[Candidate | None, float]:
        tt = set(tokens(title))
        if not tt:
            return None, 0.0
        best, best_score = None, 0.0
        # Sets the title names outright (every token of the set name present).
        # A bare "Mega Evolution ETB" cannot be a Pitch Black one, and
        # "Mega Evolution: Pitch Black ETB" cannot be the ME01 one.
        named = {gid for gid, toks in self.set_tokens.items() if toks and toks <= tt}
        if len(named) > 1:
            # series first, set last
            order = tokens(title)
            last = {gid: max(order.index(t) for t in self.set_tokens[gid]) for gid in named}
            named = {gid for gid, pos in last.items() if pos == max(last.values())}
        series = series_of(title)
        for c, pt, own in self.sealed:
            if not pt:
                continue
            if (pt & TYPE_WORDS) != (tt & TYPE_WORDS):
                continue
            if named and c.group_id not in named:
                continue
            if not named and (tt & self.set_vocab) - own:
                continue
            if series and self.set_series.get(c.group_id) not in (None, series):
                continue
            # "Mini Tin [Vaporeon]": the bracket names the variant; a title
            # that does not say which one cannot be priced as one.
            variant = set(tokens(" ".join(BRACKET_RE.findall(c.name)))) - TYPE_WORDS
            if variant and not variant & tt:
                continue
            covered = len(pt & tt) / len(pt)
            if covered < 0.75 or (len(pt & tt) < 2 and tt != pt):
                continue
            extra = len(tt - pt) / max(len(tt), 1)
            score = covered - 0.5 * extra
            if score > best_score:
                best, best_score = c, score
        if best is None or best_score < 0.5:
            return None, 0.0
        return best, min(1.0, best_score)


def attach(matcher: Matcher, listings: Iterable[Listing]) -> int:
    """Fill product_id/printing/confidence in place; return how many matched."""
    n = 0
    for li in listings:
        cand, printing, conf = matcher.match(li.title, li.variant)
        if cand is not None:
            li.product_id, li.printing, li.confidence = cand.product_id, printing, conf
            n += 1
    return n
