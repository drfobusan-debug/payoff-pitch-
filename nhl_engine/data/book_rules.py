"""Per-book settlement rules for markets where books disagree.

A 3-3 game won in a shootout is officially 4-3. Some books settle the winner's
team total at 3 (regulation), others at 4. The archive records the quote either
way; a *bet* on such a market needs the book's rule verified, and a rule older
than ``EXPIRY_DAYS`` counts as unknown again because books change data
providers and house rules between seasons.

Nothing is pre-populated. A rule enters this table only after someone has read
the book's house rules on a given date, so an empty table means every
book-rule market is ``settlement_unverified`` -- the honest default.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

EXPIRY_DAYS = 60
UNVERIFIED = "settlement_unverified"

# Markets whose ot_rule in capture.MARKET_MAP is "book_rule".
RULE_MARKETS: frozenset[str] = frozenset({"team_total", "team_total_alt"})
OT_RULES: frozenset[str] = frozenset({"reg_only", "incl_ot_so", "incl_ot"})


@dataclass(frozen=True)
class BookRule:
    book: str
    market: str
    ot_rule: str
    verified_on: str  # ISO date
    source: str = ""  # URL or note of where the house rule was read

    def expired(self, today: Date) -> bool:
        return Date.fromisoformat(self.verified_on) + timedelta(days=EXPIRY_DAYS) < today


class BookRules:
    def __init__(self, rules: list[BookRule] | None = None) -> None:
        self._rules: dict[tuple[str, str], BookRule] = {}
        for r in rules or []:
            self.add(r)

    def add(self, rule: BookRule) -> None:
        if rule.ot_rule not in OT_RULES:
            raise ValueError(f"unknown ot_rule {rule.ot_rule!r}")
        Date.fromisoformat(rule.verified_on)
        self._rules[(rule.book, rule.market)] = rule

    def resolve(self, book: str, market: str, today: Date) -> str:
        """The settlement rule to grade under, or ``settlement_unverified``."""
        rule = self._rules.get((book, market))
        if rule is None or rule.expired(today):
            return UNVERIFIED
        return rule.ot_rule

    def gate(self, book: str, market: str, today: Date) -> str | None:
        """Reason a bet must be refused, or ``None`` if the rule is known and fresh."""
        if market not in RULE_MARKETS:
            return None
        return UNVERIFIED if self.resolve(book, market, today) == UNVERIFIED else None

    def stale(self, today: Date) -> list[BookRule]:
        return [r for r in self._rules.values() if r.expired(today)]

    def all(self) -> list[BookRule]:
        return sorted(self._rules.values(), key=lambda r: (r.book, r.market))

    def __len__(self) -> int:
        return len(self._rules)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            asdict(r) for r in sorted(self._rules.values(), key=lambda r: (r.book, r.market))
        ]
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BookRules:
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        rules: list[BookRule] = []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                rules.append(
                    BookRule(
                        book=str(item["book"]),
                        market=str(item["market"]),
                        ot_rule=str(item["ot_rule"]),
                        verified_on=str(item["verified_on"]),
                        source=str(item.get("source", "")),
                    )
                )
            except (KeyError, ValueError):
                continue
        out = cls()
        for r in rules:
            try:
                out.add(r)
            except ValueError:
                continue
        return out


def rules_path(data_dir: Path) -> Path:
    return data_dir / "book_rules.json"


__all__ = [
    "EXPIRY_DAYS",
    "OT_RULES",
    "RULE_MARKETS",
    "UNVERIFIED",
    "BookRule",
    "BookRules",
    "rules_path",
]
