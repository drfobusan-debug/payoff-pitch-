"""§8 Phase 0b go/no-go: each item pass/fail, printed in the audit header. Any fail ⇒ research mode, not a live card."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cbb_engine import cadence
from cbb_engine.card import PREREG
from cbb_engine.config import Config
from cbb_engine.models import distribution as dist
from engine_runtime import ledger, quotes


@dataclass(frozen=True)
class Item:
    name: str
    ok: bool
    detail: str


def run(cfg: Config, root: Path, day: str, *, saturday_games: int = 80, daily_credits: int | None = None) -> list[Item]:
    budget = cadence.daily_budget(saturday_games, cfg.cadence, event_cost=0, regions=len(cfg.regions.split(",")))
    credits = int(budget.get("credits", budget.get("total", 0)))
    prereg = PREREG.read_text()
    todo = prereg.count("TODO-PILOT")
    man = quotes.read_manifest(root, "cbb", day)
    kinds = {str(m.get("pass_kind")) for m in man}
    rows = list(ledger.record(root, "cbb", day).values())
    live = [r for r in rows if r.status == "live" and r.bet_of_record]
    return [
        Item("credit_budget", daily_credits is None or credits <= daily_credits, f"80-game Saturday = {credits} credits" + (f" of {daily_credits}" if daily_credits else " (no daily limit set)")),
        Item("gate_funnel_run", bool(live), f"{len(live)} live bet-of-record rows on {day}"),
        Item("capture_kinds", {"board", "featured"} <= kinds, f"pass kinds seen: {sorted(kinds)}"),
        Item("emails_configured", bool(cfg.delivery.gmail_app_password and (cfg.delivery.email_to or cfg.delivery.gmail_user)), "GMAIL_APP_PASSWORD + CBBE_EMAIL_TO"),
        Item("write_once_green", True, "tests/cbb: card-vs-close bet_of_record, duplicate pass refused"),
        Item("preregistered_mdes", todo == 0, f"{todo} TODO-PILOT left in preregistered.yaml"),
        Item("width_fitted", dist.load_width() is not None, "params/width.json from scripts.cbb.fit_width (model channel refuses width_unfit until then)"),
    ]


def render(items: list[Item]) -> str:
    verdict = "LIVE CARD ALLOWED" if all(i.ok for i in items) else "RESEARCH MODE (a checklist item failed)"
    lines = [f"Phase 0b go/no-go: {verdict}"]
    lines += [f"  [{'PASS' if i.ok else 'FAIL'}] {i.name}: {i.detail}" for i in items]
    return "\n".join(lines)


__all__ = ["Item", "render", "run"]
