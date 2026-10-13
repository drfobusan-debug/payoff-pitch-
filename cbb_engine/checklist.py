"""§8 Phase 0b go/no-go: each item pass/fail, printed in the audit header. Any fail ⇒ research mode, not a live card."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from cbb_engine import cadence
from cbb_engine.card import PREREG
from cbb_engine.config import Config
from cbb_engine.models import distribution as dist
from engine_runtime import credits, ledger, quotes


@dataclass(frozen=True)
class Item:
    name: str
    ok: bool
    detail: str


def _daily_limit(root: Path, sport: str) -> int | None:
    """This sport's cap in ``<root>/credit_budget.yaml``; None when the file or the entry is missing."""
    try:
        return credits.load_budget(root / "credit_budget.yaml").sport_cap(sport)
    except credits.LedgerUnavailable:
        return None


def write_once_check() -> tuple[bool, str]:
    """Exercise the ledger guards in a scratch dir: a second write of a pass and a close-pass bet of record must both be refused."""
    row = ledger.Row(
        slate_date="2000-01-01", event_id="selftest", tip_utc="2000-01-01T00:00:00Z", market="game_total",
        side="over", line=145.5, book="draftkings", pass_id="card", pass_kind="featured", bet_of_record=True, status="live",
    )
    failed: list[str] = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger.write_once(root, "cbb", row.slate_date, "card", [row])
            try:
                ledger.write_once(root, "cbb", row.slate_date, "card", [replace(row)])
                failed.append("duplicate pass accepted")
            except FileExistsError:
                pass
            try:
                ledger.write_once(root, "cbb", row.slate_date, "close", [replace(row, pass_id="close", pass_kind="close")])
                failed.append("close pass accepted as bet of record")
            except ValueError:
                pass
    except Exception as exc:  # noqa: BLE001 - any crash is a failed check, not a crashed audit
        failed.append(f"self-test crashed: {exc!r}")
    if failed:
        return False, "; ".join(failed)
    return True, "ledger refused a duplicate pass and a close-pass bet of record"


def run(cfg: Config, root: Path, day: str, *, saturday_games: int = 80, daily_credits: int | None = None) -> list[Item]:
    budget = cadence.daily_budget(saturday_games, cfg.cadence, event_cost=0, regions=len(cfg.regions.split(",")))
    cost = int(budget.get("credits", budget.get("total", 0)))
    limit, source = (daily_credits, "--daily-credits") if daily_credits is not None else (_daily_limit(root, "cbb"), "credit_budget.yaml sports.cbb")
    prereg = PREREG.read_text()
    todo = sum(line.split("#", 1)[0].count("TODO-PILOT") for line in prereg.splitlines())
    man = quotes.read_manifest(root, "cbb", day)
    kinds = {str(m.get("pass_kind")) for m in man}
    rows = list(ledger.record(root, "cbb", day).values())
    live = [r for r in rows if r.status == "live" and r.bet_of_record]
    wo_ok, wo_detail = write_once_check()
    return [
        Item("credit_budget", limit is not None and cost <= limit, f"80-game Saturday = {cost} credits" + (f" of {limit} ({source})" if limit is not None else " (no daily limit set: add sports.cbb to credit_budget.yaml or pass --daily-credits)")),
        Item("gate_funnel_run", bool(live), f"{len(live)} live bet-of-record rows on {day}"),
        Item("capture_kinds", {"board", "featured"} <= kinds, f"pass kinds seen: {sorted(kinds)}"),
        Item("emails_configured", bool(cfg.delivery.gmail_app_password and (cfg.delivery.email_to or cfg.delivery.gmail_user)), "GMAIL_APP_PASSWORD + CBBE_EMAIL_TO"),
        Item("write_once_green", wo_ok, wo_detail),
        Item("preregistered_mdes", todo == 0, f"{todo} TODO-PILOT left in preregistered.yaml"),
        Item("width_fitted", dist.load_width() is not None, "params/width.json from scripts.cbb.fit_width (model channel refuses width_unfit until then)"),
    ]


def render(items: list[Item]) -> str:
    verdict = "LIVE CARD ALLOWED" if all(i.ok for i in items) else "RESEARCH MODE (a checklist item failed)"
    lines = [f"Phase 0b go/no-go: {verdict}"]
    lines += [f"  [{'PASS' if i.ok else 'FAIL'}] {i.name}: {i.detail}" for i in items]
    return "\n".join(lines)


__all__ = ["Item", "render", "run", "write_once_check"]
