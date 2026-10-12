"""§4.e pilot: CLV per execution quote vs the close, stale persistence, cut-10 deltas, yaml slots — offline."""

from __future__ import annotations

from pathlib import Path

from cbb_engine import history, pilot
from cbb_engine.config import SPORT
from engine_runtime import quotes

DAY = "2025-01-11"
TIP = f"{DAY}T17:00:00Z"


def _book(key: str, total: float, over: float = -110, under: float = -110) -> dict[str, object]:
    return {"key": key, "markets": [{"key": "totals", "outcomes": [
        {"name": "Over", "point": total, "price": over}, {"name": "Under", "point": total, "price": under},
    ]}]}


def _event(eid: str, sharp_total: float, dk_total: float, dk_over: float = -110, caesars_total: float | None = None) -> dict[str, object]:
    return {"id": eid, "commence_time": TIP, "home_team": "Duke Blue Devils", "away_team": "UNC Tar Heels",
            "bookmakers": [_book("pinnacle", sharp_total), _book("fanduel", sharp_total),
                           _book("caesars", caesars_total if caesars_total is not None else sharp_total),
                           _book("draftkings", dk_total, dk_over)]}


SD = 16.3


class Tape:
    """Featured passes: sharp books move 145.5 -> 146.5 at T-30 while DK holds 145.5/-110; the close
    consensus sits at 146.5 (Caesars lags at 145.5), so the held DK Over carries line value."""

    def historical_board(self, markets: tuple[str, ...], *, at: str, pass_id: str = "", **p: str) -> tuple[str, list[dict[str, object]]]:
        if "close" in pass_id:
            return at, [_event("e1", 146.5, 145.5, -120, caesars_total=145.5)]
        moved = at >= f"{DAY}T16:30:00Z"
        return at, [_event("e1", 146.5 if moved else 145.5, 145.5), _event("e2", 150.5, 150.5)]


def _write(root: Path) -> list[quotes.Quote]:
    history.pull_day(Tape(), root, DAY, [TIP], dense=True)
    return quotes.read_day(root, SPORT, DAY)


def test_clv_rows_price_each_exec_quote_against_the_close(tmp_path: Path) -> None:
    rows = pilot.clv_rows(_write(tmp_path), SD)
    over = [r for r in rows if r["event_id"] == "e1" and r["side"] == "over"]
    under = [r for r in rows if r["event_id"] == "e1" and r["side"] == "under"]
    want = pilot.clv(146.5, SD, "over", 145.5, -110)  # close consensus line 146.5 at -110/-110 -> mu 146.5
    assert over and all(r["line"] == 145.5 and abs(pilot.cents(r) - want) < 1e-9 for r in over)
    assert want > 0 > max(pilot.cents(r) for r in under)  # a point of line value vs the vig the Under eats
    assert not [r for r in rows if r["event_id"] == "e2"]  # no close for e2 -> skipped, never cross-priced
    assert abs(pilot.p_close(146.5, SD, "over", 146) + pilot.p_close(146.5, SD, "under", 146) - 1) < 1e-12
    stale = [r for r in rows if r["stale"]]
    assert stale and all(r["event_id"] == "e1" and r["line"] == 145.5 for r in stale)
    assert all(r["held_next"] for r in stale if r["held_next"] is not None)


def test_summary_and_yaml_slots(tmp_path: Path) -> None:
    shards = _write(tmp_path)
    rows = pilot.clv_rows(shards, SD)
    rows += [{**r, "game_date": "2025-01-18", "clv_cents": pilot.cents(r) + 2} for r in rows]
    rows += [{**r, "game_date": "2025-01-25", "clv_cents": pilot.cents(r) - 1} for r in rows[: len(rows) // 2]]
    deltas = pilot.featured_deltas(shards, SD)
    scores = pilot.log_scores(
        [{"game_date": DAY, "close_total": "145.5", "final_total": str(f)} for f in (130, 145, 160, 170)],
        16.3, start="2024-11-04", days=120,
    )
    rep = pilot.summarize(rows, deltas + [0.0, 1.0, -1.0], scores, cluster_inflation=1.4)
    pre = rep["prereg"]
    assert isinstance(pre, dict)
    assert set(pre) == {"mde.per_bet_clv_sd_cents", "futility.min_n", "H4.min_n", "prior_decision.mde"}
    assert pre["futility.min_n"] > 0 and rep["week5_games"] == 4
    yaml = tmp_path / "p.yaml"
    yaml.write_text(
        "hypotheses:\n  live:\n    - {id: H4, text: x, min_n: TODO-PILOT}\nprior_decision:\n  week: 5\n  mde: TODO-PILOT\n"
        "futility:\n  min_worthwhile_clv_cents: 0.5\n  min_n: TODO-PILOT\nmde:\n  per_bet_clv_sd_cents: TODO-PILOT\n  cluster_inflation: 1.4\n"
    )
    done = pilot.write_prereg(yaml, pre, note="test")
    text = yaml.read_text()
    assert len(done) == 4 and "TODO-PILOT" not in text and f"min_n: {pre['futility.min_n']}" in text
    assert pilot.write_prereg(yaml, pre, note="again") == []  # filled slots are never refit
