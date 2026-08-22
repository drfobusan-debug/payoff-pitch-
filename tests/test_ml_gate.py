"""Unit tests for the moneyline sharp-money confirmation gate."""

from __future__ import annotations

from mlb_engine.features.ml_gate import MLSharpGate


def test_gate_keeps_sharp_confirmed_side() -> None:
    gate = MLSharpGate(enabled=True, min_divergence=0.0)
    keep, reason = gate.allows(handle_pct=70.0, bets_pct=50.0)
    assert keep is True
    assert "OK" in reason


def test_gate_demotes_when_money_against() -> None:
    gate = MLSharpGate(enabled=True, min_divergence=0.0)
    keep, reason = gate.allows(handle_pct=40.0, bets_pct=60.0)
    assert keep is False
    assert "PASS" in reason


def test_gate_respects_positive_threshold() -> None:
    gate = MLSharpGate(enabled=True, min_divergence=10.0)
    keep, _ = gate.allows(handle_pct=55.0, bets_pct=50.0)
    assert keep is False
    keep, _ = gate.allows(handle_pct=65.0, bets_pct=50.0)
    assert keep is True


def test_gate_neutral_when_missing_split() -> None:
    gate = MLSharpGate(enabled=True, min_divergence=0.0)
    keep, reason = gate.allows(handle_pct=None, bets_pct=None)
    assert keep is True
    assert "neutral" in reason


def test_gate_disabled_keeps_everything() -> None:
    gate = MLSharpGate(enabled=False)
    keep, reason = gate.allows(handle_pct=10.0, bets_pct=90.0)
    assert keep is True
    assert reason == ""


def test_upgrade_promotes_sharp_backed_pass() -> None:
    gate = MLSharpGate(upgrade_divergence=5.0, max_fair_prob=0.65)
    up, reason = gate.upgrades(handle_pct=65.0, bets_pct=50.0, fair_prob=0.48)
    assert up is True
    assert "BUY" in reason


def test_upgrade_skipped_when_divergence_too_small() -> None:
    gate = MLSharpGate(upgrade_divergence=5.0)
    up, _ = gate.upgrades(handle_pct=52.0, bets_pct=50.0, fair_prob=0.48)
    assert up is False


def test_upgrade_skips_heavy_chalk() -> None:
    gate = MLSharpGate(upgrade_divergence=5.0, max_fair_prob=0.65)
    up, reason = gate.upgrades(handle_pct=70.0, bets_pct=50.0, fair_prob=0.80)
    assert up is False
    assert "chalk" in reason


def test_upgrade_disabled_never_promotes() -> None:
    gate = MLSharpGate(upgrade_enabled=False)
    up, reason = gate.upgrades(handle_pct=80.0, bets_pct=40.0, fair_prob=0.40)
    assert up is False
    assert reason == ""


def test_upgrade_requires_split() -> None:
    gate = MLSharpGate(upgrade_divergence=5.0)
    up, _ = gate.upgrades(handle_pct=None, bets_pct=None, fair_prob=0.48)
    assert up is False


def test_from_env_defaults(monkeypatch) -> None:
    for k in (
        "MLBE_ML_SHARP_GATE",
        "MLBE_ML_MIN_DIVERGENCE",
        "MLBE_ML_SHARP_UPGRADE",
        "MLBE_ML_UPGRADE_DIVERGENCE",
        "MLBE_ML_UPGRADE_MAX_FAIR",
    ):
        monkeypatch.delenv(k, raising=False)
    gate = MLSharpGate.from_env()
    assert gate.enabled is True
    assert gate.min_divergence == 0.0
    assert gate.upgrade_enabled is True
    assert gate.upgrade_divergence == 5.0
    assert gate.max_fair_prob == 0.65


def test_from_env_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_ML_SHARP_GATE", "0")
    assert MLSharpGate.from_env().enabled is False


def test_from_env_upgrade_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_ML_SHARP_UPGRADE", "0")
    assert MLSharpGate.from_env().upgrade_enabled is False

# ---- the upgrade cannot buy a price that does not pay ----------------------
def _ml_rec(
    model_prob: float,
    american: float,
    opposite: float,
    team_side: str = "home",
):
    """Price one game_ml selection through the pipeline with sharp money on it.

    Defaults to the home side because a road moneyline dog is refused on price
    alone, which would swallow every sharp-money case these tests are about.
    """
    from types import SimpleNamespace

    from mlb_engine.config import Config
    from mlb_engine.data.vsin import Split
    from mlb_engine.features.lineup_lock import LineupLockGate
    from mlb_engine.features.ml_gate import MLPenGate
    from mlb_engine.features.ml_gate import MLSharpGate as Gate
    from mlb_engine.market.ev import MarketQuote
    from mlb_engine.pipeline import Pipeline

    p = Pipeline.__new__(Pipeline)
    p.cfg = Config()
    p._calibrator = _Identity()
    p._shrink = None
    p._ml_gate = Gate.from_env()
    # The availability gates run after the sharp-money upgrade, so a hand-built
    # pipeline needs them too; neither vetoes anything without fatigue inputs.
    p._pen_gate = MLPenGate.from_env()
    p._lineup_gate = LineupLockGate.from_env()
    p._lineup_lock = None
    sel = "MIA ML" if team_side == "away" else "ATL ML"
    p._splits = {
        ("MIA @ ATL", "game_ml", sel): Split(handle_pct=80.0, bets_pct=37.0)
    }
    game = SimpleNamespace(game_date="2026-08-08", game_pk=1)
    quotes = {
        ("MIA @ ATL", "game_ml", sel): [
            MarketQuote(book="dk", american=american, opposite_american=opposite,
                        handle_pct=80.0, bets_pct=37.0)
        ]
    }
    return p._mk(
        game, "MIA @ ATL", "game", "game_ml", sel, model_prob,
        team_side=team_side, side="win", quotes=quotes,
    )


class _Identity:
    def apply(self, market: str, prob: float) -> float:
        return prob


def test_sharp_upgrade_does_not_buy_a_negative_ev_price() -> None:
    from mlb_engine.market.tiers import Tier

    # Model 44% on a -150 favourite: sharp money is heavily on it, but the price
    # is unplayable at that probability (EV ~ -0.27).
    rec = _ml_rec(0.44, american=-150.0, opposite=130.0)
    assert (rec.ev or 0.0) < 0
    assert rec.tier is Tier.PASS
    assert not any("ml-upgrade" in r for r in rec.reasons)


def test_the_sharp_upgrade_cannot_buy_a_price_the_market_makes_a_dog() -> None:
    """The devigged-probability floor overrules the upgrade, on purpose.

    This case used to be the point of the upgrade: a thin edge on a +200 dog that
    still pays, promoted because the handle is 80% on it. The graded ledger says
    that is the losing half of every split it appears in -- game_ml went -17.3%,
    plus-money buys 28.5%, and the model's disagreement with the price predicts
    losing -- so a side the market itself makes a 33% shot is refused whatever the
    handle says. Handle is the market agreeing about the side, not the price.
    """
    from mlb_engine.market.tiers import Tier

    rec = _ml_rec(0.3366, american=200.0, opposite=-220.0)
    assert (rec.ev or 0.0) > 0 and (rec.edge or 0.0) < 0.02
    assert rec.tier is Tier.PASS
    assert rec.pass_gate == "fair_floor"
    # The upgrade still ran and is still on the record; it is simply overruled.
    assert any("ml-upgrade: BUY" in r for r in rec.reasons)


def test_the_floor_is_reversible_and_the_upgrade_then_promotes(monkeypatch) -> None:
    """The floor is a stance on a price band, so it has to be switchable."""
    from mlb_engine.market.tiers import Tier

    monkeypatch.setenv("MLBE_MIN_FAIR_PROB", "0")
    rec = _ml_rec(0.3366, american=200.0, opposite=-220.0)
    assert rec.tier is Tier.MODERATE
    assert any("ml-upgrade: BUY" in r for r in rec.reasons)


def test_sharp_money_cannot_buy_a_road_dog() -> None:
    """The screen runs after the upgrade, and is meant to overrule it.

    Handle piling onto a road underdog is the market agreeing about the side,
    not about the price -- and the price is what lost 33.9% of stake on that
    cell. It keeps its own gate name rather than the blanket floor's, because the
    road-dog screen is graded on the rows it removed.
    """
    from mlb_engine.market.tiers import Tier

    road = _ml_rec(0.3366, american=200.0, opposite=-220.0, team_side="away")
    assert road.tier is Tier.PASS
    assert road.pass_gate == "away_ml_dog"
    assert any("ml-upgrade: BUY" in r for r in road.reasons)


def test_a_confirmed_moneyline_is_actually_buyable() -> None:
    """The ml gate's own OK must not read as a batter contact-quality veto.

    The two gates wrote their reason into the same local, so every moneyline the
    gate confirmed was then hard-passed by the batter-prop floor further down --
    leaving sharp-money upgrades as the only ML bets that could reach the card.
    """
    from mlb_engine.market.tiers import Tier

    rec = _ml_rec(0.65, american=-160.0, opposite=140.0)
    assert (rec.ev or 0.0) > 0 and 0.02 <= (rec.edge or 0.0) <= 0.08
    assert rec.tier is Tier.STRONG
    assert any("ml-gate: OK" in r for r in rec.reasons)
