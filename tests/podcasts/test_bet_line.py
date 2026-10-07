from dataclasses import replace

from engine_common.podcasts.extract import Pick
from engine_common.podcasts.render import bet_line, reason


def test_bet_line_reads_name_record_selection_price_edge() -> None:
    line = bet_line("Joel Meyer", "12-9 graded", "MTL ML", 120, "the price")
    assert line == "<li><b>Joel Meyer</b> (12-9 graded). <b>MTL ML +120</b> (the price)</li>"
    assert "-111</b>" in bet_line("A", "1-0", "MIN ML", -111, "the weather")
    assert "MIN ML</b>" in bet_line("A", "1-0", "MIN ML", None, "the weather")


BASE = Pick(
    pick_id="p",
    show="s",
    show_name="Show",
    host=None,
    episode_guid="g",
    episode_title="t",
    published="2026-10-01T09:00:00+00:00",
    kind="official",
    market="game_total",
    team=None,
    opponent=None,
    side="under",
    line=220.5,
    price=None,
    units=None,
    seconds=60,
    quote="",
    reason="",
    description="",
)


def test_the_edge_is_the_hosts_reason_never_the_engines() -> None:
    assert reason(replace(BASE, edge="run defense won't let them score", reason="pace")) == (
        "run defense won't let them score"
    )
    assert reason(replace(BASE, reason="the weather")) == "the weather"
    assert (
        reason(
            replace(
                BASE,
            )
        )
        == "no reason given"
    )
