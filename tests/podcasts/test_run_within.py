"""The card shortcuts cap the feed read, so a stalled read never holds the card back.

On 2026-11-24 the CBB 11:30 podcast read ran until 03:32 the next morning: neither
card was priced that day. The limit kills only the read's child process; the state
pull/push stays in the parent, outside it.
"""

from __future__ import annotations

import operator
import os
import time

from cbb_engine import cli as cbb_cli
from engine_common.podcasts.run import run_within
from nba_engine import cli as nba_cli


def test_a_stalled_read_is_stopped_and_returns_none() -> None:
    t0 = time.monotonic()
    assert run_within(1, time.sleep, 30) is None
    assert time.monotonic() - t0 < 20


def test_a_read_that_finishes_returns_its_result() -> None:
    assert run_within(30, operator.add, 1, 2) == 3


def test_a_read_that_fails_returns_none() -> None:
    assert run_within(30, int, "not a number") is None


def test_a_reader_that_dies_without_a_result_is_not_waited_on() -> None:
    t0 = time.monotonic()
    assert run_within(60, os._exit, 3) is None
    assert time.monotonic() - t0 < 20


def test_both_podcast_commands_take_the_limit() -> None:
    assert cbb_cli.build_parser().parse_args(["podcast", "--max-seconds", "60"]).max_seconds == 60
    assert nba_cli.build_parser().parse_args(["podcast", "--max-seconds", "60"]).max_seconds == 60
