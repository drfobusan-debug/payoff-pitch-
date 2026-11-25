"""The card shortcuts cap the feed read, so a stalled read never holds the card back.

On 2026-11-24 the CBB 11:30 podcast read ran until 03:32 the next morning: neither
card was priced that day. The limit kills only the read's child process; the state
pull/push stays in the parent, outside it.
"""

from __future__ import annotations

import operator
import os
import sys
import threading
import time
import types

import pytest

from cbb_engine import cli as cbb_cli
from cfb_engine import cli as cfb_cli
from engine_common.podcasts.run import max_seconds_from_env, run_within
from engine_common.podcasts.transcript import one_at_a_time, transcribe
from nba_engine import cli as nba_cli
from nhl_engine import cli as nhl_cli


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


def test_the_limit_comes_from_podcast_max_seconds(monkeypatch) -> None:
    monkeypatch.delenv("PODCAST_MAX_SECONDS", raising=False)
    assert max_seconds_from_env() == 3600
    monkeypatch.setenv("PODCAST_MAX_SECONDS", "90")
    assert max_seconds_from_env() == 90
    monkeypatch.setenv("PODCAST_MAX_SECONDS", "0")
    assert max_seconds_from_env() is None
    monkeypatch.setenv("PODCAST_MAX_SECONDS", "1h")
    assert max_seconds_from_env() == 3600


def test_cfb_and_nhl_podcast_reads_are_limited_by_default(monkeypatch) -> None:
    monkeypatch.delenv("PODCAST_MAX_SECONDS", raising=False)
    assert cfb_cli._build_parser().parse_args(["podcast"]).max_seconds == 3600
    assert nhl_cli.build_parser().parse_args(["podcast"]).max_seconds == 3600


def test_whisper_passes_take_turns(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WHISPER_LOCK", str(tmp_path / "whisper.lock"))
    held = threading.Event()

    def first() -> None:
        with one_at_a_time():
            held.set()
            time.sleep(1)

    t = threading.Thread(target=first)
    t.start()
    held.wait(5)
    t0 = time.monotonic()
    with one_at_a_time():
        waited = time.monotonic() - t0
    t.join()
    assert waited > 0.5


def test_a_queued_pass_takes_the_transcript_the_first_one_wrote(tmp_path, monkeypatch) -> None:
    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = lambda *a, **k: pytest.fail("transcribed an episode already on disk")
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setenv("WHISPER_LOCK", str(tmp_path / "whisper.lock"))
    dest = tmp_path / "ep.txt"
    dest.write_text("[00:00] done\n")
    assert transcribe(tmp_path / "gone.mp3", dest) == dest
