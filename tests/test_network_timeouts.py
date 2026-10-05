"""A scheduled pass waits on the voice and the mail server, so neither may wait forever."""

from __future__ import annotations

import asyncio
import smtplib
import sys
import types

from mlb_engine.config import Config, Credentials
from mlb_engine.output import audit_insight as ai
from mlb_engine.output import email as email_mod


def _fake_tts(monkeypatch, seen: dict, *, stall: bool) -> None:
    class _Communicate:
        def __init__(self, *a, **k):
            pass

        async def save(self, path):
            if stall:
                await asyncio.sleep(3600)
            open(path, "wb").write(b"edge")

    class _GTTS:
        def __init__(self, **k):
            seen["gtts_timeout"] = k.get("timeout")

        def save(self, path):
            open(path, "wb").write(b"gtts")

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=_Communicate))
    monkeypatch.setitem(sys.modules, "gtts", types.SimpleNamespace(gTTS=_GTTS))


def test_a_stalled_edge_voice_falls_back_instead_of_hanging(monkeypatch, tmp_path):
    seen: dict = {}
    _fake_tts(monkeypatch, seen, stall=True)
    monkeypatch.setattr(ai, "TTS_TIMEOUT_S", 0.05)
    assert ai.to_mp3("hello", tmp_path / "a.mp3") == b"gtts"
    assert seen["gtts_timeout"] == ai.GTTS_REQUEST_TIMEOUT_S


def test_a_healthy_edge_voice_is_used(monkeypatch, tmp_path):
    seen: dict = {}
    _fake_tts(monkeypatch, seen, stall=False)
    assert ai.to_mp3("hello", tmp_path / "a.mp3") == b"edge"
    assert "gtts_timeout" not in seen


def test_card_email_bounds_the_smtp_connection(monkeypatch):
    seen: dict = {}

    class _FakeSMTP:
        def __init__(self, *a, **k):
            seen["timeout"] = k.get("timeout")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self, *a):
            pass

        def send_message(self, msg):
            pass

    monkeypatch.setattr(smtplib, "SMTP_SSL", _FakeSMTP)
    cfg = Config(creds=Credentials(gmail_user="x@y.com", gmail_app_password="pw"))
    email_mod.send_card_email(cfg, subject="s", html_body="<p/>", text_body="t", to="a@b.com")
    assert seen["timeout"] == email_mod.SMTP_TIMEOUT_S
