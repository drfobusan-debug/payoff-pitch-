"""The shared HTTP session: one dropped connection must not cost the card."""

from __future__ import annotations

import requests

from mlb_engine.data import http


def test_a_request_without_a_timeout_gets_one() -> None:
    """A fetch with no deadline is how a 10:05 job is still holding a socket at noon."""
    seen: dict[str, object] = {}
    s = http.session(timeout=7.5)
    orig = requests.Session.request
    try:
        requests.Session.request = lambda self, m, u, **kw: seen.update(kw)  # type: ignore[assignment,method-assign,return-value]
        s.get("http://example.invalid")
    finally:
        requests.Session.request = orig  # type: ignore[method-assign]
    assert seen["timeout"] == 7.5


def test_an_explicit_timeout_is_left_alone() -> None:
    seen: dict[str, object] = {}
    s = http.session(timeout=7.5)
    orig = requests.Session.request
    try:
        requests.Session.request = lambda self, m, u, **kw: seen.update(kw)  # type: ignore[assignment,method-assign,return-value]
        s.get("http://example.invalid", timeout=30)
    finally:
        requests.Session.request = orig  # type: ignore[method-assign]
    assert seen["timeout"] == 30


def test_connection_resets_and_server_errors_are_retried() -> None:
    """The failure that killed the Aug 6 card was ConnectionResetError(54)."""
    retry = http.RETRY
    assert retry.connect and retry.read and retry.total
    assert 429 in retry.status_forcelist
    assert 503 in retry.status_forcelist
    assert retry.backoff_factor > 0


def test_only_idempotent_verbs_are_retried() -> None:
    """Replaying a POST could double-send; every engine fetch is a GET."""
    assert http.RETRY.allowed_methods == frozenset({"GET", "HEAD"})


def test_the_session_is_mounted_on_both_schemes() -> None:
    s = http.session()
    assert isinstance(s.adapters["https://"].max_retries, type(http.RETRY))
    assert s.adapters["http://"].max_retries.total == http.RETRY.total
