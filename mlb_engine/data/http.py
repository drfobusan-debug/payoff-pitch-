"""One HTTP session for every outbound call, with retries and a timeout.

The daily card is a single process making a few hundred requests to five hosts.
A bare ``requests.get`` treats a dropped connection as fatal, so one reset --
``ConnectionResetError(54)`` on a morning when Savant hiccups -- ends the run,
and no card means no email. Every fetch here is idempotent and read-only, so a
retry costs a second and buys the whole slate.

The timeout matters as much as the retry: a request with no deadline can hang
until the machine sleeps, which is how a scheduled 10:05 job is still holding a
socket at noon.
"""

from __future__ import annotations

import requests
from urllib3.util.retry import Retry

DEFAULT_TIMEOUT = 20.0

# Four attempts over ~3.5s of backoff. Long enough to ride out a reset or a
# rate-limit blip, short enough that a genuinely dead host fails the fetch
# rather than the slate's deadline.
RETRY = Retry(
    total=3,
    connect=3,
    read=3,
    status=3,
    backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset({"GET", "HEAD"}),
    raise_on_status=False,
    respect_retry_after_header=True,
)


class _TimeoutSession(requests.Session):
    """A session that applies ``timeout`` to any request that omits one."""

    def __init__(self, timeout: float) -> None:
        super().__init__()
        self._timeout = timeout

    def request(self, method: str, url: str, **kw: object) -> requests.Response:  # type: ignore[override]
        kw.setdefault("timeout", self._timeout)
        return super().request(method, url, **kw)  # type: ignore[arg-type]


def session(
    user_agent: str = "mlb-prediction-engine/0.1", timeout: float = DEFAULT_TIMEOUT
) -> requests.Session:
    """A retrying session. Safe to share: ``requests`` sessions are per-process."""
    s = _TimeoutSession(timeout)
    s.headers["User-Agent"] = user_agent
    adapter = requests.adapters.HTTPAdapter(max_retries=RETRY)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def get(url: str, **kw: object) -> requests.Response:
    """One-off retrying GET, for callers with no session of their own."""
    ua = kw.pop("user_agent", "mlb-prediction-engine/0.1")
    return session(user_agent=str(ua)).get(url, **kw)  # type: ignore[arg-type]
