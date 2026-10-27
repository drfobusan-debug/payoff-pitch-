import pytest


@pytest.fixture(autouse=True)
def _no_state_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CBBE_STATE_SYNC", "0")
