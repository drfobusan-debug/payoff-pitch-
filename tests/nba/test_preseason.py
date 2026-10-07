from datetime import date

from nba_engine import cli, config, state
from nba_engine.data.oddsapi import OddsAPIClient


def test_regular_season_is_the_default(monkeypatch, tmp_path):
    monkeypatch.delenv("NBAE_PRESEASON", raising=False)
    monkeypatch.setenv("NBAE_DATA_DIR", str(tmp_path))
    assert config.load_config().sport_key == "basketball_nba"
    assert config.data_dir() == tmp_path
    assert state.prefix() == "nba"


def test_preseason_switches_feed_dir_and_state_subtree(monkeypatch, tmp_path):
    monkeypatch.setenv("NBAE_PRESEASON", "1")
    monkeypatch.setenv("NBAE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("THE_ODDS_API_KEY", "k")
    assert config.data_dir() == tmp_path / "preseason"
    assert config.cache_dir() == tmp_path / "preseason" / "cache"
    assert state.prefix() == "nba/preseason"
    client = cli._client()
    assert client is not None
    assert client.base.endswith("/sports/basketball_nba_preseason")
    assert client.hist_base.endswith("/historical/sports/basketball_nba_preseason")


def test_client_calls_its_own_sport(monkeypatch):
    client = OddsAPIClient("k", sport_key="basketball_nba_preseason")
    urls: list[str] = []

    def fake(url, **params):
        urls.append(url)
        return []

    monkeypatch.setattr(client, "_get_json", fake)
    client.fetch_events(slate_date=date(2026, 10, 5))
    assert urls == ["https://api.the-odds-api.com/v4/sports/basketball_nba_preseason/events"]


def test_preseason_files_never_sync_into_the_season_tree(tmp_path):
    root = tmp_path / "data"
    p = root / "preseason" / "prices" / "2026-10-05" / "board_x.csv"
    p.parent.mkdir(parents=True)
    p.write_text("x")
    assert state._copy_missing(root, tmp_path / "nba", state.LIVE) == []
