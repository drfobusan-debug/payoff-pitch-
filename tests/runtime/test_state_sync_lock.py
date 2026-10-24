"""Every engine's state pull/push holds the one shared-worktree lock."""

from __future__ import annotations

import importlib

import pytest

from mlb_engine import state as mlb_state

ENGINES = (
    "nba_engine.state",
    "nhl_engine.state",
    "cfb_engine.state",
    "nfl_engine.state",
    "engine_common.podcasts.state",
)


def test_the_lock_is_reentrant_in_one_process(tmp_path):
    (tmp_path / ".git").mkdir()
    with mlb_state._sync_lock(tmp_path, "engine-state"):
        with mlb_state._sync_lock(tmp_path, "engine-state"):
            assert mlb_state._HELD
        assert mlb_state._HELD
    assert not mlb_state._HELD


@pytest.mark.parametrize("name", ENGINES)
def test_pull_rebuilds_the_worktree_only_under_the_lock(name, tmp_path, monkeypatch):
    mod = importlib.import_module(name)
    repo, tree = tmp_path / "repo", tmp_path / "tree"
    (repo / ".git").mkdir(parents=True)
    tree.mkdir()
    seen: list[bool] = []

    def worktree(repo_arg, branch):
        seen.append(bool(mlb_state._HELD))
        return tree

    monkeypatch.setattr(mod, "_worktree", worktree)
    mod.pull_state(tmp_path / "data", repo=repo)
    assert seen == [True]
    assert not mlb_state._HELD
