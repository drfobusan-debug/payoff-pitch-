"""The pass runner: ``capture -> price -> write_once -> grade``, every step stamped.

A ``Pass`` is one clock tick of one kind. The runner stamps ``pass_id``,
``code_sha``, ``env_lock_hash`` and the credit spend, times each step and
writes a step log beside the ledger. A step that raises is recorded as
``failed`` and re-raised by default; a missed close or capture is therefore a
logged event, never a silent CLV gap. The runner knows nothing about a sport:
the steps are callables the engine supplies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from engine_runtime.quotes import new_pass_id, now_utc

log = logging.getLogger(__name__)
Step = Callable[["Pass"], object]


def code_sha(repo: Path | None = None) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo or Path.cwd(),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def env_lock_hash(lock: Path | None) -> str:
    if lock is None or not lock.exists():
        return ""
    return hashlib.sha256(lock.read_bytes()).hexdigest()[:16]


@dataclass
class Pass:
    sport: str
    slate_date: str
    pass_kind: str
    root: Path
    pass_id: str = ""
    code_sha: str = ""
    env_lock_hash: str = ""
    started_at: str = ""
    credits: int = 0
    steps: list[dict[str, object]] = field(default_factory=list)
    context: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.started_at = self.started_at or now_utc()
        self.pass_id = self.pass_id or new_pass_id(self.pass_kind, self.started_at)

    @property
    def log_path(self) -> Path:
        return self.root / "passes" / self.sport / self.slate_date / f"{self.pass_id}.json"

    def spend(self, cost: int) -> None:
        self.credits += cost

    def run(self, name: str, step: Step, *, required: bool = True) -> object:
        t0 = time.monotonic()
        entry: dict[str, object] = {"step": name, "started": now_utc()}
        try:
            result = step(self)
        except Exception as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            self.steps.append(entry)
            self._flush()
            log.error("pass %s step %s failed: %s", self.pass_id, name, exc)
            if required:
                raise
            return None
        entry.update(status="ok", seconds=round(time.monotonic() - t0, 3))
        self.steps.append(entry)
        self._flush()
        return result

    def _flush(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(
            json.dumps(
                {
                    "pass_id": self.pass_id,
                    "sport": self.sport,
                    "slate_date": self.slate_date,
                    "pass_kind": self.pass_kind,
                    "code_sha": self.code_sha,
                    "env_lock_hash": self.env_lock_hash,
                    "started_at": self.started_at,
                    "credits": self.credits,
                    "steps": self.steps,
                },
                indent=1,
                sort_keys=True,
            )
        )


def start(
    sport: str,
    slate_date: str,
    pass_kind: str,
    root: Path,
    *,
    repo: Path | None = None,
    lock: Path | None = None,
) -> Pass:
    return Pass(
        sport,
        slate_date,
        pass_kind,
        root,
        code_sha=code_sha(repo),
        env_lock_hash=env_lock_hash(lock),
    )


__all__ = ["Pass", "Step", "code_sha", "env_lock_hash", "start"]
