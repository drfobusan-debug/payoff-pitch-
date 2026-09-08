"""A slate pass that priced no games writes no article, and removes a stale one."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.regen_slate as rs


def test_an_empty_pass_writes_nothing_and_clears_the_blocks_files(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    audit = tmp_path / "audit"
    out = tmp_path / "out"
    audit.mkdir()
    out.mkdir()
    (audit / "previews_2026-08-24_late.json").write_text(json.dumps([]))
    stale_pdf = out / "PayoffPitch_Slate_2026-08-24_late.pdf"
    stale_mp3 = out / "PayoffPitch_Slate_2026-08-24_late.mp3"
    stale_pdf.write_bytes(b"yesterday")
    stale_mp3.write_bytes(b"yesterday")
    whole = out / "PayoffPitch_Slate_2026-08-24.pdf"
    whole.write_bytes(b"slate")

    class Cfg:
        audit_dir = audit
        output_dir = out

    monkeypatch.setattr(rs, "load_config", lambda: Cfg())
    monkeypatch.setattr(
        rs,
        "build_preview_report",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("rendered")),
    )
    monkeypatch.setattr(sys, "argv", ["regen_slate", "2026-08-24", "--block", "late"])
    rs.main()
    assert not stale_pdf.exists() and not stale_mp3.exists()
    assert whole.read_bytes() == b"slate"
    assert "no late games priced" in capsys.readouterr().out
