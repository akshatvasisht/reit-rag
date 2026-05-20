"""Black-box tests for ``scripts/lint_session_tags.py``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "lint_session_tags.py"


def _run(target: Path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--paths", str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result


def test_flags_session_tag_comments(tmp_path: Path):
    sample = tmp_path / "bad.py"
    sample.write_text(
        "# Fix C added this\n"
        "# Phase 2 cleanup\n"
        "x = 1  # S3 task\n"
        "y = 2  # P0-blocker patch\n"
        "z = 3  # audit P2 finding\n"
        "a = 4  # post-review hardening\n"
        "b = 5  # (1a)check this\n"
        "c = 6  # Round 3 refactor\n"
    )

    result = _run(sample)

    assert result.returncode == 1, result.stdout + result.stderr
    out = result.stdout
    assert "Fix C added this" in out
    assert "Phase 2 cleanup" in out
    assert "S3 task" in out
    assert "P0-blocker patch" in out
    assert "audit P2 finding" in out
    assert "post-review hardening" in out
    assert "(1a)check this" in out
    assert "Round 3 refactor" in out


def test_does_not_flag_patterns_inside_strings(tmp_path: Path):
    sample = tmp_path / "clean.py"
    sample.write_text(
        'label = "Phase 1 results"\n'
        'assert "Fix A" in label or "Phase 2" in label\n'
        'sql = """\n'
        "  -- Fix B applied here\n"
        "  SELECT 1; -- S5 task\n"
        '"""\n'
        'msg = "P0-blocker patch landed"\n'
        'note = "post-review hardening completed"\n'
        "value = 1  # explain why we add one here\n"
    )

    result = _run(sample)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == ""
