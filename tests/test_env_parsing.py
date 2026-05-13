from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from src.ingestion.chart_extractor import _parse_float_env
from src.ingestion.embedder import _parse_int_env


def test_parse_int_env_invalid_value_falls_back_default(monkeypatch) -> None:
    monkeypatch.setenv("EMBED_BATCH", "not-a-number")
    assert _parse_int_env("EMBED_BATCH", 16, minimum=1) == 16


def test_parse_int_env_below_minimum_falls_back_default(monkeypatch) -> None:
    monkeypatch.setenv("EMBED_BATCH", "0")
    assert _parse_int_env("EMBED_BATCH", 16, minimum=1) == 16


def test_parse_float_env_invalid_value_falls_back_default(monkeypatch) -> None:
    monkeypatch.setenv("VISION_MIN_INTERVAL_SECONDS", "oops")
    assert _parse_float_env("VISION_MIN_INTERVAL_SECONDS", 1.4, minimum=0.0) == 1.4


def test_import_embedder_with_bad_env_does_not_crash(tmp_path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["EMBED_BATCH"] = "bad"
    proc = subprocess.run(
        [sys.executable, "-c", "import src.ingestion.embedder"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_import_chart_extractor_with_bad_env_does_not_crash() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["VISION_MIN_INTERVAL_SECONDS"] = "bad"
    proc = subprocess.run(
        [sys.executable, "-c", "import src.ingestion.chart_extractor"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
