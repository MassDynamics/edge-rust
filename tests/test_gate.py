"""The gate fails, rather than skipping, when the golden corpus is missing (review r1, R1)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_missing_corpus_fails_the_golden_suite():
    env = {k: v for k, v in os.environ.items() if k != "EDGE_RUST_ALLOW_NO_CORPUS"}
    env["MD_COUNT_CORPUS_DIR"] = "/nonexistent-edge-rust-corpus"
    golden = Path(__file__).parent / "test_edger_golden.py"
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(golden)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert r.returncode != 0, r.stdout
    assert "no edgeR runs under" in r.stdout
