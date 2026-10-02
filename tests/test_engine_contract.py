"""``edge_rust._core`` refuses what production refuses, with a Python exception and in bounded time.

Each refusal runs in a fresh interpreter under a timeout: at review r1 (39ff8d4) a zero library or
a 1e308 control hung in the Levenberg retry loop with the GIL released, beyond SIGINT's reach
(SE-1 / F1). The messages are R's (production ``.fitEdgeRModel``, edgeR 4.8.2).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest
from edge_rust import _core

SAMPLES = [f"s{j}" for j in range(6)]
COND = ["A", "A", "A", "B", "B", "B"]


def synth(ng: int = 50) -> np.ndarray:
    g = np.arange(ng)[:, None]
    j = np.arange(6)[None, :]
    return (20 + (g * 7 + j * 13) % 50).astype(np.float64)


def call(counts, controls=(), cmps=(("B", "A", "B", "A"),)):
    ids = [str(i + 1) for i in range(counts.shape[0])]
    return _core.edger_pipeline(counts, ids, SAMPLES, "condition", COND, list(controls), list(cmps))


CASES = {
    "zero_library": (
        "c = synth(); c[:, 2] = 0; call(c)",
        "library sizes should be greater than zero",
    ),
    "extreme_numeric_control": (
        "call(synth(), [('x', 'numerical', ['1e308', '-1e308', '1', '2', '1e308', '3'])])",
        "NA/NaN/Inf in foreign function call",
    ),
    "single_level_control": (
        "call(synth(), [('batch', 'categorical', ['b'] * 6)])",
        "contrasts can be applied only to factors with 2 or more levels",
    ),
    "left_equals_right": (
        "call(synth(), cmps=[('A', 'A', 'A', 'A')])",
        "contrasts are all zero",
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_refusal_is_a_value_error_in_bounded_time(case):
    body, message = CASES[case]
    src = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(__import__("pathlib").Path(__file__).parent)!r})
        from test_engine_contract import call, synth
        try:
            {body}
        except ValueError as e:
            print("VALUE_ERROR:", e)
        else:
            print("NO_ERROR")
        """
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", src], capture_output=True, text=True, timeout=60, check=False
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{case}: no answer in 60 s (the engine hangs)")
    assert "VALUE_ERROR:" in r.stdout, r.stdout + r.stderr
    assert message in r.stdout, r.stdout


def test_fortran_ordered_counts_give_the_same_answer():
    c = synth(200)
    a, b = call(np.ascontiguousarray(c)), call(np.asfortranarray(c))
    np.testing.assert_array_equal(a["F"], b["F"])
    np.testing.assert_array_equal(a["pairs"][0]["Log2FC"], b["pairs"][0]["Log2FC"])


def test_engine_panic_is_a_runtime_error():
    """SE-2: a panic must reach Python as RuntimeError, which ``except Exception`` catches."""
    with pytest.raises(RuntimeError, match="internal error in the edgeR engine"):
        _core._selftest_panic()
