"""edge_rust.run against every edgeR run in the count golden corpus.

The strict numeric gate (1e-8 relative, with the F band for everything derived from F) is the
Rust end-to-end test, ``crates/edger-core/tests/e2e.rs``; the numbers pass through the Python
layer unchanged. This test checks what the Python layer owns: row order, GroupId type, column
names, NA pattern, the ANOVA string shaping and the significance flag, with the numbers at the
tolerances the Rust test measured (F-derived columns reach 1e-5 relative where F is near 0).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import edge_rust
import numpy as np
import pandas as pd
import pytest

CORPUS = Path(os.environ.get("MD_COUNT_CORPUS_DIR", Path.home() / "wd/md-count-golden-corpus"))
RUNS = sorted(
    p.name
    for p in (CORPUS / "reference").glob("*edger_*")
    if p.name.startswith(("count_", "edge_"))
)
if not RUNS:
    pytest.skip(f"no edgeR runs under {CORPUS}", allow_module_level=True)

TIGHT = 1e-8  # AveExpr, Log2FC, MaxLog2FC
F_DERIVED = 2e-5  # PValue, AdjPValue, stat, SE, CI (see module docstring)


def manifest(run: str) -> dict:
    return json.loads((CORPUS / "runs" / run / "manifest.json").read_text())


def inputs(run: str):
    ref = CORPUS / "reference" / run
    counts = pd.read_csv(ref / "input_counts.csv", dtype={"id": str}).set_index("id")
    si = pd.read_csv(ref / "input_sample_info.csv", dtype=str)
    cmp = pd.read_csv(ref / "input_comparisons.csv", dtype=str)
    m = manifest(run)
    params = dict(m["params"], entity_type=m["entity_type"], mode=m["mode"])
    return counts, si, cmp, params


def num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s.replace("", np.nan)).to_numpy(dtype=float)


def check_numeric(name: str, got: np.ndarray, want: np.ndarray, tol: float, f_gate=False):
    na_g, na_w = np.isnan(got), np.isnan(want)
    assert (na_g == na_w).all(), f"{name}: NA pattern differs at {np.flatnonzero(na_g != na_w)[:5]}"
    a, b = got[~na_w], want[~na_w]
    gap = np.abs(a - b)
    ok = gap <= tol * np.abs(b)
    if f_gate:
        ok |= gap <= 1e-8 * np.maximum(np.abs(b), 1.0)
    assert ok.all(), f"{name}: worst rel {np.max(gap / np.abs(b)):.2e}"


def tol_for(col: str) -> tuple[float, bool]:
    stat = col.split(" ", 1)[0]
    if stat in ("AveExpr", "Log2FC", "MaxLog2FC"):
        return TIGHT, False
    if stat == "F":
        return TIGHT, True
    return F_DERIVED, False


@pytest.mark.parametrize("run", [r for r in RUNS if manifest(r)["status"] != "error"])
def test_table_matches_reference_output(run):
    counts, si, cmp, params = inputs(run)
    got = edge_rust.run(counts, si, cmp, params)
    anova = params["mode"] == "anova"
    want = pd.read_csv(
        CORPUS / "reference" / run / "reference_output.csv",
        dtype=str if anova else {"GroupId": np.int64},
        keep_default_na=not anova,
    )
    assert list(got.columns) == list(want.columns)
    assert len(got) == len(want)
    if anova:
        want = want.fillna("")
        assert list(got["GroupId"]) == list(want["GroupId"])
        assert list(got["MaxLog2FCPair"]) == list(want["MaxLog2FCPair"])
        for c in ["AveExpr", "PValue", "AdjPValue", "F", "MaxLog2FC"]:
            assert got[c].map(type).eq(str).all(), f"{c}: not strings"
    else:
        assert got["GroupId"].dtype == np.int64
        assert (got["GroupId"].to_numpy() == want["GroupId"].to_numpy()).all()
    for c in want.columns:
        if c in ("GroupId", "MaxLog2FCPair"):
            continue
        tol, f_gate = tol_for(c)
        g, w = num(got[c].astype(object)), num(want[c].astype(object))
        stat, _, label = c.partition(" ")
        if stat in ("stat", "SE", "CILeft", "CIRight"):
            # edger_f_floor rows (|F_golden| < 1e-8): F is noise there, so these are exempt.
            floor = np.abs(num(want[f"F {label}"].astype(object))) < 1e-8
            g, w = g[~floor], w[~floor]
        check_numeric(f"{run} {c}", g, w, tol, f_gate)
        if c.split(" ", 1)[0] == "AdjPValue":
            sel = ~np.isnan(w)
            assert ((g[sel] < 0.05) == (w[sel] < 0.05)).all(), f"{run} {c}: significance flag"


ERROR_RUNS = [r for r in RUNS if manifest(r)["status"] == "error"]


@pytest.mark.parametrize("run", ERROR_RUNS)
def test_expected_error(run):
    expected = manifest(run)["expected_error"]
    if run in ("edge_edger_negative", "edge_edger_protein_entity"):
        # These fail in the router before inputs are dumped: rebuild them from an ordinary run.
        counts, si, cmp, params = inputs("count_edger_airway_all_ctlnone")
        if run == "edge_edger_negative":
            counts.iloc[0, 0] = -1
        else:
            params["entity_type"] = "protein"
    else:
        counts, si, cmp, params = inputs(run)
    with pytest.raises(ValueError) as e:
        edge_rust.run(counts, si, cmp, params)
    assert expected in str(e.value)
