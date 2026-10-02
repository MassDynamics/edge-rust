"""The Python layer in ``edge_rust.run`` against production's input handling (review r1, P1 to P9).

Each test feeds ``run`` one input that production refuses or repairs, and checks the port does the
same: MDFlexiComparisons ``runEdgeRPairwiseStats`` and ``.buildCountMatrixFromLongDT``
(``R/edgeRStatsFun.R``).
"""

from __future__ import annotations

import edge_rust
import numpy as np
import pandas as pd
import pytest

COND = ["A", "A", "A", "B", "B", "B"]
SAMPLES = [f"s{j}" for j in range(6)]
CMP = pd.DataFrame({"left": ["B"], "right": ["A"]})


def counts(ng: int = 60) -> pd.DataFrame:
    g = np.arange(ng)[:, None]
    j = np.arange(6)[None, :]
    m = (20 + (g * 7 + j * 13) % 50).astype(float)
    return pd.DataFrame(m, index=[str(i + 1) for i in range(ng)], columns=SAMPLES)


def sample_info(**extra) -> pd.DataFrame:
    return pd.DataFrame({"replicate": SAMPLES, "condition": COND, **extra})


def test_missing_condition_is_rejected():
    """P1: production stops at edgeRStatsFun.R:339 instead of fitting a level called "nan"."""
    si = sample_info()
    si.loc[2, "condition"] = None
    with pytest.raises(ValueError, match="Condition column 'condition' contains missing values"):
        edge_rust.run(counts(), si, CMP, {})


@pytest.mark.parametrize("bad", [None, np.nan, ""])
def test_missing_or_empty_control_is_rejected(bad):
    """P2: an NA (or "", which importFlexiData makes NA) control drops a design row in R."""
    si = sample_info(batch=["x", "y", "x", "y", "x", "y"])
    si["batch"] = si["batch"].astype(object)
    si.loc[1, "batch"] = bad
    params = {"control_cols": {"Column": "batch", "Type": "categorical"}}
    with pytest.raises(ValueError, match="nrow\\(design\\) disagrees with ncol\\(y\\)"):
        edge_rust.run(counts(), si, CMP, params)


def test_missing_counts_are_filled_with_zero():
    """P4: production coerces NA cells to 0 (edgeRStatsFun.R:63-73); NaN and pd.NA mean NA here."""
    c = counts()
    want = c.copy()
    want.iloc[3, 1] = 0.0
    want.iloc[7, 4] = 0.0
    ref = edge_rust.run(want, sample_info(), CMP, {})
    nan = c.copy()
    nan.iloc[3, 1] = np.nan
    nan.iloc[7, 4] = np.nan
    pd.testing.assert_frame_equal(edge_rust.run(nan, sample_info(), CMP, {}), ref)
    nullable = want.astype("Int64")
    nullable.iloc[3, 1] = pd.NA
    nullable.iloc[7, 4] = pd.NA
    pd.testing.assert_frame_equal(edge_rust.run(nullable, sample_info(), CMP, {}), ref)


def test_infinite_counts_are_still_rejected():
    c = counts()
    c.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        edge_rust.run(c, sample_info(), CMP, {})


def test_sample_column_order_does_not_change_results():
    """P5: production's dcast sorts the sample columns, so the caller's order must not matter."""
    c = counts(200)
    ref = edge_rust.run(c, sample_info(), CMP, {})
    rev = c[c.columns[::-1]]
    pd.testing.assert_frame_equal(edge_rust.run(rev, sample_info(), CMP, {}), ref, rtol=0, atol=0)


def test_duplicate_gene_ids_are_rejected():
    """P7: production's dcast cannot produce a repeated gene row."""
    c = counts()
    c = pd.concat([c, c.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate gene ids"):
        edge_rust.run(c, sample_info(), CMP, {})


def test_duplicate_sample_columns_are_rejected():
    """P7: a repeated count column would fit one sample twice."""
    c = counts()
    c = pd.concat([c, c[["s0"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate sample ids"):
        edge_rust.run(c, sample_info(), CMP, {})


def test_run_does_not_modify_sample_info():
    """P8: with no replicate column the index is the sample id, and run must not rewrite it."""
    c = counts()
    c.columns = list(range(6))
    si = pd.DataFrame({"condition": COND}, index=pd.Index(range(6)))
    before = si.copy()
    edge_rust.run(c, si, CMP, {})
    pd.testing.assert_frame_equal(si, before)


def test_non_ascii_digit_group_id_is_not_converted():
    """P9: "٣" passes str.isdigit() and became GroupId 3, colliding with the real gene 3."""
    c = counts()
    c.index = list(c.index[:-1]) + ["٣"]
    out = edge_rust.run(c, sample_info(), CMP, {})
    assert not pd.api.types.is_integer_dtype(out["GroupId"])
    assert out["GroupId"].tolist().count("3") == 1
