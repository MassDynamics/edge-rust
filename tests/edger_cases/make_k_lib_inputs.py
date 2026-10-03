"""Rebuilds the inputs of the k_lib*_none cases: one sample with a library of 1 to 5 counts.

The base matrix is review r2's (300 genes, 9 samples, A / B / C, seed 7) and the edits are review
r3's (sample s5 emptied but for gene 151), so the RNG stream and every byte match the inputs those
reviews found the cases with (review overnight r2, SE2-m1). ``k_lib1_reid_none`` is
``k_lib1_one_gene_none`` with gene 151 renamed 1000, so that the stalled gene's predecessor in
numeric GroupId order (300) differs from the one in bytewise order (none: it sorts first), review
overnight r2, Minor 1. ``k_alc_none`` (sample s7 emptied but for genes 89 and 211) makes the
port's aveLogCPM fit stall on a gene that is not the first, found by a port-side search (review
overnight r2, Nit 2); its references come from generate.R. Run from this directory:

    uv run python make_k_lib_inputs.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).parent
rng = np.random.default_rng(7)
ng, samples = 300, [f"s{i}" for i in range(1, 10)]
cond = ["A"] * 3 + ["B"] * 3 + ["C"] * 3
mu = 2 ** rng.uniform(2, 11, ng)
fc = np.ones((ng, 9))
fc[rng.choice(ng, 30, replace=False)[:, None], np.arange(3, 6)] = 4
phi = 0.1
b = rng.negative_binomial(1 / phi, 1 / (1 + mu[:, None] * fc * phi)).astype(np.int64)
b[:20, :] = rng.integers(0, 3, (20, 9))  # low genes that filterByExpr drops


def write(name, counts, ids=None):
    d = root / name
    d.mkdir(exist_ok=True)
    cm = pd.DataFrame(counts, columns=samples).astype(object)
    cm.insert(0, "id", ids or [str(i + 1) for i in range(len(cm))])
    cm.to_csv(d / "input_counts.csv", index=False)
    si = pd.DataFrame(
        {
            "replicate": samples,
            "condition": cond,
            "age": ["31", "45", "52", "38", "61", "29", "44", "57", "50"],
            "batch": ["b1", "b2", "b1", "b2", "b1", "b2", "b1", "b2", "b1"],
        }
    )
    si.to_csv(d / "input_sample_info.csv", index=False)
    cmp = [("B", "A"), ("C", "A"), ("C", "B")]
    pd.DataFrame(cmp, columns=["left", "right"]).to_csv(d / "input_comparisons.csv", index=False)
    params = {"condition_col": "condition", "control_cols": None, "edger_norm_method": "none"}
    (d / "params.json").write_text(json.dumps(params) + "\n")


for k in (1, 2, 5):
    t = b.copy()
    t[:, 4] = 0
    t[150, 4] = k
    write(f"k_lib{k}_one_gene_none", t)
t = b.copy()
t[:, 4] = 0
t[150, 4] = 1
t[150, 0] = 0
write("k_lib1_rle_nogene_none", t)
t = b.copy()
t[:, 4] = 0
t[150, 4] = 1
write("k_lib1_reid_none", t, ids=[str(i + 1) if i != 150 else "1000" for i in range(ng)])
t = b.copy()
t[:, 6] = 0
t[88, 6] = 29
t[210, 6] = 22
write("k_alc_none", t)
