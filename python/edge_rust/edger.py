"""The production edgeR table around the Rust engine.

Mirrors what MDFlexiComparisons does after the engine call: the left join of the engine table to
every gene id, the integer GroupId, and for ANOVA runs ``.packageANOVAOutput``
(``R/runANOVA.R``): the omnibus columns plus ``MaxLog2FCPair`` / ``MaxLog2FC``, every column as
R's ``as.character`` with NA written as "". Rows stay in the input row order: production's final
table is ``featuresMetadata %>% left_join(stats)`` (``createResultsSummarizedExperiment.R``), so
it follows the features metadata for pairwise and ANOVA alike, and the caller passes the counts
in that order.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from edge_rust import _core

PAIR_STATS = ["Log2FC", "stat", "SE", "CILeft", "CIRight", "F", "PValue", "AdjPValue"]
ANOVA_COLUMNS = ["GroupId", "AveExpr", "PValue", "AdjPValue", "F", "MaxLog2FCPair", "MaxLog2FC"]

log = logging.getLogger(__name__)


def _control_specs(control_cols) -> list[tuple[str, str]]:
    """``control_cols`` as ``{Column, Type}`` (scalars or lists, as the job params carry it),
    a list of such dicts, or None."""
    if control_cols is None:
        return []
    if isinstance(control_cols, dict):
        cols, types = control_cols["Column"], control_cols["Type"]
        if isinstance(cols, str):
            cols, types = [cols], [types]
        return list(zip(cols, types))
    return [(c["Column"], c["Type"]) for c in control_cols]


# _is_int_id and _group_id_order have a twin in deseq2-rust's deseq2.py; change both together.
def _is_int_id(g: str) -> bool:
    """Whether ``type_convert`` would read the GroupId as an integer."""
    return g.isascii() and g.removeprefix("-").isdigit()


def _group_id_order(ids: list[str]) -> list[int]:
    """Positions of ``ids`` in GroupId order: numeric when every id is an integer, else bytewise."""
    if all(_is_int_id(g) for g in ids):
        return sorted(range(len(ids)), key=lambda i: int(ids[i]))
    return sorted(range(len(ids)), key=lambda i: ids[i].encode())


def _r_character(x) -> list[str]:
    """R's ``as.character`` of each double (``1e5`` is ``"1e+05"``), NA as "".

    The same function lives in deseq2-rust (``deseq2_rust/deseq2.py``); keep the two identical.
    """
    x = np.asarray(x, dtype=np.float64)
    out = _core.r_as_character(x.tolist())
    return ["" if na else s for na, s in zip(np.isnan(x), out)]


def run(
    counts: pd.DataFrame,
    sample_info: pd.DataFrame,
    comparisons: pd.DataFrame,
    params: dict,
    diagnostics: bool = False,
):
    """Run the edgeR QL engine and return the production output table.

    counts: genes x samples, index = gene ids, columns = sample ids; finite, non-negative. The
        output rows follow this row order, which should be the features metadata order.
    sample_info: one row per sample, sample ids in a ``replicate`` column or the index, the
        condition column and the control columns.
    comparisons: ``left``, ``right`` (output labels) and optionally ``encoded_left``,
        ``encoded_right`` (the condition values in ``sample_info``; default: left / right).
    params: ``condition_col`` (default "condition"), ``control_cols`` (``{Column, Type}``),
        ``edger_norm_method`` (default "TMM"), ``entity_type`` (default "gene"),
        ``mode`` ("discovery" or "anova").

    diagnostics: also return the fit's intermediates, as ``(table, diag)``; see ``_diag``.

    Raises ValueError with the production message when the engine refuses the input.
    """
    cc = params.get("condition_col", "condition")
    si = sample_info.set_index("replicate") if "replicate" in sample_info.columns else sample_info
    si = si.set_axis(si.index.astype(str))  # a copy: the caller's frame is left alone
    # Ids become strings below, so 1001 and "1001" are the same id.
    if pd.Index([str(g) for g in counts.index]).duplicated().any():
        raise ValueError("counts has duplicate gene ids")
    if pd.Index([str(s) for s in counts.columns]).duplicated().any():
        raise ValueError("counts has duplicate sample ids")
    # dcast orders the sample columns by id (C collation); the fit depends on that order.
    counts = counts[sorted(counts.columns, key=lambda s: str(s).encode())]
    sample_ids = [str(s) for s in counts.columns]
    missing = set(sample_ids) - set(si.index)
    if missing:
        raise ValueError(f"samples missing from sample_info: {sorted(missing)}")
    # sampleInfo[colnames(countMatrix), ]: the count matrix fixes the sample order.
    si = si.loc[sample_ids]
    if si[cc].isna().any():
        raise ValueError(
            f"Condition column '{cc}' contains missing values. "
            "Fix the sample metadata before running DE."
        )
    controls = []
    for c, t in _control_specs(params.get("control_cols")):
        # importFlexiData makes "" NA, and model.matrix drops the NA rows.
        if si[c].isna().any() or (si[c].astype(str) == "").any():
            raise ValueError("nrow(design) disagrees with ncol(y)")
        controls.append((c, t, [str(v) for v in si[c]]))
    mat = counts.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    na = np.isnan(mat)
    if na.any():
        # Production fills plain NA with 0 and stops on NaN and Inf (edgeRStatsFun.R:56-72).
        # pandas cannot tell NaN from NA, and a cell missing after a pivot arrives as NaN, so
        # every NaN is filled with 0 here: parity holds for NA only, and a literal NaN count
        # runs where production stops. Inf still stops in the engine.
        log.info("edgeR: coercing %d NA cell(s) in the count matrix to 0", int(na.sum()))
        mat[na] = 0.0
    enc_l = comparisons["encoded_left"] if "encoded_left" in comparisons else comparisons["left"]
    enc_r = comparisons["encoded_right"] if "encoded_right" in comparisons else comparisons["right"]
    cmps = [
        (str(a), str(b), str(c), str(d))
        for a, b, c, d in zip(comparisons["left"], comparisons["right"], enc_l, enc_r)
    ]
    # .buildCountMatrixFromLongDT (dcast) orders the rows by GroupId and production fits in that
    # order whatever the metadata order. The QL prior depends on it at about 1e-10, which the CIs
    # and F carry to about 1e-7 (review deseq2 r4, SE4-M1); when a one-group fit does not
    # converge, edgeR reuses the last written gene's value, and then AveExpr and df.prior move by
    # about 3e-8 with the order (review overnight r2, Minor 1).
    input_ids = [str(g) for g in counts.index]
    fit_order = _group_id_order(input_ids)
    gene_ids = [input_ids[i] for i in fit_order]
    res = _core.edger_pipeline(
        np.ascontiguousarray(mat[fit_order]),
        gene_ids,
        sample_ids,
        cc,
        [str(v) for v in si[cc]],
        controls,
        cmps,
        norm_method=params.get("edger_norm_method", "TMM"),
        entity_type=params.get("entity_type", "gene"),
        diagnostics=diagnostics,
    )

    out = {
        "GroupId": gene_ids,
        "AveExpr": res["ave_expr"],
        "F": res["F"],
        "PValue": res["PValue"],
        "AdjPValue": res["AdjPValue"],
    }
    for p in res["pairs"]:
        for s in PAIR_STATS:
            out[f"{s} {p['label']}"] = p[s]
    # left_join onto the features metadata: the input order, in both modes.
    back = np.argsort(fit_order)
    table = pd.DataFrame(out).iloc[back].reset_index(drop=True)

    if params.get("mode") == "anova":
        labels = [p["label"] for p in res["pairs"]]
        table = table[ANOVA_COLUMNS[:5]].copy()
        max_pair = [res["max_pair"][i] for i in back]
        table["MaxLog2FCPair"] = ["" if k is None else labels[k] for k in max_pair]
        table["MaxLog2FC"] = np.asarray(res["max_log2fc"])[back]
        for c in ["AveExpr", "PValue", "AdjPValue", "F", "MaxLog2FC"]:
            table[c] = _r_character(table[c])

    # type_convert(out, "integer", "GroupId"), when every id is an integer.
    if all(_is_int_id(g) for g in table["GroupId"]):
        table["GroupId"] = table["GroupId"].astype(np.int64)
    if params.get("mode") == "anova":
        table["GroupId"] = table["GroupId"].astype(str)
    if diagnostics:
        return table, _diag(res, gene_ids, sample_ids, fit_order)
    return table


def _diag(res: dict, gene_ids: list[str], sample_ids: list[str], fit_order: list[int]) -> dict:
    """The fit's intermediates, with the column names of the R reference's ``r_edger.diag/``.

    ``samples``: replicate, lib_size (after filterByExpr), norm_factor, eff_lib_size.
    ``genes`` (kept genes, input order): ave_log_cpm, s2_post, s2_prior, df_residual_adj,
    df_residual, df_prior, ``coef_<col>`` (unshrunk) and ``coefshr_<col>`` (prior count 0.125),
    natural log scale. ``design``: replicate and the design columns. ``disp``: trended_disp,
    tagwise_disp (estimateDisp). ``fitted``: fitted means, kept genes x samples. ``scalars``:
    fit_dispersion, ave_ql_dispersion, df_residual_total, top_proportion, design_columns,
    common_disp, disp_prior_df.
    """
    d = res["diag"]
    cols = list(res["design_cols"])
    ids = [gene_ids[i] for i in d["kept_idx"]]
    if all(_is_int_id(g) for g in ids):
        ids = [int(g) for g in ids]
    samples = pd.DataFrame(
        {"replicate": sample_ids, "lib_size": d["lib_size"], "norm_factor": d["norm_factor"]}
    )
    samples["eff_lib_size"] = samples["lib_size"] * samples["norm_factor"]
    nk = len(ids)
    genes = pd.DataFrame(
        {
            "id": ids,
            "ave_log_cpm": d["ave_log_cpm"],
            "s2_post": d["s2_post"],
            "s2_prior": d["s2_prior"],
            "df_residual_adj": d["df_residual_adj"],
            "df_residual": np.full(nk, d["df_residual"]),
            "df_prior": d["df_prior"],
        }
    )
    for k, c in enumerate(cols):
        genes[f"coef_{c}"] = d["unshrunk_coefficients"][:, k]
    for k, c in enumerate(cols):
        genes[f"coefshr_{c}"] = d["coefficients"][:, k]
    design = pd.DataFrame(d["design"].T, columns=cols)
    design.insert(0, "replicate", sample_ids)
    disp = pd.DataFrame(
        {"id": ids, "trended_disp": d["trended_disp"], "tagwise_disp": d["tagwise_disp"]}
    )
    fitted = pd.DataFrame(d["fitted"], index=pd.Index(ids, name="id"), columns=sample_ids)
    # The fit runs in GroupId order; report the kept genes in input order.
    back = np.argsort([fit_order[i] for i in d["kept_idx"]])
    genes = genes.iloc[back].reset_index(drop=True)
    disp = disp.iloc[back].reset_index(drop=True)
    fitted = fitted.iloc[back]
    dft = d["df_residual"] * nk
    scalars = {
        "fit_dispersion": d["fit_dispersion"],
        "ave_ql_dispersion": d["ave_ql_dispersion"],
        "df_residual_total": int(dft) if float(dft).is_integer() else dft,
        "top_proportion": None,  # glmQLFit(legacy = FALSE) leaves it NULL
        "design_columns": cols,
        "common_disp": d["common_disp"],
        "disp_prior_df": d["disp_prior_df"],
    }
    return {
        "samples": samples,
        "genes": genes,
        "design": design,
        "disp": disp,
        "fitted": fitted,
        "scalars": scalars,
    }
