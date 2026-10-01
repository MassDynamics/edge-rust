"""The production edgeR table around the Rust engine.

Mirrors what MDFlexiComparisons does after the engine call: the left join of the engine table to
every gene id (``merge(..., all.x = TRUE)``, which orders rows by GroupId as a string), the
integer GroupId, and for ANOVA runs ``.packageANOVAOutput`` (``R/runANOVA.R``): the omnibus
columns plus ``MaxLog2FCPair`` / ``MaxLog2FC``, every column as a string with NA written as "".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from edge_rust import _core

PAIR_STATS = ["Log2FC", "stat", "SE", "CILeft", "CIRight", "F", "PValue", "AdjPValue"]
ANOVA_COLUMNS = ["GroupId", "AveExpr", "PValue", "AdjPValue", "F", "MaxLog2FCPair", "MaxLog2FC"]


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


def _r_character(x: float) -> str:
    """``as.character`` of a double: 15 significant digits, NA as ""."""
    return "" if np.isnan(x) else f"{x:.15g}"


def run(
    counts: pd.DataFrame,
    sample_info: pd.DataFrame,
    comparisons: pd.DataFrame,
    params: dict,
    diagnostics: bool = False,
):
    """Run the edgeR QL engine and return the production output table.

    counts: genes x samples, index = gene ids, columns = sample ids; finite, non-negative.
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
    si.index = si.index.astype(str)
    sample_ids = [str(s) for s in counts.columns]
    missing = set(sample_ids) - set(si.index)
    if missing:
        raise ValueError(f"samples missing from sample_info: {sorted(missing)}")
    # sampleInfo[colnames(countMatrix), ]: the count matrix fixes the sample order.
    si = si.loc[sample_ids]
    controls = [
        (c, t, [str(v) for v in si[c]]) for c, t in _control_specs(params.get("control_cols"))
    ]
    enc_l = comparisons["encoded_left"] if "encoded_left" in comparisons else comparisons["left"]
    enc_r = comparisons["encoded_right"] if "encoded_right" in comparisons else comparisons["right"]
    cmps = [
        (str(a), str(b), str(c), str(d))
        for a, b, c, d in zip(comparisons["left"], comparisons["right"], enc_l, enc_r)
    ]
    gene_ids = [str(g) for g in counts.index]
    res = _core.edger_pipeline(
        np.ascontiguousarray(counts.to_numpy(dtype=np.float64)),
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
    table = pd.DataFrame(out)
    # merge(allDT, stats, by = "GroupId"): rows ordered by the character key (C collation).
    order = sorted(range(len(gene_ids)), key=lambda i: gene_ids[i].encode())
    table = table.iloc[order].reset_index(drop=True)

    if params.get("mode") == "anova":
        labels = [p["label"] for p in res["pairs"]]
        max_pair = [res["max_pair"][i] for i in order]
        max_fc = res["max_log2fc"][order]
        table = table[ANOVA_COLUMNS[:5]].copy()
        table["MaxLog2FCPair"] = ["" if k is None else labels[k] for k in max_pair]
        table["MaxLog2FC"] = max_fc
        for c in ["AveExpr", "PValue", "AdjPValue", "F", "MaxLog2FC"]:
            table[c] = [_r_character(v) for v in table[c]]

    # type_convert(out, "integer", "GroupId"), when every id is an integer.
    if all(g.lstrip("-").isdigit() for g in table["GroupId"]):
        table["GroupId"] = table["GroupId"].astype(np.int64)
    if params.get("mode") == "anova":
        table["GroupId"] = table["GroupId"].astype(str)
    if diagnostics:
        return table, _diag(res, gene_ids, sample_ids)
    return table


def _diag(res: dict, gene_ids: list[str], sample_ids: list[str]) -> dict:
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
    if all(g.lstrip("-").isdigit() for g in ids):
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
