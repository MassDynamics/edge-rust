//! The edgeR engine as MDFlexiComparisons runs it (`R/edgeRStatsFun.R`, `.fitEdgeRModel` and
//! `.edgeRStatsFun`), in the order `ref_edger` in the count corpus reference writes it out:
//! design `~ 0 + condition + controls` (`model.matrix`), `checkMatrixRank`
//! (`R/generateContrastInfos.R`), `filterByExpr`, `DGEList(keep.lib.sizes = FALSE)`,
//! `calcNormFactors`, `estimateDisp`, `glmQLFit`, then `glmQLFTest` for the omnibus contrast
//! (every level against the first) and for each comparison. The output is the engine table left
//! joined to every input gene; genes `filterByExpr` drops get NA statistics.
//!
//! The ANOVA shaping (`R/runANOVA.R`, `.packageANOVAOutput`) is a string formatting step and is
//! left to the caller; [`max_abs_log2fc`] supplies the one computed piece of it.

use crate::disp::{estimate_disp, Disp};
use crate::filter::filter_by_expr;
use crate::norm::calc_norm_factors;
use crate::ql::{glm_ql_fit, QlFit};
use crate::qltest::glm_ql_ftest;
use rnum::glibm::ln;
use rnum::linpack::qr_decompose;
use rnum::nmath::qt;
use rnum::{LimmaError, Result};

/// How a control column enters the design: categorical columns are treatment coded against
/// their first level, numerical ones enter as a single column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlKind {
    Categorical,
    Numerical,
}

/// One control column, one value per sample (in sample order).
#[derive(Debug, Clone)]
pub struct Control {
    pub name: String,
    pub kind: ControlKind,
    pub values: Vec<String>,
}

/// One comparison: `left - right`. `encoded_*` are the condition values as they appear in the
/// sample info; `left`/`right` are the labels used in the output column names.
#[derive(Debug, Clone)]
pub struct Comparison {
    pub left: String,
    pub right: String,
    pub encoded_left: String,
    pub encoded_right: String,
}

/// Engine inputs. `counts` is gene-major (`gene_ids.len() x sample_ids.len()`), columns in
/// sample order.
#[derive(Debug, Clone)]
pub struct EdgerInput {
    pub gene_ids: Vec<String>,
    pub sample_ids: Vec<String>,
    pub counts: Vec<f64>,
    pub condition_col: String,
    pub condition: Vec<String>,
    pub controls: Vec<Control>,
    pub comparisons: Vec<Comparison>,
    /// `TMM`, `RLE`, `upperquartile` or `none`.
    pub norm_method: String,
    /// Must be `gene`: edgeR is only offered for count data.
    pub entity_type: String,
}

/// Per-comparison output columns, each over every input gene.
#[derive(Debug, Clone)]
pub struct PairResult {
    /// `"<left> - <right>"`, the suffix of the output column names.
    pub label: String,
    pub log2fc: Vec<f64>,
    pub stat: Vec<f64>,
    pub se: Vec<f64>,
    pub ci_left: Vec<f64>,
    pub ci_right: Vec<f64>,
    pub f: Vec<f64>,
    pub pvalue: Vec<f64>,
    pub adj_pvalue: Vec<f64>,
}

/// The engine table, one row per input gene in input order.
#[derive(Debug, Clone)]
pub struct EdgerOutput {
    pub gene_ids: Vec<String>,
    pub ave_expr: Vec<f64>,
    pub f: Vec<f64>,
    pub pvalue: Vec<f64>,
    pub adj_pvalue: Vec<f64>,
    pub pairs: Vec<PairResult>,
    /// The design (column-major `nlib x p`) and its column names.
    pub design: Vec<f64>,
    pub design_cols: Vec<String>,
    /// `filterByExpr` keep flags, per input gene.
    pub kept: Vec<bool>,
}

/// What the fit leaves behind besides the table, for diagnostics: the kept genes' indices into
/// the input, the library sizes after filtering and the norm factors (per sample), and the
/// `estimateDisp` and `glmQLFit` results over the kept genes.
#[derive(Debug, Clone)]
pub struct EdgerDiag {
    pub kept_idx: Vec<usize>,
    pub lib_size: Vec<f64>,
    pub norm_factors: Vec<f64>,
    pub disp: Disp,
    pub ql: QlFit,
}

fn err(msg: impl Into<String>) -> LimmaError {
    LimmaError::Invalid(msg.into())
}

/// Sorted unique values. R's `factor()` sorts with the session collation, which is C in the
/// production image, so a byte-wise sort matches it.
fn levels(v: &[String]) -> Vec<String> {
    let mut l = v.to_vec();
    l.sort();
    l.dedup();
    l
}

/// `model.matrix(~ 0 + cc + ctl1 + ...)`: full indicator coding for the condition, treatment
/// coding for categorical controls, numerical controls as is. Returns the column-major design,
/// its column names and the condition levels.
pub fn build_design(input: &EdgerInput) -> Result<(Vec<f64>, Vec<String>, Vec<String>)> {
    let nlib = input.sample_ids.len();
    let mut design = Vec::new();
    let mut cols = Vec::new();
    let lv = levels(&input.condition);
    for l in &lv {
        design.extend(
            input
                .condition
                .iter()
                .map(|c| if c == l { 1.0 } else { 0.0 }),
        );
        cols.push(format!("{}{}", input.condition_col, l));
    }
    for c in &input.controls {
        if c.values.len() != nlib {
            return Err(err(format!(
                "control column '{}' has {} values for {nlib} samples",
                c.name,
                c.values.len()
            )));
        }
        match c.kind {
            ControlKind::Categorical => {
                let clv = levels(&c.values);
                if clv.len() < 2 {
                    return Err(err(
                        "contrasts can be applied only to factors with 2 or more levels",
                    ));
                }
                for l in clv.iter().skip(1) {
                    design.extend(c.values.iter().map(|v| if v == l { 1.0 } else { 0.0 }));
                    cols.push(format!("{}{}", c.name, l));
                }
            }
            ControlKind::Numerical => {
                for v in &c.values {
                    let x: f64 = v.trim().parse().map_err(|_| {
                        err(format!(
                            "control column '{}' is numerical but has value '{v}'",
                            c.name
                        ))
                    })?;
                    // model.matrix drops the NA row, and filterByExpr then stops.
                    if x.is_nan() {
                        return Err(err("nrow(design) disagrees with ncol(y)"));
                    }
                    design.push(x);
                }
                cols.push(c.name.clone());
            }
        }
    }
    Ok((design, cols, lv))
}

/// `left - right` as a design-length contrast column.
fn pair_contrast(cols: &[String], cc: &str, left: &str, right: &str) -> Result<Vec<f64>> {
    let find = |l: &str| {
        let name = format!("{cc}{l}");
        cols.iter()
            .position(|c| *c == name)
            .ok_or_else(|| err(format!("comparison level '{l}' is not a level of '{cc}'")))
    };
    let mut c = vec![0.0; cols.len()];
    c[find(left)?] += 1.0;
    c[find(right)?] -= 1.0;
    if c.iter().all(|&v| v == 0.0) {
        return Err(err("contrasts are all zero"));
    }
    Ok(c)
}

/// Spread a per-kept-gene vector to every input gene, NA where the gene was filtered out.
fn spread(v: &[f64], kept_idx: &[usize], n: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; n];
    for (k, &g) in kept_idx.iter().enumerate() {
        out[g] = v[k];
    }
    out
}

/// R's `sign()`: 0 at 0, NA at NA.
fn r_sign(x: f64) -> f64 {
    if x.is_nan() || x == 0.0 {
        x
    } else {
        x.signum()
    }
}

/// Run the engine. Errors carry the production messages (without the `md_error` markers).
pub fn run_edger(input: &EdgerInput) -> Result<EdgerOutput> {
    run_edger_diag(input).map(|(out, _)| out)
}

/// [`run_edger`], also returning the intermediate fit ([`EdgerDiag`]).
pub fn run_edger_diag(input: &EdgerInput) -> Result<(EdgerOutput, EdgerDiag)> {
    let ng = input.gene_ids.len();
    let nlib = input.sample_ids.len();
    if input.entity_type != "gene" {
        return Err(err(
            "de_method 'edgeR' is only supported for gene entity type (count data). Use de_method = 'limma' for protein, peptide, metabolite, or PTM data.",
        ));
    }
    if input.counts.len() != ng * nlib || input.condition.len() != nlib {
        return Err(err(
            "counts, gene ids, sample ids and condition values disagree in size",
        ));
    }
    let nonfinite = input.counts.iter().filter(|v| !v.is_finite()).count();
    if nonfinite > 0 {
        return Err(err(format!(
            "Count matrix contains {nonfinite} non-finite (Inf / -Inf / NaN) values. This indicates an upstream data-integrity bug; refusing to silently coerce to zero."
        )));
    }
    if input.counts.iter().any(|&v| v < 0.0) {
        return Err(err(
            "The data contains negative intensities. Please check if your data was log-transformed before starting the analysis.",
        ));
    }
    // .buildCountMatrixFromLongDT: refuse non-integer counts, then as.integer(round(x)).
    if input.counts.iter().any(|&v| (v - v.round()).abs() > 1e-6) {
        let mut sums = vec![0.0; nlib];
        for row in input.counts.chunks(nlib) {
            for (s, v) in sums.iter_mut().zip(row) {
                *s += v;
            }
        }
        if sums.iter().all(|s| (s - 1e6).abs() < 1e3) {
            return Err(err(
                "Input data appears to be CPM/TPM-normalised (non-integer values, per-sample sums ~1e6). edgeR and DESeq2 require raw integer counts. Use de_method = 'limma' for pre-normalised data, or re-upload raw counts.",
            ));
        }
        return Err(err(
            "Non-integer values detected in the count column. edgeR and DESeq2 require raw integer counts as input. Use de_method = 'limma' for pre-normalised or continuous data.",
        ));
    }
    let counts: Vec<f64> = input.counts.iter().map(|v| v.round()).collect();

    let (design, cols, lv) = build_design(input)?;
    let p = cols.len();
    // qr() is a .Fortran call, which refuses a non-finite design.
    if design.iter().any(|v| !v.is_finite()) {
        return Err(err("NA/NaN/Inf in foreign function call (arg 1)"));
    }
    // R already fails in model.matrix for a one-level factor; the engine says why.
    if lv.len() < 2 {
        return Err(err("edgeR requires at least 2 condition levels."));
    }
    // checkMatrixRank: qr(designMat)$rank < ncol(designMat).
    if qr_decompose(&design, nlib, p, 1e-7).rank < p {
        let mut preds = vec![input.condition_col.clone()];
        preds.extend(input.controls.iter().map(|c| c.name.clone()));
        return Err(err(format!(
            "Model creation failed because one or more variables '{}' are perfectly collinear. Each variable should represent unique information.",
            preds.join(", ")
        )));
    }

    // DGEList: as.integer() turned counts above .Machine$integer.max into NA.
    if counts.iter().any(|&v| v > i32::MAX as f64) {
        return Err(err("NA counts not allowed"));
    }

    let fb = filter_by_expr(&counts, nlib, &design, p)?;
    let kept_idx: Vec<usize> = (0..ng).filter(|&g| fb.keep[g]).collect();
    if kept_idx.is_empty() {
        return Err(err(
            "filterByExpr removed every gene. Check input count matrix and sample-size per condition.",
        ));
    }
    let mut y = Vec::with_capacity(kept_idx.len() * nlib);
    for &g in &kept_idx {
        y.extend_from_slice(&counts[g * nlib..(g + 1) * nlib]);
    }
    // DGEList(...)[keep, , keep.lib.sizes = FALSE]: library sizes of the kept genes.
    let mut lib = vec![0.0; nlib];
    for row in y.chunks(nlib) {
        for (l, v) in lib.iter_mut().zip(row) {
            *l += v;
        }
    }
    let nf = calc_norm_factors(&y, nlib, &lib, &input.norm_method)?;
    // A sample whose counts all sit in filtered genes: TMM stops on the NaN f75 median
    // (calcNormFactors.R:63), the other methods on the non-finite offset in estimateDisp.
    if let Some(j) = lib.iter().position(|&l| l <= 0.0) {
        let r = if input.norm_method == "TMM" {
            "missing value where TRUE/FALSE needed"
        } else {
            "offsets must be finite values"
        };
        return Err(err(format!(
            "{r} (sample '{}' has no counts in the genes filterByExpr kept)",
            input.sample_ids[j]
        )));
    }
    let lib_eff: Vec<f64> = lib
        .iter()
        .zip(&nf.norm_factors)
        .map(|(a, b)| a * b)
        .collect();
    let offset: Vec<f64> = lib_eff.iter().map(|v| ln(*v)).collect();
    // A tiny library can still give a zero or NaN norm factor. R stops when `min(offset)` is
    // not finite (makeCompressedMatrix.R:360-361); its min returns NaN if any offset is NaN.
    let min_offset = offset.iter().fold(f64::INFINITY, |m, &v| {
        if v.is_nan() || m.is_nan() {
            f64::NAN
        } else {
            m.min(v)
        }
    });
    if !min_offset.is_finite() {
        let j = offset.iter().position(|v| !v.is_finite()).unwrap_or(0);
        return Err(err(format!(
            "offsets must be finite values (sample '{}' has a non-finite offset under {})",
            input.sample_ids[j], input.norm_method
        )));
    }

    // No residual df: estimateDisp returns NA dispersions and glmQLFit then fails in an `if`.
    if p >= nlib {
        return Err(err(format!(
            "missing value where TRUE/FALSE needed (the design has {p} columns for {nlib} samples, so there are no residual degrees of freedom)"
        )));
    }

    let disp = estimate_disp(&y, nlib, &design, p, &lib_eff, &offset)?;
    let ql = glm_ql_fit(
        &y,
        nlib,
        &design,
        p,
        &offset,
        &disp.ave_logcpm,
        &disp.trended,
    )?;

    // Omnibus: cc<lv_k> - cc<lv_1> for k = 2..K, column-major p x (K - 1).
    let ncon = lv.len() - 1;
    let mut omni = Vec::with_capacity(p * ncon);
    for l in &lv[1..] {
        omni.extend(pair_contrast(&cols, &input.condition_col, l, &lv[0])?);
    }
    let om = glm_ql_ftest(&y, nlib, &design, p, &offset, &ql, &omni, ncon)?;

    // CI df as production computes it: df.prior + df.residual (unadjusted).
    let df_ci: Vec<f64> = ql.df_prior.iter().map(|d| d + ql.df_residual).collect();
    let nk = kept_idx.len();
    let mut pairs = Vec::with_capacity(input.comparisons.len());
    for cmp in &input.comparisons {
        let con = pair_contrast(
            &cols,
            &input.condition_col,
            &cmp.encoded_left,
            &cmp.encoded_right,
        )?;
        let t = glm_ql_ftest(&y, nlib, &design, p, &offset, &ql, &con, 1)?;
        let mut stat = vec![f64::NAN; nk];
        let mut se = vec![f64::NAN; nk];
        let mut lo = vec![f64::NAN; nk];
        let mut hi = vec![f64::NAN; nk];
        for g in 0..nk {
            let (f, lfc) = (t.f[g], t.logfc[g]);
            if f.is_finite() && f > 0.0 {
                stat[g] = f.sqrt() * r_sign(lfc);
                se[g] = lfc / stat[g];
                let half = qt(0.975, df_ci[g], true, false) * se[g];
                lo[g] = lfc - half;
                hi[g] = lfc + half;
            }
        }
        pairs.push(PairResult {
            label: format!("{} - {}", cmp.left, cmp.right),
            log2fc: spread(&t.logfc, &kept_idx, ng),
            stat: spread(&stat, &kept_idx, ng),
            se: spread(&se, &kept_idx, ng),
            ci_left: spread(&lo, &kept_idx, ng),
            ci_right: spread(&hi, &kept_idx, ng),
            f: spread(&t.f, &kept_idx, ng),
            pvalue: spread(&t.pvalue, &kept_idx, ng),
            adj_pvalue: spread(&t.fdr, &kept_idx, ng),
        });
    }

    let out = EdgerOutput {
        gene_ids: input.gene_ids.clone(),
        ave_expr: spread(&disp.ave_logcpm, &kept_idx, ng),
        f: spread(&om.f, &kept_idx, ng),
        pvalue: spread(&om.pvalue, &kept_idx, ng),
        adj_pvalue: spread(&om.fdr, &kept_idx, ng),
        pairs,
        design,
        design_cols: cols,
        kept: fb.keep,
    };
    let diag = EdgerDiag {
        kept_idx,
        lib_size: lib,
        norm_factors: nf.norm_factors,
        disp,
        ql,
    };
    Ok((out, diag))
}

/// `extractOutputANOVA`'s `get_max_fc` (`R/runANOVA.R`): per gene, the comparison with the
/// largest |Log2FC| (first on ties, NA ignored) and its Log2FC; `None` / NA when every Log2FC
/// is NA.
pub fn max_abs_log2fc(pairs: &[PairResult]) -> (Vec<Option<usize>>, Vec<f64>) {
    let ng = pairs.first().map_or(0, |p| p.log2fc.len());
    let mut idx = vec![None; ng];
    let mut val = vec![f64::NAN; ng];
    for g in 0..ng {
        let mut best: Option<usize> = None;
        for (k, p) in pairs.iter().enumerate() {
            let v = p.log2fc[g];
            if v.is_nan() {
                continue;
            }
            if best.is_none_or(|b| v.abs() > pairs[b].log2fc[g].abs()) {
                best = Some(k);
            }
        }
        if let Some(b) = best {
            idx[g] = Some(b);
            val[g] = pairs[b].log2fc[g];
        }
    }
    (idx, val)
}
