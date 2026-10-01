//! `topTable` (`.topTableT`, `.topTableF`) and `decideTests(method = "separate")`
//! (limma 3.66.0, `R/toptable.R`, `R/decidetests.R`).
//!
//! Rows come back in limma's sorted order, carrying the original gene index so the caller
//! can attach names.

// Index loops over parallel vectors mirror the R vector arithmetic they port.
#![allow(clippy::needless_range_loop)]

use crate::ebayes::{at, order_asc, order_desc, EBayes};
use crate::fit::MArrayLm;
use crate::linalg::p_adjust_bh;
use crate::nmath::qt;
use crate::{LimmaError, Result};

/// One row of `topTable(fit, coef = k, confint = TRUE, sort.by = "B", n = Inf)`.
#[derive(Debug, Clone, PartialEq)]
pub struct TopTableRow {
    pub gene: usize,
    pub log_fc: f64,
    pub ci_l: f64,
    pub ci_r: f64,
    pub ave_expr: f64,
    pub t: f64,
    pub p_value: f64,
    pub adj_p_value: f64,
    pub b: f64,
}

/// `topTable(fit, coef = coef, number = Inf, adjust.method = "BH", sort.by = "B",
/// confint = TRUE)` for one coefficient (0-based).
pub fn top_table_t(
    fit: &MArrayLm,
    eb: &EBayes,
    coef: usize,
    confint: f64,
) -> Result<Vec<TopTableRow>> {
    let ngenes = fit.ngenes;
    if coef >= fit.ncoef {
        return Err(LimmaError::Invalid("coef out of range".into()));
    }
    let off = coef * ngenes;
    let p: &[f64] = &eb.p_value[off..off + ngenes];
    let (adj, margin) = adj_p_and_ci_margin(fit, eb, coef, confint);
    let b: &[f64] = &eb.lods[off..off + ngenes];
    let o = order_desc(b);
    let mut rows = Vec::with_capacity(ngenes);
    for &g in &o {
        let k = off + g;
        let log_fc = fit.coefficients[k];
        rows.push(TopTableRow {
            gene: g,
            log_fc,
            ci_l: log_fc - margin[g],
            ci_r: log_fc + margin[g],
            ave_expr: fit.amean[g],
            t: eb.t[k],
            p_value: p[g],
            adj_p_value: adj[g],
            b: b[g],
        });
    }
    Ok(rows)
}

/// BH-adjusted p-values and CI half-widths for one coefficient (0-based), in gene order:
/// `p.adjust(p, "BH")` and `sqrt(s2.post) * stdev.unscaled * qt((1 + confint) / 2, df.total)`
/// (`toptable.R:224-229`). Shared by `top_table_t` and the Python binding.
pub fn adj_p_and_ci_margin(
    fit: &MArrayLm,
    eb: &EBayes,
    coef: usize,
    confint: f64,
) -> (Vec<f64>, Vec<f64>) {
    let ngenes = fit.ngenes;
    let off = coef * ngenes;
    let adj = p_adjust_bh(&eb.p_value[off..off + ngenes]);
    let alpha = (1.0 + confint) / 2.0;
    let margin = (0..ngenes)
        .map(|g| {
            eb.s2_post[g].sqrt()
                * fit.stdev_unscaled[off + g]
                * qt(alpha, eb.df_total[g], true, false)
        })
        .collect();
    (adj, margin)
}

/// One row of `topTable(fit, number = Inf)` (the F-test table).
#[derive(Debug, Clone, PartialEq)]
pub struct TopTableFRow {
    pub gene: usize,
    /// One log fold change per coefficient.
    pub coefs: Vec<f64>,
    pub ave_expr: f64,
    pub f: f64,
    pub p_value: f64,
    pub adj_p_value: f64,
}

/// `topTable(fit, coef = NULL, number = Inf, adjust.method = "BH", sort.by = "F")`.
pub fn top_table_f(fit: &MArrayLm, eb: &EBayes) -> Result<Vec<TopTableFRow>> {
    let (f, fp) = match (&eb.f, &eb.f_p_value) {
        (Some(f), Some(fp)) => (f, fp),
        _ => {
            return Err(LimmaError::Invalid(
                "F-statistics not available: design is not full rank".into(),
            ))
        }
    };
    let ngenes = fit.ngenes;
    let adj = p_adjust_bh(fp);
    let o = order_asc(fp);
    let mut rows = Vec::with_capacity(ngenes);
    for &g in &o {
        rows.push(TopTableFRow {
            gene: g,
            coefs: (0..fit.ncoef)
                .map(|j| fit.coefficients[j * ngenes + g])
                .collect(),
            ave_expr: fit.amean[g],
            f: f[g],
            p_value: fp[g],
            adj_p_value: adj[g],
        });
    }
    Ok(rows)
}

/// `decideTests(fit, method = "separate", adjust.method = "BH", p.value = p_value,
/// lfc = 0)`: an `ngenes x ncoef` column-major matrix of -1/0/1, `NaN` where the
/// p-value is `NA`.
pub fn decide_tests_separate(fit: &MArrayLm, eb: &EBayes, p_value: f64) -> Vec<f64> {
    let ngenes = fit.ngenes;
    let mut out = vec![0.0; ngenes * fit.ncoef];
    for j in 0..fit.ncoef {
        let off = j * ngenes;
        let adj = p_adjust_bh(&eb.p_value[off..off + ngenes]);
        for g in 0..ngenes {
            let k = off + g;
            out[k] = if adj[g].is_nan() {
                f64::NAN
            } else if adj[g] < p_value {
                fit.coefficients[k].signum()
            } else {
                0.0
            };
        }
    }
    out
}

/// Convenience: `df.total`-style recycled read, re-exported for callers.
pub fn recycled(v: &[f64], i: usize) -> f64 {
    at(v, i)
}
