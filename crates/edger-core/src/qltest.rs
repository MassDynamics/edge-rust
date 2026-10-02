//! `glmQLFTest` for one contrast matrix, as the reference's `ref_ql_test` writes it out from
//! edgeR 4.8.2 `R/glmQLFTest.R` and the contrast reparametrisation of `R/glmLRT.R`: the design is
//! rotated by the complete Q of `qr(contrast)`, the null model drops the first `rank` columns,
//! and the F statistic is the deviance difference over the posterior QL dispersion.

use crate::glm::glm_fit;
use crate::ql::QlFit;
use rnum::linalg::p_adjust_bh;
use rnum::linpack::qr_decompose;
use rnum::nmath::pf;

/// One QL F-test: per-gene null deviance, LR, F, df.total, p-value, BH FDR and logFC
/// (`ngenes x ncon`, gene-major, log2).
#[derive(Debug, Clone)]
pub struct QlTest {
    pub deviance_null: Vec<f64>,
    pub lr: Vec<f64>,
    pub df_test: usize,
    pub f: Vec<f64>,
    pub df_total: Vec<f64>,
    pub pvalue: Vec<f64>,
    pub fdr: Vec<f64>,
    pub ncon: usize,
    pub logfc: Vec<f64>,
}

/// `glmQLFTest(fit, contrast = contrast)` with `contrast` column-major `p x ncon`.
#[allow(clippy::too_many_arguments)]
pub fn glm_ql_ftest(
    y: &[f64],
    nlib: usize,
    x: &[f64],
    p: usize,
    offset: &[f64],
    ql: &QlFit,
    contrast: &[f64],
    ncon: usize,
) -> rnum::Result<QlTest> {
    let ng = y.len() / nlib;
    // logFC <- (fit$coefficients %*% contrast) / log(2)
    let mut logfc = vec![0.0; ng * ncon];
    for g in 0..ng {
        for k in 0..ncon {
            let mut s = 0.0;
            for l in 0..p {
                s += ql.coefficients[g * p + l] * contrast[k * p + l];
            }
            logfc[g * ncon + k] = s / std::f64::consts::LN_2;
        }
    }
    // Q <- qr.Q(qr(contrast), complete = TRUE, Dvec); design0 <- (design %*% Q)[, -(1:rank)]
    let qrc = qr_decompose(contrast, p, ncon, 1e-7);
    let r = qrc.rank;
    let p0 = p - r;
    let mut design0 = vec![0.0; nlib * p0];
    let mut e = vec![0.0; p];
    for k in 0..p0 {
        e.iter_mut().for_each(|v| *v = 0.0);
        e[r + k] = 1.0;
        let q = qrc.qy(&e);
        for l in 0..p {
            let t = q[l];
            for i in 0..nlib {
                design0[k * nlib + i] += t * x[l * nlib + i];
            }
        }
    }
    let disp = vec![ql.dispersion / ql.ave_ql_dispersion; ng];
    let null = glm_fit(y, nlib, &design0, p0, offset, &disp, None)?;
    let df_res_sum = (ng * (nlib - p)) as f64;
    let mut lr = vec![0.0; ng];
    let mut f = vec![0.0; ng];
    let mut df_total = vec![0.0; ng];
    let mut pvalue = vec![0.0; ng];
    for g in 0..ng {
        lr[g] = null.deviance[g] - ql.deviance[g];
        f[g] = lr[g] / r as f64 / ql.s2_post[g];
        let dt = ql.df_prior[g] + ql.df_residual_adj[g];
        df_total[g] = if dt.is_nan() { dt } else { dt.min(df_res_sum) };
        pvalue[g] = pf(f[g], r as f64, df_total[g], false, false);
    }
    let fdr = p_adjust_bh(&pvalue);
    Ok(QlTest {
        deviance_null: null.deviance,
        lr,
        df_test: r,
        f,
        df_total,
        pvalue,
        fdr,
        ncon,
        logfc,
    })
}
