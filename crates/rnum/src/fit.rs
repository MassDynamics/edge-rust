//! `lmFit` / `lm.series` (limma 3.66.0, `R/lmfit.R`), least-squares method only: no
//! weights, no duplicate spots, no block correlation. Matrices are column-major `f64`
//! slices; `NaN` plays R's `NA`.

use crate::linalg::chol2inv_upper;
use crate::linpack::{lm_fit, qr_decompose, Qr};
use crate::{LimmaError, Result};

/// `lm.fit`'s default tolerance.
pub const LM_TOL: f64 = 1e-7;

/// The `MArrayLM` object as `lmFit` leaves it (before `contrasts.fit` / `eBayes`).
#[derive(Debug, Clone)]
pub struct MArrayLm {
    pub ngenes: usize,
    pub ncoef: usize,
    pub narrays: usize,
    /// `ngenes x ncoef`, column-major. `NaN` marks a non-estimable coefficient.
    pub coefficients: Vec<f64>,
    /// `ngenes x ncoef`, column-major, `NaN` where the coefficient is `NaN`.
    pub stdev_unscaled: Vec<f64>,
    /// Residual standard deviation per gene; `NaN` when `df_residual == 0`.
    pub sigma: Vec<f64>,
    pub df_residual: Vec<f64>,
    /// `rank x rank`, column-major, rows/cols in `pivot[..rank]` order.
    pub cov_coefficients: Vec<f64>,
    /// 0-based column pivot of `qr(design)`.
    pub pivot: Vec<usize>,
    pub rank: usize,
    /// `rowMeans(exprs, na.rm = TRUE)`.
    pub amean: Vec<f64>,
    /// `narrays x ncoef`, column-major.
    pub design: Vec<f64>,
}

/// `rowMeans(x, na.rm = TRUE)` for a column-major `nrow x ncol` matrix; a row with no
/// finite value gives `NaN` (R gives `NaN` too).
pub fn row_means_na_rm(x: &[f64], nrow: usize, ncol: usize) -> Vec<f64> {
    let mut sums = vec![0.0; nrow];
    let mut counts = vec![0usize; nrow];
    for j in 0..ncol {
        for i in 0..nrow {
            let v = x[j * nrow + i];
            if !v.is_nan() {
                sums[i] += v;
                counts[i] += 1;
            }
        }
    }
    sums.iter()
        .zip(&counts)
        .map(|(&s, &c)| if c == 0 { f64::NAN } else { s / c as f64 })
        .collect()
}

/// `lmFit(exprs, design)` with `method = "ls"`: `exprs` is `ngenes x narrays`, `design` is
/// `narrays x ncoef`, both column-major.
pub fn lm_fit_matrix(
    exprs: &[f64],
    ngenes: usize,
    narrays: usize,
    design: &[f64],
    ncoef: usize,
) -> Result<MArrayLm> {
    if exprs.len() != ngenes * narrays {
        return Err(LimmaError::Invalid("exprs has the wrong length".into()));
    }
    if design.len() != narrays * ncoef {
        return Err(LimmaError::Invalid(
            "row dimension of design doesn't match column dimension of data object".into(),
        ));
    }
    if design.iter().any(|v| !v.is_finite()) {
        return Err(LimmaError::Invalid(
            "design must not contain NA or Inf".into(),
        ));
    }
    let amean = row_means_na_rm(exprs, ngenes, narrays);
    let mut fit = lm_series(exprs, ngenes, narrays, design, ncoef)?;
    fit.amean = amean;
    Ok(fit)
}

/// `lm.series(M, design)`: the all-finite sweep when no `NA` is present, otherwise one
/// `lm.fit` per gene on its finite observations.
pub fn lm_series(
    m: &[f64],
    ngenes: usize,
    narrays: usize,
    design: &[f64],
    nbeta: usize,
) -> Result<MArrayLm> {
    let n = narrays;
    let p = nbeta;
    let mut coefficients = vec![f64::NAN; ngenes * p];
    let mut stdev_unscaled = vec![f64::NAN; ngenes * p];
    let mut sigma = vec![f64::NAN; ngenes];
    let mut df_residual = vec![0.0; ngenes];

    let all_finite = m.iter().all(|v| v.is_finite());
    let qr: Qr = qr_decompose(design, n, p, LM_TOL);
    let rank = qr.rank;
    let cov_coefficients = qr.chol2inv();

    if all_finite {
        // lm.fit(design, t(M)): one QR, every gene through the same reflectors.
        let df = n - rank;
        let mut y = vec![0.0; n];
        for g in 0..ngenes {
            for (j, yj) in y.iter_mut().enumerate() {
                *yj = m[j * ngenes + g];
            }
            let qty = qr.qty(&y);
            let coef = qr.coef_pivoted(&qty)?;
            for (k, &b) in coef.iter().enumerate() {
                coefficients[qr.pivot[k] * ngenes + g] = b;
            }
            if df > 0 {
                let ss: f64 = qty[rank..].iter().map(|e| e * e).sum();
                sigma[g] = (ss / df as f64).sqrt();
            }
            df_residual[g] = df as f64;
        }
        for k in 0..rank {
            let sd = cov_coefficients[k * rank + k].sqrt();
            for g in 0..ngenes {
                stdev_unscaled[qr.pivot[k] * ngenes + g] = sd;
            }
        }
    } else {
        let mut x = Vec::with_capacity(n * p);
        let mut y = Vec::with_capacity(n);
        for g in 0..ngenes {
            let obs: Vec<usize> = (0..n).filter(|&j| m[j * ngenes + g].is_finite()).collect();
            let nobs = obs.len();
            if nobs == 0 {
                continue;
            }
            x.clear();
            for k in 0..p {
                for &j in &obs {
                    x.push(design[k * n + j]);
                }
            }
            y.clear();
            y.extend(obs.iter().map(|&j| m[j * ngenes + g]));
            let out = lm_fit(&x, nobs, p, &y, LM_TOL)?;
            for k in 0..p {
                coefficients[k * ngenes + g] = out.coefficients[k];
            }
            let r = out.rank;
            let cov = out.qr.chol2inv();
            for k in 0..r {
                stdev_unscaled[out.qr.pivot[k] * ngenes + g] = cov[k * r + k].sqrt();
            }
            df_residual[g] = out.df_residual as f64;
            if out.df_residual > 0 {
                let ss: f64 = out.effects[r..].iter().map(|e| e * e).sum();
                sigma[g] = (ss / out.df_residual as f64).sqrt();
            }
        }
    }

    Ok(MArrayLm {
        ngenes,
        ncoef: p,
        narrays: n,
        coefficients,
        stdev_unscaled,
        sigma,
        df_residual,
        cov_coefficients,
        pivot: qr.pivot,
        rank,
        amean: Vec::new(),
        design: design.to_vec(),
    })
}

/// `chol2inv(qr(design)$qr, size = rank)` for callers that only need the covariance.
pub fn design_cov(design: &[f64], n: usize, p: usize) -> (Vec<f64>, Vec<usize>, usize) {
    let qr = qr_decompose(design, n, p, LM_TOL);
    let k = qr.rank;
    let r: Vec<f64> = (0..k * k)
        .map(|idx| qr.qr[(idx / k) * n + idx % k])
        .collect();
    (chol2inv_upper(&r, k), qr.pivot, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_and_per_gene_branches_agree_on_finite_data() {
        // 3 genes x 4 arrays, design = intercept + slope.
        let m = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.5];
        let design = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0];
        let sweep = lm_series(&m, 3, 4, &design, 2).unwrap();
        // Force the per-gene branch with a NaN in a gene we don't compare.
        let mut m2 = m;
        m2[0] = f64::NAN;
        let per_gene = lm_series(&m2, 3, 4, &design, 2).unwrap();
        for g in 1..3 {
            for k in 0..2 {
                let a = sweep.coefficients[k * 3 + g];
                let b = per_gene.coefficients[k * 3 + g];
                assert!((a - b).abs() < 1e-12, "coef gene {g} col {k}: {a} vs {b}");
                let a = sweep.stdev_unscaled[k * 3 + g];
                let b = per_gene.stdev_unscaled[k * 3 + g];
                assert!((a - b).abs() < 1e-12, "stdev gene {g} col {k}: {a} vs {b}");
            }
            assert!((sweep.sigma[g] - per_gene.sigma[g]).abs() < 1e-12);
            assert_eq!(sweep.df_residual[g], 2.0);
            assert_eq!(per_gene.df_residual[g], 2.0);
        }
        assert_eq!(per_gene.df_residual[0], 1.0);
        // gene 2: y = 3,6,9,12.5 on x = 0..3 → slope 3.15, intercept 2.9
        assert!((sweep.coefficients[3 + 2] - 3.15).abs() < 1e-12);
        assert!((sweep.coefficients[2] - 2.9).abs() < 1e-12);
    }

    #[test]
    fn row_means_skip_nan() {
        let x = [1.0, f64::NAN, 3.0, f64::NAN];
        let r = row_means_na_rm(&x, 2, 2);
        assert_eq!(r[0], 2.0);
        assert!(r[1].is_nan());
    }
}
