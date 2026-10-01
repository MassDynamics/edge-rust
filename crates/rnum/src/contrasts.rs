//! `contrasts.fit` (limma 3.66.0, `R/contrasts.R`) with a numeric contrast matrix.

use crate::fit::MArrayLm;
use crate::linalg::{chol_upper, cov2cor, crossprod, matmul};
use crate::{LimmaError, Result};

/// `contrasts.fit(fit, contrasts)`: `contrasts` is `ncoef x ncontrasts`, column-major.
/// Returns a new `MArrayLm` whose coefficient axis is the contrasts. `pivot` and `rank`
/// are carried over untouched, as limma leaves them.
pub fn contrasts_fit(fit: &MArrayLm, contrasts: &[f64], ncontrasts: usize) -> Result<MArrayLm> {
    let ngenes = fit.ngenes;
    let mut ncoef = fit.ncoef;
    if contrasts.len() != ncoef * ncontrasts {
        return Err(LimmaError::Invalid(
            "Number of rows of contrast matrix must match number of coefficients in fit".into(),
        ));
    }
    if contrasts.iter().any(|v| v.is_nan()) {
        return Err(LimmaError::Invalid("NAs not allowed in contrasts".into()));
    }
    if ncontrasts == 0 {
        return Err(LimmaError::Invalid("contrast matrix has 0 columns".into()));
    }

    let mut contrasts = contrasts.to_vec();
    let mut coefficients = fit.coefficients.clone();
    let mut stdev_unscaled = fit.stdev_unscaled.clone();
    let mut cov_coefficients = fit.cov_coefficients.clone();
    let r = fit.rank;
    let mut cormatrix = cov2cor(&cov_coefficients, r);

    // Singular design: reduce to the estimable coefficients.
    if r < ncoef {
        let est = &fit.pivot[..r];
        for k in 0..ncoef {
            if est.contains(&k) {
                continue;
            }
            if (0..ncontrasts).any(|c| contrasts[c * ncoef + k] != 0.0) {
                return Err(LimmaError::Invalid(
                    "trying to take contrast of non-estimable coefficient".into(),
                ));
            }
        }
        contrasts = select_rows(&contrasts, ncoef, ncontrasts, est);
        coefficients = select_cols(&coefficients, ngenes, est);
        stdev_unscaled = select_cols(&stdev_unscaled, ngenes, est);
        ncoef = r;
    }

    // Drop coefficients that appear in no contrast.
    let keep: Vec<usize> = (0..ncoef)
        .filter(|&k| (0..ncontrasts).any(|c| contrasts[c * ncoef + k] != 0.0))
        .collect();
    if keep.len() < ncoef {
        contrasts = select_rows(&contrasts, ncoef, ncontrasts, &keep);
        coefficients = select_cols(&coefficients, ngenes, &keep);
        stdev_unscaled = select_cols(&stdev_unscaled, ngenes, &keep);
        cov_coefficients = select_square(&cov_coefficients, ncoef, &keep);
        cormatrix = select_square(&cormatrix, ncoef, &keep);
        ncoef = keep.len();
    }

    // NA coefficients become 0 with a huge stdev so zero contrast entries clobber them.
    let na_coef = coefficients.iter().any(|v| v.is_nan());
    if na_coef {
        for (c, s) in coefficients.iter_mut().zip(stdev_unscaled.iter_mut()) {
            if c.is_nan() {
                *c = 0.0;
                *s = 1e30;
            }
        }
    }

    let new_coefficients = matmul(&coefficients, ngenes, ncoef, &contrasts, ncontrasts);

    let orthog = if ncoef * ncoef < 2 {
        true
    } else {
        let mut all = true;
        for j in 0..ncoef {
            for i in (j + 1)..ncoef {
                if cormatrix[j * ncoef + i].abs() >= 1e-14 {
                    all = false;
                }
            }
        }
        all
    };

    let rc = chol_upper(&cov_coefficients, ncoef)?;
    let rcont = matmul(&rc, ncoef, ncoef, &contrasts, ncontrasts);
    let new_cov = crossprod(&rcont, ncoef, ncontrasts, &rcont, ncontrasts);

    let mut new_stdev = vec![0.0; ngenes * ncontrasts];
    if orthog {
        for c in 0..ncontrasts {
            for g in 0..ngenes {
                let mut acc = 0.0;
                for k in 0..ncoef {
                    let s = stdev_unscaled[k * ngenes + g];
                    let w = contrasts[c * ncoef + k];
                    acc += s * s * w * w;
                }
                new_stdev[c * ngenes + g] = acc.sqrt();
            }
        }
    } else {
        let rcor = chol_upper(&cormatrix, ncoef)?;
        let mut uc = vec![0.0; ncoef * ncontrasts];
        for g in 0..ngenes {
            for c in 0..ncontrasts {
                for k in 0..ncoef {
                    uc[c * ncoef + k] = stdev_unscaled[k * ngenes + g] * contrasts[c * ncoef + k];
                }
            }
            let ruc = matmul(&rcor, ncoef, ncoef, &uc, ncontrasts);
            for c in 0..ncontrasts {
                let ss: f64 = ruc[c * ncoef..(c + 1) * ncoef].iter().map(|v| v * v).sum();
                new_stdev[c * ngenes + g] = ss.sqrt();
            }
        }
    }

    let mut new_coefficients = new_coefficients;
    if na_coef {
        for (c, s) in new_coefficients.iter_mut().zip(new_stdev.iter_mut()) {
            if *s > 1e20 {
                *c = f64::NAN;
                *s = f64::NAN;
            }
        }
    }

    Ok(MArrayLm {
        ngenes,
        ncoef: ncontrasts,
        narrays: fit.narrays,
        coefficients: new_coefficients,
        stdev_unscaled: new_stdev,
        sigma: fit.sigma.clone(),
        df_residual: fit.df_residual.clone(),
        cov_coefficients: new_cov,
        pivot: fit.pivot.clone(),
        rank: fit.rank,
        amean: fit.amean.clone(),
        design: fit.design.clone(),
    })
}

fn select_rows(a: &[f64], nrow: usize, ncol: usize, rows: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows.len() * ncol);
    for c in 0..ncol {
        for &r in rows {
            out.push(a[c * nrow + r]);
        }
    }
    out
}

fn select_cols(a: &[f64], nrow: usize, cols: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(nrow * cols.len());
    for &c in cols {
        out.extend_from_slice(&a[c * nrow..(c + 1) * nrow]);
    }
    out
}

fn select_square(a: &[f64], n: usize, idx: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(idx.len() * idx.len());
    for &c in idx {
        for &r in idx {
            out.push(a[c * n + r]);
        }
    }
    out
}
