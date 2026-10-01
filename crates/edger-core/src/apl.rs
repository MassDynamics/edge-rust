//! `adjustedProfileLik` (edgeR 4.8.2, `R/adjustedProfileLik.R`) and its kernel
//! `compute_adj_profile_ll` (`src/compute_apl.c`): the Cox-Reid adjusted profile log-likelihood
//! of each gene at a given dispersion.

use crate::glm::{compute_xtwx, glm_fit, GlmFit};
use crate::lapack::dsytf2_upper;
use rnum::glibm::ln;
use rnum::glibm_lgamma::lgamma as lgammafn;

/// `compute_adj_profile_ll(do_adjust = TRUE)` for gene-major `y` and `mu` and one dispersion
/// per gene.
pub(crate) fn compute_apl(
    y: &[f64],
    mu: &[f64],
    nlib: usize,
    disp: &[f64],
    x: &[f64],
    p: usize,
) -> Vec<f64> {
    let low_value: f64 = 1e-10;
    let log_low_value = ln(low_value);
    let mut zw = vec![0.0; nlib];
    let mut xtwx = vec![0.0; p * p];
    let mut out = Vec::with_capacity(disp.len());
    for (tag, &d) in disp.iter().enumerate() {
        let yr = &y[tag * nlib..(tag + 1) * nlib];
        let ur = &mu[tag * nlib..(tag + 1) * nlib];
        let mut o = 0.0;
        for lib in 0..nlib {
            let curu = ur[lib];
            if curu == 0.0 {
                zw[lib] = 0.0;
                continue;
            }
            let cury = yr[lib];
            let loglik = if d > 0.0 {
                let r = 1.0 / d;
                let logmur = ln(curu + r);
                cury * ln(curu) - cury * logmur + r * ln(r) - r * logmur + lgammafn(cury + r)
                    - lgammafn(cury + 1.0)
                    - lgammafn(r)
            } else {
                cury * ln(curu) - curu - lgammafn(cury + 1.0)
            };
            o += loglik;
            zw[lib] = curu / (1.0 + d * curu);
        }
        let mut adj = 0.0;
        if p == 1 {
            for v in &zw {
                adj += v;
            }
            adj = ln(adj.abs()) / 2.0;
        } else {
            compute_xtwx(nlib, p, x, &zw, &mut xtwx);
            dsytf2_upper(&mut xtwx, p);
            for i in 0..p {
                let cur = xtwx[i * p + i];
                adj = if cur < low_value {
                    adj + log_low_value
                } else {
                    adj + ln(cur) * 0.5
                };
            }
        }
        out.push(o - adj);
    }
    out
}

/// `adjustedProfileLik(dispersion, y, design, offset, start, get.coef = TRUE)` with a shared
/// offset row: returns the APL per gene and the fit (whose coefficients seed the next grid point).
pub(crate) fn adjusted_profile_lik(
    disp: f64,
    y: &[f64],
    nlib: usize,
    x: &[f64],
    p: usize,
    offset: &[f64],
    start: Option<&[f64]>,
) -> (Vec<f64>, GlmFit) {
    let ng = y.len() / nlib;
    let dv = vec![disp; ng];
    let fit = glm_fit(y, nlib, x, p, offset, &dv, start);
    let apl = compute_apl(y, &fit.fitted, nlib, &dv, x, p);
    (apl, fit)
}
