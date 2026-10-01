//! `glmQLFit.default` with `legacy = FALSE` (edgeR 4.8.2, `R/glmQLFTest.R`), as the reference's
//! `ref_ql_fit` writes it out, with its C kernels from `src/ql_glm.c`: `compute_adjust_vec`,
//! `qr_hat`, `update_prior` (`.cxx_compute_ave_qd`), `compute_prior` and `clowess2`; plus R's
//! `rsort_with_index` (`src/main/sort.c`).

use crate::glm::{glm_fit, glm_fit_shrunk, unit_nb_deviance};
use crate::ql_weights::compute_weight;
use rnum::ebayes::{fit_f_dist_unequal_df1, order_desc, squeeze_var};
use rnum::linpack::qr_decompose;
use rnum::lowess::clowess;
use rnum::Result;

const THRESHOLD_ZERO: f64 = 1e-4;

/// The QL fit and everything the goldens record about it.
#[derive(Debug, Clone)]
pub struct QlFit {
    pub top_n: usize,
    pub dispersion_uncapped: f64,
    pub dispersion: f64,
    pub ave_ql_dispersion: f64,
    pub deviance_first: Vec<f64>,
    pub deviance: Vec<f64>,
    pub df_residual: f64,
    /// `ngenes x p`, gene-major, natural-log scale (`predFC * log(2)`).
    pub coefficients: Vec<f64>,
    pub s2: Vec<f64>,
    pub df_residual_adj: Vec<f64>,
    pub deviance_adj: Vec<f64>,
    pub s2_prior: Vec<f64>,
    pub s2_post: Vec<f64>,
    pub df_prior: Vec<f64>,
    pub fdist_scale: Vec<f64>,
    pub fdist_df2: f64,
    pub fdist_df2_shrunk: Option<Vec<f64>>,
}

/// `qr_hat`: leverages of `x` (column-major `n x p`) from `dqrdc2` (`tol = 1e-7`) and `dqrqy`.
fn qr_hat(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    let qr = qr_decompose(x, n, p, 1e-7);
    let mut h = vec![0.0; n];
    let mut e = vec![0.0; n];
    for i in 0..qr.rank {
        e.iter_mut().for_each(|v| *v = 0.0);
        e[i] = 1.0;
        let q = qr.qy(&e);
        for lib in 0..n {
            h[lib] += q[lib] * q[lib];
        }
    }
    h
}

/// `compute_adjust_vec`: adjusted deviance, df and s2 per gene.
pub(crate) fn adjust_vec(
    y: &[f64],
    mu: &[f64],
    nlib: usize,
    x: &[f64],
    p: usize,
    disp: f64,
    prior: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ng = y.len() / nlib;
    let mut df = vec![0.0; ng];
    let mut dev = vec![0.0; ng];
    let mut s2 = vec![0.0; ng];
    let mut xd = vec![0.0; nlib * p];
    let mut zw = vec![0.0; nlib];
    for tag in 0..ng {
        let yr = &y[tag * nlib..(tag + 1) * nlib];
        let ur = &mu[tag * nlib..(tag + 1) * nlib];
        for lib in 0..nlib {
            zw[lib] = (ur[lib] / (1.0 + (ur[lib] * disp / prior))).sqrt();
        }
        for i in 0..nlib * p {
            xd[i] = x[i] * zw[i % nlib];
        }
        let h = qr_hat(&xd, nlib, p);
        let mut dv = 0.0;
        let mut dff = 0.0;
        for lib in 0..nlib {
            let (wa, wk) = compute_weight(ur[lib], disp, prior);
            let mut udp = unit_nb_deviance(yr[lib], ur[lib], disp / prior);
            let mut hdp = 1.0 - h[lib];
            if hdp < THRESHOLD_ZERO {
                udp = 0.0;
                hdp = 0.0;
            }
            dv += udp * wa;
            dff += hdp * wk;
        }
        dev[tag] = dv;
        df[tag] = dff;
        s2[tag] = if dff < THRESHOLD_ZERO { 0.0 } else { dv / dff };
    }
    (df, dev, s2)
}

/// R's `rsort_with_index`: Shell sort of `x` carrying `indx`.
fn rsort_with_index(x: &mut [f64], indx: &mut [usize]) {
    let n = x.len();
    let mut h = 1;
    while h <= n / 9 {
        h = 3 * h + 1;
    }
    while h > 0 {
        for i in h..n {
            let v = x[i];
            let iv = indx[i];
            let mut j = i;
            while j >= h && rcmp_gt(x[j - h], v) {
                x[j] = x[j - h];
                indx[j] = indx[j - h];
                j -= h;
            }
            x[j] = v;
            indx[j] = iv;
        }
        h /= 3;
    }
}

/// `rcmp(a, b, nalast = TRUE) > 0`.
fn rcmp_gt(a: f64, b: f64) -> bool {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => false,
        (true, false) => true,
        (false, true) => false,
        _ => a > b,
    }
}

/// `compute_prior`: the 90th percentile of the lowess trend of `s2^(1/4)` on AveLogCPM, floored
/// at 1, to the fourth power.
fn compute_prior(ave: &[f64], s2: &[f64], df: &[f64]) -> f64 {
    let mut xx = Vec::new();
    let mut yy = Vec::new();
    for i in 0..ave.len() {
        if df[i] > 1e-8 {
            xx.push(ave[i]);
            yy.push(s2[i].sqrt().sqrt());
        }
    }
    let k = xx.len();
    let mut ind: Vec<usize> = (0..k).collect();
    rsort_with_index(&mut xx, &mut ind);
    let delta = 0.01 * (xx[k - 1] - xx[0]);
    let ys: Vec<f64> = ind.iter().map(|&i| yy[i]).collect();
    let mut ans = clowess(&xx, &ys, 0.5, 3, delta);
    let m = (k - 1) as f64 * 0.9;
    let lo = m as usize;
    ans.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let h = m - lo as f64;
    let mut p = (1.0 - h) * ans[lo] + h * ans[(lo + 1).min(k - 1)];
    if p < 1.0 {
        p = 1.0;
    }
    p * p * p * p
}

/// `update_prior` (`.cxx_compute_ave_qd`): two rounds of adjust + prior, starting from 1.
fn update_prior(
    y: &[f64],
    mu: &[f64],
    nlib: usize,
    x: &[f64],
    p: usize,
    disp: f64,
    ave: &[f64],
) -> f64 {
    let mut prior = 1.0;
    for _ in 0..2 {
        let (df, _, s2) = adjust_vec(y, mu, nlib, x, p, disp, prior);
        prior = compute_prior(ave, &s2, &df);
    }
    prior
}

/// `glmQLFit(estimateDisp(y), design, legacy = FALSE)`: the scalar NB dispersion is the mean
/// trended dispersion of the top 10% genes by AveLogCPM, capped at 4.
pub fn glm_ql_fit(
    y: &[f64],
    nlib: usize,
    x: &[f64],
    p: usize,
    offset: &[f64],
    ave: &[f64],
    trended: &[f64],
) -> Result<QlFit> {
    let ng = y.len() / nlib;
    let top_n = (0.1 * ng as f64).ceil() as usize;
    let top = order_desc(ave);
    let mut s = 0.0;
    for &i in &top[..top_n] {
        s += trended[i];
    }
    let disp_raw = s / top_n as f64;
    let disp = disp_raw.min(4.0);
    let fit0 = glm_fit(y, nlib, x, p, offset, &vec![disp; ng], None);
    let aqd = update_prior(y, &fit0.fitted, nlib, x, p, disp, ave);
    let fit = glm_fit_shrunk(y, nlib, x, p, offset, &vec![disp / aqd; ng], 0.125);
    let (df_adj, dev_adj, s2) = adjust_vec(y, &fit.fitted, nlib, x, p, disp, aqd);
    let s2_in: Vec<f64> = s2
        .iter()
        .zip(&df_adj)
        .map(|(&v, &d)| if d == 0.0 { 0.0 } else { v })
        .collect();
    let fd = fit_f_dist_unequal_df1(&s2_in, &df_adj, Some(ave), None, false, None)?;
    let sv = squeeze_var(
        &s2,
        &df_adj,
        Some(ave),
        None,
        false,
        [0.05, 0.1],
        Some(false),
    )?;
    let spread = |v: Vec<f64>| if v.len() == 1 { vec![v[0]; ng] } else { v };
    Ok(QlFit {
        top_n,
        dispersion_uncapped: disp_raw,
        dispersion: disp,
        ave_ql_dispersion: aqd,
        deviance_first: fit0.deviance,
        deviance: fit.deviance,
        df_residual: (nlib - p) as f64,
        coefficients: fit.coefficients,
        s2,
        df_residual_adj: df_adj,
        deviance_adj: dev_adj,
        s2_prior: spread(sv.var_prior),
        s2_post: sv.var_post,
        df_prior: spread(sv.df_prior),
        fdist_scale: spread(fd.scale),
        fdist_df2: fd.df2,
        fdist_df2_shrunk: fd.df2_shrunk,
    })
}
