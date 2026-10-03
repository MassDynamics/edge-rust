//! Negative binomial GLM fitting, ported from edgeR 4.8.2: `glmFit.default` (`R/glmfit.R`),
//! `designAsFactor` and `mglmOneWay` (`R/mglmOneWay.R`), `glm_one_group_vec`, `get_leven_start`
//! (null start), `fit_leven_vec` and `fit_leven_autofill` (`src/glm.c`),
//! `compute_unit_nb_deviance` (`src/compute_nbdev.c`), `addPriorCount` / `compute_offsets`
//! (`R/addPriorCount.R`, `src/add_prior_count.c`), `predFC.default` (`R/predFC.R`) and
//! `aveLogCPM.default` / `average_log_cpm` (`R/aveLogCPM.R`, `src/compute_cpm.c`).
//!
//! Layout: counts and fitted values are gene-major (`y[g * nlib + j]`), the design is
//! column-major `nlib x p` as in R, offsets are one shared row of length `nlib` (every edgeR call
//! on this path compresses them to a row), and dispersions are one value per gene. Observation
//! weights are always 1 on the Mass Dynamics path and are dropped.

use crate::lapack::{dgeqr2, dorm2r_lt, dpotrf_upper, dpotrs_upper, dtrtrs_upper, lu_solve};
use rnum::glibm::{exp, ln};
use rnum::{LimmaError, Result};

/// `compute_unit_nb_deviance`. The C's `2/3*resid` is integer division, i.e. zero.
pub fn unit_nb_deviance(y: f64, mu: f64, phi: f64) -> f64 {
    let y = y + 1e-8;
    let mu = mu + 1e-8;
    let out = if phi < 1e-4 {
        let resid = y - mu;
        2.0 * (y * ln(y / mu) - resid - 0.5 * resid * resid * phi * (1.0 + phi * (0.0 * resid - y)))
    } else {
        let product = mu * phi;
        if product > 1e6 {
            2.0 * ((y - mu) / mu - ln(y / mu)) * mu / (1.0 + product)
        } else {
            let invphi = 1.0 / phi;
            2.0 * (y * ln(y / mu) + (y + invphi) * ln((mu + invphi) / (y + invphi)))
        }
    };
    if out.is_nan() {
        out
    } else {
        out.max(0.0)
    }
}

/// One gene's one-group fit, kept apart by what the C does with its output variable.
#[derive(Debug, Clone, Copy)]
pub(crate) enum OneGroup {
    /// `glm_one_group_vec` writes `*beta`: converged, or an all-zero row (`-Inf`).
    Written(f64),
    /// `fit_one_group_mat`'s Poisson shortcut, which does not touch `ocoef`.
    Poisson(f64),
    /// Out of iterations: the C leaves `*beta` unwritten. Holds the last iterate.
    NotConverged(f64),
}

impl OneGroup {
    /// The value edgeR returns. On non-convergence `fit_one_group_mat` (`src/glm.c:154`) and
    /// `average_log_cpm` (`src/compute_cpm.c:139`) copy out an `ocoef` that was not written, which
    /// in practice still holds the last value written in the same `.Call`, i.e. the previous
    /// gene's (oracle: every probe of review overnight r1, M1). `slot` carries that value across
    /// genes. Before the first write R returns stack garbage, which cannot be matched; the last
    /// iterate is returned then (README, known differences).
    pub(crate) fn resolve(self, slot: &mut Option<f64>) -> f64 {
        match self {
            OneGroup::Written(b) => {
                *slot = Some(b);
                b
            }
            OneGroup::Poisson(b) => b,
            OneGroup::NotConverged(last) => slot.unwrap_or(last),
        }
    }
}

/// `glm_one_group_vec` for one gene. `start` NaN means R's `NA` (start from the gamma-limit
/// solution). The caller turns the result into R's value with [`OneGroup::resolve`].
pub(crate) fn one_group(
    y: &[f64],
    off: &[f64],
    disp: f64,
    maxit: usize,
    tol: f64,
    start: f64,
) -> OneGroup {
    if disp == 0.0 {
        // fit_one_group_mat's Poisson shortcut, checked before the all-zero rule as in the C.
        let sl: f64 = off.iter().map(|o| exp(*o)).sum();
        let sc: f64 = y.iter().sum();
        return OneGroup::Poisson(if sc == 0.0 {
            f64::NEG_INFINITY
        } else {
            ln(sc / sl)
        });
    }
    let n = y.len();
    let mut allzero = true;
    let mut cur;
    if start.is_nan() {
        cur = 0.0;
        let mut totweight = 0.0;
        for lib in 0..n {
            if y[lib] > 1e-10 {
                cur += y[lib] / exp(off[lib]);
                allzero = false;
            }
            totweight += 1.0;
        }
        cur = ln(cur / totweight);
    } else {
        cur = start;
        allzero = !y.iter().any(|&v| v > 1e-10);
    }
    if allzero {
        return OneGroup::Written(f64::NEG_INFINITY);
    }
    for _ in 0..maxit {
        let mut dl = 0.0;
        let mut info = 0.0;
        for lib in 0..n {
            let mu = exp(cur + off[lib]);
            let den = 1.0 + mu * disp;
            dl += (y[lib] - mu) / den;
            info += mu / den;
        }
        let step = dl / info;
        cur += step;
        if step.abs() < tol {
            return OneGroup::Written(cur);
        }
    }
    OneGroup::NotConverged(cur)
}

/// `designAsFactor`: group of each library (0-based, levels in increasing order of
/// `rowMeans(design * z^(col - 1))`) and the number of groups.
pub(crate) fn design_as_factor(x: &[f64], n: usize, p: usize) -> (Vec<usize>, usize) {
    let z: f64 = (std::f64::consts::E + std::f64::consts::PI) / 5.0;
    let v: Vec<f64> = (0..n)
        .map(|i| (0..p).map(|j| x[j * n + i] * z.powi(j as i32)).sum::<f64>() / p as f64)
        .collect();
    // factor() of a double: levels are unique(as.character(sort(unique(x)))) and each value is
    // matched on its as.character() string, which keeps 15 significant digits. Rows that differ
    // only in the last bits (a rotated design, `design %*% Q`) are one level.
    let key = rnum::rformat::r_as_character;
    let mut lv = v.clone();
    lv.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut keys: Vec<String> = Vec::new();
    for &a in &lv {
        let k = key(a);
        if !keys.contains(&k) {
            keys.push(k);
        }
    }
    let g = v
        .iter()
        .map(|&a| keys.iter().position(|b| *b == key(a)).unwrap())
        .collect();
    (g, keys.len())
}

/// A fitted GLM: coefficients (`ngenes x p`), fitted values (`ngenes x nlib`), both gene-major,
/// and the residual deviance per gene.
#[derive(Debug, Clone)]
pub struct GlmFit {
    pub coefficients: Vec<f64>,
    pub fitted: Vec<f64>,
    pub deviance: Vec<f64>,
}

/// `glmFit.default(y, design, dispersion, offset, prior.count = 0, start)`: the oneway shortcut
/// when the design is equivalent to a oneway layout, else Levenberg-Marquardt (`maxit = 250`).
pub fn glm_fit(
    y: &[f64],
    nlib: usize,
    x: &[f64],
    p: usize,
    offset: &[f64],
    disp: &[f64],
    start: Option<&[f64]>,
) -> Result<GlmFit> {
    let (group, ng) = design_as_factor(x, nlib, p);
    if ng == p {
        Ok(oneway(y, nlib, x, p, offset, disp, start, &group))
    } else {
        levenberg(y, nlib, x, p, offset, disp, start, 250, 1e-6)
    }
}

/// `glmFit.default` with `prior.count > 0`: deviance and fitted values from the unshrunk fit,
/// coefficients replaced by `predFC(..., prior.count) * log(2)`. Also returns the unshrunk
/// coefficients (`unshrunk.coefficients`).
pub fn glm_fit_shrunk(
    y: &[f64],
    nlib: usize,
    x: &[f64],
    p: usize,
    offset: &[f64],
    disp: &[f64],
    prior_count: f64,
) -> Result<(GlmFit, Vec<f64>)> {
    let mut fit = glm_fit(y, nlib, x, p, offset, disp, None)?;
    let (yy, oo) = add_prior_count(y, nlib, offset, prior_count);
    let pfc = glm_fit(&yy, nlib, x, p, &oo, disp, None)?;
    let ln2 = std::f64::consts::LN_2;
    let shrunk = pfc.coefficients.iter().map(|b| b / ln2 * ln2).collect();
    let unshrunk = std::mem::replace(&mut fit.coefficients, shrunk);
    Ok((fit, unshrunk))
}

/// `addPriorCount(y, offset, prior.count)` with a shared offset row and a scalar prior count.
pub(crate) fn add_prior_count(
    y: &[f64],
    nlib: usize,
    offset: &[f64],
    pc: f64,
) -> (Vec<f64>, Vec<f64>) {
    let (prior, off) = prior_offsets(offset, pc);
    let yy = y
        .chunks(nlib)
        .flat_map(|r| r.iter().zip(&prior).map(|(a, b)| a + b).collect::<Vec<_>>())
        .collect();
    (yy, off)
}

/// `compute_offsets(log_in = 1, log_out = 1)`: the scaled prior per library and the augmented
/// log offset `log(lib + 2 * prior)`.
fn prior_offsets(offset: &[f64], pc: f64) -> (Vec<f64>, Vec<f64>) {
    let n = offset.len();
    let lib: Vec<f64> = offset.iter().map(|o| exp(*o)).collect();
    let mut ave = 0.0;
    for l in &lib {
        ave += l;
    }
    ave /= n as f64;
    let prior: Vec<f64> = lib.iter().map(|l| pc * l / ave).collect();
    let off = lib
        .iter()
        .zip(&prior)
        .map(|(l, p)| ln(l + 2.0 * p))
        .collect();
    (prior, off)
}

/// `aveLogCPM.default(y, lib.size, prior.count, dispersion)` with a scalar dispersion.
pub fn ave_log_cpm(
    y: &[f64],
    nlib: usize,
    lib_size: &[f64],
    disp: f64,
    prior_count: f64,
) -> Vec<f64> {
    let offset: Vec<f64> = lib_size.iter().map(|l| ln(*l)).collect();
    let (prior, off) = prior_offsets(&offset, prior_count);
    let lnm = ln(1e6f64);
    let ln2 = std::f64::consts::LN_2;
    // One .Call over all genes, so one ocoef slot across them.
    let mut slot = None;
    y.chunks(nlib)
        .map(|r| {
            let yy: Vec<f64> = r.iter().zip(&prior).map(|(a, b)| a + b).collect();
            let b = one_group(&yy, &off, disp, 50, 1e-10, f64::NAN).resolve(&mut slot);
            (b + lnm) / ln2
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn oneway(
    y: &[f64],
    nlib: usize,
    x: &[f64],
    p: usize,
    offset: &[f64],
    disp: &[f64],
    start: Option<&[f64]>,
    group: &[usize],
) -> GlmFit {
    let ngenes = disp.len();
    let first: Vec<usize> = (0..p)
        .map(|g| group.iter().position(|&h| h == g).unwrap())
        .collect();
    // designunique, p x p column-major.
    let mut du = vec![0.0; p * p];
    for k in 0..p {
        for g in 0..p {
            du[k * p + g] = x[k * nlib + first[g]];
        }
    }
    let n1 = du.iter().filter(|&&v| v == 1.0).count();
    let n0 = du.iter().filter(|&&v| v == 0.0).count();
    let indicator = n1 == p && n0 == (p - 1) * p;
    let members: Vec<Vec<usize>> = (0..p)
        .map(|g| (0..nlib).filter(|&j| group[j] == g).collect())
        .collect();
    let mut coefficients = vec![0.0; ngenes * p];
    let mut fitted = vec![0.0; ngenes * nlib];
    let mut deviance = vec![0.0; ngenes];
    let mut ys = Vec::with_capacity(nlib);
    let mut os = Vec::with_capacity(nlib);
    // mglmOneWay calls mglmOneGroup once per group over all genes, so each group has its own
    // ocoef slot, carried from gene to gene.
    let mut slots: Vec<Option<f64>> = vec![None; p];
    for gene in 0..ngenes {
        let row = &y[gene * nlib..(gene + 1) * nlib];
        let mut beta = vec![0.0; p];
        for g in 0..p {
            ys.clear();
            os.clear();
            for &j in &members[g] {
                ys.push(row[j]);
                os.push(offset[j]);
            }
            let st = match start {
                None => f64::NAN,
                Some(s) => {
                    let s = &s[gene * p..(gene + 1) * p];
                    if indicator {
                        s[g]
                    } else {
                        (0..p).map(|k| s[k] * du[k * p + g]).sum()
                    }
                }
            };
            let b = one_group(&ys, &os, disp[gene], 50, 1e-10, st).resolve(&mut slots[g]);
            beta[g] = if b.is_nan() { b } else { b.max(-1e8) };
        }
        let mut dev = 0.0;
        for j in 0..nlib {
            let mu = exp(offset[j] + beta[group[j]]);
            fitted[gene * nlib + j] = mu;
            dev += unit_nb_deviance(row[j], mu, disp[gene]);
        }
        deviance[gene] = dev;
        let coef = if indicator {
            beta
        } else {
            lu_solve(&du, p, &beta).unwrap_or_else(|| vec![f64::NAN; p])
        };
        coefficients[gene * p..(gene + 1) * p].copy_from_slice(&coef);
    }
    GlmFit {
        coefficients,
        fitted,
        deviance,
    }
}

#[allow(clippy::too_many_arguments)]
fn levenberg(
    y: &[f64],
    nlib: usize,
    x: &[f64],
    p: usize,
    offset: &[f64],
    disp: &[f64],
    start: Option<&[f64]>,
    maxit: usize,
    tol: f64,
) -> Result<GlmFit> {
    let ngenes = disp.len();
    // get_leven_start: QR of the design once, then per gene the least-squares fit of a constant
    // log mean ratio, Q' and the triangular solve done per gene as in the C.
    let (qr, tau) = if start.is_none() {
        let mut qr = x.to_vec();
        let tau = dgeqr2(&mut qr, nlib, p);
        (qr, tau)
    } else {
        (Vec::new(), Vec::new())
    };
    let lib_n: Vec<f64> = offset.iter().map(|&o| exp(o)).collect();
    let mut coefficients = vec![0.0; ngenes * p];
    let mut fitted = vec![0.0; ngenes * nlib];
    let mut deviance = vec![0.0; ngenes];
    let mut ws = LevenWork::new(nlib, p);
    for gene in 0..ngenes {
        let row = &y[gene * nlib..(gene + 1) * nlib];
        let d = disp[gene];
        let mut beta: Vec<f64> = match start {
            Some(s) => s[gene * p..(gene + 1) * p].to_vec(),
            None => {
                let mut sw = 0.0;
                let mut se = 0.0;
                for lib in 0..nlib {
                    let cur_n = lib_n[lib];
                    let cw = cur_n / (1.0 + d * cur_n);
                    se += row[lib] * cw / cur_n;
                    sw += cw;
                }
                let mut effects = vec![ln(se / sw); nlib];
                dorm2r_lt(&qr, nlib, &tau, &mut effects);
                if !dtrtrs_upper(&qr, nlib, p, &mut effects) {
                    effects.iter_mut().for_each(|e| *e = f64::NAN);
                }
                effects.truncate(p);
                effects
            }
        };
        let mut mu = vec![0.0; nlib];
        let dev = fit_leven_vec(
            row, offset, d, x, p, maxit, tol, &mut beta, &mut mu, &mut ws,
        )?;
        coefficients[gene * p..(gene + 1) * p].copy_from_slice(&beta);
        fitted[gene * nlib..(gene + 1) * nlib].copy_from_slice(&mu);
        deviance[gene] = dev;
    }
    Ok(GlmFit {
        coefficients,
        fitted,
        deviance,
    })
}

struct LevenWork {
    zw: Vec<f64>,
    drv: Vec<f64>,
    dl: Vec<f64>,
    db: Vec<f64>,
    xtwx: Vec<f64>,
    xtwc: Vec<f64>,
    nbt: Vec<f64>,
    nmu: Vec<f64>,
}

impl LevenWork {
    fn new(n: usize, p: usize) -> Self {
        LevenWork {
            zw: vec![0.0; n],
            drv: vec![0.0; n],
            dl: vec![0.0; p],
            db: vec![0.0; p],
            xtwx: vec![0.0; p * p],
            xtwc: vec![0.0; p * p],
            nbt: vec![0.0; p],
            nmu: vec![0.0; n],
        }
    }
}

/// `fit_leven_autofill`: `mu = exp(offset + X beta)`, accumulated column by column like the
/// reference BLAS `dgemv` (which skips zero coefficients).
fn autofill(beta: &[f64], offset: &[f64], x: &[f64], mu: &mut [f64]) {
    let n = offset.len();
    mu.copy_from_slice(offset);
    for (j, &b) in beta.iter().enumerate() {
        if b != 0.0 {
            for i in 0..n {
                mu[i] += b * x[j * n + i];
            }
        }
    }
    for m in mu.iter_mut() {
        *m = exp(*m);
    }
}

/// `compute_xtwx` (`src/compute_apl.c`): the upper triangle of `X' W X`, column-major.
pub(crate) fn compute_xtwx(n: usize, p: usize, x: &[f64], w: &[f64], out: &mut [f64]) {
    for c1 in 0..p {
        for c2 in 0..=c1 {
            let mut s = 0.0;
            for lib in 0..n {
                s += x[c1 * n + lib] * x[c2 * n + lib] * w[lib];
            }
            out[c1 * p + c2] = s;
        }
    }
}

/// `fit_leven_vec` for one gene; returns the deviance. All-zero rows get `NA` coefficients,
/// zero means and zero deviance, as in the C. The C retries the Cholesky factorisation with a
/// growing damping until it succeeds, which never happens once `X'WX` or the damping is not
/// finite; here that is an error instead of an endless loop.
#[allow(clippy::too_many_arguments)]
fn fit_leven_vec(
    y: &[f64],
    offset: &[f64],
    disp: f64,
    x: &[f64],
    p: usize,
    maxit: usize,
    tol: f64,
    obt: &mut [f64],
    omu: &mut [f64],
    ws: &mut LevenWork,
) -> Result<f64> {
    let n = y.len();
    let ymax = y.iter().fold(0.0f64, |m, &v| if v > m { v } else { m });
    if ymax < 1e-10 {
        obt.iter_mut().for_each(|b| *b = f64::NAN);
        omu.iter_mut().for_each(|m| *m = 0.0);
        return Ok(0.0);
    }
    autofill(obt, offset, x, omu);
    let mut dev = 0.0;
    for lib in 0..n {
        dev += unit_nb_deviance(y[lib], omu[lib], disp);
    }
    let mut max_info: f64 = -1.0;
    let mut lambda = 0.0;
    let mut iter = 0;
    while {
        iter += 1;
        iter <= maxit
    } {
        for lib in 0..n {
            let cur = omu[lib];
            let denom = 1.0 + cur * disp;
            ws.zw[lib] = cur / denom;
            ws.drv[lib] = (y[lib] - cur) / denom;
        }
        compute_xtwx(n, p, x, &ws.zw, &mut ws.xtwx);
        for c in 0..p {
            let mut s = 0.0;
            for lib in 0..n {
                s += ws.drv[lib] * x[c * n + lib];
            }
            ws.dl[c] = s;
            if ws.xtwx[c * p + c] > max_info {
                max_info = ws.xtwx[c * p + c];
            }
        }
        if iter == 1 {
            lambda = max_info * 1e-6;
            if lambda < 1e-13 {
                lambda = 1e-13;
            }
        }
        let mut lev = 0;
        let mut low_dev = false;
        let mut failed = false;
        loop {
            lev += 1;
            loop {
                if !lambda.is_finite() || ws.xtwx.iter().any(|v| !v.is_finite()) {
                    return Err(LimmaError::Invalid(
                        "the NB GLM fit diverged: X'WX or its damping is not finite (non-finite counts, offsets or design)".into(),
                    ));
                }
                for c1 in 0..p {
                    for c2 in 0..=c1 {
                        ws.xtwc[c1 * p + c2] = ws.xtwx[c1 * p + c2];
                    }
                    ws.xtwc[c1 * p + c1] += lambda;
                }
                if dpotrf_upper(&mut ws.xtwc, p) {
                    break;
                }
                lambda *= 10.0;
                if lambda <= 0.0 {
                    lambda = 1e-100;
                }
            }
            ws.db.copy_from_slice(&ws.dl);
            dpotrs_upper(&ws.xtwc, p, &mut ws.db);
            for ((nb, o), d) in ws.nbt.iter_mut().zip(obt.iter()).zip(&ws.db) {
                *nb = o + d;
            }
            autofill(&ws.nbt, offset, x, &mut ws.nmu);
            let mut ndev = 0.0;
            for (&yl, &mu) in y[..n].iter().zip(&ws.nmu) {
                ndev += unit_nb_deviance(yl, mu, disp);
            }
            if ndev / ymax < 1e-13 {
                low_dev = true;
            }
            if ndev <= dev || low_dev {
                obt.copy_from_slice(&ws.nbt);
                omu.copy_from_slice(&ws.nmu);
                dev = ndev;
                break;
            }
            lambda *= 2.0;
            if lambda <= 0.0 {
                lambda = 1e-100;
            }
            if lambda / max_info > 1.0 / 1e-13 {
                failed = true;
                break;
            }
        }
        let mut divergence = 0.0;
        for c in 0..p {
            divergence += ws.dl[c] * ws.db[c];
        }
        if failed || low_dev || divergence < tol {
            break;
        }
        if lev == 1 {
            lambda /= 10.0;
        }
    }
    Ok(dev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn design_as_factor_keys_levels_with_r_as_character() {
        // Review deseq2 r4, N5: factor() matches on as.character(), which R computes with x87
        // scaling. -3.161245995276595 is "-3.1612459952766" in R, the same string as
        // -3.1612459952766, so edgeR:::designAsFactor(matrix(x)) gives 1 1 2 (2 levels) in the
        // oracle. A correctly rounded {:.14e} key splits the first two rows.
        let x = [-3.161245995276595, -3.1612459952766, 2.5];
        assert_eq!(design_as_factor(&x, 3, 1), (vec![0, 0, 1], 2));
    }

    // Review overnight r1, M1. Gene 2's one-group Newton-Raphson does not converge in 50
    // iterations at dispersion 0.8, and edgeR then returns the previous gene's coefficient. Oracle
    // (edgeR 4.8.2): mglmOneGroup(rbind(c(30, 1, 12), c(37, 2, 15)), dispersion = 0.8,
    // offset = log(lib)) and glmFit(..., matrix(1, 3, 1)) both give -8.0000944123408733 twice.
    // Gene 2 alone gives 6.95e-310 in R (stack garbage); the port keeps the last iterate there.
    #[test]
    fn non_converged_one_group_fit_reuses_the_previous_genes_coefficient() {
        let lib: [f64; 3] = [107774.0, 2.0, 104372.0];
        let off: Vec<f64> = lib.iter().map(|l| ln(*l)).collect();
        let y = [30.0, 1.0, 12.0, 37.0, 2.0, 15.0];
        let fit = glm_fit(&y, 3, &[1.0; 3], 1, &off, &[0.8, 0.8], None).unwrap();
        let want = -8.000_094_412_340_873;
        assert!((fit.coefficients[0] - want).abs() < 1e-13 * want.abs());
        assert_eq!(fit.coefficients[1], fit.coefficients[0]);
        let alone = glm_fit(&y[3..], 3, &[1.0; 3], 1, &off, &[0.8], None).unwrap();
        assert!(matches!(
            one_group(&y[3..], &off, 0.8, 50, 1e-10, f64::NAN),
            OneGroup::NotConverged(_)
        ));
        assert!(alone.coefficients[0] > -7.0 && alone.coefficients[0] < -6.0);
    }

    // The same reuse in average_log_cpm. Oracle: aveLogCPM(rbind(c(30, 1, 12), c(0, 3, 15)),
    // lib.size = lib, dispersion = 0.3725290298461914) gives 8.1410848659971418 twice.
    #[test]
    fn non_converged_ave_log_cpm_reuses_the_previous_genes_value() {
        let lib = [107774.0, 2.0, 104372.0];
        let y = [30.0, 1.0, 12.0, 0.0, 3.0, 15.0];
        let a = ave_log_cpm(&y, 3, &lib, 0.3725290298461914, 2.0);
        let want = 8.141_084_865_997_142;
        assert!((a[0] - want).abs() < 1e-13 * want);
        assert_eq!(a[1], a[0]);
    }

    #[test]
    fn oneway_and_levenberg_agree_on_a_oneway_design() {
        // Two groups of three libraries, ~0 + group.
        let y = [
            10.0, 12.0, 9.0, 30.0, 28.0, 35.0, 0.0, 1.0, 0.0, 5.0, 4.0, 6.0,
        ];
        let x = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let off: Vec<f64> = [1e6, 1.2e6, 0.9e6, 1.1e6, 1e6, 0.95e6]
            .iter()
            .map(|l: &f64| ln(*l))
            .collect();
        let disp = [0.1, 0.2];
        let a = glm_fit(&y, 6, &x, 2, &off, &disp, None).unwrap();
        let b = levenberg(&y, 6, &x, 2, &off, &disp, None, 250, 1e-14).unwrap();
        for i in 0..2 {
            assert!((a.deviance[i] - b.deviance[i]).abs() < 1e-8 * (1.0 + a.deviance[i]));
        }
        for i in 0..4 {
            assert!((a.coefficients[i] - b.coefficients[i]).abs() < 1e-6);
        }
    }

    // SE-1: a +-1e308 covariate makes X'WX non-finite; the Cholesky retry looped forever.
    #[test]
    fn levenberg_errors_on_a_non_finite_cross_product() {
        let y = [10.0, 12.0, 9.0, 30.0, 28.0, 35.0];
        let x = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e308, -1e308, 1.0, 2.0, 3.0, 1e308,
        ];
        let off = [ln(1e6); 6];
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(glm_fit(&y, 6, &x, 2, &off, &[0.1], None).is_err());
        });
        let is_err = rx
            .recv_timeout(std::time::Duration::from_secs(20))
            .expect("levenberg did not return within 20 s");
        assert!(is_err);
    }
}
