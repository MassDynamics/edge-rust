//! `squeezeVar` / `fitFDist` / `fitFDistRobustly` / `fitFDistUnequalDF1` / `eBayes`
//! (limma 3.66.0, `R/squeezeVar.R`, `R/fitFDist.R`, `R/fitFDistRobustly.R`,
//! `R/fitFDistUnequalDF1.R`, `R/ebayes.R`, `R/decidetests.R::classifyTestsF`).
//!
//! Vectors that R lets be "length 1 or length n" (`df1`, `scale`, `df.prior`) are carried
//! the same way here and read through [`at`]. `NaN` plays R's `NA`.

// Index loops over parallel vectors mirror the R vector arithmetic they port.
#![allow(clippy::needless_range_loop)]

use crate::fit::{MArrayLm, LM_TOL};
use crate::glibm::{exp, ln};
use crate::glibm_log1p::log1p;
use crate::linalg::{
    cmp_nan_last, cov2cor, cummax, eigen_symmetric, is_fullrank, mean, mean_trim, median,
    p_adjust_bh, quantile7, rank_average,
};
use crate::linpack::lm_fit;
use crate::lowess::{loess_fit, loess_fit_bounded};
use crate::nmath::{df, lgammafn, logmdigamma, pchisq, pf, pt, qf, qt, trigamma, trigamma_inverse};
use crate::optim::{brent_fmin, optimize_default_tol, uniroot, UNIROOT_DEFAULT_MAXITER};
use crate::quad::{choose_lowess_span, gauss_quad_prob_uniform};
use crate::splines::{ns_df, ns_predict};
use crate::{LimmaError, Result};

/// Element `i` of a vector R would recycle: length 1 gives the single value.
#[inline]
pub fn at(v: &[f64], i: usize) -> f64 {
    if v.len() == 1 {
        v[0]
    } else {
        v[i]
    }
}

/// `order(x)`: stable ascending, `NaN` last.
pub fn order_asc(x: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| cmp_nan_last(x[a], x[b]));
    idx
}

/// `order(x, decreasing = TRUE)`: stable descending, `NaN` last.
pub fn order_desc(x: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    // Descending on values, NaN still last: swap only the non-NaN comparison.
    idx.sort_by(|&a, &b| match (x[a].is_nan(), x[b].is_nan()) {
        (false, false) => cmp_nan_last(x[b], x[a]),
        _ => cmp_nan_last(x[a], x[b]),
    });
    idx
}

fn n_unique(x: &[f64]) -> usize {
    let mut s: Vec<f64> = x.to_vec();
    s.sort_by(|a, b| cmp_nan_last(*a, *b));
    s.dedup();
    s.len()
}

/// R's `sum()`: the image's x87 long double accumulator for finite input (non-finite input
/// gives the same Inf or NaN either way).
fn sum(x: &[f64]) -> f64 {
    if x.iter()
        .all(|v| *v == 0.0 || (1e-300..1e300).contains(&v.abs()))
    {
        crate::ldouble::sum(x)
    } else {
        x.iter().sum()
    }
}

// ---------------------------------------------------------------------------------------
// fitFDist

/// `fitFDist`'s result: `scale` has length 1 without a covariate, `n` with one.
#[derive(Debug, Clone)]
pub struct FDistFit {
    pub scale: Vec<f64>,
    pub df2: f64,
}

/// `fitFDist(x, df1, covariate)`: moment estimation of a scaled F-distribution's scale and
/// `df2` given `df1` (length 1 or `n`).
pub fn fit_f_dist(x: &[f64], df1: &[f64], covariate: Option<&[f64]>) -> Result<FDistFit> {
    let n = x.len();
    let na = FDistFit {
        scale: vec![f64::NAN],
        df2: f64::NAN,
    };
    if n == 0 {
        return Ok(na);
    }
    if n == 1 {
        return Ok(FDistFit {
            scale: vec![x[0]],
            df2: 0.0,
        });
    }

    // Check df1
    let mut ok = vec![true; n];
    if df1.len() == 1 {
        if !(df1[0].is_finite() && df1[0] > 1e-15) {
            return Ok(na);
        }
    } else {
        if df1.len() != n {
            return Err(LimmaError::Invalid(
                "x and df1 have different lengths".into(),
            ));
        }
        for i in 0..n {
            ok[i] = df1[i].is_finite() && df1[i] > 1e-15;
        }
    }

    // Check covariate
    let mut cov: Option<Vec<f64>> = None;
    if let Some(c) = covariate {
        if c.len() != n {
            return Err(LimmaError::Invalid(
                "x and covariate must be of same length".into(),
            ));
        }
        if c.iter().any(|v| v.is_nan()) {
            return Err(LimmaError::Invalid(
                "NA covariate values not allowed".into(),
            ));
        }
        let mut c = c.to_vec();
        if !c.iter().all(|v| v.is_finite()) {
            let fin: Vec<f64> = c.iter().copied().filter(|v| v.is_finite()).collect();
            if fin.is_empty() {
                for v in c.iter_mut() {
                    *v = v.signum();
                }
            } else {
                let lo = fin.iter().cloned().fold(f64::INFINITY, f64::min);
                let hi = fin.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                for v in c.iter_mut() {
                    if *v == f64::NEG_INFINITY {
                        *v = lo - 1.0;
                    } else if *v == f64::INFINITY {
                        *v = hi + 1.0;
                    }
                }
            }
        }
        cov = Some(c);
    }

    // Remove missing or infinite or negative values and zero degrees of freedom
    for i in 0..n {
        ok[i] = ok[i] && x[i].is_finite() && x[i] > -1e-15;
    }
    let nok = ok.iter().filter(|&&b| b).count();
    if nok == 1 {
        let i = ok.iter().position(|&b| b).unwrap();
        return Ok(FDistFit {
            scale: vec![x[i]],
            df2: 0.0,
        });
    }
    if nok == 0 {
        return Err(LimmaError::Invalid("fitFDist: no usable variances".into()));
    }
    let notallok = nok < n;
    let xs: Vec<f64> = (0..n).filter(|&i| ok[i]).map(|i| x[i]).collect();
    let df1s: Vec<f64> = if df1.len() > 1 {
        (0..n).filter(|&i| ok[i]).map(|i| df1[i]).collect()
    } else {
        df1.to_vec()
    };
    let (covs, cov_notok): (Option<Vec<f64>>, Vec<f64>) = match &cov {
        Some(c) => (
            Some((0..n).filter(|&i| ok[i]).map(|i| c[i]).collect()),
            (0..n).filter(|&i| !ok[i]).map(|i| c[i]).collect(),
        ),
        None => (None, Vec::new()),
    };

    // Set df for spline trend
    let mut splinedf = 1usize;
    if let Some(c) = &covs {
        splinedf = 1 + (nok >= 3) as usize + (nok >= 6) as usize + (nok >= 30) as usize;
        splinedf = splinedf.min(n_unique(c));
        // If covariate takes only one unique value or insufficient observations, recall
        // with NULL covariate
        if splinedf < 2 {
            let out = fit_f_dist(&xs, &df1s, None)?;
            return Ok(FDistFit {
                scale: vec![out.scale[0]; n],
                df2: out.df2,
            });
        }
    }

    // Avoid exactly zero values
    let mut xs: Vec<f64> = xs.iter().map(|&v| v.max(0.0)).collect();
    let mut m = median(&xs);
    if m == 0.0 {
        // warning: More than half of residual variances are exactly zero
        m = 1.0;
    }
    for v in xs.iter_mut() {
        *v = v.max(1e-5 * m);
    }

    // Better to work on with log(F)
    let e: Vec<f64> = (0..nok)
        .map(|i| ln(xs[i]) + logmdigamma(at(&df1s, i) / 2.0))
        .collect();

    let emean: Vec<f64>;
    let mut evar: f64;
    match &covs {
        None => {
            let em = mean(&e);
            evar =
                sum(&e.iter().map(|v| (v - em) * (v - em)).collect::<Vec<_>>()) / (nok - 1) as f64;
            emean = vec![em];
        }
        Some(c) => {
            let design = ns_df(c, splinedf, true)?;
            let fit = lm_fit(&design.basis, nok, design.ncol, &e, LM_TOL)?;
            if notallok {
                let design2 = ns_predict(&design, &cov_notok)?;
                let mut em = vec![0.0; n];
                let mut k_ok = 0;
                let mut k_no = 0;
                for i in 0..n {
                    if ok[i] {
                        em[i] = fit.fitted_values[k_ok];
                        k_ok += 1;
                    } else {
                        let mut acc = 0.0;
                        for j in 0..design2.ncol {
                            acc += design2.basis[j * design2.n + k_no] * fit.coefficients[j];
                        }
                        em[i] = acc;
                        k_no += 1;
                    }
                }
                emean = em;
            } else {
                emean = fit.fitted_values.clone();
            }
            let resid = &fit.effects[fit.rank..];
            evar = mean(&resid.iter().map(|v| v * v).collect::<Vec<_>>());
        }
    }

    // Estimate scale and df2
    let tri: Vec<f64> = if df1s.len() == 1 {
        vec![trigamma(df1s[0] / 2.0)]
    } else {
        df1s.iter().map(|d| trigamma(d / 2.0)).collect()
    };
    evar -= mean(&tri);
    if evar > 0.0 {
        let df2 = 2.0 * trigamma_inverse(evar);
        let lmd = logmdigamma(df2 / 2.0);
        let scale = emean.iter().map(|em| exp(em - lmd)).collect();
        Ok(FDistFit { scale, df2 })
    } else {
        let scale = match covs {
            // Use simple pooled variance, which is MLE of the scale in this case.
            None => vec![mean(&xs)],
            Some(_) => emean.iter().map(|&em| exp(em)).collect(),
        };
        Ok(FDistFit {
            scale,
            df2: f64::INFINITY,
        })
    }
}

// ---------------------------------------------------------------------------------------
// fitFDistRobustly

/// `fitFDistRobustly`'s result: `scale` length 1 or `n`, `df2_shrunk` length `n`.
#[derive(Debug, Clone)]
pub struct FDistRobustFit {
    pub scale: Vec<f64>,
    pub df2: f64,
    pub df2_shrunk: Vec<f64>,
}

struct Moments {
    mean: f64,
    var: f64,
}

/// `fitFDistRobustly(x, df1, covariate, winsor.tail.p)`.
pub fn fit_f_dist_robustly(
    x: &[f64],
    df1: &[f64],
    covariate: Option<&[f64]>,
    winsor_tail_p: [f64; 2],
) -> Result<FDistRobustFit> {
    let n = x.len();

    // Eliminate cases of no useful data
    if n < 2 {
        return Ok(FDistRobustFit {
            scale: vec![f64::NAN],
            df2: f64::NAN,
            df2_shrunk: vec![f64::NAN; n],
        });
    }
    if n == 2 {
        let f = fit_f_dist(x, df1, covariate)?;
        return Ok(FDistRobustFit {
            df2_shrunk: vec![f.df2; n],
            scale: f.scale,
            df2: f.df2,
        });
    }
    if df1.len() != 1 && df1.len() != n {
        return Err(LimmaError::Invalid(
            "x and df1 are different lengths".into(),
        ));
    }
    if let Some(c) = covariate {
        if c.len() != n {
            return Err(LimmaError::Invalid(
                "x and covariate are different lengths".into(),
            ));
        }
        if !c.iter().all(|v| v.is_finite()) {
            return Err(LimmaError::Invalid(
                "covariate contains NA or infinite values".into(),
            ));
        }
    }

    // Treat zero df1 values as non-informative cases. Similarly for missing values of x.
    let ok: Vec<bool> = (0..n)
        .map(|i| !x[i].is_nan() && at(df1, i).is_finite() && at(df1, i) > 1e-6)
        .collect();
    if !ok.iter().all(|&b| b) {
        let idx_ok: Vec<usize> = (0..n).filter(|&i| ok[i]).collect();
        let xs: Vec<f64> = idx_ok.iter().map(|&i| x[i]).collect();
        let df1s: Vec<f64> = if df1.len() > 1 {
            idx_ok.iter().map(|&i| df1[i]).collect()
        } else {
            df1.to_vec()
        };
        let covs: Option<Vec<f64>> = covariate.map(|c| idx_ok.iter().map(|&i| c[i]).collect());
        // With two values left, R's `n == 2` return is fitFDist's list, which has no df2.shrunk,
        // so `df2.shrunk[ok] <- fit$df2.shrunk` stops.
        if idx_ok.len() == 2 {
            return Err(LimmaError::Invalid("replacement has length zero".into()));
        }
        let fit = fit_f_dist_robustly(&xs, &df1s, covs.as_deref(), winsor_tail_p)?;
        let mut df2_shrunk = vec![fit.df2; n];
        for (k, &i) in idx_ok.iter().enumerate() {
            df2_shrunk[i] = fit.df2_shrunk[k];
        }
        let scale = match (covariate, &covs) {
            (Some(c), Some(cs)) => {
                let cov2: Vec<f64> = (0..n).filter(|&i| !ok[i]).map(|i| c[i]).collect();
                let logscale: Vec<f64> = fit.scale.iter().map(|&s| ln(s)).collect();
                let interp = crate::linalg::approx_rule2_ties_mean(cs, &logscale, &cov2);
                let mut scale = vec![0.0; n];
                let mut k_ok = 0;
                let mut k_no = 0;
                for i in 0..n {
                    if ok[i] {
                        scale[i] = fit.scale[k_ok];
                        k_ok += 1;
                    } else {
                        scale[i] = exp(interp[k_no]);
                        k_no += 1;
                    }
                }
                scale
            }
            _ => fit.scale,
        };
        return Ok(FDistRobustFit {
            scale,
            df2: fit.df2,
            df2_shrunk,
        });
    }

    // Avoid zero or negative x values
    let mut x = x.to_vec();
    let m = median(&x);
    if m <= 0.0 {
        return Err(LimmaError::Invalid("Variances are mostly <= 0".into()));
    }
    for v in x.iter_mut() {
        if *v < m * 1e-12 {
            *v = m * 1e-12;
        }
    }

    // Store non-robust estimates
    let non_robust = fit_f_dist(&x, df1, covariate)?;

    // Check winsor.tail.p
    let prob = [winsor_tail_p[0], 1.0 - winsor_tail_p[1]];
    if winsor_tail_p[0] < 1.0 / n as f64 && winsor_tail_p[1] < 1.0 / n as f64 {
        return Ok(FDistRobustFit {
            df2_shrunk: vec![non_robust.df2; n],
            scale: non_robust.scale,
            df2: non_robust.df2,
        });
    }

    // Transform x to constant df1
    let d1: f64;
    if df1.len() > 1 {
        let df1max = df1.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let low: Vec<usize> = (0..n).filter(|&i| df1[i] < df1max - 1e-14).collect();
        if !low.is_empty() {
            let df2 = non_robust.df2;
            for &i in &low {
                let s = at(&non_robust.scale, i);
                let f = x[i] / s;
                let pupper = pf(f, df1[i], df2, false, true);
                let plower = pf(f, df1[i], df2, true, true);
                let f = if pupper < plower {
                    qf(pupper, df1max, df2, false, true)
                } else {
                    qf(plower, df1max, df2, true, true)
                };
                x[i] = f * s;
            }
        }
        d1 = df1max;
    } else {
        d1 = df1[0];
    }

    // Better to work with log(F)
    let z: Vec<f64> = x.iter().map(|&v| ln(v)).collect();

    // Demean or Detrend
    let ztrend: Vec<f64>;
    let zresid: Vec<f64>;
    match covariate {
        None => {
            let t = mean_trim(&z, winsor_tail_p[1]);
            ztrend = vec![t];
            zresid = z.iter().map(|v| v - t).collect();
        }
        Some(c) => {
            let fitted = loess_fit(&z, c, None, 0.4, 4);
            zresid = z.iter().zip(&fitted).map(|(a, b)| a - b).collect();
            ztrend = fitted;
        }
    }

    // Moments of Winsorized residuals
    let zrq = quantile7(&zresid, &prob);
    let zwins: Vec<f64> = zresid.iter().map(|&v| v.max(zrq[0]).min(zrq[1])).collect();
    let zwmean = mean(&zwins);
    let zwvar = mean(
        &zwins
            .iter()
            .map(|v| (v - zwmean) * (v - zwmean))
            .collect::<Vec<_>>(),
    ) * n as f64
        / (n - 1) as f64;

    // Theoretical Winsorized moments
    let (gnodes, gweights) = gauss_quad_prob_uniform(128);
    let linkfun = |x: f64| x / (1.0 + x);
    let linkinv = |x: f64| x / (1.0 - x);
    let winsorized_moments = |df2: f64| -> Moments {
        let fq = [
            qf(winsor_tail_p[0], d1, df2, true, false),
            qf(1.0 - winsor_tail_p[1], d1, df2, true, false),
        ];
        let zq = [ln(fq[0]), ln(fq[1])];
        let q = [linkfun(fq[0]), linkfun(fq[1])];
        let q21 = q[1] - q[0];
        let mut fz: Vec<(f64, f64)> = Vec::with_capacity(gnodes.len());
        for (&node, &w) in gnodes.iter().zip(&gweights) {
            let nd = q[0] + q21 * node;
            let fnode = linkinv(nd);
            let znode = ln(fnode);
            let f = df(fnode, d1, df2, false) / ((1.0 - nd) * (1.0 - nd));
            fz.push((w * f, znode));
        }
        let wfz: Vec<f64> = fz.iter().map(|&(wf, znode)| wf * znode).collect();
        let m = q21 * sum(&wfz) + sum(&[zq[0] * winsor_tail_p[0], zq[1] * winsor_tail_p[1]]);
        // R: g$weights*f*(znodes-m)^2, the square formed first
        let wfv: Vec<f64> = fz
            .iter()
            .map(|&(wf, znode)| wf * ((znode - m) * (znode - m)))
            .collect();
        let v = q21 * sum(&wfv)
            + sum(&[
                (zq[0] - m) * (zq[0] - m) * winsor_tail_p[0],
                (zq[1] - m) * (zq[1] - m) * winsor_tail_p[1],
            ]);
        Moments { mean: m, var: v }
    };

    // Try df2==Inf
    let mom = winsorized_moments(f64::INFINITY);
    let funval_inf = ln(zwvar / mom.var);
    if funval_inf <= 0.0 {
        let df2 = f64::INFINITY;
        // Correct trend for bias
        let ztrendcorrected: Vec<f64> = ztrend.iter().map(|t| t + zwmean - mom.mean).collect();
        let s20: Vec<f64> = ztrendcorrected.iter().map(|&v| exp(v)).collect();
        // Posterior df for outliers
        let fstat: Vec<f64> = (0..n)
            .map(|i| exp(z[i] - at(&ztrendcorrected, i)))
            .collect();
        let tail_p: Vec<f64> = fstat
            .iter()
            .map(|f| pchisq(f * d1, d1, false, false))
            .collect();
        let r = rank_average(&fstat);
        let df_pooled = n as f64 * d1;
        let mut df2_shrunk = vec![df2; n];
        let mut any_o = false;
        for i in 0..n {
            let emp = (n as f64 - r[i] + 0.5) / n as f64;
            let pno = (tail_p[i] / emp).min(1.0);
            if pno < 1.0 {
                df2_shrunk[i] = pno * df_pooled;
                any_o = true;
            }
        }
        if any_o {
            let o = order_asc(&tail_p);
            let ordered: Vec<f64> = o.iter().map(|&i| df2_shrunk[i]).collect();
            let cm = cummax(&ordered);
            for (k, &i) in o.iter().enumerate() {
                df2_shrunk[i] = cm[k];
            }
        }
        return Ok(FDistRobustFit {
            scale: s20,
            df2,
            df2_shrunk,
        });
    }

    // Estimate df2 by matching variance of zwins
    let fun = |x: f64| -> f64 {
        let mom = winsorized_moments(linkinv(x));
        ln(zwvar / mom.var)
    };

    // Use non-robust estimate as lower bound for df2
    if non_robust.df2 == f64::INFINITY {
        return Ok(FDistRobustFit {
            df2_shrunk: vec![non_robust.df2; n],
            scale: non_robust.scale,
            df2: non_robust.df2,
        });
    }
    let rbx = linkfun(non_robust.df2);
    let funval_low = fun(rbx);
    let df2 = if funval_low >= 0.0 {
        non_robust.df2
    } else {
        let u = uniroot(
            fun,
            rbx,
            1.0,
            funval_low,
            funval_inf,
            1e-8,
            UNIROOT_DEFAULT_MAXITER,
        )
        .ok_or_else(|| LimmaError::Invalid("fitFDistRobustly: uniroot failed".into()))?;
        linkinv(u.root)
    };

    // Correct ztrend for bias
    let mom = winsorized_moments(df2);
    let ztrendcorrected: Vec<f64> = ztrend.iter().map(|t| t + zwmean - mom.mean).collect();
    let s20: Vec<f64> = ztrendcorrected.iter().map(|&v| exp(v)).collect();

    // Posterior df for outliers
    let zresid: Vec<f64> = (0..n).map(|i| z[i] - at(&ztrendcorrected, i)).collect();
    let fstat: Vec<f64> = zresid.iter().map(|&v| exp(v)).collect();
    let log_tail_p: Vec<f64> = fstat.iter().map(|&f| pf(f, d1, df2, false, true)).collect();
    let r = rank_average(&fstat);
    let ln_n = ln(n as f64);
    let mut log_pno = vec![0.0; n];
    let mut any_neg = false;
    for i in 0..n {
        let log_emp = ln(n as f64 - r[i] + 0.5) - ln_n;
        log_pno[i] = (log_tail_p[i] - log_emp).min(0.0);
        if log_pno[i] < 0.0 {
            any_neg = true;
        }
    }
    let pno: Vec<f64> = log_pno.iter().map(|&v| exp(v)).collect();
    let pout: Vec<f64> = log_pno.iter().map(|v| -v.exp_m1()).collect();
    let df2_shrunk = if any_neg {
        // Find df2.outlier to make maxFstat the median of the distribution
        let min_log_tail_p = log_tail_p.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut df2_shrunk: Vec<f64>;
        if min_log_tail_p == f64::NEG_INFINITY {
            df2_shrunk = pno.iter().map(|p| p * df2).collect();
        } else {
            let mut df2_outlier = ln(0.5) / min_log_tail_p * df2;
            // Iterate for accuracy
            let max_f = fstat.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let new_log_tail_p = pf(max_f, d1, df2_outlier, false, true);
            df2_outlier *= ln(0.5) / new_log_tail_p;
            df2_shrunk = (0..n)
                .map(|i| pno[i] * df2 + pout[i] * df2_outlier)
                .collect();
        }
        // Force df2.shrunk to be monotonic in TailP
        monotone_in(&mut df2_shrunk, &order_asc(&log_tail_p));
        df2_shrunk
    } else {
        vec![df2; n]
    };

    Ok(FDistRobustFit {
        scale: s20,
        df2,
        df2_shrunk,
    })
}

/// The shared tail of the robust fits: `df2.ordered <- df2.shrunk[o]; m <- cumsum/(1:n);
/// imin <- which.min(m); df2.ordered[1:imin] <- m[imin]; df2.shrunk[o] <- cummax(df2.ordered)`.
fn monotone_in(df2_shrunk: &mut [f64], o: &[usize]) {
    let n = o.len();
    let mut ordered: Vec<f64> = o.iter().map(|&i| df2_shrunk[i]).collect();
    // R's cumsum() accumulates in long double and rounds each partial sum
    let mut acc = crate::ldouble::Ld::ZERO;
    let mut imin = 0;
    let mut mmin = f64::INFINITY;
    for (k, v) in ordered.iter().enumerate() {
        acc = acc.add(crate::ldouble::Ld::from_f64(*v));
        let m = acc.to_f64() / (k + 1) as f64;
        if m < mmin {
            mmin = m;
            imin = k;
        }
    }
    let _ = n;
    for v in ordered.iter_mut().take(imin + 1) {
        *v = mmin;
    }
    let cm = cummax(&ordered);
    for (k, &i) in o.iter().enumerate() {
        df2_shrunk[i] = cm[k];
    }
}

// ---------------------------------------------------------------------------------------
// fitFDistUnequalDF1

/// `fitFDistUnequalDF1`'s result: `df2_shrunk` is present only when the robust path found
/// right outliers (R's `df2.shrunk` is otherwise `NULL`).
#[derive(Debug, Clone)]
pub struct FDistUnequalFit {
    pub scale: Vec<f64>,
    pub df2: f64,
    pub df2_outlier: Option<f64>,
    pub df2_shrunk: Option<Vec<f64>>,
}

/// `fitFDistUnequalDF1(x, df1, covariate, span, robust, prior.weights)`.
pub fn fit_f_dist_unequal_df1(
    x: &[f64],
    df1: &[f64],
    covariate: Option<&[f64]>,
    span: Option<f64>,
    robust: bool,
    prior_weights: Option<&[f64]>,
) -> Result<FDistUnequalFit> {
    let n = x.len();
    if df1.len() != 1 && df1.len() != n {
        return Err(LimmaError::Invalid(
            "x and df1 are different lengths".into(),
        ));
    }
    if df1.iter().any(|v| v.is_nan()) {
        return Err(LimmaError::Invalid("NA df1 values".into()));
    }
    let mut covariate = covariate;
    if let Some(c) = covariate {
        if c.len() != n {
            return Err(LimmaError::Invalid(
                "x and covariate are different lengths".into(),
            ));
        }
        if c.iter().any(|v| v.is_nan()) {
            // R's message really says prior.weights here (fitFDistUnequalDF1.R:17 copy-paste).
            return Err(LimmaError::Invalid(
                "prior.weights contain NA values".into(),
            ));
        }
    }
    let mut pw: Option<Vec<f64>> = prior_weights.map(|w| w.to_vec());
    if let Some(w) = &pw {
        if w.len() != n {
            return Err(LimmaError::Invalid(
                // R says covariate here too (fitFDistUnequalDF1.R:22 copy-paste).
                "x and covariate are different lengths".into(),
            ));
        }
        if w.iter().any(|v| v.is_nan()) {
            return Err(LimmaError::Invalid(
                "prior.weights contain NA values".into(),
            ));
        }
        if w.iter().any(|&v| v < 0.0) {
            return Err(LimmaError::Invalid("prior.weights are negative".into()));
        }
    }

    // Check for NAs in x
    let mut x = x.to_vec();
    if x.iter().any(|v| v.is_nan()) {
        let w = pw.get_or_insert_with(|| vec![1.0; n]);
        for i in 0..n {
            if x[i].is_nan() {
                w[i] = 0.0;
                x[i] = 0.0;
            }
        }
    }

    // Treat small df1 values as un-informative
    let mut df1 = df1.to_vec();
    if df1.iter().cloned().fold(f64::INFINITY, f64::min) < 0.01 {
        let w = pw.get_or_insert_with(|| vec![1.0; n]);
        for i in 0..n {
            if at(&df1, i) < 0.01 {
                w[i] = 0.0;
            }
        }
        for d in df1.iter_mut() {
            if *d < 0.01 {
                *d = 1.0;
            }
        }
    }

    // Check there are some informative x values
    let mut informative: Vec<bool> = x.iter().map(|&v| v > 0.0).collect();
    if let Some(w) = &pw {
        for i in 0..n {
            if w[i] == 0.0 {
                informative[i] = false;
            }
        }
    }
    let n_informative = informative.iter().filter(|&&b| b).count();
    let mut robust = robust;
    if n_informative < 2 {
        return Ok(FDistUnequalFit {
            scale: vec![f64::NAN],
            df2: f64::NAN,
            df2_outlier: None,
            df2_shrunk: None,
        });
    }
    // R sets prior.weights to NULL here but leaves PriorWeights as it was. When PriorWeights is
    // TRUE (any NA x or df1 < 0.01), `w * NULL` is numeric(0), so emean is 0/0 = NaN and the
    // likelihood is sum(NULL * ...) = 0 for every par. optimize() then returns the minimum of a
    // constant, the scale is NaN and every moderated statistic downstream is NaN. Reproduced
    // as is (round-3 review, golden edge_two_informative_*).
    let mut null_weights = false;
    if n_informative == 2 {
        covariate = None;
        robust = false;
        null_weights = pw.is_some();
        pw = None;
    }

    // Avoid exactly zero x values for moment estimation
    let xi: Vec<f64> = (0..n).filter(|&i| informative[i]).map(|i| x[i]).collect();
    let m = median(&xi);
    let xpos: Vec<f64> = x.iter().map(|&v| v.max(1e-12 * m)).collect();

    // Work on with log(F)
    let z: Vec<f64> = xpos.iter().map(|&v| ln(v)).collect();

    // Average log(F) adjusted for d1
    let d1: Vec<f64> = df1.iter().map(|d| d / 2.0).collect();
    let e: Vec<f64> = (0..n).map(|i| z[i] + logmdigamma(at(&d1, i))).collect();
    let mut w: Vec<f64> = (0..n).map(|i| 1.0 / trigamma(at(&d1, i))).collect();
    if let Some(pwv) = &pw {
        for i in 0..n {
            w[i] *= pwv[i];
        }
    }
    let emean: Vec<f64> = match covariate {
        None if null_weights => vec![f64::NAN],
        None => {
            let sw: f64 = sum(&w);
            let swe: f64 = sum(&(0..n).map(|i| w[i] * e[i]).collect::<Vec<_>>());
            vec![swe / sw]
        }
        Some(c) => {
            let span = span.unwrap_or_else(|| choose_lowess_span(n, 500.0, 0.3, 1.0 / 3.0));
            let q75 = quantile7(&w, &[0.75])[0];
            let wn: Vec<f64> = w.iter().map(|v| v / q75).collect();
            loess_fit_bounded(&e, c, Some(&wn), span, 1, 1e-8, 1e2)
        }
    };

    // Log-likelihood function
    let d1x: Vec<f64> = (0..n).map(|i| at(&d1, i) * xpos[i]).collect();
    let minus_twice_log_lik = |par: f64| -> f64 {
        if null_weights {
            return -2.0 * 0.0;
        }
        let d2 = par / (1.0 - par);
        let lmd = logmdigamma(d2);
        let lg_d2 = lgammafn(d2);
        let mut terms = vec![0.0; n];
        for i in 0..n {
            let d1i = at(&d1, i);
            let d2s20 = d2 * exp(at(&emean, i) - lmd);
            let term =
                -(d1i + d2) * log1p(d1x[i] / d2s20) - d1i * ln(d2s20) + lgammafn(d1i + d2) - lg_d2;
            terms[i] = match &pw {
                Some(pwv) => pwv[i] * term,
                None => term,
            };
        }
        -2.0 * sum(&terms)
    };

    // Optimization
    let minimum = brent_fmin(0.5, 0.9998, minus_twice_log_lik, optimize_default_tol());
    let d2 = minimum / (1.0 - minimum);
    let lmd = logmdigamma(d2);
    let s20: Vec<f64> = emean.iter().map(|em| exp(em - lmd)).collect();

    // Finish here if robust=FALSE
    if !robust {
        return Ok(FDistUnequalFit {
            scale: s20,
            df2: 2.0 * d2,
            df2_outlier: None,
            df2_shrunk: None,
        });
    }

    // Use FDR to identify two-sided outliers
    let df2 = 2.0 * d2;
    let fstat: Vec<f64> = (0..n).map(|i| x[i] / at(&s20, i)).collect();
    let right_p: Vec<f64> = (0..n)
        .map(|i| pf(fstat[i], at(&df1, i), df2, false, false))
        .collect();
    let mut left_p: Vec<f64> = right_p.iter().map(|p| 1.0 - p).collect();
    if left_p.iter().cloned().fold(f64::INFINITY, f64::min) < 0.001 {
        for i in 0..n {
            if left_p[i] < 0.001 {
                left_p[i] = pf(fstat[i], at(&df1, i), df2, true, false);
            }
        }
    }
    let two_sided: Vec<f64> = (0..n).map(|i| 2.0 * left_p[i].min(right_p[i])).collect();
    let mut fdr = p_adjust_bh(&two_sided);
    for v in fdr.iter_mut() {
        if *v > 0.3 {
            *v = 1.0;
        }
    }

    // If no outliers, return non-robust estimates
    if fdr.iter().cloned().fold(f64::INFINITY, f64::min) == 1.0 {
        return Ok(FDistUnequalFit {
            scale: s20,
            df2,
            df2_outlier: None,
            df2_shrunk: None,
        });
    }

    // Refit F-distribution with FDR as prior weights
    let outpw = fit_f_dist_unequal_df1(&x, &df1, covariate, None, false, Some(&fdr))?;
    let s20 = outpw.scale;
    let df2 = outpw.df2;

    // Use qqplot-type method to identify right outliers
    let r = rank_average(&fstat);
    let pno: Vec<f64> = (0..n)
        .map(|i| {
            let uniform_p = (n as f64 - r[i] + 0.5) / n as f64;
            (right_p[i] / uniform_p).min(1.0)
        })
        .collect();

    // If no right outliers, return robust estimates without df2 shrinkage
    if pno.iter().cloned().fold(f64::INFINITY, f64::min) == 1.0 {
        return Ok(FDistUnequalFit {
            scale: s20,
            df2,
            df2_outlier: None,
            df2_shrunk: None,
        });
    }

    // Posterior df for right outliers
    let mut imin = 0;
    for i in 1..n {
        if right_p[i] < right_p[imin] {
            imin = i;
        }
    }
    let min_right_p = right_p[imin];
    let df2_outlier;
    let mut df2_shrunk: Vec<f64>;
    if min_right_p == 0.0 {
        df2_outlier = 0.0;
        df2_shrunk = pno.iter().map(|p| p * df2).collect();
    } else {
        let mut d = ln(0.5) / ln(min_right_p) * df2;
        // Iterate for accuracy
        let new_log_right_p = pf(fstat[imin], at(&df1, imin), d, false, true);
        d *= ln(0.5) / new_log_right_p;
        df2_outlier = d;
        df2_shrunk = (0..n).map(|i| pno[i] * df2 + (1.0 - pno[i]) * d).collect();
    }

    // Force df2.shrunk to be monotonic in RightP
    monotone_in(&mut df2_shrunk, &order_asc(&right_p));

    Ok(FDistUnequalFit {
        scale: s20,
        df2,
        df2_outlier: Some(df2_outlier),
        df2_shrunk: Some(df2_shrunk),
    })
}

// ---------------------------------------------------------------------------------------
// squeezeVar

/// `squeezeVar`'s result. `df_prior` and `var_prior` have length 1 or `n`.
#[derive(Debug, Clone)]
pub struct SqueezeVar {
    pub df_prior: Vec<f64>,
    pub var_prior: Vec<f64>,
    pub var_post: Vec<f64>,
}

/// `squeezeVar(var, df, covariate, span, robust, winsor.tail.p, legacy)`.
pub fn squeeze_var(
    var: &[f64],
    df: &[f64],
    covariate: Option<&[f64]>,
    span: Option<f64>,
    robust: bool,
    winsor_tail_p: [f64; 2],
    legacy: Option<bool>,
) -> Result<SqueezeVar> {
    let n = var.len();
    if n == 0 {
        return Err(LimmaError::Invalid("var is empty".into()));
    }
    if n < 3 {
        return Ok(SqueezeVar {
            df_prior: vec![0.0],
            var_prior: var.to_vec(),
            var_post: var.to_vec(),
        });
    }
    // When df==0, guard against missing or infinite values in var
    let mut var = var.to_vec();
    if df.len() > 1 {
        for i in 0..n {
            if df[i] == 0.0 {
                var[i] = 0.0;
            }
        }
    }
    let mut legacy = legacy;
    if span.is_some() {
        legacy = Some(false);
    }
    let legacy = legacy.unwrap_or_else(|| {
        let dfp: Vec<f64> = df.iter().cloned().filter(|&d| d > 0.0).collect();
        if dfp.is_empty() {
            return false;
        }
        let lo = dfp.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = dfp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        lo == hi
    });

    let (var_prior, df_prior): (Vec<f64>, Vec<f64>) = if legacy {
        if robust {
            let fit = fit_f_dist_robustly(&var, df, covariate, winsor_tail_p)?;
            (fit.scale, fit.df2_shrunk)
        } else {
            let fit = fit_f_dist(&var, df, covariate)?;
            (fit.scale, vec![fit.df2])
        }
    } else {
        let fit = fit_f_dist_unequal_df1(&var, df, covariate, span, robust, None)?;
        let df_prior = fit.df2_shrunk.unwrap_or_else(|| vec![fit.df2]);
        (fit.scale, df_prior)
    };
    if df_prior.iter().any(|v| v.is_nan()) {
        return Err(LimmaError::Invalid("Could not estimate prior df".into()));
    }
    let var_post = squeeze_var_post(&var, df, &var_prior, &df_prior);
    Ok(SqueezeVar {
        df_prior,
        var_prior,
        var_post,
    })
}

/// `.squeezeVar(var, df, var.prior, df.prior)`.
pub fn squeeze_var_post(var: &[f64], df: &[f64], var_prior: &[f64], df_prior: &[f64]) -> Vec<f64> {
    let n = var.len();
    let m = df_prior.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if m.is_finite() {
        return (0..n)
            .map(|i| {
                let d = at(df, i);
                let dp = at(df_prior, i);
                (d * var[i] + dp * at(var_prior, i)) / (d + dp)
            })
            .collect();
    }
    let mut var_post: Vec<f64> = (0..n).map(|i| at(var_prior, i)).collect();
    let m = df_prior.iter().cloned().fold(f64::INFINITY, f64::min);
    if m > 1e100 {
        return var_post;
    }
    for i in 0..n {
        let dp = at(df_prior, i);
        if dp.is_finite() {
            let d = at(df, i);
            var_post[i] = (d * var[i] + dp * var_post[i]) / (d + dp);
        }
    }
    var_post
}

// ---------------------------------------------------------------------------------------
// eBayes

/// `eBayes` arguments with limma's defaults.
#[derive(Debug, Clone)]
pub struct EBayesOptions {
    pub proportion: f64,
    pub stdev_coef_lim: [f64; 2],
    pub trend: bool,
    pub span: Option<f64>,
    pub robust: bool,
    pub winsor_tail_p: [f64; 2],
    pub legacy: Option<bool>,
}

impl Default for EBayesOptions {
    fn default() -> Self {
        EBayesOptions {
            proportion: 0.01,
            stdev_coef_lim: [0.1, 4.0],
            trend: false,
            span: None,
            robust: false,
            winsor_tail_p: [0.05, 0.1],
            legacy: None,
        }
    }
}

/// What `eBayes` adds to the `MArrayLM`. Matrices are `ngenes x ncoef` column-major.
#[derive(Debug, Clone)]
pub struct EBayes {
    /// Length 1 or `ngenes`.
    pub df_prior: Vec<f64>,
    /// Length 1 or `ngenes`.
    pub s2_prior: Vec<f64>,
    /// Length `ncoef`.
    pub var_prior: Vec<f64>,
    pub s2_post: Vec<f64>,
    pub t: Vec<f64>,
    pub df_total: Vec<f64>,
    pub p_value: Vec<f64>,
    pub lods: Vec<f64>,
    /// `None` when the design is not full rank.
    pub f: Option<Vec<f64>>,
    pub f_p_value: Option<Vec<f64>>,
    pub f_df1: Option<f64>,
    /// R warnings `eBayes` raised, in order.
    pub warnings: Vec<&'static str>,
}

pub const VAR_PRIOR_WARNING: &str = "Estimation of var.prior failed - set to default value";
pub const RECYCLE_WARNING: &str =
    "number of items to replace is not a multiple of replacement length";

/// `eBayes(fit, ...)`.
pub fn ebayes(fit: &MArrayLm, opts: &EBayesOptions) -> Result<EBayes> {
    let ngenes = fit.ngenes;
    let ncoef = fit.ncoef;
    if fit
        .df_residual
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        == 0.0
    {
        return Err(LimmaError::Invalid(
            "No residual degrees of freedom in linear model fits".into(),
        ));
    }
    if !fit.sigma.iter().any(|v| v.is_finite()) {
        return Err(LimmaError::Invalid(
            "No finite residual standard deviations".into(),
        ));
    }
    let covariate: Option<&[f64]> = if opts.trend {
        if fit.amean.is_empty() {
            return Err(LimmaError::Invalid(
                "Need Amean component in fit to estimate trend".into(),
            ));
        }
        Some(&fit.amean)
    } else {
        None
    };

    // Moderated t-statistic
    let sigma2: Vec<f64> = fit.sigma.iter().map(|s| s * s).collect();
    let sq = squeeze_var(
        &sigma2,
        &fit.df_residual,
        covariate,
        opts.span,
        opts.robust,
        opts.winsor_tail_p,
        opts.legacy,
    )?;
    let s2_prior = sq.var_prior;
    let s2_post = sq.var_post;
    let df_prior = sq.df_prior;
    let mut t = vec![0.0; ngenes * ncoef];
    for j in 0..ncoef {
        for g in 0..ngenes {
            let k = j * ngenes + g;
            t[k] = fit.coefficients[k] / fit.stdev_unscaled[k] / s2_post[g].sqrt();
        }
    }
    let df_pooled: f64 = sum(&fit
        .df_residual
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .collect::<Vec<_>>());
    let df_total: Vec<f64> = (0..ngenes)
        .map(|g| (fit.df_residual[g] + at(&df_prior, g)).min(df_pooled))
        .collect();
    let mut p_value = vec![0.0; ngenes * ncoef];
    for j in 0..ncoef {
        for g in 0..ngenes {
            let k = j * ngenes + g;
            p_value[k] = 2.0 * pt(-t[k].abs(), df_total[g], true, false);
        }
    }

    // B-statistic
    let med = median(&s2_prior);
    let var_prior_lim = [
        opts.stdev_coef_lim[0] * opts.stdev_coef_lim[0] / med,
        opts.stdev_coef_lim[1] * opts.stdev_coef_lim[1] / med,
    ];
    let mut var_prior = vec![0.0; ncoef];
    for j in 0..ncoef {
        var_prior[j] = tmixture_vector(
            &t[j * ngenes..(j + 1) * ngenes],
            &fit.stdev_unscaled[j * ngenes..(j + 1) * ngenes],
            &df_total,
            opts.proportion,
            Some(var_prior_lim),
        );
    }
    let mut warnings = Vec::new();
    let n_na = var_prior.iter().filter(|v| v.is_nan()).count();
    if n_na > 0 {
        // `out$var.prior[is.na(out$var.prior)] <- 1/out$s2.prior`, recycled as R does.
        let mut k = 0;
        for v in var_prior.iter_mut() {
            if v.is_nan() {
                *v = 1.0 / at(&s2_prior, k % s2_prior.len());
                k += 1;
            }
        }
        if n_na % s2_prior.len() != 0 {
            warnings.push(RECYCLE_WARNING);
        }
        warnings.push(VAR_PRIOR_WARNING);
    }
    let mut lods = vec![0.0; ngenes * ncoef];
    let log_odds = ln(opts.proportion / (1.0 - opts.proportion));
    for j in 0..ncoef {
        for g in 0..ngenes {
            let k = j * ngenes + g;
            let su2 = fit.stdev_unscaled[k] * fit.stdev_unscaled[k];
            let r = (su2 + var_prior[j]) / su2;
            let t2 = t[k] * t[k];
            let kernel = if at(&df_prior, g) > 1e6 {
                t2 * (1.0 - 1.0 / r) / 2.0
            } else {
                let dft = df_total[g];
                (1.0 + dft) / 2.0 * ln((t2 + dft) / (t2 / r + dft))
            };
            lods[k] = log_odds - ln(r) / 2.0 + kernel;
        }
    }

    // Overall F-statistic (classifyTestsF with fstat.only = TRUE)
    let (f, f_p_value, f_df1) = if is_fullrank(
        &fit.design,
        fit.narrays,
        fit.design.len() / fit.narrays.max(1),
    ) {
        let df2: Vec<f64> = (0..ngenes)
            .map(|g| at(&df_prior, g) + fit.df_residual[g])
            .collect();
        let (fstat, df1) = f_stat(&t, ngenes, ncoef, &fit.cov_coefficients);
        let fp: Vec<f64> = (0..ngenes)
            .map(|g| pf(fstat[g], df1, df2[g], false, false))
            .collect();
        (Some(fstat), Some(fp), Some(df1))
    } else {
        (None, None, None)
    };

    Ok(EBayes {
        df_prior,
        s2_prior,
        var_prior,
        s2_post,
        t,
        df_total,
        p_value,
        lods,
        f,
        f_p_value,
        f_df1,
        warnings,
    })
}

/// `classifyTestsF(fit, fstat.only = TRUE)`: the moderated F per gene and its `df1`.
/// `cov_coefficients` is `ntests x ntests` column-major.
pub fn f_stat(
    t: &[f64],
    ngenes: usize,
    ntests: usize,
    cov_coefficients: &[f64],
) -> (Vec<f64>, f64) {
    if ntests == 1 {
        return (t.iter().map(|v| v * v).collect(), 1.0);
    }
    let n = ntests;
    let mut cov = cov_coefficients.to_vec();
    // Adjust any coefficient variances exactly zero (usually caused by an all zero contrast)
    for i in 0..n {
        if cov[i * n + i] == 0.0 {
            cov[i * n + i] = 1.0;
        }
    }
    let cor = cov2cor(&cov, n);
    let (values, vectors) = eigen_symmetric(&cor, n);
    let r = values.iter().filter(|&&v| v / values[0] > 1e-8).count();
    let sqrt_r = (r as f64).sqrt();
    // Q <- .matvec(E$vectors[,1:r], 1/sqrt(E$values[1:r])) / sqrt(r)
    let mut q = vec![0.0; n * r];
    for j in 0..r {
        let s = 1.0 / values[j].sqrt() / sqrt_r;
        for k in 0..n {
            q[j * n + k] = vectors[j * n + k] * s;
        }
    }
    let mut f = vec![0.0; ngenes];
    for g in 0..ngenes {
        let mut acc = 0.0;
        for j in 0..r {
            let mut proj = 0.0;
            for k in 0..n {
                proj += t[k * ngenes + g] * q[j * n + k];
            }
            acc += proj * proj;
        }
        f[g] = acc;
    }
    (f, r as f64)
}

/// `tmixture.vector(tstat, stdev.unscaled, df, proportion, v0.lim)`.
pub fn tmixture_vector(
    tstat: &[f64],
    stdev_unscaled: &[f64],
    df: &[f64],
    proportion: f64,
    v0_lim: Option<[f64; 2]>,
) -> f64 {
    // Remove missing values
    let keep: Vec<usize> = (0..tstat.len()).filter(|&i| !tstat[i].is_nan()).collect();
    let mut tstat: Vec<f64> = keep.iter().map(|&i| tstat[i].abs()).collect();
    let stdev: Vec<f64> = keep.iter().map(|&i| stdev_unscaled[i]).collect();
    let df: Vec<f64> = keep.iter().map(|&i| df[i]).collect();

    let ngenes = tstat.len();
    let ntarget = (proportion / 2.0 * ngenes as f64).ceil() as usize;
    if ntarget < 1 {
        return f64::NAN;
    }
    // If ntarget is v small, ensure p at least matches selected proportion
    let p = (ntarget as f64 / ngenes as f64).max(proportion);

    // Method requires that df be equal
    let max_df = df.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    for i in 0..ngenes {
        if df[i] < max_df {
            let tail_p = pt(tstat[i], df[i], false, true);
            tstat[i] = qt(tail_p, max_df, false, true);
        }
    }

    // Select top statistics
    let o = order_desc(&tstat);
    let o = &o[..ntarget];
    let top: Vec<f64> = o.iter().map(|&i| tstat[i]).collect();
    let v1: Vec<f64> = o.iter().map(|&i| stdev[i] * stdev[i]).collect();

    // Compare to order statistics
    let mut v0 = vec![0.0; ntarget];
    for k in 0..ntarget {
        let r = (k + 1) as f64;
        let p0 = 2.0 * pt(top[k], max_df, false, false);
        let ptarget = ((r - 0.5) / ngenes as f64 - (1.0 - p) * p0) / p;
        if ptarget > p0 {
            let qtarget = qt(ptarget / 2.0, max_df, false, false);
            v0[k] = v1[k] * ((top[k] / qtarget) * (top[k] / qtarget) - 1.0);
        }
    }
    if let Some(lim) = v0_lim {
        for v in v0.iter_mut() {
            *v = v.max(lim[0]).min(lim[1]);
        }
    }
    mean(&v0)
}
