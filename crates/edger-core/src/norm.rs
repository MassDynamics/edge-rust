//! `calcNormFactors.default` (edgeR 4.8.2, `R/calcNormFactors.R`) for the methods production
//! offers except `TMMwsp`: `TMM` (`.calcFactorTMM`), `RLE` (`.calcFactorRLE`), `upperquartile`
//! (`.calcFactorQuantile`) and `none`, as the reference's `plain_norm_factors` writes them out.

use rnum::linalg::{mean, median, quantile7, rank_average};
use rnum::{LimmaError, Result};

/// Normalisation factors (scaled to unit geometric mean) and, for TMM, the 0-based reference
/// column.
#[derive(Debug, Clone)]
pub struct NormFactors {
    pub norm_factors: Vec<f64>,
    pub ref_column: Option<usize>,
}

/// `.calcFactorTMM(obs, ref, libsize.obs, libsize.ref, logratioTrim = 0.3, sumTrim = 0.05)`.
fn tmm(obs: &[f64], rf: &[f64], n_o: f64, n_r: f64) -> f64 {
    let mut log_r = Vec::new();
    let mut abs_e = Vec::new();
    let mut v = Vec::new();
    for (&o, &r) in obs.iter().zip(rf) {
        let lr = ((o / n_o) / (r / n_r)).log2();
        let ae = ((o / n_o).log2() + (r / n_r).log2()) / 2.0;
        let vv = (n_o - o) / n_o / o + (n_r - r) / n_r / r;
        if lr.is_finite() && ae.is_finite() && ae > -1e10 {
            log_r.push(lr);
            abs_e.push(ae);
            v.push(vv);
        }
    }
    if log_r.iter().fold(0.0f64, |m, x| m.max(x.abs())) < 1e-6 {
        return 1.0;
    }
    let n = log_r.len() as f64;
    let lo_l = (n * 0.3).floor() + 1.0;
    let hi_l = n + 1.0 - lo_l;
    let lo_s = (n * 0.05).floor() + 1.0;
    let hi_s = n + 1.0 - lo_s;
    let rl = rank_average(&log_r);
    let re = rank_average(&abs_e);
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..log_r.len() {
        if rl[i] >= lo_l && rl[i] <= hi_l && re[i] >= lo_s && re[i] <= hi_s {
            let a = log_r[i] / v[i];
            let b = 1.0 / v[i];
            if !a.is_nan() {
                num += a;
            }
            if !b.is_nan() {
                den += b;
            }
        }
    }
    let f = num / den;
    if f.is_nan() {
        1.0
    } else {
        2f64.powf(f)
    }
}

/// `calcNormFactors(y, lib.size, method)`; `counts` gene-major. `TMMwsp` is not ported.
pub fn calc_norm_factors(
    counts: &[f64],
    nlib: usize,
    lib: &[f64],
    method: &str,
) -> Result<NormFactors> {
    let x: Vec<&[f64]> = counts
        .chunks(nlib)
        .filter(|r| r.iter().any(|&v| v > 0.0))
        .collect();
    let mut method = method;
    if !matches!(method, "TMM" | "RLE" | "upperquartile" | "none") {
        return Err(LimmaError::Invalid(format!(
            "edger_norm_method '{method}' is not supported by the Rust edgeR engine"
        )));
    }
    if x.is_empty() || nlib == 1 {
        method = "none";
    }
    let col = |j: usize| -> Vec<f64> { x.iter().map(|r| r[j]).collect() };
    let q75 = || -> Vec<f64> {
        (0..nlib)
            .map(|j| quantile7(&col(j), &[0.75])[0] / lib[j])
            .collect()
    };
    let mut ref_column = None;
    let f: Vec<f64> = match method {
        "TMM" => {
            let f75 = q75();
            let r = if median(&f75) < 1e-20 {
                let s: Vec<f64> = (0..nlib)
                    .map(|j| col(j).iter().map(|v| v.sqrt()).sum())
                    .collect();
                argmax(&s)
            } else {
                let m = mean(&f75);
                let d: Vec<f64> = f75.iter().map(|v| -(v - m).abs()).collect();
                argmax(&d)
            };
            ref_column = Some(r);
            let rc = col(r);
            (0..nlib)
                .map(|j| tmm(&col(j), &rc, lib[j], lib[r]))
                .collect()
        }
        "RLE" => {
            let gm: Vec<f64> = x
                .iter()
                .map(|r| (r.iter().map(|v| v.ln()).sum::<f64>() / nlib as f64).exp())
                .collect();
            (0..nlib)
                .map(|j| {
                    let u: Vec<f64> = x
                        .iter()
                        .zip(&gm)
                        .filter(|(_, &g)| g > 0.0)
                        .map(|(r, &g)| r[j] / g)
                        .collect();
                    median(&u) / lib[j]
                })
                .collect()
        }
        "upperquartile" => q75(),
        _ => vec![1.0; nlib],
    };
    let lm = mean(&f.iter().map(|v| v.ln()).collect::<Vec<_>>()).exp();
    Ok(NormFactors {
        norm_factors: f.iter().map(|v| v / lm).collect(),
        ref_column,
    })
}

/// R's `which.max`: first index of the maximum.
fn argmax(x: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..x.len() {
        if x[i] > x[best] {
            best = i;
        }
    }
    best
}
