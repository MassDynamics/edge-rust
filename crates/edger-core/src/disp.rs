//! `estimateDisp.default` with a design (edgeR 4.8.2, `R/estimateDisp.R`), as the reference's
//! `ref_estimate_disp` writes it out: APL grid over 21 points, `WLEB` (`R/WLEB.R`) with the
//! locfit trend, `squeezeVar` for the prior df, then common, trended and tagwise dispersions.
//! Also `.comboGroups` and `.residDF` (`R/residDF.R`).

use crate::apl::adjusted_profile_lik;
use crate::glm::{ave_log_cpm, glm_fit};
use crate::interp::maximize_interpolant;
use rnum::ebayes::squeeze_var;
use rnum::linpack::qr_decompose;
use rnum::locfit::{locfit, LocfitOptions};
use rnum::quad::choose_lowess_span;
use rnum::Result;
use std::collections::BTreeMap;

pub const NGRID: usize = 21;

/// Everything `estimateDisp` computes that the goldens record.
#[derive(Debug, Clone)]
pub struct Disp {
    pub sel: Vec<bool>,
    /// `nsel x 21`, row-major.
    pub l0: Vec<f64>,
    pub m0: Vec<f64>,
    pub glmfit005_deviance: Vec<f64>,
    pub prior_deviance: Vec<f64>,
    pub prior_df_residual: Vec<f64>,
    pub prior_s2: Vec<f64>,
    pub ave_logcpm: Vec<f64>,
    pub trended: Vec<f64>,
    pub tagwise: Vec<f64>,
    pub common: f64,
    pub overall_log2: f64,
    pub span: f64,
    pub prior_df: f64,
    pub prior_n: f64,
    pub squeezevar_legacy: bool,
    pub squeezevar_var_prior: Vec<f64>,
}

/// `.comboGroups(truths)`: rows with the same TRUE/FALSE pattern, grouped (gene-major `truths`).
pub(crate) fn combo_groups(truths: &[bool], ncol: usize) -> Vec<Vec<usize>> {
    let mut groups: BTreeMap<Vec<bool>, Vec<usize>> = BTreeMap::new();
    for (i, r) in truths.chunks(ncol).enumerate() {
        groups.entry(r.to_vec()).or_default().push(i);
    }
    groups.into_values().collect()
}

/// Rank of `qr(x[rows, ], tol = 1e-7)`.
fn qr_rank_rows(
    x: &[f64],
    n: usize,
    p: usize,
    rows: &[bool],
) -> (usize, Vec<usize>, Vec<f64>, usize) {
    let idx: Vec<usize> = (0..n).filter(|&i| rows[i]).collect();
    let m = idx.len();
    let mut sub = Vec::with_capacity(m * p);
    for j in 0..p {
        for &i in &idx {
            sub.push(x[j * n + i]);
        }
    }
    let qr = qr_decompose(&sub, m, p, 1e-7);
    (qr.rank, qr.pivot, sub, m)
}

/// `.residDF(zero, design)`.
pub(crate) fn resid_df(zero: &[bool], nlib: usize, x: &[f64], p: usize) -> Vec<f64> {
    let ng = zero.len() / nlib;
    let nzero: Vec<usize> = zero
        .chunks(nlib)
        .map(|r| r.iter().filter(|&&z| z).count())
        .collect();
    let mut df: Vec<f64> = vec![nlib as f64 - p as f64; ng];
    let mut some: Vec<usize> = Vec::new();
    for g in 0..ng {
        if nzero[g] == nlib {
            df[g] = 0.0;
        } else if nzero[g] > 0 {
            some.push(g);
        }
    }
    if !some.is_empty() {
        let zero2: Vec<bool> = some
            .iter()
            .flat_map(|&g| zero[g * nlib..(g + 1) * nlib].to_vec())
            .collect();
        for grp in combo_groups(&zero2, nlib) {
            let z = &zero2[grp[0] * nlib..(grp[0] + 1) * nlib];
            let nz: Vec<bool> = z.iter().map(|&v| !v).collect();
            let (rank, _, _, _) = qr_rank_rows(x, nlib, p, &nz);
            for &i in &grp {
                let g = some[i];
                df[g] = ((nlib - nzero[g]) as f64 - rank as f64).max(0.0);
            }
        }
    }
    df
}

/// `locfitByCol(y, x, span, degree = 0)` on a row-major `n x ncol` matrix.
fn locfit_by_col(y: &[f64], ncol: usize, x: &[f64], span: f64) -> Result<Vec<f64>> {
    let n = x.len();
    if span * (n as f64) < 2.0 || n <= 1 {
        return Ok(y.to_vec());
    }
    let opts = LocfitOptions {
        alpha: span,
        deg: 0,
        ..Default::default()
    };
    let mut out = vec![0.0; y.len()];
    for j in 0..ncol {
        let col: Vec<f64> = (0..n).map(|i| y[i * ncol + j]).collect();
        let f = locfit(x, &col, None, &opts)?.fitted();
        for i in 0..n {
            out[i * ncol + j] = f[i];
        }
    }
    Ok(out)
}

/// `estimateDisp(y, design)` on the kept counts (gene-major), with the effective library sizes
/// and their log as the offset row.
pub fn estimate_disp(
    y: &[f64],
    nlib: usize,
    x: &[f64],
    p: usize,
    lib_eff: &[f64],
    offset: &[f64],
) -> Result<Disp> {
    let ntags = y.len() / nlib;
    let sel: Vec<bool> = y
        .chunks(nlib)
        .map(|r| r.iter().sum::<f64>() >= 5.0)
        .collect();
    let sely: Vec<f64> = y
        .chunks(nlib)
        .zip(&sel)
        .filter(|(_, &s)| s)
        .flat_map(|(r, _)| r.to_vec())
        .collect();
    let nsel = sely.len() / nlib;
    let pts: Vec<f64> = (0..NGRID).map(|i| -10.0 + i as f64).collect();
    let grid: Vec<f64> = pts.iter().map(|t| 0.1 * 2f64.powf(*t)).collect();
    let mut l0 = vec![0.0; nsel * NGRID];

    let fit005 = glm_fit(&sely, nlib, x, p, offset, &vec![0.05; nsel], None);
    let zerofit: Vec<bool> = sely
        .iter()
        .zip(&fit005.fitted)
        .map(|(&c, &m)| c < 1e-4 && m < 1e-4)
        .collect();
    for subg in combo_groups(&zerofit, nlib) {
        let nz: Vec<bool> = zerofit[subg[0] * nlib..(subg[0] + 1) * nlib]
            .iter()
            .map(|&z| !z)
            .collect();
        if !nz.iter().any(|&v| v) {
            continue;
        }
        let (redesign, rp, libs): (Vec<f64>, usize, Vec<usize>) = if nz.iter().all(|&v| v) {
            (x.to_vec(), p, (0..nlib).collect())
        } else {
            let (rank, pivot, sub, m) = qr_rank_rows(x, nlib, p, &nz);
            let mut rd = Vec::with_capacity(m * rank);
            for &j in &pivot[..rank] {
                rd.extend_from_slice(&sub[j * m..(j + 1) * m]);
            }
            if m == rank {
                continue;
            }
            (rd, rank, (0..nlib).filter(|&i| nz[i]).collect())
        };
        let nl = libs.len();
        let mut cury = Vec::with_capacity(subg.len() * nl);
        for &g in &subg {
            for &l in &libs {
                cury.push(sely[g * nlib + l]);
            }
        }
        let curo: Vec<f64> = libs.iter().map(|&l| offset[l]).collect();
        let mut last: Option<Vec<f64>> = None;
        for (i, &dv) in grid.iter().enumerate() {
            let (apl, fit) =
                adjusted_profile_lik(dv, &cury, nl, &redesign, rp, &curo, last.as_deref());
            for (k, &g) in subg.iter().enumerate() {
                l0[g * NGRID + i] = apl[k];
            }
            last = Some(fit.coefficients);
        }
    }

    let colsum: Vec<f64> = (0..NGRID)
        .map(|j| (0..nsel).map(|g| l0[g * NGRID + j]).sum())
        .collect();
    let overall = maximize_interpolant(&pts, &colsum)[0];
    let common = 0.1 * 2f64.powf(overall);
    let ave = ave_log_cpm(y, nlib, lib_eff, common, 2.0);
    let ave_sel: Vec<f64> = ave
        .iter()
        .zip(&sel)
        .filter(|(_, &s)| s)
        .map(|(&a, _)| a)
        .collect();

    let span = choose_lowess_span(nsel, 50.0, 0.3, 1.0 / 3.0);
    let m0 = locfit_by_col(&l0, NGRID, &ave_sel, span)?;
    let trend: Vec<f64> = maximize_interpolant(&pts, &m0)
        .iter()
        .map(|t| 0.1 * 2f64.powf(*t))
        .collect();
    let mut imin = 0;
    for i in 1..nsel {
        if ave_sel[i] < ave_sel[imin] {
            imin = i;
        }
    }
    let mut trended = vec![trend[imin]; ntags];
    let mut k = 0;
    for g in 0..ntags {
        if sel[g] {
            trended[g] = trend[k];
            k += 1;
        }
    }

    let fit2 = glm_fit(&sely, nlib, x, p, offset, &trend, None);
    let zero2: Vec<bool> = sely
        .iter()
        .zip(&fit2.fitted)
        .map(|(&c, &m)| c < 1e-4 && m < 1e-4)
        .collect();
    let dfr = resid_df(&zero2, nlib, x, p);
    let s2: Vec<f64> = fit2
        .deviance
        .iter()
        .zip(&dfr)
        .map(|(&d, &df)| if df == 0.0 { 0.0 } else { (d / df).max(0.0) })
        .collect();
    let dfp: Vec<f64> = dfr.iter().cloned().filter(|&d| d > 0.0).collect();
    let sv_legacy = !dfp.is_empty()
        && dfp.iter().cloned().fold(f64::INFINITY, f64::min)
            == dfp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sv = squeeze_var(&s2, &dfr, Some(&ave_sel), None, false, [0.05, 0.1], None)?;
    let prior_df = sv.df_prior[0];
    let prior_n = prior_df / (nlib as f64 - p as f64);
    let mut tagwise = trended.clone();
    if prior_n <= 1e6 {
        let pn = prior_n.min(1e6);
        let l0a: Vec<f64> = l0.iter().zip(&m0).map(|(a, b)| a + pn * b).collect();
        let ind = maximize_interpolant(&pts, &l0a);
        let mut k = 0;
        for g in 0..ntags {
            if sel[g] {
                tagwise[g] = 0.1 * 2f64.powf(ind[k]);
                k += 1;
            }
        }
    }
    Ok(Disp {
        sel,
        l0,
        m0,
        glmfit005_deviance: fit005.deviance,
        prior_deviance: fit2.deviance,
        prior_df_residual: dfr,
        prior_s2: s2,
        ave_logcpm: ave,
        trended,
        tagwise,
        common,
        overall_log2: overall,
        span,
        prior_df,
        prior_n,
        squeezevar_legacy: sv_legacy,
        squeezevar_var_prior: sv.var_prior,
    })
}
