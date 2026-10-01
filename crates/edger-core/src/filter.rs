//! `filterByExpr.default` (edgeR 4.8.2, `R/filterByExpr.R`) with a design and the production
//! defaults `min.count = 10`, `min.total.count = 15`, `large.n = 10`, `min.prop = 0.7`, plus the
//! `hat()` leverage it uses (R `stats::hat`, `src/library/stats/R/lm.influence.R`) and the
//! package CPM (`cpm.default`, `src/compute_cpm.c`). Shared with the DESeq2 port.

use rnum::linalg::median;
use rnum::linpack::qr_decompose;

/// What `filterByExpr` computed, per gene and overall.
#[derive(Debug, Clone)]
pub struct FilterResult {
    pub keep: Vec<bool>,
    pub n_above_cutoff: Vec<f64>,
    pub total: Vec<f64>,
    pub min_sample_size: f64,
    pub cpm_cutoff: f64,
    pub lib_size: Vec<f64>,
}

/// `hat(x, intercept = TRUE)`: leverages of `cbind(1, x)` from its LINPACK QR (`tol = 1e-7`).
/// `x` is column-major `n x p`.
pub fn hat(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut xi = vec![1.0; n];
    xi.extend_from_slice(x);
    let qr = qr_decompose(&xi, n, p + 1, 1e-7);
    let mut h = vec![0.0; n];
    for k in 0..qr.rank {
        let mut e = vec![0.0; n];
        e[k] = 1.0;
        let q = qr.qy(&e);
        for i in 0..n {
            h[i] += q[i] * q[i];
        }
    }
    h
}

/// `filterByExpr(DGEList(counts), design)`. `counts` is gene-major (`nlib` per gene), the
/// design column-major `nlib x p`.
pub fn filter_by_expr(counts: &[f64], nlib: usize, design: &[f64], p: usize) -> FilterResult {
    let ngenes = counts.len().checked_div(nlib).unwrap_or(0);
    let mut lib_size = vec![0.0; nlib];
    for row in counts.chunks(nlib) {
        for (l, v) in lib_size.iter_mut().zip(row) {
            *l += v;
        }
    }
    let h = hat(design, nlib, p);
    let hmax = h.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut mss = 1.0 / hmax;
    if mss > 10.0 {
        mss = 10.0 + (mss - 10.0) * 0.7;
    }
    let cpm_cutoff = 10.0 / median(&lib_size) * 1e6;
    let tol = 1e-14;
    let mut keep = Vec::with_capacity(ngenes);
    let mut n_above_cutoff = Vec::with_capacity(ngenes);
    let mut total = Vec::with_capacity(ngenes);
    for row in counts.chunks(nlib) {
        let n = row
            .iter()
            .zip(&lib_size)
            .filter(|(&v, &l)| v * 1e6 / l >= cpm_cutoff)
            .count() as f64;
        let t: f64 = row.iter().sum();
        keep.push(n >= mss - tol && t >= 15.0 - tol);
        n_above_cutoff.push(n);
        total.push(t);
    }
    FilterResult {
        keep,
        n_above_cutoff,
        total,
        min_sample_size: mss,
        cpm_cutoff,
        lib_size,
    }
}
