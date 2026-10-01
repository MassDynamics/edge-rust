//! One-predictor local regression as R's `locfit` package computes it (locfit 1.5-9.12), for the
//! two callers in this fleet:
//!
//! - edgeR's `locfitByCol` (`R/locfitByCol.R`), reached from `WLEB(trend.method="locfit")`:
//!   `fitted(locfit(y ~ x, weights = w, alpha = span, deg = 0))`.
//! - DESeq2's local dispersion fit (`R/core.R`, `estimateDispersionsFit(fitType="local")`):
//!   `locfit(logDisps ~ logMeans, weights = w)` (degree 2, `alpha = 0.7`), then `predict()`.
//!
//! What is ported, with the C source it comes from (`locfit/src`):
//!
//! - Bandwidth (`lf_nbhd.c`: `nbhd`, `compbandwid`, `kordstat`): `nn = (int)(n*alpha + 1e-12)`;
//!   `h` is the `nn`-th smallest `|x_i - x0|` when `nn < n`, else `max|x_i - x0| * nn/n`.
//! - Weights (`weight.c`: `weightsph`, `W`): tricube `(1-u^3)^3` for `u = |x_i - x0|/h <= 1`;
//!   when `h == 0` only points at distance 0 get weight 1. Prior weights enter the likelihood.
//! - Local fit (`locfit.c`: `lfinit`, `reginit`, `lfiter`; `lf_fitfun.c`: `fitfun`;
//!   `family.c` gaussian/identity; `m_max.c`: `max_nr`): basis `[1, dx, dx^2/2, ...]`, start at
//!   the weighted mean, one Newton step (`likereg` with the gaussian family returns `NR_BREAK`
//!   after the first step), solved by `jacob_solve` with `JAC_EIGD` (`m_jacob.c`) and the Jacobi
//!   eigen decomposition `eig_dec`/`eig_solve` (`m_eigen.c`), ported literally including the
//!   quirk that a coordinate below the rank tolerance is passed through instead of zeroed.
//! - Evaluation structure `ev = "tree"` (`ev_atree.c`: `atree_guessnv`, `atree_split`,
//!   `atree_grow`, `atree_start`, `atree_int`; `ev_main.c`: `newsplit`), with `cut = 0.8`,
//!   pseudo-vertices, and the vertex cap (`newsplit: out of vertex space`).
//! - Interpolation (`ev_interp.c`: `exvval`, `exvvalpv`, `rectcell_interp`, `linear_interp`,
//!   `hermite2`, `dointpoint`): linear between vertex values for degree 0, cubic Hermite on
//!   value and slope for degree >= 1, the end cell extended linearly outside the fitted range.
//!
//! - The parametric component (`pcomp.c`: `compparcomp`, `subparcomp`, `addparcomp`): a global
//!   degree-`deg` fit at the prior-weighted mean of `x` (all points, unit kernel weights) is
//!   subtracted from each vertex's value and slope before interpolating and added back at the
//!   evaluation point. Inside the data range this is an identity up to rounding; outside it, it
//!   is what makes locfit extrapolate with the global polynomial rather than linearly.
//!
//! Deliberately not ported:
//!
//! - Variance, confidence bands, degrees of freedom (`comp_vari`, `lf_vcov`): neither caller
//!   reads them.
//! - Anything with more than one predictor, other families, links, kernels or `ev` types.
//!
//! See `crates/edger-core/tests/stage_locfit.rs` for the edgeR gate.

use crate::{LimmaError, Result};

/// Options for [`locfit`]. Defaults are locfit's own (`alpha = 0.7`, `deg = 2`, `cut = 0.8`,
/// `maxk = 100`).
#[derive(Debug, Clone, Copy)]
pub struct LocfitOptions {
    /// Nearest-neighbour fraction (`alpha`, edgeR's `span`).
    pub alpha: f64,
    /// Local polynomial degree (`deg`).
    pub deg: usize,
    /// Tree refinement cut (`ev = rbox(cut = ...)`).
    pub cut: f64,
    /// Vertex-space inflation (`maxk`), as a percentage.
    pub maxk: usize,
}

impl Default for LocfitOptions {
    fn default() -> Self {
        LocfitOptions {
            alpha: 0.7,
            deg: 2,
            cut: 0.8,
            maxk: 100,
        }
    }
}

/// A vertex of the adaptive tree.
#[derive(Debug, Clone, Copy)]
struct Vertex {
    x: f64,
    /// Bandwidth used at this vertex (mean of the parents for a pseudo-vertex).
    h: f64,
    /// Local fit value and slope (the slope is only used for `deg >= 1`).
    val: f64,
    slope: f64,
    /// Pseudo-vertex: not fitted, its value is interpolated from the parent cell.
    pseudo: bool,
}

/// A cell of the tree: its two end vertices and, once split, `(midpoint, left, right)`.
#[derive(Debug, Clone, Copy)]
struct Cell {
    lo: usize,
    hi: usize,
    split: Option<(usize, usize, usize)>,
}

/// A fitted `locfit` object, ready to evaluate with [`Locfit::predict`].
#[derive(Debug, Clone)]
pub struct Locfit {
    vertices: Vec<Vertex>,
    cells: Vec<Cell>,
    hasd: bool,
    x: Vec<f64>,
    /// Parametric component: centre and coefficients on `[1, dx, dx^2/2, ...]`.
    xbar: f64,
    pc: Vec<f64>,
}

struct Data<'a> {
    x: &'a [f64],
    y: &'a [f64],
    w: Option<&'a [f64]>,
    nn: usize,
    deg: usize,
    cut: f64,
    nvm: usize,
    flo: f64,
    fhi: f64,
}

impl Data<'_> {
    fn prwt(&self, i: usize) -> f64 {
        match self.w {
            Some(w) => w[i],
            None => 1.0,
        }
    }
}

/// Fit `locfit(y ~ x, weights = w, alpha, deg)` with the default `ev = "tree"` structure.
///
/// `w = None` means unit prior weights. Errors on length mismatch, empty input, non-finite `x`,
/// or when the tree outgrows locfit's vertex space (R's `newsplit: out of vertex space`).
pub fn locfit(x: &[f64], y: &[f64], w: Option<&[f64]>, opts: &LocfitOptions) -> Result<Locfit> {
    let n = x.len();
    if n == 0 || y.len() != n || w.is_some_and(|w| w.len() != n) {
        return Err(LimmaError::Invalid(
            "locfit: x, y and weights must be non-empty and the same length".into(),
        ));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(LimmaError::Invalid("locfit: non-finite x".into()));
    }
    let flo = x.iter().copied().fold(f64::INFINITY, f64::min);
    let fhi = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // atree_guessnv with d = 1 (vc = 2).
    let cut = if opts.cut < 0.01 { 0.01 } else { opts.cut };
    let alp = opts.alpha;
    let mut nvm: usize = 1 << 30;
    if alp > 0.0 {
        let a0 = if alp > 1.0 { 1.0 } else { 1.0 / alp };
        let cu = cut.min(1.0);
        nvm = nvm.min(((5.0 * a0 / cu + 1.0) * 2.0) as usize);
    }
    if nvm == 1 << 30 {
        nvm = 102 * 2;
    }
    let nvm = (opts.maxk as f64 / 100.0 * nvm as f64) as usize;

    let d = Data {
        x,
        y,
        w,
        nn: (n as f64 * alp + 1e-12) as usize,
        deg: opts.deg,
        cut,
        nvm,
        flo,
        fhi,
    };

    // compparcomp: global fit at the prior-weighted mean of x, unit kernel weights.
    let mut sw = 0.0;
    let mut sx = 0.0;
    for (i, &xi) in x.iter().enumerate() {
        sw += d.prwt(i);
        sx += xi * d.prwt(i);
    }
    let xbar = sx / sw;
    let all: Vec<usize> = (0..n).collect();
    let pc = local_fit(&d, xbar, &all, &vec![1.0; n]).unwrap_or_else(|| vec![0.0; d.deg + 1]);

    // atree_start: fit both ends, then grow.
    let mut vertices = Vec::new();
    for xv in [flo, fhi] {
        let (h, val, slope) = fit_vertex(&d, xv, xbar, &pc);
        vertices.push(Vertex {
            x: xv,
            h,
            val,
            slope,
            pseudo: false,
        });
    }
    let mut cells = vec![Cell {
        lo: 0,
        hi: 1,
        split: None,
    }];
    grow(&d, (xbar, &pc), &mut vertices, &mut cells, 0)?;

    Ok(Locfit {
        vertices,
        cells,
        hasd: opts.deg > 0,
        x: x.to_vec(),
        xbar,
        pc,
    })
}

/// `atree_split` (d = 1): does the cell with these end vertices need splitting?
fn needs_split(d: &Data, vs: &[Vertex], lo: usize, hi: usize) -> bool {
    let mut hmin = 0.0;
    for i in [lo, hi] {
        let h = vs[i].h;
        if h > 0.0 && (hmin == 0.0 || h < hmin) {
            hmin = h;
        }
    }
    let (ll, ur) = (vs[lo].x, vs[hi].x);
    let score = if hmin == 0.0 {
        2.0 * (ur - ll) / (d.fhi - d.flo)
    } else {
        (ur - ll) / hmin
    };
    d.cut < score
}

/// `atree_grow` + `newsplit` (d = 1): recursive midpoint splitting, left half first.
fn grow(d: &Data, pc: (f64, &[f64]), vs: &mut Vec<Vertex>, cells: &mut Vec<Cell>, c: usize) -> Result<()> {
    let (lo, hi) = (cells[c].lo, cells[c].hi);
    if !needs_split(d, vs, lo, hi) {
        return Ok(());
    }
    let le = vs[hi].x - vs[lo].x;
    let pv = le < d.cut * vs[lo].h.min(vs[hi].h);
    if vs.len() == d.nvm {
        return Err(LimmaError::Invalid("newsplit: out of vertex space".into()));
    }
    let xm = (vs[lo].x + vs[hi].x) / 2.0;
    let v = if pv {
        Vertex {
            x: xm,
            h: (vs[lo].h + vs[hi].h) / 2.0,
            val: 0.0,
            slope: 0.0,
            pseudo: true,
        }
    } else {
        let (h, val, slope) = fit_vertex(d, xm, pc.0, pc.1);
        Vertex {
            x: xm,
            h,
            val,
            slope,
            pseudo: false,
        }
    };
    let mid = vs.len();
    vs.push(v);
    let left = cells.len();
    cells.push(Cell {
        lo,
        hi: mid,
        split: None,
    });
    let right = left + 1;
    cells.push(Cell {
        lo: mid,
        hi,
        split: None,
    });
    cells[c].split = Some((mid, left, right));
    grow(d, pc, vs, cells, left)?;
    grow(d, pc, vs, cells, right)
}

/// `procvraw` at one vertex: bandwidth, local fit, minus the parametric component.
/// Returns `(h, value, slope)`.
fn fit_vertex(d: &Data, x0: f64, xbar: f64, pc: &[f64]) -> (f64, f64, f64) {
    let n = d.x.len();
    let di: Vec<f64> = d.x.iter().map(|&xi| (xi - x0).abs()).collect();
    // compbandwid
    let h = if d.nn == 0 {
        0.0
    } else if d.nn < n {
        let mut s = di.clone();
        s.sort_by(|a, b| a.total_cmp(b));
        s[d.nn - 1]
    } else {
        di.iter().copied().fold(0.0, f64::max) * (d.nn as f64 / n as f64)
    };
    // nbhd: keep the points with positive weight, in data order.
    let mut ind = Vec::new();
    let mut wt = Vec::new();
    for i in 0..n {
        let w = if h == 0.0 {
            if di[i] == 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            let u = di[i] / h;
            if u > 1.0 {
                0.0
            } else {
                let t = 1.0 - u * u * u;
                t * t * t
            }
        };
        if w > 0.0 {
            ind.push(i);
            wt.push(w);
        }
    }
    // LF_NOPT: locfit leaves the coefficients unset; report zero.
    let cf = local_fit(d, x0, &ind, &wt).unwrap_or_else(|| vec![0.0; d.deg + 1]);
    // subparcomp
    let (pv, ps) = pc_eval(pc, x0 - xbar);
    let slope = if cf.len() > 1 { cf[1] - ps } else { 0.0 };
    (h, cf[0] - pv, slope)
}

/// The parametric component and its slope at `dx = x - xbar` (`fitfun` + `innerprod`).
fn pc_eval(pc: &[f64], dx: f64) -> (f64, f64) {
    let p = pc.len();
    let mut f = vec![0.0; p];
    f[0] = 1.0;
    for j in 1..p {
        f[j] = f[j - 1] * dx / j as f64;
    }
    let v: f64 = pc.iter().zip(&f).map(|(a, b)| a * b).sum();
    // derivative basis: [0, 1, dx, dx^2/2, ...]
    let mut g = vec![0.0; p];
    if p > 1 {
        g[1] = 1.0;
        for j in 2..p {
            g[j] = g[j - 1] * dx / (j - 1) as f64;
        }
    }
    let s: f64 = pc.iter().zip(&g).map(|(a, b)| a * b).sum();
    (v, s)
}

/// `locfit()` at `x0` over the points `ind` with kernel weights `wt`: `reginit` start, one
/// Newton step through `jacob_solve` (JAC_EIGD). `None` when no point has weight (`LF_NOPT`).
fn local_fit(d: &Data, x0: f64, ind: &[usize], wt: &[f64]) -> Option<Vec<f64>> {
    let p = d.deg + 1;
    // reginit (gaussian, identity link).
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    for (k, &i) in ind.iter().enumerate() {
        let pw = d.prwt(i);
        s1 += wt[k] * pw * d.y[i];
        s0 += wt[k] * pw;
    }
    if s0 == 0.0 {
        return None;
    }
    let mut cf = vec![0.0; p];
    cf[0] = s1 / s0;
    // likereg at the start: Z = sum w prwt X X', f1 = sum w X prwt (y - theta).
    let mut z = vec![0.0; p * p];
    let mut f1 = vec![0.0; p];
    let mut xr = vec![0.0; p];
    for (k, &i) in ind.iter().enumerate() {
        let dx = d.x[i] - x0;
        xr[0] = 1.0;
        for j in 1..p {
            xr[j] = xr[j - 1] * dx / j as f64;
        }
        let theta: f64 = xr.iter().zip(&cf).map(|(a, b)| a * b).sum();
        let pw = d.prwt(i);
        let res = pw * (d.y[i] - theta);
        let ww = wt[k];
        for a in 0..p {
            f1[a] += ww * xr[a] * res;
            for b in 0..p {
                z[a * p + b] += ww * pw * xr[a] * xr[b];
            }
        }
    }
    // jacob_solve with JAC_EIGD.
    let dg: Vec<f64> = (0..p)
        .map(|i| {
            let zi = z[i * (p + 1)];
            if zi <= 0.0 {
                0.0
            } else {
                1.0 / zi.sqrt()
            }
        })
        .collect();
    for i in 0..p {
        for j in 0..p {
            z[i * p + j] *= dg[i] * dg[j];
        }
    }
    let q = eig_dec(&mut z, p);
    for (f, g) in f1.iter_mut().zip(&dg) {
        *f *= g;
    }
    let rank = eig_solve(&z, &q, &mut f1, p);
    for (f, g) in f1.iter_mut().zip(&dg) {
        *f *= g;
    }
    if rank > 0 {
        for (c, f) in cf.iter_mut().zip(&f1) {
            *c += f;
        }
    }
    Some(cf)
}

/// `eig_dec` (m_eigen.c): cyclic Jacobi, at most 20 sweeps. `x` (row-major `d x d`) is
/// overwritten with the diagonalised matrix; returns the rotation matrix `P`.
fn eig_dec(x: &mut [f64], d: usize) -> Vec<f64> {
    let mut p = vec![0.0; d * d];
    for i in 0..d {
        p[i * d + i] = 1.0;
    }
    for _ in 0..20 {
        let mut ms = false;
        for i in 0..d {
            for j in (i + 1)..d {
                if x[i * d + j] * x[i * d + j] > 1.0e-15 * (x[i * d + i] * x[j * d + j]).abs() {
                    let mut c = (x[j * d + j] - x[i * d + i]) / 2.0;
                    let mut s = -x[i * d + j];
                    let r = (c * c + s * s).sqrt();
                    c /= r;
                    s = ((1.0 - c) / 2.0).sqrt() * if s > 0.0 { 1.0 } else { -1.0 };
                    c = ((1.0 + c) / 2.0).sqrt();
                    for k in 0..d {
                        let u = x[i * d + k];
                        let v = x[j * d + k];
                        x[i * d + k] = u * c + v * s;
                        x[j * d + k] = v * c - u * s;
                    }
                    for k in 0..d {
                        let u = x[k * d + i];
                        let v = x[k * d + j];
                        x[k * d + i] = u * c + v * s;
                        x[k * d + j] = v * c - u * s;
                    }
                    x[i * d + j] = 0.0;
                    x[j * d + i] = 0.0;
                    for k in 0..d {
                        let u = p[k * d + i];
                        let v = p[k * d + j];
                        p[k * d + i] = u * c + v * s;
                        p[k * d + j] = v * c - u * s;
                    }
                    ms = true;
                }
            }
        }
        if !ms {
            break;
        }
    }
    p
}

/// `eig_solve` (m_eigen.c) with `e_tol = 1e-8 * max diag`. Returns the rank.
fn eig_solve(dmat: &[f64], q: &[f64], x: &mut [f64], d: usize) -> usize {
    let mut mx = dmat[0];
    for i in 1..d {
        if dmat[i * (d + 1)] > mx {
            mx = dmat[i * (d + 1)];
        }
    }
    let tol = 1.0e-8 * mx;
    let mut w = vec![0.0; d];
    for (i, wi) in w.iter_mut().enumerate() {
        for j in 0..d {
            *wi += q[j * d + i] * x[j];
        }
    }
    let mut rank = 0;
    for (i, wi) in w.iter_mut().enumerate() {
        if dmat[i * d + i] > tol {
            *wi /= dmat[i * (d + 1)];
            rank += 1;
        }
    }
    for (i, xi) in x.iter_mut().enumerate() {
        *xi = 0.0;
        for j in 0..d {
            *xi += q[i * d + j] * w[j];
        }
    }
    rank
}

/// `hermite2` (ev_interp.c).
fn hermite2(x: f64, z: f64) -> [f64; 4] {
    if z == 0.0 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let h = x / z;
    if h < 0.0 {
        return [1.0, 0.0, h, 0.0];
    }
    if h > 1.0 {
        return [0.0, 1.0, 0.0, h - 1.0];
    }
    let p1 = h * h * (3.0 - 2.0 * h);
    [1.0 - p1, p1, h * (1.0 - h) * (1.0 - h), h * h * (h - 1.0)]
}

impl Locfit {
    /// `exvval` with `what = PCOEF`: value, plus slope when the fit has derivatives.
    fn exvval(&self, v: usize) -> [f64; 2] {
        let vx = &self.vertices[v];
        [vx.val, if self.hasd { vx.slope } else { 0.0 }]
    }

    /// Evaluate the fit at `x` (`predict.locfit` -> `dointpoint`): the tree interpolant plus the
    /// parametric component. Outside the data range the end cell's interpolant is extended
    /// linearly, as `hermite2`/`linear_interp` do, and the global polynomial is added back.
    pub fn predict(&self, x: f64) -> f64 {
        self.interpolate(x) + pc_eval(&self.pc, x - self.xbar).0
    }

    /// `atree_int`: the tree interpolant of the vertex values (parametric component removed).
    fn interpolate(&self, x: f64) -> f64 {
        let mut c = 0usize;
        let mut vl = self.exvval(self.cells[0].lo);
        let mut vr = self.exvval(self.cells[0].hi);
        while let Some((mid, left, right)) = self.cells[c].split {
            let ll = self.vertices[self.cells[c].lo].x;
            let ur = self.vertices[self.cells[c].hi].x;
            let h = ur - ll;
            let vm = if self.vertices[mid].pseudo {
                // exvvalpv
                if self.hasd {
                    let f0 = (vl[0] + vr[0]) / 2.0 + h * (vl[1] - vr[1]) / 8.0;
                    let f1 = 1.5 * (vr[0] - vl[0]) / h - (vl[1] + vr[1]) / 4.0;
                    [f0, f1]
                } else {
                    [(vl[0] + vr[0]) / 2.0, 0.0]
                }
            } else {
                self.exvval(mid)
            };
            if 2.0 * (x - ll) < h {
                vr = vm;
                c = left;
            } else {
                vl = vm;
                c = right;
            }
        }
        let ll = self.vertices[self.cells[c].lo].x;
        let ur = self.vertices[self.cells[c].hi].x;
        if self.hasd {
            let mut phi = hermite2(x - ll, ur - ll);
            phi[2] *= ur - ll;
            phi[3] *= ur - ll;
            phi[0] * vl[0] + phi[1] * vr[0] + phi[2] * vl[1] + phi[3] * vr[1]
        } else {
            // linear_interp
            let dd = ur - ll;
            if dd == 0.0 {
                return vl[0];
            }
            let hh = x - ll;
            ((dd - hh) * vl[0] + hh * vr[0]) / dd
        }
    }

    /// Evaluate at several points (`predict(fit, newdata)`).
    pub fn predict_many(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|&x| self.predict(x)).collect()
    }

    /// Fitted values at the data points (`fitted.locfit`).
    pub fn fitted(&self) -> Vec<f64> {
        self.predict_many(&self.x)
    }

    /// Number of tree vertices, pseudo-vertices included (locfit's `fp$nv`).
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degree_zero_reproduces_a_constant() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.37).sin() * 3.0).collect();
        let y = vec![2.5; 50];
        let opts = LocfitOptions {
            alpha: 0.3,
            deg: 0,
            ..Default::default()
        };
        let f = locfit(&x, &y, None, &opts).unwrap();
        for v in f.fitted() {
            assert!((v - 2.5).abs() < 1e-12);
        }
    }

    #[test]
    fn degree_two_matches_r_on_a_quadratic() {
        // R 4.5.0, locfit 1.5-9.12: x <- (0:79)/7; y <- 1 - 0.5*x + 0.25*x^2
        // predict(locfit(y ~ x), c(0, 2.8214285714285716, 5.5, 11.285714285714286)).
        // Not exact at the ends: the one-sided local design is near-singular and locfit's
        // eigen solve passes the small direction through undivided.
        let x: Vec<f64> = (0..80).map(|i| i as f64 / 7.0).collect();
        let y: Vec<f64> = x.iter().map(|&v| 1.0 - 0.5 * v + 0.25 * v * v).collect();
        let f = locfit(&x, &y, None, &LocfitOptions::default()).unwrap();
        let got = f.predict_many(&[0.0, 2.8214285714285716, 5.5, 11.285714285714286]);
        let want = [
            0.99999996132834923,
            1.5794005090170637,
            5.8124999999896128,
            27.19897979886062,
        ];
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() <= 1e-13 * w.abs(), "{g} vs {w}");
        }
    }

    #[test]
    fn weighted_fits_match_r() {
        // R 4.5.0, locfit 1.5-9.12:
        // i <- 0:199; x <- sin(i*1.7)*4 + i/50; y <- cos(i*0.9) + 0.1*x^2 - 0.3*x
        // w <- 1 + (i %% 7)/3
        // predict(locfit(y ~ x, weights = w), c(-3.9, -1.234, 0, 2.5, 7.3, 9))
        // predict(locfit(y ~ x, weights = w, alpha = 0.3, deg = 0), c(-3.9, -1.234, 0, 2.5, 7.3))
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 1.7).sin() * 4.0 + i as f64 / 50.0).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.9).cos() + 0.1 * x[i] * x[i] - 0.3 * x[i])
            .collect();
        let w: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64 / 3.0).collect();
        let f2 = locfit(&x, &y, Some(&w), &LocfitOptions::default()).unwrap();
        let got = f2.predict_many(&[-3.9, -1.234, 0.0, 2.5, 7.3, 9.0]);
        let want = [
            2.416726078833467,
            0.49167716869886235,
            -0.026824384464920682,
            -0.17859354174577674,
            3.0659188793571168,
            5.2269393158652262,
        ];
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() <= 1e-12 * w.abs(), "deg 2: {g} vs {w}");
        }
        let opts = LocfitOptions {
            alpha: 0.3,
            deg: 0,
            ..Default::default()
        };
        let f0 = locfit(&x, &y, Some(&w), &opts).unwrap();
        let got = f0.predict_many(&[-3.9, -1.234, 0.0, 2.5, 7.3]);
        let want = [
            1.3081583758750677,
            0.48185497143669015,
            0.025059043135785664,
            -0.12817023159359109,
            2.1254135107890617,
        ];
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() <= 1e-12 * w.abs(), "deg 0: {g} vs {w}");
        }
    }
}
