//! Natural cubic spline basis `splines::ns()` as limma's `fitFDist` uses it
//! (`ns(covariate, df = splinedf, intercept = TRUE)` and `predict.ns`).
//! Ported from R 4.5.0 `src/library/splines/{src/splines.c, R/splines.R,
//! R/splineClasses.R}`.

#![allow(clippy::needless_range_loop)]

use crate::linalg::{cmp_nan_last, quantile7};
use crate::linpack::qr_decompose;
use crate::{LimmaError, Result};

const ORD: usize = 4;

struct Spl<'a> {
    knots: &'a [f64],
    curs: isize,
    boundary: bool,
    ldel: [f64; ORD],
    rdel: [f64; ORD],
    a: [f64; ORD],
}

impl<'a> Spl<'a> {
    fn set_cursor(&mut self, x: f64) -> isize {
        self.curs = -1;
        self.boundary = false;
        let nk = self.knots.len();
        for i in 0..nk {
            if self.knots[i] >= x {
                self.curs = i as isize;
            }
            if self.knots[i] > x {
                break;
            }
        }
        if self.curs > (nk - ORD) as isize {
            let last_legit = nk - ORD;
            if x == self.knots[last_legit] {
                self.boundary = true;
                self.curs = last_legit as isize;
            }
        }
        self.curs
    }

    fn diff_table(&mut self, x: f64, ndiff: usize) {
        let c = self.curs as usize;
        for i in 0..ndiff {
            self.rdel[i] = self.knots[c + i] - x;
            self.ldel[i] = x - self.knots[c - (i + 1)];
        }
    }

    fn basis_funcs(&mut self, x: f64, b: &mut [f64]) {
        self.diff_table(x, ORD - 1);
        b[0] = 1.0;
        for j in 1..=ORD - 1 {
            let mut saved = 0.0;
            for r in 0..j {
                let den = self.rdel[r] + self.ldel[j - 1 - r];
                if den != 0.0 {
                    let term = b[r] / den;
                    b[r] = saved + self.rdel[r] * term;
                    saved = self.ldel[j - 1 - r] * term;
                } else {
                    if r != 0 || self.rdel[r] != 0.0 {
                        b[r] = saved;
                    }
                    saved = 0.0;
                }
            }
            b[j] = saved;
        }
    }

    /// `evaluate()` from splines.c: value or `nder`-th derivative of the
    /// spline with coefficients `self.a` at `x`.
    fn evaluate(&mut self, x: f64, nder: usize) -> f64 {
        let ti = self.curs as usize; // index into knots
        let mut outer = ORD - 1;
        if self.boundary && nder == ORD - 1 {
            return 0.0;
        }
        let mut nder = nder;
        while nder > 0 {
            nder -= 1;
            // for(inner = outer, apt = a, lpt = ti - outer; inner--; apt++, lpt++)
            for k in 0..outer {
                let lpt = ti - outer + k;
                self.a[k] = outer as f64 * (self.a[k + 1] - self.a[k])
                    / (self.knots[lpt + outer] - self.knots[lpt]);
            }
            outer -= 1;
        }
        self.diff_table(x, outer);
        while outer > 0 {
            outer -= 1;
            // for(apt = a, lpt = ldel + outer, rpt = rdel, inner = outer + 1; inner--; lpt--, rpt++, apt++)
            for k in 0..=outer {
                let l = self.ldel[outer - k];
                let r = self.rdel[k];
                self.a[k] = (self.a[k + 1] * l + self.a[k] * r) / (r + l);
            }
        }
        self.a[0]
    }
}

/// `splineDesign(knots, x, ord = 4, derivs)` (dense, `outer.ok = FALSE`).
/// `derivs` is recycled over `x`. Returns column-major `nx x (nk - 4)`.
pub fn spline_design(knots: &[f64], x: &[f64], derivs: &[usize]) -> Result<Vec<f64>> {
    let nk = knots.len();
    if nk < 2 * ORD - 1 {
        return Err(LimmaError::Invalid(format!(
            "need at least 2*ord -1 (={}) knots",
            2 * ORD - 1
        )));
    }
    let mut kn = knots.to_vec();
    kn.sort_by(|a, b| cmp_nan_last(*a, *b));
    let degree = ORD - 1;
    for &xi in x {
        if xi < kn[ORD - 1] || kn[nk - degree - 1] < xi {
            return Err(LimmaError::Invalid(format!(
                "the 'x' data must be in the range {} to {} unless you set 'outer.ok = TRUE'",
                kn[ORD - 1],
                kn[nk - degree - 1]
            )));
        }
    }
    let nx = x.len();
    let ncoef = nk - ORD;
    let mut design = vec![0.0; nx * ncoef];
    let mut sp = Spl {
        knots: &kn,
        curs: -1,
        boundary: false,
        ldel: [0.0; ORD],
        rdel: [0.0; ORD],
        a: [0.0; ORD],
    };
    let nd = derivs.len().max(1);
    for (i, &xi) in x.iter().enumerate() {
        sp.set_cursor(xi);
        let io = sp.curs - ORD as isize;
        let der = if derivs.is_empty() { 0 } else { derivs[i % nd] };
        let mut vals = [0.0; ORD];
        if io < 0 || io > nk as isize {
            for v in vals.iter_mut() {
                *v = f64::NAN;
            }
        } else if der > 0 {
            if der >= ORD {
                return Err(LimmaError::Invalid(format!(
                    "derivs = {} >= ord = {}, but should be in {{0,..,ord-1}}",
                    der, ORD
                )));
            }
            for ii in 0..ORD {
                sp.a = [0.0; ORD];
                sp.a[ii] = 1.0;
                vals[ii] = sp.evaluate(xi, der);
            }
        } else {
            sp.basis_funcs(xi, &mut vals);
        }
        // jj = 1:ord + offset  (offset = io), 1-based columns io+1 .. io+ord
        for j in 0..ORD {
            let col = io as usize + j;
            if col < ncoef {
                design[col * nx + i] = vals[j];
            }
        }
    }
    Ok(design)
}

/// A fitted `ns()` basis with the attributes needed by `predict.ns`.
#[derive(Debug, Clone)]
pub struct NsBasis {
    /// Column-major `n x ncol`.
    pub basis: Vec<f64>,
    pub n: usize,
    pub ncol: usize,
    pub knots: Vec<f64>,
    pub boundary_knots: [f64; 2],
    pub intercept: bool,
}

fn seq_interior(n_iknots: usize) -> Vec<f64> {
    // seq.int(0, 1, length.out = nIknots + 2)[-c(1, nIknots + 2)]
    let len = n_iknots + 2;
    let by = 1.0 / (len - 1) as f64;
    (1..=n_iknots).map(|i| i as f64 * by).collect()
}

/// `splines::ns(x, df = df, intercept = intercept)` with default
/// `Boundary.knots = range(x)`; `x` must be finite.
pub fn ns_df(x: &[f64], df: usize, intercept: bool) -> Result<NsBasis> {
    if x.is_empty() {
        return Err(LimmaError::Invalid("ns: empty x".into()));
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in x {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let boundary = if x.len() == 1 {
        [x[0] * 7.0 / 8.0, x[0] * 9.0 / 8.0]
    } else {
        [lo, hi]
    };
    let n_iknots = (df as isize - 1 - intercept as isize).max(0) as usize;
    let mut knots = if n_iknots > 0 {
        quantile7(x, &seq_interior(n_iknots))
    } else {
        Vec::new()
    };
    // shove interior knots matching boundary knots inside
    if !knots.is_empty() {
        let kmin = knots.iter().cloned().fold(f64::INFINITY, f64::min);
        let kmax = knots.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if kmin == boundary[0] {
            let piv = boundary[0];
            let above = knots
                .iter()
                .cloned()
                .filter(|&k| k > piv)
                .fold(f64::INFINITY, f64::min);
            if above.is_infinite() {
                return Err(LimmaError::Invalid(
                    "all interior knots match left boundary knot".into(),
                ));
            }
            for k in knots.iter_mut() {
                if *k == piv {
                    *k += (above - piv) / 8.0;
                }
            }
        }
        if kmax == boundary[1] {
            let piv = boundary[1];
            let below = knots
                .iter()
                .cloned()
                .filter(|&k| k < piv)
                .fold(f64::NEG_INFINITY, f64::max);
            if below.is_infinite() {
                return Err(LimmaError::Invalid(
                    "all interior knots match right boundary knot".into(),
                ));
            }
            for k in knots.iter_mut() {
                if *k == piv {
                    *k -= (piv - below) / 8.0;
                }
            }
        }
    }
    ns_with_knots(x, &knots, boundary, intercept, false)
}

/// `predict.ns` / `ns(x, knots = , Boundary.knots = , intercept = )`.
pub fn ns_predict(basis: &NsBasis, newx: &[f64]) -> Result<NsBasis> {
    ns_with_knots(
        newx,
        &basis.knots,
        basis.boundary_knots,
        basis.intercept,
        true,
    )
}

fn ns_with_knots(
    x: &[f64],
    knots: &[f64],
    boundary: [f64; 2],
    intercept: bool,
    boundary_given: bool,
) -> Result<NsBasis> {
    let nx = x.len();
    let n_iknots = knots.len();
    let mut aknots: Vec<f64> = Vec::with_capacity(n_iknots + 8);
    for _ in 0..4 {
        aknots.push(boundary[0]);
        aknots.push(boundary[1]);
    }
    aknots.extend_from_slice(knots);
    aknots.sort_by(|a, b| cmp_nan_last(*a, *b));
    let ncoef = n_iknots + 4;

    let mut basis = vec![0.0; nx * ncoef];
    let outside: Vec<bool> = if boundary_given {
        x.iter()
            .map(|&v| v < boundary[0] || v > boundary[1])
            .collect()
    } else {
        vec![false; nx]
    };
    if outside.iter().any(|&o| o) {
        // linear extrapolation beyond the boundary knots
        for side in 0..2 {
            let k_pivot = boundary[side];
            let rows: Vec<usize> = (0..nx)
                .filter(|&i| {
                    outside[i]
                        && if side == 0 {
                            x[i] < boundary[0]
                        } else {
                            x[i] > boundary[1]
                        }
                })
                .collect();
            if rows.is_empty() {
                continue;
            }
            let tt = spline_design(&aknots, &[k_pivot, k_pivot], &[0, 1])?; // 2 x ncoef
            for &i in &rows {
                let d = x[i] - k_pivot;
                for j in 0..ncoef {
                    basis[j * nx + i] = tt[j * 2] + d * tt[j * 2 + 1];
                }
            }
        }
        let inside: Vec<usize> = (0..nx).filter(|&i| !outside[i]).collect();
        if !inside.is_empty() {
            let xin: Vec<f64> = inside.iter().map(|&i| x[i]).collect();
            let b = spline_design(&aknots, &xin, &[0])?;
            for (r, &i) in inside.iter().enumerate() {
                for j in 0..ncoef {
                    basis[j * nx + i] = b[j * xin.len() + r];
                }
            }
        }
    } else {
        basis = spline_design(&aknots, x, &[0])?;
    }
    // const: second derivatives at the boundary knots, 2 x ncoef
    let mut konst = spline_design(&aknots, &boundary, &[2, 2])?;
    let mut ncol_full = ncoef;
    if !intercept {
        // drop the first column of both
        konst = konst[2..].to_vec();
        basis = basis[nx..].to_vec();
        ncol_full -= 1;
    }
    // qr.const <- qr(t(const)); basis <- t(qr.qty(qr.const, t(basis)))[, -(1:2)]
    // t(const) is ncol_full x 2
    let mut tconst = vec![0.0; ncol_full * 2];
    for j in 0..ncol_full {
        tconst[j] = konst[j * 2];
        tconst[ncol_full + j] = konst[j * 2 + 1];
    }
    let qr = qr_decompose(&tconst, ncol_full, 2, 1e-7);
    let ncol = ncol_full - 2;
    let mut out = vec![0.0; nx * ncol];
    let mut row = vec![0.0; ncol_full];
    for i in 0..nx {
        for j in 0..ncol_full {
            row[j] = basis[j * nx + i];
        }
        let qty = qr.qty(&row);
        for j in 0..ncol {
            out[j * nx + i] = qty[j + 2];
        }
    }
    Ok(NsBasis {
        basis: out,
        n: nx,
        ncol,
        knots: knots.to_vec(),
        boundary_knots: boundary,
        intercept,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ns_dimensions_and_predict_matches_inside() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 / 7.0).collect();
        let b = ns_df(&x, 4, true).unwrap();
        assert_eq!(b.ncol, 4);
        assert_eq!(b.knots.len(), 2);
        let p = ns_predict(&b, &x).unwrap();
        for (a, c) in b.basis.iter().zip(&p.basis) {
            assert!((a - c).abs() < 1e-12);
        }
        // extrapolation is linear beyond the boundary
        let e = ns_predict(&b, &[-1.0, -2.0, -3.0]).unwrap();
        for j in 0..e.ncol {
            let d1 = e.basis[j * 3 + 1] - e.basis[j * 3];
            let d2 = e.basis[j * 3 + 2] - e.basis[j * 3 + 1];
            assert!((d1 - d2).abs() < 1e-10);
        }
    }

    #[test]
    fn ns_df1_is_linear_in_x() {
        // df = 1, intercept = TRUE: R bumps to 0 interior knots, which
        // gives 4 - 2 = 2 columns spanning the linear functions.
        let x: Vec<f64> = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let b = ns_df(&x, 1, true).unwrap();
        assert_eq!(b.ncol, 2);
        // second differences ~ 0 for equally spaced x
        for j in 0..2 {
            for i in 2..x.len() {
                let c = &b.basis[j * x.len()..];
                let dd = c[i] - 2.0 * c[i - 1] + c[i - 2];
                assert!(dd.abs() < 1e-12, "not linear: {}", dd);
            }
        }
    }
}
