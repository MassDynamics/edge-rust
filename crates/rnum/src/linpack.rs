//! LINPACK QR with limited column pivoting, as used by R's `qr()` and
//! `lm.fit()` (`src/appl/dqrdc2.f`, `dqrsl.f`, `dqrls.f`).
//!
//! R decides which design columns are *non-estimable* by `dqrdc2`'s pivot
//! rule (a column whose remaining norm falls below `tol` times its original
//! norm is moved to the end). limma's `lm.series` relies on that exact rule
//! per gene when NAs mask different rows, so the Fortran is ported
//! line-for-line rather than replaced by a Householder QR from a crate.
//!
//! Matrices are column-major `Vec<f64>` with explicit `(n, p)`.

use crate::{LimmaError, Result};

/// Reference BLAS `DNRM2` as of LAPACK 3.10 (Blue's scaled sum of squares), unit stride.
///
/// R 4.5 ships this version in its reference BLAS, so `qr()` and `lm.fit()` norms
/// match R bit for bit.
pub fn dnrm2(x: &[f64]) -> f64 {
    let tsml = 2f64.powi(-511);
    let tbig = 2f64.powi(486);
    let ssml = 2f64.powi(537);
    let sbig = 2f64.powi(-538);
    if x.is_empty() {
        return 0.0;
    }
    let (mut asml, mut amed, mut abig) = (0.0f64, 0.0f64, 0.0f64);
    let mut notbig = true;
    for &v in x {
        let ax = v.abs();
        if ax > tbig {
            abig += (ax * sbig) * (ax * sbig);
            notbig = false;
        } else if ax < tsml {
            if notbig {
                asml += (ax * ssml) * (ax * ssml);
            }
        } else {
            amed += ax * ax;
        }
    }
    let (scl, sumsq);
    if abig > 0.0 {
        if amed > 0.0 || amed > f64::MAX || amed.is_nan() {
            abig += (amed * sbig) * sbig;
        }
        scl = 1.0 / sbig;
        sumsq = abig;
    } else if asml > 0.0 {
        if amed > 0.0 || amed > f64::MAX || amed.is_nan() {
            let amed = amed.sqrt();
            let asml = asml.sqrt() / ssml;
            let (ymin, ymax) = if asml > amed {
                (amed, asml)
            } else {
                (asml, amed)
            };
            scl = 1.0;
            sumsq = ymax * ymax * (1.0 + (ymin / ymax) * (ymin / ymax));
        } else {
            scl = 1.0 / ssml;
            sumsq = asml;
        }
    } else {
        scl = 1.0;
        sumsq = amed;
    }
    scl * sumsq.sqrt()
}

fn ddot(x: &[f64], y: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() {
        s += x[i] * y[i];
    }
    s
}

/// `y += a * x` (reference `daxpy` skips `a == 0`).
fn daxpy(a: f64, x: &[f64], y: &mut [f64]) {
    if a == 0.0 {
        return;
    }
    for i in 0..x.len() {
        y[i] += a * x[i];
    }
}

/// Result of `dqrdc2`: the packed QR in `qr` (column-major `n x p`),
/// `qraux`, the 0-based column `pivot`, and the `rank`.
#[derive(Debug, Clone)]
pub struct Qr {
    pub qr: Vec<f64>,
    pub n: usize,
    pub p: usize,
    pub qraux: Vec<f64>,
    pub pivot: Vec<usize>,
    pub rank: usize,
    pub tol: f64,
}

/// `dqrdc2`: Householder QR with limited column pivoting. `x` is
/// column-major `n x p` and is overwritten with the packed decomposition.
/// Returns the rank `k`.
pub fn dqrdc2(
    x: &mut [f64],
    n: usize,
    p: usize,
    tol: f64,
    qraux: &mut [f64],
    jpvt: &mut [usize],
    work: &mut [f64],
) -> usize {
    // work is p x 2, column-major: work[j] = work(j,1), work[p + j] = work(j,2)
    let col = |j: usize| j * n;
    if n > 0 {
        for j in 0..p {
            let nrm = dnrm2(&x[col(j)..col(j) + n]);
            qraux[j] = nrm;
            work[j] = nrm;
            work[p + j] = if nrm == 0.0 { 1.0 } else { nrm };
        }
    }
    let lup = n.min(p);
    let mut k = p + 1; // 1-based as in Fortran
    for l in 1..=lup {
        // l is 1-based here; li = l - 1 is the 0-based index.
        loop {
            let li = l - 1;
            if l >= k || qraux[li] >= work[p + li] * tol {
                break;
            }
            // Move column l to the end, shifting the others left.
            for i in 0..n {
                let t = x[col(li) + i];
                for j in (l + 1)..=p {
                    x[col(j - 2) + i] = x[col(j - 1) + i];
                }
                x[col(p - 1) + i] = t;
            }
            let i = jpvt[li];
            let t = qraux[li];
            let tt = work[li];
            let ttt = work[p + li];
            for j in (l + 1)..=p {
                jpvt[j - 2] = jpvt[j - 1];
                qraux[j - 2] = qraux[j - 1];
                work[j - 2] = work[j - 1];
                work[p + j - 2] = work[p + j - 1];
            }
            jpvt[p - 1] = i;
            qraux[p - 1] = t;
            work[p - 1] = tt;
            work[p + p - 1] = ttt;
            k -= 1;
        }
        let li = l - 1;
        if l != n {
            // Householder transformation for column l.
            let mut nrmxl = dnrm2(&x[col(li) + li..col(li) + n]);
            if nrmxl != 0.0 {
                let xll = x[col(li) + li];
                if xll != 0.0 {
                    nrmxl = nrmxl.abs() * if xll < 0.0 { -1.0 } else { 1.0 };
                }
                let s = 1.0 / nrmxl;
                for i in li..n {
                    x[col(li) + i] *= s;
                }
                x[col(li) + li] += 1.0;
                // Apply the transformation to the remaining columns,
                // updating the norms.
                for j in (l + 1)..=p {
                    let ji = j - 1;
                    let (head, tail) = x.split_at_mut(col(ji));
                    let xl = &head[col(li) + li..col(li) + n];
                    let xj = &mut tail[li..n];
                    let t = -ddot(xl, xj) / xl[0];
                    daxpy(t, xl, xj);
                    if qraux[ji] != 0.0 {
                        let r = xj[0].abs() / qraux[ji];
                        let mut tt = 1.0 - r * r;
                        tt = tt.max(0.0);
                        let t = tt;
                        if t.abs() >= 1e-6 {
                            qraux[ji] *= t.sqrt();
                        } else {
                            qraux[ji] = dnrm2(&xj[1..]);
                            work[ji] = qraux[ji];
                        }
                    }
                }
                // Save the transformation.
                qraux[li] = x[col(li) + li];
                x[col(li) + li] = -nrmxl;
            }
        }
    }
    (k - 1).min(n)
}

/// R's `qr(x, tol)` (LINPACK, the default `LAPACK = FALSE`).
pub fn qr_decompose(x: &[f64], n: usize, p: usize, tol: f64) -> Qr {
    assert_eq!(x.len(), n * p, "qr_decompose: x is not n x p");
    let mut qr = x.to_vec();
    let mut qraux = vec![0.0; p];
    let mut pivot: Vec<usize> = (0..p).collect();
    let mut work = vec![0.0; 2 * p];
    let rank = if p == 0 {
        0
    } else {
        dqrdc2(&mut qr, n, p, tol, &mut qraux, &mut pivot, &mut work)
    };
    Qr {
        qr,
        n,
        p,
        qraux,
        pivot,
        rank,
        tol,
    }
}

impl Qr {
    /// Column `j` of the packed matrix, rows `from..n`.
    fn col(&self, j: usize, from: usize) -> &[f64] {
        &self.qr[j * self.n + from..j * self.n + self.n]
    }

    /// Apply the j-th Householder reflector (with the diagonal temporarily
    /// replaced by `qraux[j]`, as `dqrsl` does) to `v[j..n]`.
    fn apply_reflector(&self, j: usize, v: &mut [f64]) {
        if self.qraux[j] == 0.0 {
            return;
        }
        let xj = self.col(j, j);
        let d = self.qraux[j];
        // ddot with the virtual column (d, xj[1..])
        let mut dot = d * v[j];
        for i in 1..xj.len() {
            dot += xj[i] * v[j + i];
        }
        let t = -dot / d;
        if t == 0.0 {
            return;
        }
        v[j] += t * d;
        for i in 1..xj.len() {
            v[j + i] += t * xj[i];
        }
    }

    /// `qr.qty`: returns `Q' y` for one column `y` of length `n`.
    pub fn qty(&self, y: &[f64]) -> Vec<f64> {
        assert_eq!(y.len(), self.n);
        let mut v = y.to_vec();
        let ju = self.rank.min(self.n.saturating_sub(1));
        for j in 0..ju {
            self.apply_reflector(j, &mut v);
        }
        v
    }

    /// `qr.qy`: returns `Q y`.
    pub fn qy(&self, y: &[f64]) -> Vec<f64> {
        assert_eq!(y.len(), self.n);
        let mut v = y.to_vec();
        let ju = self.rank.min(self.n.saturating_sub(1));
        for j in (0..ju).rev() {
            self.apply_reflector(j, &mut v);
        }
        v
    }

    /// Back-solve `R b = qty[0..rank]` (the `cb` branch of `dqrsl`).
    /// Coefficients are in *pivoted* order, length `rank`.
    pub(crate) fn coef_pivoted(&self, qty: &[f64]) -> Result<Vec<f64>> {
        let k = self.rank;
        let mut b: Vec<f64> = qty[..k].to_vec();
        for j in (0..k).rev() {
            let rjj = self.qr[j * self.n + j];
            if rjj == 0.0 {
                return Err(LimmaError::Invalid(format!(
                    "dqrsl: zero diagonal in R at column {}",
                    j + 1
                )));
            }
            b[j] /= rjj;
            let t = -b[j];
            if j > 0 {
                let (head, _) = b.split_at_mut(j);
                daxpy(t, &self.qr[j * self.n..j * self.n + j], head);
            }
        }
        Ok(b)
    }

    /// `chol2inv(qr, size = rank)`: `(R'R)^{-1}` for the leading
    /// `rank x rank` block of `R`, column-major. Columns are in pivoted order.
    pub fn chol2inv(&self) -> Vec<f64> {
        let k = self.rank;
        let r = |i: usize, j: usize| self.qr[j * self.n + i];
        crate::linalg::chol2inv_upper(
            &(0..k * k)
                .map(|idx| r(idx % k, idx / k))
                .collect::<Vec<_>>(),
            k,
        )
    }
}

/// Result of R's `lm.fit(x, y)` for one response vector.
#[derive(Debug, Clone)]
pub struct LmFit {
    /// Length `p`, in the original column order; `NaN` marks a
    /// non-estimable coefficient (R's `NA`).
    pub coefficients: Vec<f64>,
    pub residuals: Vec<f64>,
    /// `Q' y` (length `n`); rows `rank..n` are the residual effects.
    pub effects: Vec<f64>,
    pub fitted_values: Vec<f64>,
    pub rank: usize,
    pub df_residual: usize,
    pub qr: Qr,
}

/// `lm.fit(x, y, tol = 1e-7)` for a single response. `x` is column-major
/// `n x p`.
pub fn lm_fit(x: &[f64], n: usize, p: usize, y: &[f64], tol: f64) -> Result<LmFit> {
    if n == 0 {
        return Err(LimmaError::Invalid("0 (non-NA) cases".into()));
    }
    if y.len() != n {
        return Err(LimmaError::Invalid("incompatible dimensions".into()));
    }
    let qr = qr_decompose(x, n, p, tol);
    let k = qr.rank;
    let (effects, residuals, fitted, coef_piv) = if k > 0 {
        let qty = qr.qty(y);
        let b = qr.coef_pivoted(&qty)?;
        // rsd = Q * (0,...,0, qty[k..]); xb = Q * (qty[..k], 0, ...)
        let mut rsd_in = vec![0.0; n];
        rsd_in[k..].copy_from_slice(&qty[k..]);
        let rsd = qr.qy(&rsd_in);
        let mut xb_in = vec![0.0; n];
        xb_in[..k].copy_from_slice(&qty[..k]);
        let xb = qr.qy(&xb_in);
        (qty, rsd, xb, b)
    } else {
        (y.to_vec(), y.to_vec(), vec![0.0; n], Vec::new())
    };
    // Un-pivot: coef[r2] <- NA; coef[pivot] <- coef
    let mut coefficients = vec![f64::NAN; p];
    for (j, &bj) in coef_piv.iter().enumerate() {
        coefficients[qr.pivot[j]] = bj;
    }
    // lm.fit returns fitted.values = y - residuals (not xb from dqrsl).
    let fitted_values: Vec<f64> = y.iter().zip(&residuals).map(|(a, b)| a - b).collect();
    let _ = fitted;
    Ok(LmFit {
        coefficients,
        residuals,
        effects,
        fitted_values,
        rank: k,
        df_residual: n - k,
        qr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_rank_small() {
        // x = [1 1 1 1; 0 1 2 3]', y = 1 + 2*x2
        let n = 4;
        let p = 2;
        let x = vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 3.0, 5.0, 7.0];
        let f = lm_fit(&x, n, p, &y, 1e-7).unwrap();
        assert_eq!(f.rank, 2);
        assert!((f.coefficients[0] - 1.0).abs() < 1e-12);
        assert!((f.coefficients[1] - 2.0).abs() < 1e-12);
        assert!(f.residuals.iter().all(|r| r.abs() < 1e-12));
        assert_eq!(f.df_residual, 2);
    }

    #[test]
    fn rank_deficient_pivots_last_column() {
        // Third column is the sum of the first two.
        let n = 5;
        let p = 3;
        let mut x = vec![0.0; n * p];
        for i in 0..n {
            x[i] = 1.0;
            x[n + i] = i as f64;
            x[2 * n + i] = 1.0 + i as f64;
        }
        let y = vec![2.0, 3.1, 3.9, 5.2, 6.0];
        let f = lm_fit(&x, n, p, &y, 1e-7).unwrap();
        assert_eq!(f.rank, 2);
        assert_eq!(f.qr.pivot, vec![0, 1, 2]);
        assert!(f.coefficients[2].is_nan());
        assert!(f.coefficients[0].is_finite());
        assert_eq!(f.df_residual, 3);
        let ci = f.qr.chol2inv();
        assert_eq!(ci.len(), 4);
        // (X'X)^-1 for [1, i]: X'X = [[5,10],[10,30]] -> inv = [[0.6,-0.2],[-0.2,0.1]]
        assert!((ci[0] - 0.6).abs() < 1e-12);
        assert!((ci[1] + 0.2).abs() < 1e-12);
        assert!((ci[3] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn deficient_middle_column_moves_to_end() {
        // Second column is zero: pivot must become [0, 2, 1].
        let n = 4;
        let p = 3;
        let x = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 3.0, 5.0, 7.0];
        let f = lm_fit(&x, n, p, &y, 1e-7).unwrap();
        assert_eq!(f.rank, 2);
        assert_eq!(f.qr.pivot, vec![0, 2, 1]);
        assert!(f.coefficients[1].is_nan());
        assert!((f.coefficients[2] - 2.0).abs() < 1e-12);
    }
}
