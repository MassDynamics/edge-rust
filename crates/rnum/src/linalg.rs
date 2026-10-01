//! Small dense linear algebra and R vector helpers used by the limma port.
//! All matrices are column-major `Vec<f64>` with explicit dimensions.

use crate::{LimmaError, Result};

/// `t(a) %*% b` for `a` (`n x p`) and `b` (`n x q`); result `p x q`.
pub fn crossprod(a: &[f64], n: usize, p: usize, b: &[f64], q: usize) -> Vec<f64> {
    let mut out = vec![0.0; p * q];
    for j in 0..q {
        for i in 0..p {
            let mut s = 0.0;
            for k in 0..n {
                s += a[i * n + k] * b[j * n + k];
            }
            out[j * p + i] = s;
        }
    }
    out
}

/// `a %*% b` for `a` (`n x p`) and `b` (`p x q`); result `n x q`.
pub fn matmul(a: &[f64], n: usize, p: usize, b: &[f64], q: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * q];
    for j in 0..q {
        for k in 0..p {
            let bkj = b[j * p + k];
            if bkj == 0.0 {
                continue;
            }
            for i in 0..n {
                out[j * n + i] += a[k * n + i] * bkj;
            }
        }
    }
    out
}

/// Transpose of an `n x p` matrix.
pub fn transpose(a: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * p];
    for j in 0..p {
        for i in 0..n {
            out[i * p + j] = a[j * n + i];
        }
    }
    out
}

/// Upper Cholesky factor `U` with `U'U = a` (R's `chol()`, LAPACK dpotrf).
/// Only the upper triangle of `a` is read.
pub fn chol_upper(a: &[f64], n: usize) -> Result<Vec<f64>> {
    let mut u = vec![0.0; n * n];
    for j in 0..n {
        let mut s = a[j * n + j];
        for k in 0..j {
            s -= u[j * n + k] * u[j * n + k];
        }
        if s <= 0.0 || !s.is_finite() {
            return Err(LimmaError::Invalid(format!(
                "the leading minor of order {} is not positive",
                j + 1
            )));
        }
        let ujj = s.sqrt();
        u[j * n + j] = ujj;
        for i in (j + 1)..n {
            let mut t = a[i * n + j];
            for k in 0..j {
                t -= u[j * n + k] * u[i * n + k];
            }
            u[i * n + j] = t / ujj;
        }
    }
    Ok(u)
}

/// Inverse of an upper-triangular `n x n` matrix (LAPACK dtrtri, upper,
/// non-unit).
pub fn inv_upper(r: &[f64], n: usize) -> Result<Vec<f64>> {
    let mut inv = vec![0.0; n * n];
    for j in 0..n {
        let rjj = r[j * n + j];
        if rjj == 0.0 {
            return Err(LimmaError::Invalid(format!(
                "element ({}, {}) is zero, so the inverse cannot be computed",
                j + 1,
                j + 1
            )));
        }
        inv[j * n + j] = 1.0 / rjj;
        // Column j of the inverse, rows i < j: solve R[i..j, i..j] x = -R[i..j, j] * inv[j,j]
        for i in (0..j).rev() {
            let mut s = 0.0;
            for k in (i + 1)..=j {
                s += r[k * n + i] * inv[j * n + k];
            }
            inv[j * n + i] = -s / r[i * n + i];
        }
    }
    Ok(inv)
}

/// `chol2inv(R)`: `(R'R)^{-1}` from an upper-triangular `R` (`n x n`).
/// Returns the full symmetric matrix, column-major.
pub fn chol2inv_upper(r: &[f64], n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let inv = inv_upper(r, n).expect("chol2inv: singular R");
    // (R'R)^-1 = R^-1 R^-T
    let mut out = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..=j {
            let mut s = 0.0;
            for k in j..n {
                s += inv[k * n + i] * inv[k * n + j];
            }
            out[j * n + i] = s;
            out[i * n + j] = s;
        }
    }
    out
}

/// `cov2cor(v)`: scale a covariance matrix to a correlation matrix.
pub fn cov2cor(v: &[f64], n: usize) -> Vec<f64> {
    let is: Vec<f64> = (0..n).map(|i| 1.0 / v[i * n + i].sqrt()).collect();
    let mut r = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            r[j * n + i] = is[i] * v[j * n + i] * is[j];
        }
    }
    for i in 0..n {
        r[i * n + i] = 1.0;
    }
    r
}

/// Eigen-decomposition of a symmetric `n x n` matrix by cyclic Jacobi.
/// Returns `(values, vectors)` with values in decreasing order and vectors
/// column-major (column `i` belongs to `values[i]`).
pub fn eigen_symmetric(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut m = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let at = |m: &[f64], i: usize, j: usize| m[j * n + i];
    for _sweep in 0..100 {
        let mut off = 0.0;
        for j in 0..n {
            for i in 0..j {
                off += at(&m, i, j) * at(&m, i, j);
            }
        }
        if off == 0.0 || off < 1e-300 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = at(&m, p, q);
                if apq == 0.0 {
                    continue;
                }
                let app = at(&m, p, p);
                let aqq = at(&m, q, q);
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = at(&m, k, p);
                    let akq = at(&m, k, q);
                    m[p * n + k] = c * akp - s * akq;
                    m[q * n + k] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = at(&m, p, k);
                    let aqk = at(&m, q, k);
                    m[k * n + p] = c * apk - s * aqk;
                    m[k * n + q] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[p * n + k];
                    let vkq = v[q * n + k];
                    v[p * n + k] = c * vkp - s * vkq;
                    v[q * n + k] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut idx: Vec<usize> = (0..n).collect();
    let vals: Vec<f64> = (0..n).map(|i| m[i * n + i]).collect();
    idx.sort_by(|&a, &b| cmp_nan_last(vals[b], vals[a]));
    let values: Vec<f64> = idx.iter().map(|&i| vals[i]).collect();
    let mut vectors = vec![0.0; n * n];
    for (newj, &oldj) in idx.iter().enumerate() {
        vectors[newj * n..newj * n + n].copy_from_slice(&v[oldj * n..oldj * n + n]);
    }
    (values, vectors)
}

/// limma `is.fullrank(x)`.
pub fn is_fullrank(x: &[f64], n: usize, p: usize) -> bool {
    if p == 0 {
        return false;
    }
    let xtx = crossprod(x, n, p, x, p);
    let (e, _) = eigen_symmetric(&xtx, p);
    e[0] > 0.0 && (e[p - 1] / e[0]).abs() > 1e-13
}

// ---------------------------------------------------------------------------
// R vector helpers
// ---------------------------------------------------------------------------

/// Ascending order with `NaN` last. Agrees with `partial_cmp` on non-`NaN` values (so
/// `-0.0 == 0.0`, unlike `total_cmp`) and is a total order, so sorting never panics.
pub fn cmp_nan_last(a: f64, b: f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        _ => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
    }
}

fn sorted(x: &[f64]) -> Vec<f64> {
    let mut s = x.to_vec();
    s.sort_by(|a, b| cmp_nan_last(*a, *b));
    s
}

/// `median(x)` of finite values (no NA handling: caller filters).
pub fn median(x: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 {
        return f64::NAN;
    }
    let s = sorted(x);
    let half = n.div_ceil(2);
    if n % 2 == 1 {
        s[half - 1]
    } else {
        (s[half - 1] + s[half]) / 2.0
    }
}

/// `quantile(x, probs, type = 7)`.
pub fn quantile7(x: &[f64], probs: &[f64]) -> Vec<f64> {
    let n = x.len();
    let s = sorted(x);
    probs
        .iter()
        .map(|&p| {
            let index = 1.0 + (n.max(1) - 1) as f64 * p;
            let lo = index.floor();
            let hi = index.ceil();
            let xlo = s[lo as usize - 1];
            let xhi = s[hi as usize - 1];
            let mut qs = xlo;
            if index > lo && xhi != xlo {
                let h = index - lo;
                qs = (1.0 - h) * xlo + h * xhi;
            }
            qs
        })
        .collect()
}

/// `mean(x, trim = trim)`.
pub fn mean_trim(x: &[f64], trim: f64) -> f64 {
    let n = x.len();
    if n == 0 {
        return f64::NAN;
    }
    if trim <= 0.0 {
        return mean(x);
    }
    if trim >= 0.5 {
        return median(x);
    }
    let lo = (n as f64 * trim).floor() as usize + 1;
    let hi = n + 1 - lo;
    let s = sorted(x);
    mean(&s[lo - 1..hi])
}

/// R's `mean()` for doubles: two-pass with the long-double refinement
/// (long double is double on arm64; the second pass is kept regardless).
pub fn mean(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mut s = 0.0;
    for &v in x {
        s += v;
    }
    s /= n;
    if s.is_finite() {
        let mut t = 0.0;
        for &v in x {
            t += v - s;
        }
        s += t / n;
    }
    s
}

/// `rank(x)` with `ties.method = "average"` (1-based, as doubles).
pub fn rank_average(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| cmp_nan_last(x[a], x[b]));
    let mut r = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && x[idx[j + 1]] == x[idx[i]] {
            j += 1;
        }
        let avg = (i + j + 2) as f64 / 2.0;
        for k in i..=j {
            r[idx[k]] = avg;
        }
        i = j + 1;
    }
    r
}

/// `cummax(x)`.
pub fn cummax(x: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    let mut m = f64::NEG_INFINITY;
    for &v in x {
        if v.is_nan() {
            m = f64::NAN;
        }
        if v > m {
            m = v;
        }
        out.push(m);
    }
    out
}

/// `p.adjust(p, method = "BH")` over the finite entries; `NaN` entries stay `NaN`.
pub fn p_adjust_bh(p: &[f64]) -> Vec<f64> {
    let idx: Vec<usize> = (0..p.len()).filter(|&i| !p[i].is_nan()).collect();
    let n = idx.len();
    let mut out = vec![f64::NAN; p.len()];
    if n == 0 {
        return out;
    }
    // order(p, decreasing = TRUE)
    let mut o = idx.clone();
    o.sort_by(|&a, &b| cmp_nan_last(p[b], p[a]));
    let mut running = f64::INFINITY;
    for (k, &i) in o.iter().enumerate() {
        let rank = (n - k) as f64;
        let v = n as f64 / rank * p[i];
        if v < running {
            running = v;
        }
        out[i] = running.min(1.0);
    }
    out
}

/// `approx(x, y, xout, rule = 2, ties = mean)`: linear interpolation with
/// constant extrapolation, after averaging `y` over tied `x`.
pub fn approx_rule2_ties_mean(x: &[f64], y: &[f64], xout: &[f64]) -> Vec<f64> {
    // regularize: sort by x, average ties
    let mut idx: Vec<usize> = (0..x.len())
        .filter(|&i| x[i].is_finite() && y[i].is_finite())
        .collect();
    idx.sort_by(|&a, &b| cmp_nan_last(x[a], x[b]));
    let mut ux: Vec<f64> = Vec::new();
    let mut uy: Vec<f64> = Vec::new();
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && x[idx[j + 1]] == x[idx[i]] {
            j += 1;
        }
        let ys: Vec<f64> = (i..=j).map(|k| y[idx[k]]).collect();
        ux.push(x[idx[i]]);
        uy.push(mean(&ys));
        i = j + 1;
    }
    let n = ux.len();
    xout.iter()
        .map(|&v| {
            if v.is_nan() || n == 0 {
                return f64::NAN;
            }
            if v < ux[0] {
                return uy[0];
            }
            if v > ux[n - 1] {
                return uy[n - 1];
            }
            // binary search as in R's approx1
            let mut i = 0usize;
            let mut j = n - 1;
            while i < j.saturating_sub(1) {
                let ij = (i + j) / 2;
                if v < ux[ij] {
                    j = ij;
                } else {
                    i = ij;
                }
            }
            if v == ux[j] {
                return uy[j];
            }
            if v == ux[i] {
                return uy[i];
            }
            uy[i] + (uy[j] - uy[i]) * ((v - ux[i]) / (ux[j] - ux[i]))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chol_and_chol2inv_roundtrip() {
        // a = [[4,2],[2,3]]
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let u = chol_upper(&a, 2).unwrap();
        assert!((u[0] - 2.0).abs() < 1e-15);
        assert!((u[2] - 1.0).abs() < 1e-15);
        assert!((u[3] - 2.0f64.sqrt()).abs() < 1e-15);
        let inv = chol2inv_upper(&u, 2);
        // inverse of a = 1/8 * [[3,-2],[-2,4]]
        assert!((inv[0] - 0.375).abs() < 1e-14);
        assert!((inv[1] + 0.25).abs() < 1e-14);
        assert!((inv[3] - 0.5).abs() < 1e-14);
    }

    #[test]
    fn eigen_symmetric_2x2() {
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let (e, v) = eigen_symmetric(&a, 2);
        assert!((e[0] - 3.0).abs() < 1e-14);
        assert!((e[1] - 1.0).abs() < 1e-14);
        assert!((v[0].abs() - v[1].abs()).abs() < 1e-14);
    }

    #[test]
    fn r_helpers() {
        assert_eq!(quantile7(&[1.0, 2.0, 3.0, 4.0], &[0.5]), vec![2.5]);
        assert_eq!(quantile7(&[5.0, 1.0, 3.0], &[0.25, 0.75]), vec![2.0, 4.0]);
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(mean_trim(&[1.0, 2.0, 3.0, 100.0, 4.0], 0.2), 3.0);
        assert_eq!(
            rank_average(&[10.0, 20.0, 10.0, 5.0]),
            vec![2.5, 4.0, 2.5, 1.0]
        );
        assert_eq!(cummax(&[1.0, 3.0, 2.0, 5.0]), vec![1.0, 3.0, 3.0, 5.0]);
        let adj = p_adjust_bh(&[0.01, 0.04, 0.03, f64::NAN]);
        assert!((adj[0] - 0.03).abs() < 1e-15);
        assert!((adj[1] - 0.04).abs() < 1e-15);
        assert!((adj[2] - 0.04).abs() < 1e-15);
        assert!(adj[3].is_nan());
        let a = approx_rule2_ties_mean(
            &[1.0, 2.0, 2.0, 3.0],
            &[1.0, 2.0, 4.0, 5.0],
            &[0.0, 1.5, 2.0, 2.5, 9.0],
        );
        assert_eq!(a, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn cmp_nan_last_matches_partial_cmp_on_non_nan() {
        let vals = [
            f64::NEG_INFINITY,
            -2.5,
            -0.0,
            0.0,
            1e-300,
            3.0,
            3.0,
            f64::INFINITY,
        ];
        for &a in &vals {
            for &b in &vals {
                assert_eq!(cmp_nan_last(a, b), a.partial_cmp(&b).unwrap(), "{a} vs {b}");
            }
        }
        // Stable sort: equal keys (incl. -0.0 == 0.0) keep their input order, as with partial_cmp.
        let x = [3.0, 0.0, -0.0, -2.5, 3.0, 1e-300];
        let mut by_new: Vec<usize> = (0..x.len()).collect();
        let mut by_old = by_new.clone();
        by_new.sort_by(|&i, &j| cmp_nan_last(x[i], x[j]));
        by_old.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());
        assert_eq!(by_new, by_old);
    }

    #[test]
    fn cmp_nan_last_puts_nan_last() {
        let mut x = [f64::NAN, 2.0, f64::NEG_INFINITY, f64::NAN, -1.0];
        x.sort_by(|a, b| cmp_nan_last(*a, *b));
        assert_eq!(&x[..3], &[f64::NEG_INFINITY, -1.0, 2.0]);
        assert!(x[3].is_nan() && x[4].is_nan());
    }
}
