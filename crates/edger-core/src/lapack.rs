//! The few dense LAPACK kernels edgeR's C code calls, ported from R 4.5.0's bundled
//! `src/modules/lapack/dlapack.f`: Cholesky (`DPOTRF` upper, `DPOTRS`) for `fit_leven_vec` in
//! `edgeR/src/glm.c`, the Bunch-Kaufman factorisation (`DSYTF2` upper, which `DSYTRF` runs for
//! n < 64) for `compute_adj_profile_ll` in `edgeR/src/compute_apl.c`, and an LU solve with partial
//! pivoting for `solve(designunique, t(beta))` in `edgeR/R/mglmOneWay.R`.
//!
//! All matrices are column-major `n x n`.

/// `DPOTRF('U')`: overwrite the upper triangle of `a` with `U` such that `A = U'U`. Returns
/// `false` when a pivot is not positive (LAPACK's `info > 0`). The unblocked column order of
/// `DPOTF2`; R's `DPOTRF2` recursion agrees to rounding.
pub(crate) fn dpotrf_upper(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut ajj = a[j * n + j];
        for k in 0..j {
            ajj -= a[j * n + k] * a[j * n + k];
        }
        if ajj <= 0.0 || ajj.is_nan() {
            a[j * n + j] = ajj;
            return false;
        }
        let ajj = ajj.sqrt();
        a[j * n + j] = ajj;
        for c in j + 1..n {
            let mut s = a[c * n + j];
            for k in 0..j {
                s -= a[j * n + k] * a[c * n + k];
            }
            a[c * n + j] = s / ajj;
        }
    }
    true
}

/// `DPOTRS('U')` for one right-hand side: solve `U'U x = b` in place.
pub(crate) fn dpotrs_upper(u: &[f64], n: usize, b: &mut [f64]) {
    // U' y = b (forward), then U x = y (backward).
    for i in 0..n {
        let mut t = b[i];
        for k in 0..i {
            t -= u[i * n + k] * b[k];
        }
        b[i] = t / u[i * n + i];
    }
    for k in (0..n).rev() {
        b[k] /= u[k * n + k];
        let bk = b[k];
        for i in 0..k {
            b[i] -= bk * u[k * n + i];
        }
    }
}

/// `DSYTF2('U')`: Bunch-Kaufman `A = U D U'` on the upper triangle of `a`, in place. Only the
/// diagonal of the result is read by edgeR (as the log-determinant), so the pivot vector is not
/// returned. Singular pivots are left in place (LAPACK's `info > 0`), as edgeR ignores them.
pub(crate) fn dsytf2_upper(a: &mut [f64], n: usize) {
    let alpha = (1.0 + 17f64.sqrt()) / 8.0;
    let at = |i: usize, j: usize| j * n + i;
    // k is 1-based as in the Fortran.
    let mut k = n;
    while k >= 1 {
        let mut kstep = 1;
        let absakk = a[at(k - 1, k - 1)].abs();
        let (imax, colmax) = if k > 1 {
            let mut im = 1;
            let mut cm = a[at(0, k - 1)].abs();
            for i in 2..k {
                let v = a[at(i - 1, k - 1)].abs();
                if v > cm {
                    cm = v;
                    im = i;
                }
            }
            (im, cm)
        } else {
            (0, 0.0)
        };
        // A zero (or NaN) column is left in place: LAPACK sets info = k and kp = k, which only
        // feeds the pivot vector.
        if !(absakk.max(colmax) == 0.0 || absakk.is_nan()) {
            let kp;
            if absakk >= alpha * colmax {
                kp = k;
            } else {
                // rowmax over row imax: columns imax+1..k, then column imax rows 1..imax-1.
                let mut rowmax: f64 = 0.0;
                for j in imax + 1..=k {
                    rowmax = rowmax.max(a[at(imax - 1, j - 1)].abs());
                }
                if imax > 1 {
                    let mut cm = a[at(0, imax - 1)].abs();
                    for i in 2..imax {
                        cm = cm.max(a[at(i - 1, imax - 1)].abs());
                    }
                    rowmax = rowmax.max(cm);
                }
                if absakk >= alpha * colmax * (colmax / rowmax) {
                    kp = k;
                } else if a[at(imax - 1, imax - 1)].abs() >= alpha * rowmax {
                    kp = imax;
                } else {
                    kp = imax;
                    kstep = 2;
                }
            }
            let kk = k + 1 - kstep;
            if kp != kk {
                // Interchange rows and columns kk and kp in the leading k x k submatrix.
                for i in 1..kp {
                    a.swap(at(i - 1, kk - 1), at(i - 1, kp - 1));
                }
                for j in kp + 1..kk {
                    a.swap(at(j - 1, kk - 1), at(kp - 1, j - 1));
                }
                a.swap(at(kk - 1, kk - 1), at(kp - 1, kp - 1));
                if kstep == 2 {
                    a.swap(at(k - 2, k - 1), at(kp - 1, k - 1));
                }
            }
            if kstep == 1 {
                let r1 = 1.0 / a[at(k - 1, k - 1)];
                // DSYR upper: A(1:k-1,1:k-1) -= r1 * x x', x = A(1:k-1, k).
                for j in 1..k {
                    let xj = a[at(j - 1, k - 1)];
                    if xj != 0.0 {
                        let temp = -r1 * xj;
                        for i in 1..=j {
                            let xi = a[at(i - 1, k - 1)];
                            a[at(i - 1, j - 1)] += xi * temp;
                        }
                    }
                }
                for i in 1..k {
                    a[at(i - 1, k - 1)] *= r1;
                }
            } else if k > 2 {
                let mut d12 = a[at(k - 2, k - 1)];
                let d22 = a[at(k - 2, k - 2)] / d12;
                let d11 = a[at(k - 1, k - 1)] / d12;
                let t = 1.0 / (d11 * d22 - 1.0);
                d12 = t / d12;
                for j in (1..=k - 2).rev() {
                    let wkm1 = d12 * (d11 * a[at(j - 1, k - 2)] - a[at(j - 1, k - 1)]);
                    let wk = d12 * (d22 * a[at(j - 1, k - 1)] - a[at(j - 1, k - 2)]);
                    for i in (1..=j).rev() {
                        a[at(i - 1, j - 1)] = a[at(i - 1, j - 1)]
                            - a[at(i - 1, k - 1)] * wk
                            - a[at(i - 1, k - 2)] * wkm1;
                    }
                    a[at(j - 1, k - 1)] = wk;
                    a[at(j - 1, k - 2)] = wkm1;
                }
            }
        }
        if k <= kstep {
            break;
        }
        k -= kstep;
    }
}

/// `solve(a, b)` for one right-hand side: LU with partial pivoting (LAPACK `DGESV`'s
/// algorithm). Returns `None` for an exactly singular `a`.
pub(crate) fn lu_solve(a: &[f64], n: usize, b: &[f64]) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut x = b.to_vec();
    let at = |i: usize, j: usize| j * n + i;
    for c in 0..n {
        let mut p = c;
        let mut best = m[at(c, c)].abs();
        for r in c + 1..n {
            let v = m[at(r, c)].abs();
            if v > best {
                best = v;
                p = r;
            }
        }
        if best == 0.0 {
            return None;
        }
        if p != c {
            for j in 0..n {
                m.swap(at(c, j), at(p, j));
            }
            x.swap(c, p);
        }
        let piv = m[at(c, c)];
        for r in c + 1..n {
            let l = m[at(r, c)] / piv;
            m[at(r, c)] = l;
            if l != 0.0 {
                for j in c + 1..n {
                    m[at(r, j)] -= l * m[at(c, j)];
                }
                x[r] -= l * x[c];
            }
        }
    }
    for i in (0..n).rev() {
        let mut t = x[i];
        for j in i + 1..n {
            t -= m[at(i, j)] * x[j];
        }
        x[i] = t / m[at(i, i)];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_solves_a_spd_system() {
        // A = [[4,2,0.6],[2,5,1],[0.6,1,3]], b = A * [1, -2, 0.5]
        let a = [4.0, 2.0, 0.6, 2.0, 5.0, 1.0, 0.6, 1.0, 3.0];
        let x = [1.0, -2.0, 0.5];
        let b: Vec<f64> = (0..3)
            .map(|i| (0..3).map(|j| a[j * 3 + i] * x[j]).sum())
            .collect();
        let mut u = a;
        assert!(dpotrf_upper(&mut u, 3));
        let mut s = b.clone();
        dpotrs_upper(&u, 3, &mut s);
        for i in 0..3 {
            assert!((s[i] - x[i]).abs() < 1e-14);
        }
        let l = lu_solve(&a, 3, &b).unwrap();
        for i in 0..3 {
            assert!((l[i] - x[i]).abs() < 1e-14);
        }
    }

    #[test]
    fn bunch_kaufman_diagonal_gives_the_log_determinant_of_a_spd_matrix() {
        let a = [4.0, 2.0, 0.6, 2.0, 5.0, 1.0, 0.6, 1.0, 3.0];
        let det = 4.0 * (5.0 * 3.0 - 1.0) - 2.0 * (2.0 * 3.0 - 0.6) + 0.6 * (2.0 - 3.0);
        let mut f = a;
        dsytf2_upper(&mut f, 3);
        let prod: f64 = (0..3).map(|i| f[i * 3 + i]).product();
        assert!((prod - det).abs() < 1e-12 * det);
    }
}
