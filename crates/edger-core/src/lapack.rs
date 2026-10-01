//! The few dense LAPACK kernels edgeR's C code calls, ported from R 4.5.0's bundled
//! `src/modules/lapack/dlapack.f`: Cholesky (`DPOTRF` upper, `DPOTRS`) for `fit_leven_vec` and
//! Householder QR (`DGEQRF`, `DORMQR`, `DTRTRS`) for `get_leven_start` in `edgeR/src/glm.c`, the
//! Bunch-Kaufman factorisation (`DSYTF2` upper, which `DSYTRF` runs for n < 64) for `compute_adj_profile_ll` in `edgeR/src/compute_apl.c`, and an LU solve with partial
//! pivoting for `solve(designunique, t(beta))` in `edgeR/R/mglmOneWay.R`.
//!
//! All matrices are column-major `n x n`.

/// `DPOTRF('U')`: overwrite the upper triangle of `a` with `U` such that `A = U'U`. Returns
/// `false` when a pivot is not positive (LAPACK's `info > 0`). For n < 64 LAPACK 3.12's `DPOTRF`
/// runs the recursive `DPOTRF2`, whose `DTRSM` / `DSYRK` updates round differently from the
/// column order of `DPOTF2`, so this follows the recursion.
pub(crate) fn dpotrf_upper(a: &mut [f64], n: usize) -> bool {
    dpotrf2_upper(a, n, 0, n)
}

/// `DPOTRF2('U')` on the `n x n` block of `a` (leading dimension `lda`) starting at `(o, o)`.
fn dpotrf2_upper(a: &mut [f64], lda: usize, o: usize, n: usize) -> bool {
    let at = |i: usize, j: usize| (o + j) * lda + o + i;
    if n == 0 {
        return true;
    }
    if n == 1 {
        let v = a[at(0, 0)];
        if v <= 0.0 || v.is_nan() {
            return false;
        }
        a[at(0, 0)] = v.sqrt();
        return true;
    }
    let n1 = n / 2;
    let n2 = n - n1;
    if !dpotrf2_upper(a, lda, o, n1) {
        return false;
    }
    // DTRSM('L', 'U', 'T', 'N', n1, n2, 1, A11, A12): A12 := inv(A11') A12.
    for j in 0..n2 {
        for i in 0..n1 {
            let mut t = a[at(i, n1 + j)];
            for k in 0..i {
                t -= a[at(k, i)] * a[at(k, n1 + j)];
            }
            a[at(i, n1 + j)] = t / a[at(i, i)];
        }
    }
    // DSYRK('U', 'T', n2, n1, -1, A12, 1, A22): A22 := A22 - A12' A12.
    for j in 0..n2 {
        for i in 0..=j {
            let mut t = 0.0;
            for l in 0..n1 {
                t += a[at(l, n1 + i)] * a[at(l, n1 + j)];
            }
            a[at(n1 + i, n1 + j)] += -t;
        }
    }
    dpotrf2_upper(a, lda, o + n1, n2)
}

/// Reference BLAS `DNRM2` as of LAPACK 3.10 (Blue's scaled sum of squares), unit stride.
fn dnrm2(x: &[f64]) -> f64 {
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

/// `DLAPY2`: `sqrt(x^2 + y^2)` avoiding overflow.
fn dlapy2(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return if y.is_nan() { y } else { x };
    }
    let w = x.abs().max(y.abs());
    let z = x.abs().min(y.abs());
    if z == 0.0 || w > f64::MAX {
        w
    } else {
        w * (1.0 + (z / w) * (z / w)).sqrt()
    }
}

/// `DLARFG`: generate the elementary reflector for `(alpha, x)`; returns `tau`, overwrites
/// `alpha` with `beta` and `x` with `v(2:n)`.
fn dlarfg(alpha: &mut f64, x: &mut [f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let mut xnorm = dnrm2(x);
    if xnorm == 0.0 {
        return 0.0;
    }
    let mut beta = -dlapy2(*alpha, xnorm).copysign(*alpha);
    let safmin = f64::MIN_POSITIVE / f64::EPSILON * 2.0; // DLAMCH('S') / DLAMCH('E')
    let mut knt = 0;
    if beta.abs() < safmin {
        let rsafmn = 1.0 / safmin;
        loop {
            knt += 1;
            x.iter_mut().for_each(|v| *v *= rsafmn);
            beta *= rsafmn;
            *alpha *= rsafmn;
            if !(beta.abs() < safmin && knt < 20) {
                break;
            }
        }
        xnorm = dnrm2(x);
        beta = -dlapy2(*alpha, xnorm).copysign(*alpha);
    }
    let tau = (beta - *alpha) / beta;
    let s = 1.0 / (*alpha - beta);
    x.iter_mut().for_each(|v| *v *= s);
    for _ in 0..knt {
        beta *= safmin;
    }
    *alpha = beta;
    tau
}

/// `DLARF1F('L')`: apply `H = I - tau v v'` (with `v(1) = 1` implied, `v[0]` unread) from the
/// left to the `m x n` matrix `c` (leading dimension `ldc`).
fn dlarf1f_left(v: &[f64], tau: f64, c: &mut [f64], m: usize, n: usize, ldc: usize) {
    if tau == 0.0 {
        return;
    }
    let mut lastv = m;
    while lastv > 1 && v[lastv - 1] == 0.0 {
        lastv -= 1;
    }
    // ILADLC: the last column of C(1:lastv, :) with a nonzero.
    let mut lastc = n;
    if n > 0 && c[(n - 1) * ldc] == 0.0 && c[(n - 1) * ldc + lastv - 1] == 0.0 {
        while lastc > 0
            && c[(lastc - 1) * ldc..(lastc - 1) * ldc + lastv]
                .iter()
                .all(|&e| e == 0.0)
        {
            lastc -= 1;
        }
    }
    if lastc == 0 {
        return;
    }
    if lastv == 1 {
        for j in 0..lastc {
            c[j * ldc] *= 1.0 - tau;
        }
        return;
    }
    let mut work = vec![0.0; lastc];
    for (j, w) in work.iter_mut().enumerate() {
        let mut t = 0.0;
        for i in 1..lastv {
            t += c[j * ldc + i] * v[i];
        }
        *w = t + c[j * ldc];
    }
    for (j, w) in work.iter().enumerate() {
        c[j * ldc] += -tau * w;
    }
    for (j, w) in work.iter().enumerate() {
        if *w != 0.0 {
            let t = -tau * w;
            for i in 1..lastv {
                c[j * ldc + i] += v[i] * t;
            }
        }
    }
}

/// `DGEQR2`: Householder QR of the column-major `m x n` matrix `a` in place (what `DGEQRF` runs
/// when min(m, n) <= 32); returns `tau`.
pub(crate) fn dgeqr2(a: &mut [f64], m: usize, n: usize) -> Vec<f64> {
    let k = m.min(n);
    let mut tau = vec![0.0; k];
    for i in 0..k {
        let (head, rest) = a.split_at_mut((i + 1) * m);
        let col = &mut head[i * m + i..i * m + m];
        let (alpha, x) = col.split_first_mut().unwrap();
        tau[i] = dlarfg(alpha, x);
        if i + 1 < n {
            dlarf1f_left(col, tau[i], &mut rest[i..], m - i, n - i - 1, m);
        }
    }
    tau
}

/// `DORMQR('L', 'T')` for one right-hand side (`DORM2R` when k <= 32): `c := Q' c`.
pub(crate) fn dorm2r_lt(qr: &[f64], m: usize, tau: &[f64], c: &mut [f64]) {
    for (i, &t) in tau.iter().enumerate() {
        dlarf1f_left(&qr[i * m + i..i * m + m], t, &mut c[i..], m - i, 1, m);
    }
}

/// `DTRTRS('U', 'N', 'N')` for one right-hand side on the leading `k x k` upper triangle of `a`
/// (leading dimension `lda`): solve `R x = b` in place. Returns `false` on a zero diagonal.
pub(crate) fn dtrtrs_upper(a: &[f64], lda: usize, k: usize, b: &mut [f64]) -> bool {
    if (0..k).any(|i| a[i * lda + i] == 0.0) {
        return false;
    }
    for j in (0..k).rev() {
        if b[j] != 0.0 {
            b[j] /= a[j * lda + j];
            let bj = b[j];
            for i in 0..j {
                b[i] -= bj * a[j * lda + i];
            }
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

/// R's `qr(x, tol)` (LINPACK `dqrdc2`, `src/appl/dqrdc2.f`) with R 4.5's BLAS `dnrm2` (Blue's
/// algorithm, from the LAPACK 3.12 reference BLAS). `rnum::linpack::qr_decompose` uses the older
/// scaled-sum `dnrm2`, which differs from the reference image in the last bit, so edgeR's QRs
/// go through this copy. `x` is column-major `n x p`.
pub(crate) fn qr_decompose_r45(x: &[f64], n: usize, p: usize, tol: f64) -> rnum::linpack::Qr {
    let mut x = x.to_vec();
    let col = |j: usize| j * n;
    let mut qraux = vec![0.0; p];
    let mut jpvt: Vec<usize> = (0..p).collect();
    let mut work = vec![0.0; 2 * p];
    for j in 0..p {
        let nrm = dnrm2(&x[col(j)..col(j) + n]);
        qraux[j] = nrm;
        work[j] = nrm;
        work[p + j] = if nrm == 0.0 { 1.0 } else { nrm };
    }
    let mut k = p + 1;
    for l in 1..=n.min(p) {
        let li = l - 1;
        while l < k && qraux[li] < work[p + li] * tol {
            for i in 0..n {
                let t = x[col(li) + i];
                for j in (l + 1)..=p {
                    x[col(j - 2) + i] = x[col(j - 1) + i];
                }
                x[col(p - 1) + i] = t;
            }
            let (i, t, tt, ttt) = (jpvt[li], qraux[li], work[li], work[p + li]);
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
        if l == n {
            continue;
        }
        let mut nrmxl = dnrm2(&x[col(li) + li..col(li) + n]);
        if nrmxl == 0.0 {
            continue;
        }
        let xll = x[col(li) + li];
        if xll != 0.0 {
            nrmxl = nrmxl.abs() * if xll < 0.0 { -1.0 } else { 1.0 };
        }
        let s = 1.0 / nrmxl;
        for i in li..n {
            x[col(li) + i] *= s;
        }
        x[col(li) + li] += 1.0;
        for j in (l + 1)..=p {
            let ji = j - 1;
            let (head, tail) = x.split_at_mut(col(ji));
            let xl = &head[col(li) + li..col(li) + n];
            let xj = &mut tail[li..n];
            let mut dot = 0.0;
            for i in 0..xl.len() {
                dot += xl[i] * xj[i];
            }
            let t = -dot / xl[0];
            if t != 0.0 {
                for i in 0..xl.len() {
                    xj[i] += t * xl[i];
                }
            }
            if qraux[ji] != 0.0 {
                let r = xj[0].abs() / qraux[ji];
                let tt = (1.0 - r * r).max(0.0);
                if tt.abs() >= 1e-6 {
                    qraux[ji] *= tt.sqrt();
                } else {
                    qraux[ji] = dnrm2(&xj[1..]);
                    work[ji] = qraux[ji];
                }
            }
        }
        qraux[li] = x[col(li) + li];
        x[col(li) + li] = -nrmxl;
    }
    let rank = if p == 0 { 0 } else { (k - 1).min(n) };
    rnum::linpack::Qr {
        qr: x,
        n,
        p,
        qraux,
        pivot: jpvt,
        rank,
        tol,
    }
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
