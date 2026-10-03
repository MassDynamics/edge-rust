//! `loess(y ~ x, span, degree)` for one predictor, gaussian family, as R 4.5 `stats::loess`
//! computes it with its defaults: `surface = "interpolate"`, `statistics = "approximate"`,
//! `cell = 0.2`, `iterations = 1`.
//!
//! This is a line-by-line port of the parts of Cleveland's `loessf.f` that the default path
//! reaches for `D = 1`: the bounding box (`ehg126`), the k-d tree (`ehg124`, `ehg129`,
//! `ehg106`, `ehg125`), the local fits at the vertices (`ehg139`, `ehg127`) and the cubic
//! Hermite interpolation (`ehg128`). The local fit uses LINPACK `dqrdc`, `dqrsl` and `dsvdc`
//! on top of the reference BLAS that ships with R 4.5 (the new `dnrm2` and `drotg` from
//! `blas2.f90`), all ported privately here so the floating point operation order matches R.
//!
//! The trace-of-hat statistics that `"1.approx"` also computes do not touch the fitted
//! surface and are not ported. Indices are 1-based internally to mirror the Fortran.

/// A fitted one-predictor loess surface (k-d tree plus vertex values and slopes).
#[derive(Debug, Clone)]
pub struct Loess {
    // 1-based arrays, element 0 unused.
    vert: Vec<f64>,
    vval0: Vec<f64>,
    vval1: Vec<f64>,
    a: Vec<usize>,
    xi: Vec<f64>,
    lo: Vec<usize>,
    hi: Vec<usize>,
    c: Vec<[usize; 2]>,
    xmin: f64,
    xmax: f64,
    fitted: Vec<f64>,
}

impl Loess {
    /// Fitted values at the data points, as `fitted(loess(...))`.
    pub fn fitted(&self) -> &[f64] {
        &self.fitted
    }

    /// `predict(fit, newdata)`: interpolated surface, `NaN` (R's `NA`) outside the data range.
    pub fn predict(&self, newx: &[f64]) -> Vec<f64> {
        newx.iter()
            .map(|&z| {
                if z >= self.xmin && z <= self.xmax {
                    self.ehg128(z)
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    fn ehg128(&self, z: f64) -> f64 {
        let mut j = 1usize;
        while self.a[j] != 0 {
            j = if z <= self.xi[j] {
                self.lo[j]
            } else {
                self.hi[j]
            };
        }
        let ll = self.c[j][0];
        let ur = self.c[j][1];
        let (vll, vur) = (self.vert[ll], self.vert[ur]);
        let h = (z - vll) / (vur - vll);
        let omh = 1.0 - h;
        let phi0 = (omh * omh) * (1.0 + 2.0 * h);
        let phi1 = (h * h) * (3.0 - 2.0 * h);
        let psi0 = h * (omh * omh);
        let psi1 = (h * h) * (h - 1.0);
        phi0 * self.vval0[ll]
            + phi1 * self.vval0[ur]
            + (psi0 * self.vval1[ll] + psi1 * self.vval1[ur]) * (vur - vll)
    }
}

/// Fit `loess(y ~ x, span = span, degree = degree)` with unit weights and R's defaults
/// (`cell = 0.2`, gaussian, interpolate). `degree` is 0, 1 or 2.
pub fn loess(x: &[f64], y: &[f64], span: f64, degree: usize) -> Result<Loess, String> {
    loess_cell(x, y, span, degree, 0.2)
}

/// As [`loess`] with an explicit `loess.control(cell = )`.
#[allow(clippy::int_plus_one)]
pub fn loess_cell(
    x: &[f64],
    y: &[f64],
    span: f64,
    degree: usize,
    cell: f64,
) -> Result<Loess, String> {
    let n = x.len();
    if y.len() != n {
        return Err("x and y lengths differ".into());
    }
    if degree > 2 {
        return Err("'degree' must be 0, 1 or 2".into());
    }
    if x.iter().chain(y.iter()).any(|v| !v.is_finite()) {
        return Err("NA/NaN/Inf in 'x' or 'y'".into());
    }
    let nvmax = n.max(200);
    let nf_f = (n as f64 * span + 1e-5).floor();
    let nf = (nf_f as usize).min(n);
    if nf_f <= 0.0 || nf == 0 {
        return Err("span is too small".into());
    }
    let k = degree + 1;
    if k > nf - 1 {
        return Err("span too small: fewer data values than degrees of freedom".into());
    }
    let ncmax = nvmax;
    let vc = 2usize;
    // loess_raw sets v(2) = span * cell; fc = ifloor(n * v(2)); fd = v(3) * dnrm2 = 0.
    let fcell = span * cell;
    let fc = ifloor(n as f64 * fcell);
    let fd = 0.0;

    let x1: Vec<f64> = std::iter::once(0.0).chain(x.iter().copied()).collect();
    let y1: Vec<f64> = std::iter::once(0.0).chain(y.iter().copied()).collect();

    // ehg126: bounding box with a 0.5% margin.
    let machin = f64::MAX;
    let mut alpha = machin;
    let mut beta = -machin;
    for &t in &x1[1..] {
        alpha = alpha.min(t);
        beta = beta.max(t);
    }
    let mu = 0.005 * (beta - alpha).max(1e-10 * alpha.abs().max(beta.abs()) + 1e-30);
    let mut vert = vec![0.0; nvmax + 1];
    vert[1] = alpha - mu;
    vert[vc] = beta + mu;
    let mut nv = vc;
    let mut nc = 1usize;
    let mut a = vec![0usize; ncmax + 1];
    let mut xi = vec![0.0; ncmax + 1];
    let mut lo = vec![0usize; ncmax + 2];
    let mut hi = vec![0usize; ncmax + 2];
    let mut c = vec![[0usize; 2]; ncmax + 1];
    c[1] = [1, 2];
    let mut pi: Vec<usize> = (0..=n).collect();

    // ehg124: build the k-d tree (dd = d = 1).
    {
        let mut p = 1usize;
        let mut l = 1usize;
        let mut u = n;
        lo[p] = l;
        hi[p] = u;
        while p <= nc {
            let diag = vert[c[p][1]] - vert[c[p][0]];
            let diam = (diag * diag).sqrt();
            let mut leaf = if (u - l) + 1 <= fc { true } else { diam <= fd };
            if !leaf {
                leaf = if ncmax < nc + 2 {
                    true
                } else {
                    (nvmax as f64) < nv as f64 + vc as f64 / 2.0
                };
            }
            let mut m = 0usize;
            if !leaf {
                // ehg129 / IDAMAX: one dimension, so k = 1.
                m = ((l + u) as f64 / 2.0) as usize;
                ehg106(l, u, m, &x1, &mut pi);
                let mut offset: i64 = 0;
                loop {
                    let mo = m as i64 + offset;
                    if mo >= u as i64 || mo < l as i64 {
                        break;
                    }
                    let (lower, check, upper);
                    if offset < 0 {
                        lower = l;
                        check = mo as usize;
                        upper = check;
                    } else {
                        lower = mo as usize + 1;
                        check = lower;
                        upper = u;
                    }
                    ehg106(lower, upper, check, &x1, &mut pi);
                    if x1[pi[mo as usize]] == x1[pi[mo as usize + 1]] {
                        offset = -offset;
                        if offset >= 0 {
                            offset += 1;
                        }
                    } else {
                        m = mo as usize;
                        break;
                    }
                }
                leaf = vert[c[p][0]] == x1[pi[m]] || vert[c[p][1]] == x1[pi[m]];
            }
            if leaf {
                a[p] = 0;
            } else {
                a[p] = 1;
                xi[p] = x1[pi[m]];
                nc += 1;
                lo[p] = nc;
                lo[nc] = l;
                hi[nc] = m;
                nc += 1;
                hi[p] = nc;
                lo[nc] = m + 1;
                hi[nc] = u;
                // ehg125 with d = 1, k = 1: one new vertex at xi(p), deduplicated.
                let t = xi[p];
                let mut h = nv + 1;
                vert[h] = t;
                let mut mm = 0usize;
                let mut matched = false;
                for cand in 1..=nv {
                    if vert[cand] == vert[h] {
                        matched = true;
                        mm = cand;
                        break;
                    }
                }
                if matched {
                    h -= 1;
                } else {
                    mm = h;
                }
                let (f0, f1) = (c[p][0], c[p][1]);
                c[lo[p]] = [f0, mm];
                c[hi[p]] = [mm, f1];
                nv = h;
                if nv > nvmax {
                    return Err("loess k-d tree exceeded nvmax".into());
                }
            }
            p += 1;
            l = lo[p];
            u = hi[p];
        }
    }

    // ehg139: local fits at each vertex; psi persists across vertices as in Fortran.
    let mut psi: Vec<usize> = (0..=n).collect();
    let mut vval0 = vec![0.0; nv + 1];
    let mut vval1 = vec![0.0; nv + 1];
    let rw = vec![1.0; n + 1];
    for l in 1..=nv {
        let s = ehg127(vert[l], n, nf, span, &x1, &mut psi, &y1, &rw, k, degree)?;
        vval0[l] = s[0];
        vval1[l] = s[1];
    }

    let mut fit = Loess {
        vert,
        vval0,
        vval1,
        a,
        xi,
        lo,
        hi,
        c,
        xmin: alpha,
        xmax: beta,
        fitted: Vec::new(),
    };
    fit.fitted = x.iter().map(|&z| fit.ehg128(z)).collect();
    Ok(fit)
}

fn ifloor(x: f64) -> usize {
    let mut i = x as i64;
    if x < i as f64 {
        i -= 1;
    }
    i.max(0) as usize
}

/// Floyd-Rivest style selection on `p[pi[il..=ir]]` so that `pi[k]` holds the k-th value.
fn ehg106(il: usize, ir: usize, k: usize, p: &[f64], pi: &mut [usize]) {
    let mut l = il;
    let mut r = ir;
    while l < r {
        let t = p[pi[k]];
        let mut i = l;
        let mut j = r;
        pi.swap(l, k);
        if t < p[pi[r]] {
            pi.swap(l, r);
        }
        while i < j {
            pi.swap(i, j);
            i += 1;
            j -= 1;
            while p[pi[i]] < t {
                i += 1;
            }
            while t < p[pi[j]] {
                j -= 1;
            }
        }
        if p[pi[l]] == t {
            pi.swap(l, j);
        } else {
            j += 1;
            pi.swap(r, j);
        }
        if j <= k {
            l = j + 1;
        }
        if k <= j {
            r = j - 1;
        }
    }
}

/// Local weighted regression at vertex `q`; returns (value, slope).
#[allow(clippy::too_many_arguments)]
fn ehg127(
    q: f64,
    n: usize,
    nf: usize,
    f: f64,
    x: &[f64],
    psi: &mut [usize],
    y: &[f64],
    rw: &[f64],
    k: usize,
    tdeg: usize,
) -> Result<[f64; 2], String> {
    let machep = f64::EPSILON;
    let mut dist = vec![0.0; n + 1];
    for i in 1..=n {
        let dx = x[i] - q;
        dist[i] += dx * dx;
    }
    ehg106(1, n, nf, &dist, psi);
    let rho = dist[psi[nf]] * 1f64.max(f);
    if rho <= 0.0 {
        return Err("loess: span too small".into());
    }
    let mut w = vec![0.0; nf + 1];
    for i in 1..=nf {
        w[i] = (dist[psi[i]] / rho).sqrt();
    }
    for i in 1..=nf {
        let w3 = (w[i] * w[i]) * w[i];
        let omw = 1.0 - w3;
        w[i] = (rw[psi[i]] * ((omw * omw) * omw)).sqrt();
    }
    // b(nf, k), column-major, 1-based.
    let ld = nf;
    let idx = |i: usize, j: usize| (i - 1) + (j - 1) * ld;
    let mut b = vec![0.0; nf * k];
    for i in 1..=nf {
        b[idx(i, 1)] = w[i];
    }
    if tdeg >= 1 {
        for i in 1..=nf {
            b[idx(i, 2)] = w[i] * (x[psi[i]] - q);
        }
    }
    if tdeg >= 2 {
        for i in 1..=nf {
            let dx = x[psi[i]] - q;
            b[idx(i, 3)] = w[i] * (dx * dx);
        }
    }
    let mut eta = vec![0.0; nf];
    for i in 1..=nf {
        eta[i - 1] = w[i] * y[psi[i]];
    }
    let mut colnor = [1.0f64; 15];
    for j in 1..=k {
        let mut scal = 0.0;
        for i in 1..=nf {
            scal += b[idx(i, j)] * b[idx(i, j)];
        }
        scal = scal.sqrt();
        if 0.0 < scal {
            for i in 1..=nf {
                b[idx(i, j)] /= scal;
            }
            colnor[j - 1] = scal;
        } else {
            colnor[j - 1] = 1.0;
        }
    }
    let mut qraux = vec![0.0; k];
    dqrdc0(&mut b, ld, nf, k, &mut qraux);
    dqrsl_qty(&mut b, ld, nf, k, &qraux, &mut eta);
    // u = upper triangle of the k x k block of b.
    let mut u = vec![0.0; k * k];
    for i in 1..=k {
        for j in i..=k {
            u[(i - 1) + (j - 1) * k] = b[idx(i, j)];
        }
    }
    let mut sigma = vec![0.0; k];
    let mut vmat = vec![0.0; k * k];
    let info = dsvdc21(&mut u, k, &mut sigma, &mut vmat);
    if info != 0 {
        return Err("loess: dsvdc failed".into());
    }
    let tol = sigma[0] * (100.0 * machep);
    for j in 0..k {
        for i3 in 0..k {
            vmat[j + i3 * k] /= colnor[j];
        }
    }
    let mut dgamma = vec![0.0; k];
    for j in 0..k {
        dgamma[j] = if tol < sigma[j] {
            ddot(&u[j * k..j * k + k], &eta[..k]) / sigma[j]
        } else {
            0.0
        };
    }
    let mut s = [0.0; 2];
    for (j, sj) in s.iter_mut().enumerate() {
        *sj = if j < k {
            let mut acc = 0.0;
            for i3 in 0..k {
                acc += vmat[j + i3 * k] * dgamma[i3];
            }
            acc
        } else {
            0.0
        };
    }
    Ok(s)
}

// ---------------------------------------------------------------------------------------
// Reference BLAS (R 4.5) and LINPACK pieces, 0-based slices; operation order as in Fortran.

fn ddot(x: &[f64], y: &[f64]) -> f64 {
    let mut t = 0.0;
    for i in 0..x.len() {
        t += x[i] * y[i];
    }
    t
}

fn daxpy(da: f64, x: &[f64], y: &mut [f64]) {
    if da == 0.0 {
        return;
    }
    for i in 0..y.len() {
        y[i] += da * x[i];
    }
}

/// R 4.5 `blas2.f90` DNRM2 (Blue's scaled sum of squares).
fn dnrm2(x: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    let tsml = 2f64.powi(-511);
    let tbig = 2f64.powi(486);
    let ssml = 2f64.powi(537);
    let sbig = 2f64.powi(-538);
    let maxn = f64::MAX;
    let mut notbig = true;
    let (mut asml, mut amed, mut abig) = (0.0f64, 0.0f64, 0.0f64);
    for &v in x {
        let ax = v.abs();
        if ax > tbig {
            let t = ax * sbig;
            abig += t * t;
            notbig = false;
        } else if ax < tsml {
            if notbig {
                let t = ax * ssml;
                asml += t * t;
            }
        } else {
            amed += ax * ax;
        }
    }
    let (scl, sumsq);
    if abig > 0.0 {
        if amed > 0.0 || amed > maxn || amed.is_nan() {
            abig += (amed * sbig) * sbig;
        }
        scl = 1.0 / sbig;
        sumsq = abig;
    } else if asml > 0.0 {
        if amed > 0.0 || amed > maxn || amed.is_nan() {
            let amed2 = amed.sqrt();
            let asml2 = asml.sqrt() / ssml;
            let (ymin, ymax) = if asml2 > amed2 {
                (amed2, asml2)
            } else {
                (asml2, amed2)
            };
            scl = 1.0;
            let r = ymin / ymax;
            sumsq = (ymax * ymax) * (1.0 + r * r);
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

/// R 4.5 `blas2.f90` DROTG. Returns (r, z, c, s) for inputs (a, b).
fn drotg(a: f64, b: f64) -> (f64, f64, f64, f64) {
    let safmin = 2f64.powi(-1022);
    let safmax = 2f64.powi(1023);
    let anorm = a.abs();
    let bnorm = b.abs();
    if bnorm == 0.0 {
        (a, 0.0, 1.0, 0.0)
    } else if anorm == 0.0 {
        (b, 1.0, 0.0, 1.0)
    } else {
        let scl = safmax.min(safmin.max(anorm).max(bnorm));
        let sigma = if anorm > bnorm {
            1f64.copysign(a)
        } else {
            1f64.copysign(b)
        };
        let (as_, bs) = (a / scl, b / scl);
        let r = sigma * (scl * (as_ * as_ + bs * bs).sqrt());
        let c = a / r;
        let s = b / r;
        let z = if anorm > bnorm {
            s
        } else if c != 0.0 {
            1.0 / c
        } else {
            1.0
        };
        (r, z, c, s)
    }
}

fn drot(x: &mut [f64], xo: usize, yo: usize, n: usize, c: f64, s: f64) {
    for i in 0..n {
        let dx = x[xo + i];
        let dy = x[yo + i];
        let t = c * dx + s * dy;
        x[yo + i] = c * dy - s * dx;
        x[xo + i] = t;
    }
}

fn dsign(a: f64, b: f64) -> f64 {
    if b >= 0.0 {
        a.abs()
    } else {
        -a.abs()
    }
}

/// LINPACK `dqrdc` with `job = 0` (no pivoting). `x` is ld x p column-major.
fn dqrdc0(x: &mut [f64], ld: usize, n: usize, p: usize, qraux: &mut [f64]) {
    let lup = n.min(p);
    for l in 1..=lup {
        qraux[l - 1] = 0.0;
        if l == n {
            continue;
        }
        let cl = (l - 1) * ld;
        let mut nrmxl = dnrm2(&x[cl + l - 1..cl + n]);
        if nrmxl == 0.0 {
            continue;
        }
        if x[cl + l - 1] != 0.0 {
            nrmxl = dsign(nrmxl, x[cl + l - 1]);
        }
        let sc = 1.0 / nrmxl;
        if sc != 1.0 {
            for v in &mut x[cl + l - 1..cl + n] {
                *v *= sc;
            }
        }
        x[cl + l - 1] += 1.0;
        for j in l + 1..=p {
            let cj = (j - 1) * ld;
            let (left, right) = x.split_at_mut(cj);
            let xl = &left[cl + l - 1..cl + n];
            let xj = &mut right[l - 1..n];
            let t = -ddot(xl, xj) / xl[0];
            daxpy(t, xl, xj);
        }
        qraux[l - 1] = x[cl + l - 1];
        x[cl + l - 1] = -nrmxl;
    }
}

/// LINPACK `dqrsl` with `job = 1000`: overwrite `y` with Q' y.
fn dqrsl_qty(x: &mut [f64], ld: usize, n: usize, k: usize, qraux: &[f64], y: &mut [f64]) {
    let ju = k.min(n - 1);
    for j in 1..=ju {
        if qraux[j - 1] == 0.0 {
            continue;
        }
        let cj = (j - 1) * ld;
        let temp = x[cj + j - 1];
        x[cj + j - 1] = qraux[j - 1];
        let xj = &x[cj + j - 1..cj + n];
        let t = -ddot(xj, &y[j - 1..n]) / xj[0];
        daxpy(t, xj, &mut y[j - 1..n]);
        x[cj + j - 1] = temp;
    }
}

/// LINPACK `dsvdc` with `job = 21` on a square p x p matrix `x` (ld = p), with `x` and `u`
/// aliased as in `ehg127`: on return `x` holds U, `s` the singular values, `v` holds V.
#[allow(clippy::unnecessary_min_or_max)]
fn dsvdc21(x: &mut [f64], p: usize, s: &mut [f64], v: &mut [f64]) -> usize {
    let n = p;
    let ld = p;
    // 1-based accessors.
    let ix = |i: usize, j: usize| (i - 1) + (j - 1) * ld;
    let maxit = 30;
    let ncu = n.min(p);
    let mut e = vec![0.0; p + 1];
    let mut ss = vec![0.0; p + 2];
    let mut work = vec![0.0; n + 1];
    let nct = (n - 1).min(p);
    let nrt = 0usize.max((p as i64 - 2).min(n as i64).max(0) as usize);
    let lu = nct.max(nrt);
    for l in 1..=lu {
        let lp1 = l + 1;
        if l <= nct {
            let c0 = ix(l, l);
            ss[l] = dnrm2(&x[c0..c0 + n - l + 1]);
            if ss[l] != 0.0 {
                if x[c0] != 0.0 {
                    ss[l] = dsign(ss[l], x[c0]);
                }
                let sc = 1.0 / ss[l];
                if sc != 1.0 {
                    for t in &mut x[c0..c0 + n - l + 1] {
                        *t *= sc;
                    }
                }
                x[c0] += 1.0;
            }
            ss[l] = -ss[l];
        }
        for j in lp1..=p {
            if l <= nct && ss[l] != 0.0 {
                let c0 = ix(l, l);
                let cj = ix(l, j);
                let (left, right) = x.split_at_mut(cj);
                let xl = &left[c0..c0 + n - l + 1];
                let xj = &mut right[..n - l + 1];
                let t = -ddot(xl, xj) / xl[0];
                daxpy(t, xl, xj);
            }
            e[j] = x[ix(l, j)];
        }
        // wantu: u(i,l) = x(i,l) is a no-op because u and x are aliased.
        if l <= nrt {
            e[l] = dnrm2(&e[lp1..=p]);
            if e[l] != 0.0 {
                if e[lp1] != 0.0 {
                    e[l] = dsign(e[l], e[lp1]);
                }
                let sc = 1.0 / e[l];
                if sc != 1.0 {
                    for t in &mut e[lp1..=p] {
                        *t *= sc;
                    }
                }
                e[lp1] += 1.0;
            }
            e[l] = -e[l];
            if !(lp1 > n || e[l] == 0.0) {
                for w in work.iter_mut().take(n + 1).skip(lp1) {
                    *w = 0.0;
                }
                for j in lp1..=p {
                    let ej = e[j];
                    if ej != 0.0 {
                        for i in lp1..=n {
                            work[i] += ej * x[ix(i, j)];
                        }
                    }
                }
                for j in lp1..=p {
                    let t = -e[j] / e[lp1];
                    if t != 0.0 {
                        for i in lp1..=n {
                            x[ix(i, j)] += t * work[i];
                        }
                    }
                }
            }
            for i in lp1..=p {
                v[ix(i, l)] = e[i];
            }
        }
    }
    let mut m = p.min(n + 1);
    let nctp1 = nct + 1;
    let nrtp1 = nrt + 1;
    if nct < p {
        ss[nctp1] = x[ix(nctp1, nctp1)];
    }
    if n < m {
        ss[m] = 0.0;
    }
    if nrtp1 < m {
        e[nrtp1] = x[ix(nrtp1, m)];
    }
    e[m] = 0.0;
    // wantu
    if ncu >= nctp1 {
        for j in nctp1..=ncu {
            for i in 1..=n {
                x[ix(i, j)] = 0.0;
            }
            x[ix(j, j)] = 1.0;
        }
    }
    for ll in 1..=nct {
        let l = nct - ll + 1;
        if ss[l] != 0.0 {
            let lp1 = l + 1;
            for j in lp1..=ncu {
                let c0 = ix(l, l);
                let cj = ix(l, j);
                let (left, right) = x.split_at_mut(cj);
                let ul = &left[c0..c0 + n - l + 1];
                let uj = &mut right[..n - l + 1];
                let t = -ddot(ul, uj) / ul[0];
                daxpy(t, ul, uj);
            }
            let c0 = ix(l, l);
            for t in &mut x[c0..c0 + n - l + 1] {
                *t = -*t;
            }
            x[c0] += 1.0;
            for i in 1..l {
                x[ix(i, l)] = 0.0;
            }
        } else {
            for i in 1..=n {
                x[ix(i, l)] = 0.0;
            }
            x[ix(l, l)] = 1.0;
        }
    }
    // wantv
    for ll in 1..=p {
        let l = p - ll + 1;
        let lp1 = l + 1;
        if l <= nrt && e[l] != 0.0 {
            for j in lp1..=p {
                let c0 = ix(lp1, l);
                let cj = ix(lp1, j);
                let (left, right) = v.split_at_mut(cj);
                let vl = &left[c0..c0 + p - l];
                let vj = &mut right[..p - l];
                let t = -ddot(vl, vj) / vl[0];
                daxpy(t, vl, vj);
            }
        }
        for i in 1..=p {
            v[ix(i, l)] = 0.0;
        }
        v[ix(l, l)] = 1.0;
    }
    let mm = m;
    let mut iter = 0;
    let mut info = 0usize;
    while m != 0 {
        if iter >= maxit {
            info = m;
            break;
        }
        let mut l = 0usize;
        for ll in 1..=m {
            l = m - ll;
            if l == 0 {
                break;
            }
            let test = ss[l].abs() + ss[l + 1].abs();
            let ztest = test + e[l].abs();
            let acc = (test - ztest).abs() / (1.0e-100 + test);
            if acc <= 1.0e-15 {
                e[l] = 0.0;
                break;
            }
        }
        let kase;
        if l == m - 1 {
            kase = 4;
        } else {
            let lp1 = l + 1;
            let mp1 = m + 1;
            let mut ls = l;
            for lls in lp1..=mp1 {
                // LINPACK's `m - lls + lp1`, reordered so the usize never goes through -1.
                ls = m + lp1 - lls;
                if ls == l {
                    break;
                }
                let mut test = 0.0;
                if ls != m {
                    test += e[ls].abs();
                }
                if ls != l + 1 {
                    test += e[ls - 1].abs();
                }
                let ztest = test + ss[ls].abs();
                let acc = (test - ztest).abs() / (1.0e-100 + test);
                if acc <= 1.0e-15 {
                    ss[ls] = 0.0;
                    break;
                }
            }
            if ls == l {
                kase = 3;
            } else if ls == m {
                kase = 1;
            } else {
                kase = 2;
                l = ls;
            }
        }
        l += 1;
        match kase {
            1 => {
                let mm1 = m - 1;
                let mut f = e[m - 1];
                e[m - 1] = 0.0;
                for kk in l..=mm1 {
                    let k = mm1 - kk + l;
                    let (r, z, cs, sn) = drotg(ss[k], f);
                    ss[k] = r;
                    f = z;
                    if k != l {
                        f = -sn * e[k - 1];
                        e[k - 1] *= cs;
                    }
                    drot(v, ix(1, k), ix(1, m), p, cs, sn);
                }
                let _ = f;
            }
            2 => {
                let mut f = e[l - 1];
                e[l - 1] = 0.0;
                for k in l..=m {
                    let (r, _z, cs, sn) = drotg(ss[k], f);
                    ss[k] = r;
                    f = -sn * e[k];
                    e[k] *= cs;
                    drot(x, ix(1, k), ix(1, l - 1), n, cs, sn);
                }
            }
            3 => {
                let scale = ss[m]
                    .abs()
                    .max(ss[m - 1].abs())
                    .max(e[m - 1].abs())
                    .max(ss[l].abs())
                    .max(e[l].abs());
                let sm = ss[m] / scale;
                let smm1 = ss[m - 1] / scale;
                let emm1 = e[m - 1] / scale;
                let sl = ss[l] / scale;
                let el = e[l] / scale;
                let b = ((smm1 + sm) * (smm1 - sm) + emm1 * emm1) / 2.0;
                let smem = sm * emm1;
                let c = smem * smem;
                let mut shift = 0.0;
                if !(b == 0.0 && c == 0.0) {
                    shift = (b * b + c).sqrt();
                    if b < 0.0 {
                        shift = -shift;
                    }
                    shift = c / (b + shift);
                }
                let mut f = (sl + sm) * (sl - sm) + shift;
                let mut g = sl * el;
                let mm1 = m - 1;
                for k in l..=mm1 {
                    let (r, z, cs, sn) = drotg(f, g);
                    f = r;
                    g = z;
                    let _ = g;
                    if k != l {
                        e[k - 1] = f;
                    }
                    f = cs * ss[k] + sn * e[k];
                    e[k] = cs * e[k] - sn * ss[k];
                    g = sn * ss[k + 1];
                    ss[k + 1] *= cs;
                    drot(v, ix(1, k), ix(1, k + 1), p, cs, sn);
                    let (r, z, cs, sn) = drotg(f, g);
                    f = r;
                    g = z;
                    let _ = g;
                    ss[k] = f;
                    f = cs * e[k] + sn * ss[k + 1];
                    ss[k + 1] = -sn * e[k] + cs * ss[k + 1];
                    g = sn * e[k + 1];
                    let _ = g;
                    e[k + 1] *= cs;
                    if k < n {
                        drot(x, ix(1, k), ix(1, k + 1), n, cs, sn);
                    }
                }
                e[m - 1] = f;
                iter += 1;
            }
            _ => {
                if ss[l] < 0.0 {
                    ss[l] = -ss[l];
                    for i in 1..=p {
                        v[ix(i, l)] = -v[ix(i, l)];
                    }
                }
                while l != mm && ss[l] < ss[l + 1] {
                    ss.swap(l, l + 1);
                    if l < p {
                        for i in 1..=p {
                            v.swap(ix(i, l), ix(i, l + 1));
                        }
                    }
                    if l < n {
                        for i in 1..=n {
                            x.swap(ix(i, l), ix(i, l + 1));
                        }
                    }
                    l += 1;
                }
                iter = 0;
                m -= 1;
            }
        }
    }
    s.copy_from_slice(&ss[1..=p]);
    info
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The negligible-singular-value scan in `dsvdc21` runs `lls` up to `m + 1` with `l = 0`, where
    /// LINPACK's `ls = m - lls + lp1` goes through -1. Debug builds panic on that intermediate; a
    /// generic full-rank matrix reaches it on the first sweep.
    #[test]
    fn dsvdc21_full_rank_scan_does_not_underflow() {
        let mut x = vec![4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0];
        let (mut s, mut v) = (vec![0.0; 3], vec![0.0; 9]);
        assert_eq!(dsvdc21(&mut x, 3, &mut s, &mut v), 0);
        // Symmetric positive definite: singular values are the eigenvalues, sum = trace 15,
        // product = det 67.
        assert!((s.iter().sum::<f64>() - 15.0).abs() < 1e-12);
        assert!((s.iter().product::<f64>() - 67.0).abs() < 1e-10);
        assert!(s.windows(2).all(|w| w[0] >= w[1]));
    }
}
