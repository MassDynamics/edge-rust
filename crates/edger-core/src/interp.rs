//! `maximizeInterpolant` (edgeR 4.8.2, `R/maximizeInterpolant.R`), i.e. `max_interpolant` and
//! `find_max` (`src/interpolator.c`) over the FMM cubic spline of `src/fmm_spline.c`.

/// `fmm_spline`: Forsythe, Malcolm and Moler end conditions. Written 1-based like the C.
fn fmm_spline(x: &[f64], y: &[f64], b: &mut [f64], c: &mut [f64], d: &mut [f64]) {
    let n = x.len();
    if n < 2 {
        return;
    }
    // 1-based views.
    let xx = |i: usize| x[i - 1];
    let yy = |i: usize| y[i - 1];
    macro_rules! B {
        ($i:expr) => {
            b[$i - 1]
        };
    }
    macro_rules! C {
        ($i:expr) => {
            c[$i - 1]
        };
    }
    macro_rules! D {
        ($i:expr) => {
            d[$i - 1]
        };
    }
    if n < 3 {
        let t = yy(2) - yy(1);
        B!(1) = t / (xx(2) - xx(1));
        B!(2) = B!(1);
        C!(1) = 0.0;
        C!(2) = 0.0;
        D!(1) = 0.0;
        D!(2) = 0.0;
        return;
    }
    let nm1 = n - 1;
    D!(1) = xx(2) - xx(1);
    C!(2) = (yy(2) - yy(1)) / D!(1);
    for i in 2..n {
        D!(i) = xx(i + 1) - xx(i);
        B!(i) = 2.0 * (D!(i - 1) + D!(i));
        C!(i + 1) = (yy(i + 1) - yy(i)) / D!(i);
        C!(i) = C!(i + 1) - C!(i);
    }
    B!(1) = -D!(1);
    B!(n) = -D!(nm1);
    C!(1) = 0.0;
    C!(n) = 0.0;
    if n > 3 {
        C!(1) = C!(3) / (xx(4) - xx(2)) - C!(2) / (xx(3) - xx(1));
        C!(n) = C!(nm1) / (xx(n) - xx(n - 2)) - C!(n - 2) / (xx(nm1) - xx(n - 3));
        C!(1) = C!(1) * D!(1) * D!(1) / (xx(4) - xx(1));
        C!(n) = -C!(n) * D!(nm1) * D!(nm1) / (xx(n) - xx(n - 3));
    }
    for i in 2..=n {
        let t = D!(i - 1) / B!(i - 1);
        B!(i) -= t * D!(i - 1);
        C!(i) -= t * C!(i - 1);
    }
    C!(n) /= B!(n);
    for i in (1..=nm1).rev() {
        C!(i) = (C!(i) - D!(i) * C!(i + 1)) / B!(i);
    }
    B!(n) = (yy(n) - yy(n - 1)) / D!(n - 1) + D!(n - 1) * (C!(n - 1) + 2.0 * C!(n));
    for i in 1..=nm1 {
        B!(i) = (yy(i + 1) - yy(i)) / D!(i) - D!(i) * (C!(i + 1) + 2.0 * C!(i));
        D!(i) = (C!(i + 1) - C!(i)) / D!(i);
        C!(i) *= 3.0;
    }
    C!(n) *= 3.0;
    D!(n) = D!(nm1);
}

/// `find_max`: the abscissa of the maximum of the interpolating spline through `(x, y)`.
pub fn find_max(x: &[f64], y: &[f64]) -> f64 {
    let npts = x.len();
    let mut maxed = -1.0;
    let mut maxed_at: Option<usize> = None;
    for (i, &yi) in y[..npts].iter().enumerate() {
        if maxed_at.is_none() || yi > maxed {
            maxed = yi;
            maxed_at = Some(i);
        }
    }
    let m = maxed_at.expect("find_max: no points");
    let mut x_max = x[m];
    let mut b = vec![0.0; npts];
    let mut c = vec![0.0; npts];
    let mut d = vec![0.0; npts];
    fmm_spline(x, y, &mut b, &mut c, &mut d);
    let mut side = |i: usize, width: f64| {
        let (ld, lc, lb) = (d[i], c[i], b[i]);
        let delta = lc * lc - 3.0 * ld * lb;
        if delta >= 0.0 {
            let numerator = -lc - delta.sqrt();
            let chosen = numerator / (3.0 * ld);
            if chosen > 0.0 && chosen < width {
                let temp = ((ld * chosen + lc) * chosen + lb) * chosen + y[i];
                if temp > maxed {
                    maxed = temp;
                    x_max = chosen + x[i];
                }
            }
        }
    };
    if m > 0 {
        side(m - 1, x[m] - x[m - 1]);
    }
    if m < npts - 1 {
        side(m, x[m + 1] - x[m]);
    }
    x_max
}

/// `maximizeInterpolant(x, y)` for a row-major `ntag x length(x)` likelihood matrix.
pub fn maximize_interpolant(x: &[f64], y: &[f64]) -> Vec<f64> {
    y.chunks(x.len()).map(|r| find_max(x, r)).collect()
}
