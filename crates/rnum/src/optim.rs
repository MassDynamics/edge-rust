//! The two scalar optimisers limma's prior fits reach through R: `optimize()` and
//! `uniroot()`. Both are ports of the C behind those R functions, in
//! `R-4.5.0/src/library/stats/src/`:
//!
//! - [`brent_fmin`]  `optimize.c: Brent_fmin`, called by `fitFDistUnequalDF1` as
//!   `optimize(minusTwiceLogLik, c(1/2, 0.9998))` (`fitFDistUnequalDF1.R:98`).
//! - [`zeroin2`]     `zeroin.c: R_zeroin2`, called by `fitFDistRobustly` through
//!   `uniroot(fun, interval = c(rbx, 1), tol = 1e-8, f.lower = , f.upper = )`
//!   (`fitFDistRobustly.R:176`).
//!
//! The iteration order and every floating-point expression follow the C, so that the
//! sequence of points each optimiser evaluates is the one R evaluates.

/// `optimize()`'s default `tol`: `.Machine$double.eps^0.25`.
pub fn optimize_default_tol() -> f64 {
    f64::EPSILON.powf(0.25)
}

/// `uniroot()`'s default `maxiter`.
pub const UNIROOT_DEFAULT_MAXITER: i32 = 1000;

/// `Brent_fmin(ax, bx, f, tol)`: the abscissa of a minimum of `f` on `[ax, bx]`, by golden
/// section search and successive parabolic interpolation. This is the whole of R's
/// `optimize()` apart from the argument checks.
pub fn brent_fmin<F: FnMut(f64) -> f64>(ax: f64, bx: f64, mut f: F, tol: f64) -> f64 {
    // c is the squared inverse of the golden ratio.
    let c = (3.0 - 5.0f64.sqrt()) * 0.5;

    // eps is approximately the square root of the relative machine precision.
    let eps = f64::EPSILON.sqrt();

    let mut a = ax;
    let mut b = bx;
    let mut v = a + c * (b - a);
    let mut w = v;
    let mut x = v;

    let mut d: f64 = 0.0;
    let mut e: f64 = 0.0;
    let mut fx = f(x);
    let mut fv = fx;
    let mut fw = fx;
    let tol3 = tol / 3.0;

    loop {
        let xm = (a + b) * 0.5;
        let tol1 = eps * x.abs() + tol3;
        let t2 = tol1 * 2.0;

        // check stopping criterion
        if (x - xm).abs() <= t2 - (b - a) * 0.5 {
            break;
        }
        let mut p: f64 = 0.0;
        let mut q: f64 = 0.0;
        let mut r: f64 = 0.0;
        if e.abs() > tol1 {
            // fit parabola
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = (q - r) * 2.0;
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            r = e;
            e = d;
        }

        let mut u;
        if p.abs() >= (q * 0.5 * r).abs() || p <= q * (a - x) || p >= q * (b - x) {
            // a golden-section step
            e = if x < xm { b - x } else { a - x };
            d = c * e;
        } else {
            // a parabolic-interpolation step
            d = p / q;
            u = x + d;

            // f must not be evaluated too close to ax or bx
            if u - a < t2 || b - u < t2 {
                d = tol1;
                if x >= xm {
                    d = -d;
                }
            }
        }

        // f must not be evaluated too close to x
        if d.abs() >= tol1 {
            u = x + d;
        } else if d > 0.0 {
            u = x + tol1;
        } else {
            u = x - tol1;
        }

        let fu = f(u);

        // update a, b, v, w, and x
        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            w = x;
            x = u;
            fv = fw;
            fw = fx;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    x
}

/// What [`zeroin2`] reports back, mirroring the `*Tol` / `*Maxit` out-parameters of the C
/// and the `root`, `iter`, `estim.prec` fields of `uniroot()`'s result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Zeroin {
    pub root: f64,
    /// Iterations used; `-1` when `maxit` ran out (R's `uniroot` then warns).
    pub iter: i32,
    pub estim_prec: f64,
}

/// `R_zeroin2(ax, bx, fa, fb, f, tol, maxit)`: Brent's root finder on `[ax, bx]`, given
/// the function values at both ends. `fa` and `fb` must bracket a root (opposite signs),
/// which `uniroot()` checks before calling this.
pub fn zeroin2<F: FnMut(f64) -> f64>(
    ax: f64,
    bx: f64,
    fa: f64,
    fb: f64,
    mut f: F,
    tol: f64,
    maxit: i32,
) -> Zeroin {
    let mut a = ax;
    let mut b = bx;
    let mut fa = fa;
    let mut fb = fb;
    let mut c = a;
    let mut fc = fa;
    let mut remaining = maxit + 1;

    // First test if we have found a root at an endpoint
    if fa == 0.0 {
        return Zeroin {
            root: a,
            iter: 0,
            estim_prec: 0.0,
        };
    }
    if fb == 0.0 {
        return Zeroin {
            root: b,
            iter: 0,
            estim_prec: 0.0,
        };
    }

    while remaining > 0 {
        remaining -= 1;
        // Distance from the last but one to the last approximation
        let prev_step = b - a;

        if fc.abs() < fb.abs() {
            // Swap data for b to be the best approximation
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol_act = 2.0 * f64::EPSILON * b.abs() + tol / 2.0;
        let mut new_step = (c - b) / 2.0;

        if new_step.abs() <= tol_act || fb == 0.0 {
            // Acceptable approx. is found
            return Zeroin {
                root: b,
                iter: maxit + 1 - remaining - 1,
                estim_prec: (c - b).abs(),
            };
        }

        // Decide if the interpolation can be tried
        if prev_step.abs() >= tol_act && fa.abs() > fb.abs() {
            let cb = c - b;
            let (mut p, mut q);
            if a == c {
                // If we have only two distinct points linear interpolation can only be applied
                let t1 = fb / fa;
                p = cb * t1;
                q = 1.0 - t1;
            } else {
                // Quadric inverse interpolation
                q = fa / fc;
                let t1 = fb / fc;
                let t2 = fb / fa;
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0));
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0);
            }
            if p > 0.0 {
                // p was calculated with the opposite sign; make p positive
                q = -q;
            } else {
                // and assign possible minus to q
                p = -p;
            }

            if p < (0.75 * cb * q - (tol_act * q).abs() / 2.0) && p < (prev_step * q / 2.0).abs() {
                // If b+p/q falls in [b,c] and isn't too large it is accepted
                new_step = p / q;
            }
        }

        if new_step.abs() < tol_act {
            // Adjust the step to be not less than tolerance
            new_step = if new_step > 0.0 { tol_act } else { -tol_act };
        }
        // Save the previous approx.
        a = b;
        fa = fb;
        // Do step to a new approxim.
        b += new_step;
        fb = f(b);
        if (fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0) {
            // Adjust c for it to have a sign opposite to that of b
            c = a;
            fc = fa;
        }
    }
    // failed!
    Zeroin {
        root: b,
        iter: -1,
        estim_prec: (c - b).abs(),
    }
}

/// R's `uniroot(f, interval = c(lower, upper), tol =, maxiter =, f.lower =, f.upper =)`,
/// restricted to the argument set limma uses (no `extendInt`, no `check.conv`). Returns
/// `None` where R stops with "f() values at end points not of opposite sign".
pub fn uniroot<F: FnMut(f64) -> f64>(
    f: F,
    lower: f64,
    upper: f64,
    f_lower: f64,
    f_upper: f64,
    tol: f64,
    maxiter: i32,
) -> Option<Zeroin> {
    if lower.is_nan() || upper.is_nan() || lower >= upper {
        return None;
    }
    if f_lower.is_nan() || f_upper.is_nan() {
        return None;
    }
    if f_lower * f_upper > 0.0 {
        return None;
    }
    Some(zeroin2(lower, upper, f_lower, f_upper, f, tol, maxiter))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brent_fmin_finds_the_minimum_of_a_shifted_parabola() {
        // R: optimize(function(x) (x - 1/3)^2, c(0, 1))$minimum -> 0.3333333
        let x = brent_fmin(
            0.0,
            1.0,
            |x| (x - 1.0 / 3.0).powi(2),
            optimize_default_tol(),
        );
        assert!((x - 1.0 / 3.0).abs() < 1e-6, "x = {x}");
    }

    #[test]
    fn brent_fmin_reproduces_r_optimize_on_a_documented_example() {
        // ?optimize: f <- function (x, a) (x - a)^2; optimize(f, c(0, 1), a = 1/3)
        // R prints $minimum 0.3333333 and $objective 0. The evaluation sequence is R's,
        // so the result agrees with R to the tolerance R itself promises (about 1e-4 in x
        // for a quadratic, far better in practice).
        let x = brent_fmin(
            0.0,
            1.0,
            |x| (x - 1.0 / 3.0) * (x - 1.0 / 3.0),
            optimize_default_tol(),
        );
        assert!((x - 0.3333333).abs() < 1e-7, "x = {x}");
    }

    #[test]
    fn zeroin2_finds_the_root_of_a_cubic_like_uniroot() {
        // ?uniroot: f <- function (x, a) x - a; uniroot(f, c(0, 1), tol = 0.0001, a = 1/3)
        let f = |x: f64| x - 1.0 / 3.0;
        let z = uniroot(f, 0.0, 1.0, f(0.0), f(1.0), 1e-4, UNIROOT_DEFAULT_MAXITER).unwrap();
        // R reports $root 0.3333333, $f.root 0, $iter 1 and $estim.prec 0.6666667: the
        // linear interpolation lands on the root exactly, and the precision estimate is
        // then the width of the bracket it never had to shrink.
        assert_eq!(z.root, 1.0 / 3.0);
        assert_eq!(z.iter, 1);
        assert!((z.estim_prec - 2.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn zeroin2_returns_an_endpoint_that_is_already_a_root() {
        let z = zeroin2(0.0, 1.0, 0.0, 1.0, |x| x, 1e-8, 1000);
        assert_eq!(
            z,
            Zeroin {
                root: 0.0,
                iter: 0,
                estim_prec: 0.0
            }
        );
    }

    #[test]
    fn uniroot_refuses_an_interval_that_does_not_bracket() {
        assert!(uniroot(|x| x + 2.0, 0.0, 1.0, 2.0, 3.0, 1e-8, 1000).is_none());
    }

    #[test]
    fn zeroin2_reports_failure_when_maxit_runs_out() {
        let f = |x: f64| x * x * x - 0.3;
        let z = zeroin2(0.0, 1.0, f(0.0), f(1.0), f, 1e-300, 1);
        assert_eq!(z.iter, -1);
        // ... and with room to iterate it converges to the cube root of 0.3.
        let z = uniroot(f, 0.0, 1.0, f(0.0), f(1.0), 1e-12, UNIROOT_DEFAULT_MAXITER).unwrap();
        assert!((z.root - 0.3f64.cbrt()).abs() < 1e-11, "root = {}", z.root);
    }
}
