//! Port of R's `src/nmath/pt.c` (`pt`), `qt.c` (`qt`) and `dt.c` (`dt`), all from R 4.5.0.
//! Proved against `scalar/pt.csv` and `scalar/qt.csv` in `tests/scalar_t.rs`; `dt` has no
//! golden and is pinned by the unit tests at the bottom of this file against its two
//! closed forms (df = 1 is Cauchy, df = Inf is the normal).
//!
//! `qt` is Hill's (1970) Algorithm 396 with R's additions: `expm1` for Lozy's remark,
//! Hill's (1981) two-term Taylor refinement, and an inversion of `pt` for `0 < df < 1`.

use super::consts::{M_1_PI, M_1_SQRT_2PI, M_LN2, M_LN_SQRT_2PI, M_PI, M_PI_2, M_SQRT2};
use super::dpq::{
    r_d_0, r_d_cval, r_d_lexp, r_d_log, r_d_lval, r_d_qiv, r_dt_0, r_dt_1, r_dt_qiv,
    r_q_p01_boundaries,
};
use super::gamma::{bd0, lbeta, stirlerr};
use super::pbeta::pbeta;
use super::pnorm::{dnorm, pnorm, qnorm};

/// `Rtanpi` in `src/nmath/cospi.c`: `tan(pi * x)`, exact when `x = k/4` for integer `k`,
/// NaN at the half-integers. R's `qt` reaches it as `tanpi`; on platforms with a libm
/// `tanpi` R uses that instead, which agrees to rounding.
fn tanpi(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if !x.is_finite() {
        return f64::NAN;
    }
    // tan(pi(x + k)) == tan(pi x)  for all integer k
    let mut x = x % 1.0;
    // map (-1,1] --> (-1/2, 1/2] :
    if x <= -0.5 {
        x += 1.0;
    } else if x > 0.5 {
        x -= 1.0;
    }
    if x == 0.0 {
        0.0
    } else if x == 0.5 {
        f64::NAN
    } else if x == 0.25 {
        1.0
    } else if x == -0.25 {
        -1.0
    } else {
        (M_PI * x).tan()
    }
}

/// `pt` in `src/nmath/pt.c`: `P[T <= x]` where `T ~ t_n` (the t distribution with `n`
/// degrees of freedom), or the upper tail, or either on the log scale.
pub fn pt(x: f64, n: f64, lower_tail: bool, log_p: bool) -> f64 {
    let mut lower_tail = lower_tail;
    if x.is_nan() || n.is_nan() {
        return x + n;
    }
    if n <= 0.0 {
        return f64::NAN;
    }

    if !x.is_finite() {
        return if x < 0.0 {
            r_dt_0(lower_tail, log_p)
        } else {
            r_dt_1(lower_tail, log_p)
        };
    }
    if !n.is_finite() {
        return pnorm(x, 0.0, 1.0, lower_tail, log_p);
    }

    let nx = 1.0 + (x / n) * x;
    // FIXME: This test is probably losing rather than gaining precision,
    // now that pbeta(*, log_p = TRUE) is much better.
    // Note however that a version of this test *is* needed for x*x > D_MAX
    let mut val = if nx > 1e100 {
        // <==>  x*x > 1e100 * n
        // Danger of underflow. So use Abramowitz & Stegun 26.5.4
        // pbeta(z, a, b) ~ z^a(1-z)^b / aB(a,b) ~ z^a / aB(a,b),
        // with z = 1/nx,  a = n/2,  b= 1/2 :
        let lval = -0.5 * n * (2.0 * x.abs().ln() - n.ln()) - lbeta(0.5 * n, 0.5) - (0.5 * n).ln();
        if log_p {
            lval
        } else {
            lval.exp()
        }
    } else if n > x * x {
        pbeta(
            x * x / (n + x * x),
            0.5,
            n / 2.0,
            /*lower_tail*/ false,
            log_p,
        )
    } else {
        pbeta(1.0 / nx, n / 2.0, 0.5, /*lower_tail*/ true, log_p)
    };

    // Use "1 - v"  if	lower_tail  and	 x > 0 (but not both):
    if x <= 0.0 {
        lower_tail = !lower_tail;
    }

    if log_p {
        if lower_tail {
            (-0.5 * val.exp()).ln_1p()
        } else {
            val - M_LN2 // = log(.5* pbeta(....))
        }
    } else {
        val /= 2.0;
        r_d_cval(val, lower_tail)
    }
}

/// `qt` in `src/nmath/qt.c`: the quantile function of the t distribution with `ndf`
/// degrees of freedom. Hill's (1970) Algorithm 396, supplemented by inversion of `pt` for
/// `0 < ndf < 1`, with Hill's (1981) two-term Taylor improvement.
pub fn qt(p: f64, ndf: f64, lower_tail: bool, log_p: bool) -> f64 {
    const EPS: f64 = 1.0e-12;

    if p.is_nan() || ndf.is_nan() {
        return p + ndf;
    }

    r_q_p01_boundaries!(p, f64::NEG_INFINITY, f64::INFINITY, lower_tail, log_p);

    if ndf <= 0.0 {
        return f64::NAN;
    }

    if ndf < 1.0 {
        // based on qnt
        const ACCU: f64 = 1e-13;
        const EPS_BIG: f64 = 1e-11; // must be > accu

        let mut iter = 0;

        let p = r_dt_qiv(p, lower_tail, log_p);

        // Invert pt(.) :
        // 1. finding an upper and lower bound
        if p > 1.0 - f64::EPSILON {
            return f64::INFINITY;
        }
        let mut pp = super::arith::fmin2(1.0 - f64::EPSILON, p * (1.0 + EPS_BIG));
        let mut ux = 1.0;
        while ux < f64::MAX && pt(ux, ndf, true, false) < pp {
            ux *= 2.0;
        }
        pp = p * (1.0 - EPS_BIG);
        let mut lx = -1.0;
        while lx > -f64::MAX && pt(lx, ndf, true, false) > pp {
            lx *= 2.0;
        }

        // 2. interval (lx,ux)  halving
        //    regula falsi failed on qt(0.1, 0.1)
        let mut nx;
        loop {
            nx = 0.5 * (lx + ux);
            if pt(nx, ndf, true, false) > p {
                ux = nx;
            } else {
                lx = nx;
            }
            iter += 1;
            if !((ux - lx) / nx.abs() > ACCU && iter < 1000) {
                break;
            }
        }

        return 0.5 * (lx + ux);
    }

    // Old comment:
    // FIXME: "This test should depend on  ndf  AND p  !!
    // -----  and in fact should be replaced by
    // something like Abramowitz & Stegun 26.7.5 (p.949)"
    //
    // That would say that if the qnorm value is x then
    // the result is about x + (x^3+x)/4df + (5x^5+16x^3+3x)/96df^2
    // The differences are tiny even if x ~ 1e5, and qnorm is not
    // that accurate in the extreme tails.
    if ndf > 1e20 {
        return qnorm(p, 0.0, 1.0, lower_tail, log_p);
    }

    #[allow(non_snake_case)]
    let mut P = r_d_qiv(p, log_p); // if exp(p) underflows, we fix below

    let neg = (!lower_tail || P < 0.5) && (lower_tail || P > 0.5);
    let is_neg_lower = lower_tail == neg; // both TRUE or FALSE == !xor
    if neg {
        P = 2.0
            * (if log_p {
                if lower_tail {
                    P
                } else {
                    -p.exp_m1()
                }
            } else {
                r_d_lval(p, lower_tail)
            });
    } else {
        P = 2.0
            * (if log_p {
                if lower_tail {
                    -p.exp_m1()
                } else {
                    P
                }
            } else {
                r_d_cval(p, lower_tail)
            });
    }
    // 0 <= P <= 1 ; P = 2*min(P', 1 - P')  in all cases

    let mut q;
    if (ndf - 2.0).abs() < EPS {
        // df ~= 2
        if P > f64::MIN_POSITIVE {
            if 3.0 * P < f64::EPSILON {
                // P ~= 0
                q = 1.0 / P.sqrt();
            } else if P > 0.9 {
                // P ~= 1
                q = (1.0 - P) * (2.0 / (P * (2.0 - P))).sqrt();
            } else {
                // eps/3 <= P <= 0.9
                q = (2.0 / (P * (2.0 - P)) - 2.0).sqrt();
            }
        } else {
            // P << 1, q = 1/sqrt(P) = ...
            if log_p {
                q = if is_neg_lower {
                    (-p / 2.0).exp() / M_SQRT2
                } else {
                    1.0 / (-p.exp_m1()).sqrt()
                };
            } else {
                q = f64::INFINITY;
            }
        }
    } else if ndf < 1.0 + EPS {
        // df ~= 1  (df < 1 excluded above): Cauchy
        if P == 1.0 {
            q = 0.0; // some versions of tanpi give Inf, some NaN
        } else if P > 0.0 {
            q = 1.0 / tanpi(P / 2.0); // == - tan((P+1) * M_PI_2) -- suffers for P ~= 0
        } else {
            // P = 0, but maybe = 2*exp(p) !
            if log_p {
                // 1/tan(e) ~ 1/e
                q = if is_neg_lower {
                    M_1_PI * (-p).exp()
                } else {
                    -1.0 / (M_PI * p.exp_m1())
                };
            } else {
                q = f64::INFINITY;
            }
        }
    } else {
        // -- usual case;  including, e.g.,  df = 1.1
        let mut x = 0.0;
        let mut y;
        let mut log_p2 = 0.0; // -Wall
        let a = 1.0 / (ndf - 0.5);
        let b = 48.0 / (a * a);
        let mut c = ((20700.0 * a / b - 98.0) * a - 16.0) * a + 96.36;
        let d = ((94.5 / (b + c) - 3.0) / b + 1.0) * (a * M_PI_2).sqrt() * ndf;
        let p_ok1 = P > f64::MIN_POSITIVE || !log_p;
        let mut p_ok = p_ok1; // when true (after check below), use "normal scale": log_p=FALSE
        if p_ok1 {
            y = (d * P).powf(2.0 / ndf);
            p_ok = y >= f64::EPSILON;
        } else {
            y = 0.0; // set below; the C leaves it uninitialised on this path
        }
        if !p_ok {
            // log.p && P very.small  ||  (d*P)^(2/df) =: y < eps_c
            log_p2 = if is_neg_lower {
                r_d_log(p, log_p)
            } else {
                r_d_lexp(p, log_p)
            }; // == log(P / 2)
            x = (d.ln() + M_LN2 + log_p2) / ndf;
            y = (2.0 * x).exp();
        }

        if (ndf < 2.1 && P > 0.5) || y > 0.05 + a {
            // P > P0(df)
            // Asymptotic inverse expansion about normal
            if p_ok {
                x = qnorm(
                    0.5 * P,
                    0.0,
                    1.0,
                    /*lower_tail*/ true,
                    /*log_p*/ false,
                );
            } else {
                // log_p && P underflowed
                x = qnorm(log_p2, 0.0, 1.0, lower_tail, /*log_p*/ true);
            }

            y = x * x;
            if ndf < 5.0 {
                c += 0.3 * (ndf - 4.5) * (x + 0.6);
            }
            // Left to right as the C associates it, `(... + b) + c`; `c += ... + b` would
            // sum `b` and `c` first and round differently.
            #[allow(clippy::assign_op_pattern)]
            {
                c = (((0.05 * d * x - 5.0) * x - 7.0) * x - 2.0) * x + b + c;
            }
            y = (((((0.4 * y + 6.3) * y + 36.0) * y + 94.5) / c - y - 3.0) / b + 1.0) * x;
            y = (a * y * y).exp_m1();
            q = (ndf * y).sqrt();
        } else if !p_ok && x < -M_LN2 * f64::MANTISSA_DIGITS as f64 {
            // 0.5* log(DBL_EPSILON)
            // y above might have underflown
            q = ndf.sqrt() * (-x).exp();
        } else {
            // re-use 'y' from above
            y = ((1.0 / (((ndf + 6.0) / (ndf * y) - 0.089 * d - 0.822) * (ndf + 2.0) * 3.0)
                + 0.5 / (ndf + 4.0))
                * y
                - 1.0)
                * (ndf + 1.0)
                / (ndf + 2.0)
                + 1.0 / y;
            q = (ndf * y).sqrt();
        }

        // Now apply 2-term Taylor expansion improvement (1-term = Newton):
        // as by Hill (1981) [ref.above]

        // FIXME: This can be far from optimal when log_p = TRUE
        //      but is still needed, e.g. for qt(-2, df=1.01, log=TRUE).
        //	Probably also improvable when  lower_tail = FALSE

        if p_ok1 {
            #[allow(non_snake_case)]
            let M = ((f64::MAX / 2.0).sqrt() - ndf).abs();
            let mut it = 0;
            // C: while(it++ < 10 && (y = dt(q, ndf, FALSE)) > 0 &&
            //          R_FINITE(x = (pt(q, ndf, FALSE, FALSE) - P/2) / y) &&
            //          fabs(x) > 1e-14*fabs(q))
            loop {
                it += 1;
                if it > 10 {
                    break;
                }
                y = dt(q, ndf, false);
                // Negated, as in the C: a NaN `y` or `x` must leave the loop.
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                if !(y > 0.0) {
                    break;
                }
                x = (pt(q, ndf, false, false) - P / 2.0) / y;
                if !x.is_finite() {
                    break;
                }
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                if !(x.abs() > 1e-14 * q.abs()) {
                    break;
                }
                // Newton (=Taylor 1 term):
                //  q += x;
                // Taylor 2-term :
                #[allow(non_snake_case)]
                let F = if q.abs() < M {
                    q * (ndf + 1.0) / (2.0 * (q * q + ndf))
                } else {
                    (ndf + 1.0) / (2.0 * (q + ndf / q))
                };
                let del_q = x * (1.0 + x * F);
                if del_q.is_finite() && (q + del_q).is_finite() {
                    q += del_q;
                } else if x.is_finite() && (q + x).is_finite() {
                    q += x;
                } else {
                    // FIXME??  if  q+x = +/-Inf is *better* than q should use it
                    break; // cannot improve  q  with a Newton/Taylor step
                }
            }
        }
    }
    if neg {
        -q
    } else {
        q
    }
}

/// `dt` in `src/nmath/dt.c`: the density of the t distribution with `n` degrees of
/// freedom, evaluated as
/// `sqrt(n/2) / ((n+1)/2) * Gamma((n+3)/2) / Gamma((n+2)/2) * (1+x^2/n)^(-n/2) / sqrt(2 pi (1+x^2/n))`,
/// a form that is stable for all `n`, including `n -> 0` and `n -> infinity`.
pub fn dt(x: f64, n: f64, give_log: bool) -> f64 {
    if x.is_nan() || n.is_nan() {
        return x + n;
    }
    if n <= 0.0 {
        return f64::NAN;
    }
    if !x.is_finite() {
        return r_d_0(give_log);
    }
    if !n.is_finite() {
        return dnorm(x, 0.0, 1.0, give_log);
    }

    let t = -bd0(n / 2.0, (n + 1.0) / 2.0) + stirlerr((n + 1.0) / 2.0) - stirlerr(n / 2.0);
    let x2n = x * x / n; // in  [0, Inf]
    let mut ax = 0.0; // <- -Wpedantic
    let l_x2n; // := log(sqrt(1 + x2n)) = log(1 + x2n)/2
    let u;
    let lrg_x2n = x2n > 1.0 / f64::EPSILON;
    if lrg_x2n {
        // large x^2/n :
        ax = x.abs();
        l_x2n = ax.ln() - n.ln() / 2.0; // = log(x2n)/2 = 1/2 * log(x^2 / n)
        u = //  log(1 + x2n) * n/2 =  n * log(1 + x2n)/2 =
            n * l_x2n;
    } else if x2n > 0.2 {
        l_x2n = (1.0 + x2n).ln() / 2.0;
        u = n * l_x2n;
    } else {
        l_x2n = x2n.ln_1p() / 2.0;
        u = -bd0(n / 2.0, (n + x * x) / 2.0) + x * x / 2.0;
    }

    // old: return  R_D_fexp(M_2PI*(1+x2n), t-u);

    // R_D_fexp(f,x) :=  (give_log ? -0.5*log(f)+(x) : exp(x)/sqrt(f))
    // f = 2pi*(1+x2n)
    //  ==> 0.5*log(f) = log(2pi)/2 + log(1+x2n)/2 = log(2pi)/2 + l_x2n
    //	     1/sqrt(f) = 1/sqrt(2pi * (1+ x^2 / n))
    //		       = 1/sqrt(2pi)/(|x|/sqrt(n)*sqrt(1+1/x2n))
    //		       = M_1_SQRT_2PI * sqrt(n)/ (|x|*sqrt(1+1/x2n))
    if give_log {
        return t - u - (M_LN_SQRT_2PI + l_x2n);
    }

    // else :  if(lrg_x2n) : sqrt(1 + 1/x2n) ='= sqrt(1) = 1
    let i_sqrt_ = if lrg_x2n {
        n.sqrt() / ax
    } else {
        (-l_x2n).exp()
    };
    (t - u).exp() * M_1_SQRT_2PI * i_sqrt_
}

#[cfg(test)]
mod tests {
    use super::super::consts::M_2PI;
    use super::*;

    fn assert_rel(what: &str, got: f64, want: f64, tol: f64) {
        let rel = (got - want).abs() / want.abs();
        assert!(
            rel <= tol,
            "{what}: got {got:e}, want {want:e} (rel {rel:e})"
        );
    }

    /// `dt` has no golden: pin it to the two closed forms it collapses to. df = 1 is the
    /// standard Cauchy, `1 / (pi (1 + x^2))`; df = Inf is `dnorm`, `exp(-x^2/2) / sqrt(2 pi)`.
    /// The df = Inf path is a direct dispatch to `dnorm`, so the closed form is checked
    /// there too; a large finite df (1e10) is checked against the normal at the looser
    /// `O(1/df)` distance the t density actually sits from it.
    #[test]
    fn dt_matches_cauchy_and_normal_closed_forms() {
        let xs = [-10.0, -3.5, -1.0, -0.3, 0.0, 0.3, 1.0, 3.5, 10.0];
        for &x in &xs {
            let cauchy = 1.0 / (M_PI * (1.0 + x * x));
            assert_rel(&format!("dt({x}, 1)"), dt(x, 1.0, false), cauchy, 1e-14);
            assert_rel(
                &format!("dt({x}, 1, log)"),
                dt(x, 1.0, true),
                cauchy.ln(),
                1e-14,
            );

            let normal = (-x * x / 2.0).exp() / M_2PI.sqrt();
            assert_rel(
                &format!("dt({x}, Inf)"),
                dt(x, f64::INFINITY, false),
                normal,
                1e-14,
            );
            assert_rel(
                &format!("dt({x}, Inf, log)"),
                dt(x, f64::INFINITY, true),
                normal.ln(),
                1e-14,
            );
        }
        // Approaching the normal from a finite df: the relative gap is O(x^4 / df).
        for &x in &xs {
            let normal = (-x * x / 2.0).exp() / M_2PI.sqrt();
            assert_rel(&format!("dt({x}, 1e10)"), dt(x, 1e10, false), normal, 1e-6);
        }
    }

    #[test]
    fn dt_edges_follow_r() {
        assert!(dt(f64::NAN, 1.0, false).is_nan());
        assert!(dt(0.0, f64::NAN, false).is_nan());
        assert!(dt(0.0, 0.0, false).is_nan());
        assert!(dt(0.0, -1.0, false).is_nan());
        assert_eq!(dt(f64::INFINITY, 3.0, false), 0.0);
        assert_eq!(dt(f64::NEG_INFINITY, 3.0, true), f64::NEG_INFINITY);
        // The large-x^2/n branch: dt(1e10, 1) = 1/(pi (1 + 1e20)) to rounding.
        assert_rel(
            "dt(1e10, 1)",
            dt(1e10, 1.0, false),
            1.0 / (M_PI * (1.0 + 1e20)),
            1e-14,
        );
    }

    #[test]
    fn tanpi_is_exact_at_quarter_points() {
        assert_eq!(tanpi(0.0), 0.0);
        assert_eq!(tanpi(0.25), 1.0);
        assert_eq!(tanpi(-0.25), -1.0);
        assert_eq!(tanpi(1.25), 1.0);
        assert!(tanpi(0.5).is_nan());
        assert!(tanpi(f64::INFINITY).is_nan());
        assert_rel("tanpi(0.1)", tanpi(0.1), (M_PI * 0.1).tan(), 1e-15);
    }

    #[test]
    fn pt_and_qt_edges_follow_r() {
        assert!(pt(f64::NAN, 1.0, true, false).is_nan());
        assert!(pt(0.0, 0.0, true, false).is_nan());
        assert_eq!(pt(f64::INFINITY, 1.0, true, false), 1.0);
        assert_eq!(pt(f64::NEG_INFINITY, 1.0, true, false), 0.0);
        assert_eq!(pt(0.0, 5.0, true, false), 0.5);
        assert!(qt(f64::NAN, 1.0, true, false).is_nan());
        assert!(qt(0.5, 0.0, true, false).is_nan());
        assert!(qt(1.5, 1.0, true, false).is_nan());
        assert_eq!(qt(0.0, 1.0, true, false), f64::NEG_INFINITY);
        assert_eq!(qt(1.0, 1.0, true, false), f64::INFINITY);
        assert_eq!(qt(0.5, 3.0, true, false), 0.0);
        // df ~= 1 is the Cauchy: qt(0.75, 1) = 1 exactly via tanpi(0.25).
        assert_eq!(qt(0.75, 1.0, true, false), 1.0);
        // df > 1e20 dispatches to qnorm.
        assert_eq!(
            qt(0.3, 1e21, true, false),
            qnorm(0.3, 0.0, 1.0, true, false)
        );
    }
}
