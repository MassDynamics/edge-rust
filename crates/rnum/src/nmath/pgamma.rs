//! Port of R's `src/nmath/pgamma.c` (`pgamma`, `pgamma_raw` and its static helpers),
//! `qgamma.c` (`qgamma`, `qchisq_appr`), `pchisq.c` (`pchisq`) and `qchisq.c` (`qchisq`),
//! all from R 4.5.0. Proved against `scalar/pgamma.csv`, `scalar/qgamma.csv`,
//! `scalar/pchisq.csv` and `scalar/qchisq.csv` in `tests/scalar_pgamma.rs`.
//!
//! `pgamma.c` also holds `log1pmx`, `lgamma1p`, `logspace_add`, `logspace_sub` and
//! `dpois_wrap`; those are ported in `gamma.rs` next to `dpois_raw` and are called from
//! there rather than duplicated here.
//!
//! `pgamma_raw` is Morten Welinder's redesign (originally for Gnumeric): a small-`x`
//! series (A&S 6.5.29), a Poisson-density-scaled upper series / lower continued fraction
//! on either side of `alph`, and Smith's asymptotic expansion (`ppois_asymp`) when `x` is
//! near `alph`. `qgamma` is AS 91 (Best & Roberts 1975) with R core's log/tail extensions
//! and final Newton steps.

// The C's `do { ... } while (cond)` loops are written `loop { ...; if !(cond) { break } }`
// below. Clippy would have `!(a > b)` as `a <= b`, but the two differ on a NaN, where the
// C loop stops and the rewrite would spin forever; the negation keeps the C's behaviour.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use crate::nmath::arith::fmax2;
use crate::nmath::consts::M_LN2;
use crate::nmath::dpq::{
    r_d_0, r_d_1, r_dt_0, r_dt_1, r_dt_clog, r_dt_log, r_dt_qiv, r_log1_exp, r_p_bounds_01,
    r_q_p01_boundaries, r_q_p01_check,
};
use crate::nmath::gamma::{dgamma, dpois_raw, dpois_wrap, lgamma1p, lgammafn, log1pmx};
use crate::nmath::pnorm::{dnorm, pnorm, qnorm};

/// `pgamma.c`: `scalefactor := (2^32)^8 = 2^256 = 1.157921e+77`.
const SCALEFACTOR: f64 = {
    let s = 4294967296.0_f64;
    let s2 = s * s;
    let s4 = s2 * s2;
    s4 * s4
};

/// `pgamma.c: pgamma_smallx` — Abramowitz and Stegun 6.5.29 [right].
fn pgamma_smallx(x: f64, alph: f64, lower_tail: bool, log_p: bool) -> f64 {
    let mut sum = 0.0;
    let mut c = alph;
    let mut n = 0.0;
    let mut term;

    /*
     * Relative to 6.5.29 all terms have been multiplied by alph
     * and the first, thus being 1, is omitted.
     */
    loop {
        n += 1.0;
        c *= -x / n;
        term = c / (alph + n);
        sum += term;
        if !(term.abs() > f64::EPSILON * sum.abs()) {
            break;
        }
    }

    if lower_tail {
        let f1 = if log_p { sum.ln_1p() } else { 1.0 + sum };
        let f2 = if alph > 1.0 {
            let d = dpois_raw(alph, x, log_p);
            if log_p {
                d + x
            } else {
                d * x.exp()
            }
        } else if log_p {
            alph * x.ln() - lgamma1p(alph)
        } else {
            x.powf(alph) / lgamma1p(alph).exp()
        };
        if log_p {
            f1 + f2
        } else {
            f1 * f2
        }
    } else {
        let lf2 = alph * x.ln() - lgamma1p(alph);
        if log_p {
            r_log1_exp(sum.ln_1p() + lf2)
        } else {
            let f1m1 = sum;
            let f2m1 = lf2.exp_m1();
            -(f1m1 + f2m1 + f1m1 * f2m1)
        }
    }
} /* pgamma_smallx() */

/// `pgamma.c: pd_upper_series`.
fn pd_upper_series(x: f64, mut y: f64, log_p: bool) -> f64 {
    let mut term = x / y;
    let mut sum = term;

    loop {
        y += 1.0;
        term *= x / y;
        sum += term;
        if !(term > sum * f64::EPSILON) {
            break;
        }
    }

    /* sum =  \sum_{n=1}^ oo  x^n     / (y*(y+1)*...*(y+n-1))
     *	   =  \sum_{n=0}^ oo  x^(n+1) / (y*(y+1)*...*(y+n))
     *	   =  x/y * (1 + \sum_{n=1}^oo	x^n / ((y+1)*...*(y+n)))
     *	   ~  x/y +  o(x/y)   {which happens when alph -> Inf}
     */
    if log_p {
        sum.ln()
    } else {
        sum
    }
}

/// `pgamma.c: pd_lower_cf` — continued fraction for calculation of the scaled upper-tail
/// `F_{gamma}  ~=  (y / d) * [1 +  (1-y)/d +  O( ((1-y)/d)^2 ) ]`.
fn pd_lower_cf(y: f64, d: f64) -> f64 {
    const MAX_IT: f64 = 200000.0;

    let mut f = 0.0; /* -Wall */

    if y == 0.0 {
        return 0.0;
    }

    let mut f0 = y / d;
    /* Needed, e.g. for  pgamma(10^c(100,295), shape= 1.1, log=TRUE): */
    if (y - 1.0).abs() < d.abs() * f64::EPSILON {
        /* includes y < d = Inf */
        return f0;
    }

    if f0 > 1.0 {
        f0 = 1.0;
    }
    let mut c2 = y;
    let mut c4 = d; /* original (y,d), *not* potentially scaled ones!*/

    let mut a1 = 0.0;
    let mut b1 = 1.0;
    let mut a2 = y;
    let mut b2 = d;

    // `while NEEDED_SCALE`
    while b2 > SCALEFACTOR {
        a1 /= SCALEFACTOR;
        b1 /= SCALEFACTOR;
        a2 /= SCALEFACTOR;
        b2 /= SCALEFACTOR;
    }

    let mut i = 0.0;
    let mut of = -1.0; /* far away */
    while i < MAX_IT {
        i += 1.0;
        c2 -= 1.0;
        let mut c3 = i * c2;
        c4 += 2.0;
        /* c2 = y - i,  c3 = i(y - i),  c4 = d + 2i,  for i odd */
        a1 = c4 * a2 + c3 * a1;
        b1 = c4 * b2 + c3 * b1;

        i += 1.0;
        c2 -= 1.0;
        c3 = i * c2;
        c4 += 2.0;
        /* c2 = y - i,  c3 = i(y - i),  c4 = d + 2i,  for i even */
        a2 = c4 * a1 + c3 * a2;
        b2 = c4 * b1 + c3 * b2;

        // `if NEEDED_SCALE`
        if b2 > SCALEFACTOR {
            a1 /= SCALEFACTOR;
            b1 /= SCALEFACTOR;
            a2 /= SCALEFACTOR;
            b2 /= SCALEFACTOR;
        }

        if b2 != 0.0 {
            f = a2 / b2;
            /* convergence check: relative; "absolute" for very small f : */
            if (f - of).abs() <= f64::EPSILON * fmax2(f0, f.abs()) {
                return f;
            }
            of = f;
        }
    }

    // MATHLIB_WARNING(" ** NON-convergence in pgamma()'s pd_lower_cf() f= %g.\n", f);
    f /* should not happen ... */
} /* pd_lower_cf() */

/// `pgamma.c: pd_lower_series`.
fn pd_lower_series(lambda: f64, mut y: f64) -> f64 {
    let mut term = 1.0;
    let mut sum = 0.0;

    while y >= 1.0 && term > sum * f64::EPSILON {
        term *= y / lambda;
        sum += term;
        y -= 1.0;
    }
    /* sum =  \sum_{n=0}^ oo  y*(y-1)*...*(y - n) / lambda^(n+1)
     *	   =  y/lambda * (1 + \sum_{n=1}^Inf  (y-1)*...*(y-n) / lambda^n)
     *	   ~  y/lambda + o(y/lambda)
     */

    if y != y.floor() {
        /*
         * The series does not converge as the terms start getting
         * bigger (besides flipping sign) for y < -lambda.
         */
        /* FIXME: in quite few cases, adding  term*f  has no effect (f too small)
         *	  and is unnecessary e.g. for pgamma(4e12, 121.1) */
        let f = pd_lower_cf(y, lambda + 1.0 - y);
        sum += term * f;
    }

    sum
} /* pd_lower_series() */

/// `pgamma.c: dpnorm` — the ratio `dnorm(x, 0, 1, FALSE) / pnorm(x, 0, 1, lower_tail,
/// FALSE)` with higher accuracy than doing it directly (Abramowitz & Stegun 26.2.12).
///
/// So as not to repeat a pnorm call, expects `lp == pnorm(x, 0, 1, lower_tail, TRUE)`, but
/// uses it only in the non-critical case where either `x` is small or `p == exp(lp)` is
/// close to 1.
fn dpnorm(mut x: f64, mut lower_tail: bool, lp: f64) -> f64 {
    if x < 0.0 {
        x = -x;
        lower_tail = !lower_tail;
    }

    if x > 10.0 && !lower_tail {
        let mut term = 1.0 / x;
        let mut sum = term;
        let x2 = x * x;
        let mut i = 1.0;

        loop {
            term *= -i / x2;
            sum += term;
            i += 2.0;
            if !(term.abs() > f64::EPSILON * sum) {
                break;
            }
        }

        1.0 / sum
    } else {
        let d = dnorm(x, 0.0, 1.0, false);
        d / lp.exp()
    }
}

/// `pgamma.c: ppois_asymp` — asymptotic expansion to calculate the probability that a
/// Poisson variate has value `<= x`. Various assertions about this are made (without
/// proof) at <http://members.aol.com/iandjmsmith/PoissonApprox.htm>.
fn ppois_asymp(x: f64, lambda: f64, lower_tail: bool, log_p: bool) -> f64 {
    const COEFS_A: [f64; 8] = [
        -1e99, /* placeholder used for 1-indexing */
        2.0 / 3.0,
        -4.0 / 135.0,
        8.0 / 2835.0,
        16.0 / 8505.0,
        -8992.0 / 12629925.0,
        -334144.0 / 492567075.0,
        698752.0 / 1477701225.0,
    ];

    const COEFS_B: [f64; 8] = [
        -1e99, /* placeholder */
        1.0 / 12.0,
        1.0 / 288.0,
        -139.0 / 51840.0,
        -571.0 / 2488320.0,
        163879.0 / 209018880.0,
        5246819.0 / 75246796800.0,
        -534703531.0 / 902961561600.0,
    ];

    let dfm = lambda - x;
    /* If lambda is large, the distribution is highly concentrated
       about lambda.  So representation error in x or lambda can lead
       to arbitrarily large values of pt_ and hence divergence of the
       coefficients of this approximation.
    */
    let pt_ = -log1pmx(dfm / x);
    let mut s2pt = (2.0 * x * pt_).sqrt();
    if dfm < 0.0 {
        s2pt = -s2pt;
    }

    let mut res12 = 0.0;
    let mut res1_term = x.sqrt();
    let mut res1_ig = res1_term;
    let mut res2_term = s2pt;
    let mut res2_ig = res2_term;
    for i in 1..8 {
        let fi = i as f64;
        res12 += res1_ig * COEFS_A[i];
        res12 += res2_ig * COEFS_B[i];
        res1_term *= pt_ / fi;
        res2_term *= 2.0 * pt_ / (2.0 * fi + 1.0);
        res1_ig = res1_ig / x + res1_term;
        res2_ig = res2_ig / x + res2_term;
    }

    let mut elfb = x;
    let mut elfb_term = 1.0;
    for coef_b in &COEFS_B[1..] {
        elfb += elfb_term * coef_b;
        elfb_term /= x;
    }
    if !lower_tail {
        elfb = -elfb;
    }

    let f = res12 / elfb;

    let np = pnorm(s2pt, 0.0, 1.0, !lower_tail, log_p);

    if log_p {
        let n_d_over_p = dpnorm(s2pt, !lower_tail, np);
        np + (f * n_d_over_p).ln_1p()
    } else {
        let nd = dnorm(s2pt, 0.0, 1.0, log_p);
        np + f * nd
    }
} /* ppois_asymp() */

/// `pgamma.c: pgamma_raw` — `pgamma` with `scale = 1`, assuming `(x, alph)` are not NA
/// and `alph > 0`.
pub(crate) fn pgamma_raw(x: f64, alph: f64, lower_tail: bool, log_p: bool) -> f64 {
    r_p_bounds_01!(x, 0.0, f64::INFINITY, lower_tail, log_p);

    let res = if x < 1.0 {
        pgamma_smallx(x, alph, lower_tail, log_p)
    } else if x <= alph - 1.0 && x < 0.8 * (alph + 50.0) {
        /* incl. large alph compared to x */
        let sum = pd_upper_series(x, alph, log_p); /* = x/alph + o(x/alph) */
        let d = dpois_wrap(alph, x, log_p);
        if !lower_tail {
            if log_p {
                r_log1_exp(d + sum)
            } else {
                1.0 - d * sum
            }
        } else if log_p {
            sum + d
        } else {
            sum * d
        }
    } else if alph - 1.0 < x && alph < 0.8 * (x + 50.0) {
        /* incl. large x compared to alph */
        let d = dpois_wrap(alph, x, log_p);
        let sum = if alph < 1.0 {
            if x * f64::EPSILON > 1.0 - alph {
                r_d_1(log_p)
            } else {
                let f = pd_lower_cf(alph, x - (alph - 1.0)) * x / alph;
                /* = [alph/(x - alph+1) + o(alph/(x-alph+1))] * x/alph = 1 + o(1) */
                if log_p {
                    f.ln()
                } else {
                    f
                }
            }
        } else {
            let sum = pd_lower_series(x, alph - 1.0); /* = (alph-1)/x + o((alph-1)/x) */
            if log_p {
                sum.ln_1p()
            } else {
                1.0 + sum
            }
        };
        if !lower_tail {
            if log_p {
                sum + d
            } else {
                sum * d
            }
        } else if log_p {
            r_log1_exp(d + sum)
        } else {
            1.0 - d * sum
        }
    } else {
        /* x >= 1 and x fairly near alph. */
        ppois_asymp(alph - 1.0, x, !lower_tail, log_p)
    };

    /*
     * We lose a fair amount of accuracy to underflow in the cases
     * where the final result is very close to DBL_MIN.	 In those
     * cases, simply redo via log space.
     */
    if !log_p && res < f64::MIN_POSITIVE / f64::EPSILON {
        /* with(.Machine, double.xmin / double.eps) #|-> 1.002084e-292 */
        pgamma_raw(x, alph, lower_tail, true).exp()
    } else {
        res
    }
}

/// `pgamma` in `src/nmath/pgamma.c`: the distribution function of the gamma distribution
/// with shape `alph` and scale `scale` (the regularised incomplete gamma function, A&S
/// 6.5.1), lower or upper tail, optionally on the log scale.
pub fn pgamma(mut x: f64, alph: f64, scale: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x.is_nan() || alph.is_nan() || scale.is_nan() {
        return x + alph + scale;
    }
    if alph < 0.0 || scale <= 0.0 {
        return f64::NAN;
    }
    x /= scale;
    if x.is_nan() {
        /* eg. original x = scale = +Inf */
        return x;
    }
    if alph == 0.0 {
        /* limit case; useful e.g. in pnchisq() */
        return if x <= 0.0 {
            r_dt_0(lower_tail, log_p)
        } else {
            r_dt_1(lower_tail, log_p)
        }; /* <= assert  pgamma(0,0) ==> 0 */
    }
    pgamma_raw(x, alph, lower_tail, log_p)
}

/// `qgamma.c: qchisq_appr` — the AS 91 starting approximation to the chi-squared quantile
/// with `nu` degrees of freedom; `g = log Gamma(nu/2)` and `tol` is AS 91's `EPS1`.
pub(crate) fn qchisq_appr(
    p: f64,
    nu: f64,
    g: f64, /* = log Gamma(nu/2) */
    lower_tail: bool,
    log_p: bool,
    tol: f64, /* EPS1 */
) -> f64 {
    const C7: f64 = 4.67;
    const C8: f64 = 6.66;
    const C9: f64 = 6.73;
    const C10: f64 = 13.32;

    /* test arguments and initialise */

    if p.is_nan() || nu.is_nan() {
        return p + nu;
    }
    r_q_p01_check!(p, log_p);
    if nu <= 0.0 {
        return f64::NAN;
    }

    let alpha = 0.5 * nu; /* = [pq]gamma() shape */
    let c = alpha - 1.0;

    let p1 = r_dt_log(p, lower_tail, log_p);
    if nu < (-1.24) * p1 {
        /* for small chi-squared */
        /* log(alpha) + g = log(alpha) + log(gamma(alpha)) =
         *        = log(alpha*gamma(alpha)) = lgamma(alpha+1) suffers from
         *  catastrophic cancellation when alpha << 1
         */
        let lgam1pa = if alpha < 0.5 {
            lgamma1p(alpha)
        } else {
            alpha.ln() + g
        };
        ((lgam1pa + p1) / alpha + M_LN2).exp()
    } else if nu > 0.32 {
        /*  using Wilson and Hilferty estimate */

        let x = qnorm(p, 0.0, 1.0, lower_tail, log_p);
        let p1 = 2.0 / (9.0 * nu);
        let mut ch = nu * (x * p1.sqrt() + 1.0 - p1).powf(3.0);

        /* approximation for p tending to 1: */
        if ch > 2.2 * nu + 6.0 {
            ch = -2.0 * (r_dt_clog(p, lower_tail, log_p) - c * (0.5 * ch).ln() + g);
        }
        ch
    } else {
        /* "small nu" : 1.24*(-log(p)) <= nu <= 0.32 */

        let mut ch = 0.4;
        let a = r_dt_clog(p, lower_tail, log_p) + g + c * M_LN2;
        loop {
            let q = ch;
            let p1 = 1.0 / (1.0 + ch * (C7 + ch));
            let p2 = ch * (C9 + ch * (C8 + ch));
            let t = -0.5 + (C7 + 2.0 * ch) * p1 - (C9 + ch * (C10 + 3.0 * ch)) / p2;
            ch -= (1.0 - (a + 0.5 * ch).exp() * p2 * p1) / t;
            if !((q - ch).abs() > tol * ch.abs()) {
                break;
            }
        }
        ch
    }
}

/// `qgamma` in `src/nmath/qgamma.c`: the quantile function of the gamma distribution with
/// shape `alpha` and scale `scale`. Applied Statistics algorithm AS 91 (`ppchi2`), via
/// `pgamma_raw` (AS 239), with R core's `lower_tail`/`log_p` handling, non-trivial results
/// for `p` outside `[0.000002, 0.999998]`, and final Newton step(s).
pub fn qgamma(mut p: f64, alpha: f64, scale: f64, lower_tail: bool, mut log_p: bool) -> f64 {
    const EPS1: f64 = 1e-2;
    const EPS2: f64 = 5e-7; /* final precision of AS 91 */
    const EPS_N: f64 = 1e-15; /* precision of Newton step / iterations */

    const MAXIT: i32 = 1000; /* was 20 */

    const P_MIN: f64 = 1e-100; /* was 0.000002 = 2e-6 */
    const P_MAX: f64 = 1.0 - 1e-14; /* was (1-1e-12) and 0.999998 = 1 - 2e-6 */

    const I420: f64 = 1.0 / 420.0;
    const I2520: f64 = 1.0 / 2520.0;
    const I5040: f64 = 1.0 / 5040.0;

    let mut max_it_newton = 1;

    /* test arguments and initialise */

    if p.is_nan() || alpha.is_nan() || scale.is_nan() {
        return p + alpha + scale;
    }
    r_q_p01_boundaries!(p, 0.0, f64::INFINITY, lower_tail, log_p);

    if alpha < 0.0 || scale <= 0.0 {
        return f64::NAN;
    }

    if alpha == 0.0 {
        /* all mass at 0 : */
        return 0.0;
    }

    if alpha < 1e-10 {
        /* Warning seems unnecessary now: */
        max_it_newton = 7; /* may still be increased below */
    }

    let mut p_ = r_dt_qiv(p, lower_tail, log_p); /* lower_tail prob (in any case) */

    let mut g = lgammafn(alpha); /* log Gamma(v/2) */

    /*----- Phase I : Starting Approximation */
    let mut ch = qchisq_appr(
        p,
        /* nu= 'df' =  */ 2.0 * alpha,
        /* lgamma(nu/2)= */ g,
        lower_tail,
        log_p,
        /* tol= */ EPS1,
    );
    // The C's `goto END` targets leave this block; what follows it is the `END:` label.
    'end: {
        if !ch.is_finite() {
            /* forget about all iterations! */
            max_it_newton = 0;
            break 'end;
        }
        if ch < EPS2 {
            /* Corrected according to AS 91; MM, May 25, 1999 */
            max_it_newton = 20;
            break 'end; /* and do Newton steps */
        }

        /* FIXME: This (cutoff to {0, +Inf}) is far from optimal
         * -----  when log_p or !lower_tail, but NOT doing it can be even worse */
        // Kept as the C writes it rather than `!(P_MIN..=P_MAX).contains(&p_)`: a NaN
        // `p_` falls through here in C, and the rewrite would trap it.
        #[allow(clippy::manual_range_contains)]
        if p_ > P_MAX || p_ < P_MIN {
            /* did return ML_POSINF or 0.;	much better: */
            max_it_newton = 20;
            break 'end; /* and do Newton steps */
        }

        /*----- Phase II: Iteration
         *	Call pgamma() [AS 239]	and calculate seven term taylor series
         */
        let c = alpha - 1.0;
        let s6 = (120.0 + c * (346.0 + 127.0 * c)) * I5040; /* used below, is "const" */

        let ch0 = ch; /* save initial approx. */
        for _ in 1..=MAXIT {
            let q = ch;
            let p1 = 0.5 * ch;
            let p2 = p_ - pgamma_raw(p1, alpha, /*lower_tail*/ true, /*log_p*/ false);
            if !p2.is_finite() || ch <= 0.0 {
                ch = ch0;
                max_it_newton = 27;
                break 'end; /*was  return ML_NAN;*/
            }

            let t = p2 * (alpha * M_LN2 + g + p1 - c * ch.ln()).exp();
            let b = t / ch;
            let a = 0.5 * t - b * c;
            let s1 =
                (210.0 + a * (140.0 + a * (105.0 + a * (84.0 + a * (70.0 + 60.0 * a))))) * I420;
            let s2 = (420.0 + a * (735.0 + a * (966.0 + a * (1141.0 + 1278.0 * a)))) * I2520;
            let s3 = (210.0 + a * (462.0 + a * (707.0 + 932.0 * a))) * I2520;
            let s4 =
                (252.0 + a * (672.0 + 1182.0 * a) + c * (294.0 + a * (889.0 + 1740.0 * a))) * I5040;
            let s5 = (84.0 + 2264.0 * a + c * (1175.0 + 606.0 * a)) * I2520;

            ch += t
                * (1.0 + 0.5 * t * s1
                    - b * c * (s1 - b * (s2 - b * (s3 - b * (s4 - b * (s5 - b * s6))))));
            if (q - ch).abs() < EPS2 * ch {
                break 'end;
            }
            if (q - ch).abs() > 0.1 * ch {
                /* diverging? -- also forces ch > 0 */
                if ch < q {
                    ch = 0.9 * q;
                } else {
                    ch = 1.1 * q;
                }
            }
        }
        /* no convergence in MAXIT iterations -- but we add Newton now... */
    }

    /* END:
     * PR# 2214 :	 From: Morten Welinder <terra@diku.dk>, Fri, 25 Oct 2002 16:50
     * --------	 To: R-bugs@biostat.ku.dk     Subject: qgamma precision
     *
     * With a final Newton step, double accuracy, e.g. for (p= 7e-4; nu= 0.9)
     *
     * Improved (MM): - only if rel.Err > EPS_N (= 1e-15);
     *		    - also for lower_tail = FALSE	 or log_p = TRUE
     *		    - optionally *iterate* Newton
     */
    let mut x = 0.5 * scale * ch;
    if max_it_newton != 0 {
        /* always use log scale */
        if !log_p {
            p = p.ln();
            log_p = true;
        }
        if x == 0.0 {
            let one_p = 1.0 + 1e-7;
            let one_m = 1.0 - 1e-7;
            x = f64::MIN_POSITIVE;
            p_ = pgamma(x, alpha, scale, lower_tail, log_p);
            if (lower_tail && p_ > p * one_p) || (!lower_tail && p_ < p * one_m) {
                return 0.0;
            }
            /* else:  continue, using x = DBL_MIN instead of  0  */
        } else {
            p_ = pgamma(x, alpha, scale, lower_tail, log_p);
        }
        if p_ == f64::NEG_INFINITY {
            return 0.0; /* PR#14710 */
        }
        for i in 1..=max_it_newton {
            let p1 = p_ - p;
            if p1.abs() < (EPS_N * p).abs() {
                break;
            }
            /* else */
            g = dgamma(x, alpha, scale, log_p);
            if g == r_d_0(log_p) {
                break;
            }
            /* else :
             * delta x = f(x)/f'(x);
             * if(log_p) f(x) := log P(x) - p; f'(x) = d/dx log P(x) = P' / P
             * ==> f(x)/f'(x) = f*P / P' = f*exp(p_) / P' (since p_ = log P(x))
             */
            let mut t = if log_p { p1 * (p_ - g).exp() } else { p1 / g }; /* = "delta x" */
            t = if lower_tail { x - t } else { x + t };
            p_ = pgamma(t, alpha, scale, lower_tail, log_p);
            if (p_ - p).abs() > p1.abs() || (i > 1 && (p_ - p).abs() == p1.abs())
            /* <- against flip-flop */
            {
                /* no improvement */
                break;
            } /* else : */
            x = t;
        }
    }

    x
}

/// `pchisq` in `src/nmath/pchisq.c`: the distribution function of the chi-squared
/// distribution with `df` degrees of freedom, `pgamma(x, df/2, 2)`.
pub fn pchisq(x: f64, df: f64, lower_tail: bool, log_p: bool) -> f64 {
    pgamma(x, df / 2.0, 2.0, lower_tail, log_p)
}

/// `qchisq` in `src/nmath/qchisq.c`: the quantile function of the chi-squared distribution
/// with `df` degrees of freedom, `qgamma(p, df/2, 2)`.
pub fn qchisq(p: f64, df: f64, lower_tail: bool, log_p: bool) -> f64 {
    qgamma(p, 0.5 * df, 2.0, lower_tail, log_p)
}
