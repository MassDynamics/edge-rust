//! Port of R's `src/nmath/pnorm.c` (`pnorm5`, `pnorm_both`), `qnorm.c` (`qnorm5`) and
//! `dnorm.c` (`dnorm4`), all from R 4.5.0. Proved against `scalar/pnorm.csv` and
//! `scalar/qnorm.csv` in `tests/scalar_pnorm.rs`; `dnorm` has no golden and is pinned by
//! the unit tests at the bottom of this file.
//!
//! `pnorm_both` is Cody's (1969, 1993) rational Chebyshev approximation with Maechler's
//! `_both`/`log_p` extensions; `qnorm` is Wichura's AS 241 with Maechler's (2022)
//! asymptotic tail formulas for `r > 27`, which R only reaches with `log_p = TRUE`.

// AS 241's and Cody's coefficients are transcribed verbatim from the C, at more digits
// than a double holds, so they can be diffed against the source.
#![allow(clippy::excessive_precision)]

use super::consts::{M_1_SQRT_2PI, M_2PI, M_LN2, M_LN_SQRT_2PI, M_SQRT2, M_SQRT_32};
use super::dpq::{
    r_d_0, r_d_1, r_dt_0, r_dt_1, r_dt_civ, r_dt_qiv, r_forceint, r_q_p01_boundaries,
};

/// C's `ldexp(x, n)` = `x * 2^n`. Not in `std`; a multiply by a power of two is exact for
/// the small `n` used here, which is what the C leans on ("== (x / 2) perfectly").
fn ldexp(x: f64, n: i32) -> f64 {
    x * 2f64.powi(n)
}

/// `pnorm5` in `src/nmath/pnorm.c`: the normal distribution function
/// `P[X <= x]` (or the upper tail, or either on the log scale) for `X ~ N(mu, sigma^2)`.
pub fn pnorm(x: f64, mu: f64, sigma: f64, lower_tail: bool, log_p: bool) -> f64 {
    // Note: The structure of these checks has been carefully thought through.
    // For example, if x == mu and sigma == 0, we get the correct answer 1.
    if x.is_nan() || mu.is_nan() || sigma.is_nan() {
        return x + mu + sigma;
    }
    if !x.is_finite() && mu == x {
        return f64::NAN; // x-mu is NaN
    }
    if sigma <= 0.0 {
        if sigma < 0.0 {
            return f64::NAN;
        }
        // sigma = 0 :
        return if x < mu {
            r_dt_0(lower_tail, log_p)
        } else {
            r_dt_1(lower_tail, log_p)
        };
    }
    let p = (x - mu) / sigma;
    if !p.is_finite() {
        return if x < mu {
            r_dt_0(lower_tail, log_p)
        } else {
            r_dt_1(lower_tail, log_p)
        };
    }
    let x = p;

    let (mut p, mut cp) = (0.0, 0.0);
    pnorm_both(x, &mut p, &mut cp, if lower_tail { 0 } else { 1 }, log_p);

    if lower_tail {
        p
    } else {
        cp
    }
}

/// `pnorm_both` in `src/nmath/pnorm.c`: both tails of the standard normal at once.
///
/// `i_tail` in {0,1,2} means: "lower", "upper", or "both":
/// if(lower) return  *cum := P[X <= x]
/// if(upper) return *ccum := P[X >  x] = 1 - P[X <= x]
///
/// As in C, a tail that was not asked for is left untouched — including `ccum` in the
/// `log_p` branches where the C only sets it on one side of zero.
// The `-38.4674 < x && x < 8.2924` tests are open on both ends; `Range::contains` is not.
#[allow(clippy::manual_range_contains)]
pub(crate) fn pnorm_both(x: f64, cum: &mut f64, ccum: &mut f64, i_tail: i32, log_p: bool) {
    const A: [f64; 5] = [
        2.2352520354606839287,
        161.02823106855587881,
        1067.6894854603709582,
        18154.981253343561249,
        0.065682337918207449113,
    ];
    const B: [f64; 4] = [
        47.20258190468824187,
        976.09855173777669322,
        10260.932208618978205,
        45507.789335026729956,
    ];
    const C: [f64; 9] = [
        0.39894151208813466764,
        8.8831497943883759412,
        93.506656132177855979,
        597.27027639480026226,
        2494.5375852903726711,
        6848.1904505362823326,
        11602.651437647350124,
        9842.7148383839780218,
        1.0765576773720192317e-8,
    ];
    const D: [f64; 8] = [
        22.266688044328115691,
        235.38790178262499861,
        1519.377599407554805,
        6485.558298266760755,
        18615.571640885098091,
        34900.952721145977266,
        38912.003286093271411,
        19685.429676859990727,
    ];
    const P: [f64; 6] = [
        0.21589853405795699,
        0.1274011611602473639,
        0.022235277870649807,
        0.001421619193227893466,
        2.9112874951168792e-5,
        0.02307344176494017303,
    ];
    const Q: [f64; 5] = [
        1.28426009614491121,
        0.468238212480865118,
        0.0659881378689285515,
        0.00378239633202758244,
        7.29751555083966205e-5,
    ];

    if x.is_nan() {
        *cum = x;
        *ccum = x;
        return;
    }

    // Consider changing these :
    let eps = f64::EPSILON * 0.5;

    // i_tail in {0,1,2} =^= {lower, upper, both}
    let lower = i_tail != 1;
    let upper = i_tail != 0;

    let (mut xden, mut xnum, mut temp);

    let y = x.abs();
    if y <= 0.67448975 {
        // qnorm(3/4) = .6744.... -- earlier had 0.66291
        if y > eps {
            let xsq = x * x;
            xnum = A[4] * xsq;
            xden = xsq;
            for (a, b) in A[..3].iter().zip(&B[..3]) {
                xnum = (xnum + a) * xsq;
                xden = (xden + b) * xsq;
            }
        } else {
            xnum = 0.0;
            xden = 0.0;
        }

        temp = x * (xnum + A[3]) / (xden + B[3]);
        if lower {
            *cum = 0.5 + temp;
        }
        if upper {
            *ccum = 0.5 - temp;
        }
        if log_p {
            if lower {
                *cum = cum.ln();
            }
            if upper {
                *ccum = ccum.ln();
            }
        }
    } else if y <= M_SQRT_32 {
        // Evaluate pnorm for 0.674.. = qnorm(3/4) < |x| <= sqrt(32) ~= 5.657

        xnum = C[8] * y;
        xden = y;
        for (c, d) in C[..7].iter().zip(&D[..7]) {
            xnum = (xnum + c) * y;
            xden = (xden + d) * y;
        }
        temp = (xnum + C[7]) / (xden + D[7]);

        do_del(y, x, temp, lower, upper, log_p, cum, ccum);
        swap_tail(x, lower, cum, ccum);
    }
    // else	  |x| > sqrt(32) = 5.657 :
    // the next two case differentiations were really for lower=T, log=F
    // Particularly	 *not*	for  log_p !
    //
    // Cody had (-37.5193 < x  &&  x < 8.2924) ; R originally had y < 50
    //
    // Note that we do want symmetry(0), lower/upper -> hence use y
    //
    // NB: allowing "DENORMS" ==> boundaries at +/- 38.4674  <--> qnorm(log(2^-1074), log.p=TRUE)
    // --                               rather than 37.5193 (up to R 4.4.x)
    else if (log_p && y < 1e170) // avoid underflow below
        || (lower && -38.4674 < x && x < 8.2924)
        || (upper && -8.2924 < x && x < 38.4674)
    {
        // Evaluate pnorm for x in (-37.5, -5.657) union (5.657, 37.5)
        let xsq = 1.0 / (x * x); // (1./x)*(1./x) might be better
        xnum = P[5] * xsq;
        xden = xsq;
        for (p, q) in P[..4].iter().zip(&Q[..4]) {
            xnum = (xnum + p) * xsq;
            xden = (xden + q) * xsq;
        }
        temp = xsq * (xnum + P[4]) / (xden + Q[4]);
        temp = (M_1_SQRT_2PI - temp) / y;

        do_del(x, x, temp, lower, upper, log_p, cum, ccum);
        swap_tail(x, lower, cum, ccum);
    } else {
        // large |x| such that probs are 0 or 1
        if x > 0.0 {
            *cum = r_d_1(log_p);
            *ccum = r_d_0(log_p);
        } else {
            *cum = r_d_0(log_p);
            *ccum = r_d_1(log_p);
        }
    }
    // NO_DENORMS is not defined in R's build: denormalised results are returned as is.
}

/// `d_2(x)` in `pnorm.c`: `ldexp(x, -1)` == `x / 2` "perfectly".
fn d_2(x: f64) -> f64 {
    ldexp(x, -1)
}

/// The `do_del(X)` macro in `pnorm.c`. `xx` is the macro's `X` (`y` or `x` at the two
/// call sites); `x` is the signed argument the `ccum` condition reads.
#[allow(clippy::too_many_arguments)]
fn do_del(
    xx: f64,
    x: f64,
    temp: f64,
    lower: bool,
    upper: bool,
    log_p: bool,
    cum: &mut f64,
    ccum: &mut f64,
) {
    let xsq = ldexp(ldexp(xx, 4).trunc(), -4);
    let del = (xx - xsq) * (xx + xsq);
    if log_p {
        *cum = (-xsq * d_2(xsq)) - d_2(del) + temp.ln();
        if (lower && x > 0.0) || (upper && x <= 0.0) {
            *ccum = (-(-xsq * d_2(xsq)).exp() * (-d_2(del)).exp() * temp).ln_1p();
        }
    } else {
        *cum = (-xsq * d_2(xsq)).exp() * (-d_2(del)).exp() * temp;
        *ccum = 1.0 - *cum;
    }
}

/// The `swap_tail` macro in `pnorm.c`: for `x > 0`, swap `ccum <--> cum`.
fn swap_tail(x: f64, lower: bool, cum: &mut f64, ccum: &mut f64) {
    if x > 0.0 {
        let temp = *cum;
        if lower {
            *cum = *ccum;
        }
        *ccum = temp;
    }
}

/// `qnorm5` in `src/nmath/qnorm.c`: the normal quantile function, Wichura's AS 241 improved
/// for the very extreme tail (and `log_p = TRUE`) after Maechler (2022).
pub fn qnorm(p: f64, mu: f64, sigma: f64, lower_tail: bool, log_p: bool) -> f64 {
    if p.is_nan() || mu.is_nan() || sigma.is_nan() {
        return p + mu + sigma;
    }
    r_q_p01_boundaries!(p, f64::NEG_INFINITY, f64::INFINITY, lower_tail, log_p);

    if sigma < 0.0 {
        return f64::NAN;
    }
    if sigma == 0.0 {
        return mu;
    }

    let p_ = r_dt_qiv(p, lower_tail, log_p); // real lower_tail prob. p
    let q = p_ - 0.5;

    // -- use AS 241 ---
    // double ppnd16_(double *p, long *ifault)
    //      ALGORITHM AS241  APPL. STATIST. (1988) VOL. 37, NO. 3
    //
    //      Produces the normal deviate Z corresponding to a given lower
    //      tail area of P; Z is accurate to about 1 part in 10**16.
    //
    //      (original fortran code used PARAMETER(..) for the coefficients
    //       and provided hash codes for checking them...)
    let val;
    if q.abs() <= 0.425 {
        // |p~ - 0.5| <= .425  <==> 0.075 <= p~ <= 0.925
        let r = 0.180625 - q * q; // = .425^2 - q^2  >= 0
        val = q
            * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r
                + 67265.770927008700853)
                * r
                + 45921.953931549871457)
                * r
                + 13731.693765509461125)
                * r
                + 1971.5909503065514427)
                * r
                + 133.14166789178437745)
                * r
                + 3.387132872796366608)
            / (((((((r * 5226.495278852854561 + 28729.085735721942674) * r
                + 39307.89580009271061)
                * r
                + 21213.794301586595867)
                * r
                + 5394.1960214247511077)
                * r
                + 687.1870074920579083)
                * r
                + 42.313330701600911252)
                * r
                + 1.0);
    } else {
        // closer than 0.075 from {0,1} boundary :
        //  r := log(p~);  p~ = min(p, 1-p) < 0.075 :
        let lp = if log_p && ((lower_tail && q <= 0.0) || (!lower_tail && q > 0.0)) {
            p
        } else {
            (if q > 0.0 {
                r_dt_civ(p, lower_tail, log_p) // 1-p
            } else {
                p_ // = R_DT_Iv(p) ^=  p
            })
            .ln()
        };
        // r = sqrt( - log(min(p,1-p)) )  <==>  min(p, 1-p) = exp( - r^2 ) :
        let mut r = (-lp).sqrt();
        let mut v;
        if r <= 5.0 {
            // <==> min(p,1-p) >= exp(-25) ~= 1.3888e-11
            r += -1.6;
            v = (((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r
                + 0.24178072517745061177)
                * r
                + 1.27045825245236838258)
                * r
                + 3.64784832476320460504)
                * r
                + 5.7694972214606914055)
                * r
                + 4.6303378461565452959)
                * r
                + 1.42343711074968357734)
                / (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r
                    + 0.0151986665636164571966)
                    * r
                    + 0.14810397642748007459)
                    * r
                    + 0.68976733498510000455)
                    * r
                    + 1.6763848301838038494)
                    * r
                    + 2.05319162663775882187)
                    * r
                    + 1.0);
        } else if r <= 27.0 {
            // p is very close to  0 or 1: r in (5, 27] :
            //  r >   5 <==> min(p,1-p)  < exp(-25) = 1.3888..e-11
            //  r <= 27 <==> min(p,1-p) >= exp(-27^2) = exp(-729) ~= 2.507972e-317
            // i.e., we are just barely in the range where min(p, 1-p) has not yet underflowed to zero.
            // Wichura, p.478: minimax rational approx R_3(t) is for 5 <= t <= 27  (t :== r)
            r += -5.0;
            v = (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r
                + 0.0012426609473880784386)
                * r
                + 0.026532189526576123093)
                * r
                + 0.29656057182850489123)
                * r
                + 1.7848265399172913358)
                * r
                + 5.4637849111641143699)
                * r
                + 6.6579046435011037772)
                / (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r
                    + 1.8463183175100546818e-5)
                    * r
                    + 7.868691311456132591e-4)
                    * r
                    + 0.0148753612908506148525)
                    * r
                    + 0.13692988092273580531)
                    * r
                    + 0.59983220655588793769)
                    * r
                    + 1.0);
        } else {
            // r > 27: p is *really* close to 0 or 1 .. practically only when log_p =TRUE
            if r >= 6.4e8 {
                // p is *very extremely* close to 0 or 1
                // Using the asymptotical formula ("0-th order"): qn = sqrt(2*s)
                v = r * M_SQRT2;
            } else {
                let s2 = -ldexp(lp, 1); // = -2*lp = 2s
                let mut x2 = s2 - (M_2PI * s2).ln(); // = xs_1
                                                     // if(r >= 36000.)  # <==> s >= 36000^2   use x2 = xs_1  above
                if r < 36000.0 {
                    x2 = s2 - (M_2PI * x2).ln() - 2.0 / (2.0 + x2); // == xs_2
                    if r < 840.0 {
                        // 27 < r < 840
                        x2 = s2 - (M_2PI * x2).ln()
                            + 2.0 * (-(1.0 - 1.0 / (4.0 + x2)) / (2.0 + x2)).ln_1p(); // == xs_3
                        if r < 109.0 {
                            // 27 < r < 109
                            x2 = s2 - (M_2PI * x2).ln()
                                + 2.0
                                    * (-(1.0 - (1.0 - 5.0 / (6.0 + x2)) / (4.0 + x2)) / (2.0 + x2))
                                        .ln_1p(); // == xs_4
                            if r < 55.0 {
                                // 27 < r < 55
                                x2 = s2 - (M_2PI * x2).ln()
                                    + 2.0
                                        * (-(1.0
                                            - (1.0 - (5.0 - 9.0 / (8.0 + x2)) / (6.0 + x2))
                                                / (4.0 + x2))
                                            / (2.0 + x2))
                                            .ln_1p(); // == xs_5
                            }
                        }
                    }
                }
                v = x2.sqrt();
            }
        }
        if q < 0.0 {
            v = -v;
        }
        val = v;
    }
    mu + sigma * val
}

/// `dnorm4` in `src/nmath/dnorm.c`: the normal density, on the log scale when `give_log`.
pub fn dnorm(x: f64, mu: f64, sigma: f64, give_log: bool) -> f64 {
    if x.is_nan() || mu.is_nan() || sigma.is_nan() {
        return x + mu + sigma;
    }
    if sigma < 0.0 {
        return f64::NAN;
    }
    if !sigma.is_finite() {
        return r_d_0(give_log);
    }
    if !x.is_finite() && mu == x {
        return f64::NAN; // x-mu is NaN
    }
    if sigma == 0.0 {
        return if x == mu {
            f64::INFINITY
        } else {
            r_d_0(give_log)
        };
    }
    let x = (x - mu) / sigma;

    if !x.is_finite() {
        return r_d_0(give_log);
    }

    let x = x.abs();
    if x >= 2.0 * f64::MAX.sqrt() {
        return r_d_0(give_log);
    }
    if give_log {
        return -(M_LN_SQRT_2PI + 0.5 * x * x + sigma.ln());
    }
    //  M_1_SQRT_2PI = 1 / sqrt(2 * pi)
    // MATHLIB_FAST_dnorm is not defined in R's build: the "more accurate, less fast" path.
    if x < 5.0 {
        return M_1_SQRT_2PI * (-0.5 * x * x).exp() / sigma;
    }

    // ELSE:
    //
    // x*x  may lose upto about two digits accuracy for "large" x
    // Morten Welinder's proposal for PR#15620
    // https://bugs.r-project.org/show_bug.cgi?id=15620
    //
    // -- 1 --  No hoop jumping when we underflow to zero anyway:
    //
    //  -x^2/2 <         log(2)*.Machine$double.min.exp  <==>
    //     x   > sqrt(-2*log(2)*.Machine$double.min.exp) =IEEE= 37.64031
    // but "thanks" to denormalized numbers, underflow happens a bit later,
    //  effective.D.MIN.EXP <- with(.Machine, double.min.exp + double.ulp.digits)
    // for IEEE, DBL_MIN_EXP is -1022 but "effective" is -1074
    // ==> boundary = sqrt(-2*log(2)*(.Machine$double.min.exp + .Machine$double.ulp.digits))
    //              =IEEE=  38.58601
    // [on one x86_64 platform, effective boundary a bit lower: 38.56804]
    //
    // C's DBL_MIN_EXP is -1021 and DBL_MANT_DIG 53, as `f64::MIN_EXP` and
    // `f64::MANTISSA_DIGITS` are here, so the boundary is the C's 38.56804.
    if x > (-2.0 * M_LN2 * ((f64::MIN_EXP + 1 - f64::MANTISSA_DIGITS as i32) as f64)).sqrt() {
        return 0.0;
    }

    // Now, to get full accuracy, split x into two parts,
    //  x = x1+x2, such that |x2| <= 2^-16.
    // Assuming that we are using IEEE doubles, that means that
    // x1*x1 is error free for x<1024 (but we have x < 38.6 anyway).
    //
    // If we do not have IEEE this is still an improvement over the naive formula.
    let x1 = //  R_forceint(x * 65536) / 65536 =
        ldexp(r_forceint(ldexp(x, 16)), -16);
    let x2 = x - x1;
    M_1_SQRT_2PI / sigma * ((-0.5 * x1 * x1).exp() * ((-0.5 * x2 - x1) * x2).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dnorm_at_zero_is_one_over_sqrt_2pi() {
        assert_eq!(dnorm(0.0, 0.0, 1.0, false), M_1_SQRT_2PI);
        assert_eq!(dnorm(0.0, 0.0, 1.0, true), -M_LN_SQRT_2PI);
        // Scaling by sigma, and the split-x path (|x| >= 5) against the log path: two
        // different roundings of the same value, so a few ulps apart.
        assert_eq!(dnorm(1.0, 1.0, 2.0, false), M_1_SQRT_2PI / 2.0);
        let (got, want) = (
            dnorm(7.0, 0.0, 1.0, false),
            dnorm(7.0, 0.0, 1.0, true).exp(),
        );
        assert!((got - want).abs() <= 1e-14 * want, "{got:e} vs {want:e}");
        // Past the underflow boundary the split path returns an exact 0.
        assert_eq!(dnorm(39.0, 0.0, 1.0, false), 0.0);
    }

    #[test]
    fn dnorm_nan_and_inf_corners() {
        assert!(dnorm(f64::NAN, 0.0, 1.0, false).is_nan());
        assert!(dnorm(0.0, f64::NAN, 1.0, false).is_nan());
        assert!(dnorm(0.0, 0.0, f64::NAN, false).is_nan());
        assert!(dnorm(0.0, 0.0, -1.0, false).is_nan());
        assert!(dnorm(f64::INFINITY, f64::INFINITY, 1.0, false).is_nan());
        assert_eq!(dnorm(0.0, 0.0, f64::INFINITY, false), 0.0);
        assert_eq!(dnorm(0.0, 0.0, f64::INFINITY, true), f64::NEG_INFINITY);
        assert_eq!(dnorm(f64::INFINITY, 0.0, 1.0, false), 0.0);
        assert_eq!(dnorm(f64::NEG_INFINITY, 0.0, 1.0, true), f64::NEG_INFINITY);
        assert_eq!(dnorm(1.0, 1.0, 0.0, false), f64::INFINITY);
        assert_eq!(dnorm(2.0, 1.0, 0.0, false), 0.0);
    }

    #[test]
    fn pnorm_nan_and_inf_corners() {
        assert!(pnorm(f64::NAN, 0.0, 1.0, true, false).is_nan());
        assert!(pnorm(0.0, 0.0, -1.0, true, false).is_nan());
        assert!(pnorm(f64::INFINITY, f64::INFINITY, 1.0, true, false).is_nan());
        assert_eq!(pnorm(f64::INFINITY, 0.0, 1.0, true, false), 1.0);
        assert_eq!(pnorm(f64::NEG_INFINITY, 0.0, 1.0, true, false), 0.0);
        assert_eq!(pnorm(f64::NEG_INFINITY, 0.0, 1.0, false, false), 1.0);
        assert_eq!(
            pnorm(f64::NEG_INFINITY, 0.0, 1.0, true, true),
            f64::NEG_INFINITY
        );
        assert_eq!(pnorm(f64::INFINITY, 0.0, 1.0, true, true), 0.0);
        // sigma = 0: a step at mu, with x == mu on the "1" side.
        assert_eq!(pnorm(1.0, 1.0, 0.0, true, false), 1.0);
        assert_eq!(pnorm(0.5, 1.0, 0.0, true, false), 0.0);
        assert_eq!(pnorm(0.0, 0.0, 1.0, true, false), 0.5);
        assert_eq!(pnorm(0.0, 0.0, 1.0, true, true), -M_LN2);
        // Beyond +/-38.4674 the lower tail is an exact 0 / 1, and the log tail keeps going.
        assert_eq!(pnorm(-40.0, 0.0, 1.0, true, false), 0.0);
        assert_eq!(pnorm(40.0, 0.0, 1.0, true, false), 1.0);
        assert!(pnorm(-40.0, 0.0, 1.0, true, true) < -800.0);
    }

    #[test]
    fn qnorm_nan_and_boundary_corners() {
        assert!(qnorm(f64::NAN, 0.0, 1.0, true, false).is_nan());
        assert!(qnorm(0.5, 0.0, -1.0, true, false).is_nan());
        assert!(qnorm(1.5, 0.0, 1.0, true, false).is_nan());
        assert!(qnorm(0.5, 0.0, 1.0, true, true).is_nan());
        assert_eq!(qnorm(0.0, 0.0, 1.0, true, false), f64::NEG_INFINITY);
        assert_eq!(qnorm(1.0, 0.0, 1.0, true, false), f64::INFINITY);
        assert_eq!(qnorm(0.0, 0.0, 1.0, false, false), f64::INFINITY);
        assert_eq!(
            qnorm(f64::NEG_INFINITY, 0.0, 1.0, true, true),
            f64::NEG_INFINITY
        );
        assert_eq!(qnorm(0.5, 3.0, 0.0, true, false), 3.0);
        assert_eq!(qnorm(0.5, 0.0, 1.0, true, false), 0.0);
        assert_eq!(qnorm(0.5, 2.0, 3.0, true, false), 2.0);
        // The r > 27 asymptotic tail is only reachable with log_p; round-trip it.
        for lp in [-800.0, -1e4, -1e6, -1e12, -1e18] {
            let q = qnorm(lp, 0.0, 1.0, true, true);
            assert!(q < -39.0, "lp = {lp:e}: q = {q}");
            let back = pnorm(q, 0.0, 1.0, true, true);
            if back.is_finite() {
                assert!(
                    (back - lp).abs() <= 1e-10 * lp.abs(),
                    "lp = {lp:e}: back = {back:e}"
                );
            }
        }
    }
}
