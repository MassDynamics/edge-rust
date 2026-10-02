//! M1: `brent_fmin` against R's `optimize` (optimize.c `fcn1`); the bound and pf cases lock
//! behaviour that already matched.
// The literals are R's bits, written out at 18 digits on purpose (review r2, R2-7).
#![allow(clippy::excessive_precision)]
use rnum::optim::{brent_fmin, optimize_default_tol};

// Identical bits in R: optimize(function(x) x, c(0.5, 0.9998)) and its negation.
#[test]
fn brent_at_the_fitfdist_bounds_matches_r_bits() {
    let tol = optimize_default_tol();
    assert_eq!(brent_fmin(0.5, 0.9998, |x| x, tol), 0.500045802965765618);
    assert_eq!(brent_fmin(0.5, 0.9998, |x| -x, tol), 0.999754204479256892);
}

// M1: fcn1 maps NA, NaN and +Inf to DBL_MAX and -Inf to -DBL_MAX, so a flat non-finite
// objective walks to the upper bound as in R.
#[test]
fn brent_treats_a_non_finite_objective_like_r_optimize() {
    let tol = optimize_default_tol();
    assert_eq!(
        brent_fmin(0.5, 0.9998, |_| f64::NAN, tol),
        0.999754204479256892
    );
    assert_eq!(
        brent_fmin(0.5, 0.9998, |_| f64::INFINITY, tol),
        0.999754204479256892
    );
    assert_eq!(
        brent_fmin(0.5, 0.9998, |_| f64::NEG_INFINITY, tol),
        0.999754204479256892
    );
}

// M1: a NaN on part of the interval is treated as DBL_MAX there, so the minimum is found on the
// finite part, bit for bit (R: optimize(function(x) if (x > 0.7) NaN else (x - 0.6)^2, c(0.5, 0.9998))).
#[test]
fn brent_avoids_a_nan_region_like_r_optimize() {
    let tol = optimize_default_tol();
    let x = brent_fmin(
        0.5,
        0.9998,
        |x| {
            if x > 0.7 {
                f64::NAN
            } else {
                (x - 0.6) * (x - 0.6)
            }
        },
        tol,
    );
    assert_eq!(x, 0.599999999999999978);
}

// R2-5: fcn1 keeps the sign of -Inf (-DBL_MAX), so a -Inf region is a minimum, bit for bit
// (R 4.5.0: optimize(function(x) if (x > 0.7) -Inf else (x - 0.6)^2, c(0.5, 0.9998))).
#[test]
fn brent_treats_minus_inf_as_minus_dbl_max_like_r_optimize() {
    let tol = optimize_default_tol();
    let x = brent_fmin(
        0.5,
        0.9998,
        |x| {
            if x > 0.7 {
                f64::NEG_INFINITY
            } else {
                (x - 0.6) * (x - 0.6)
            }
        },
        tol,
    );
    assert_eq!(x, 0.80896336796935731);
}

// glmQLFTest: a negative F gives p = 1.
#[test]
fn pf_upper_of_negative_f_is_one() {
    assert_eq!(rnum::nmath::pf(-0.5, 1.0, 10.0, false, false), 1.0);
}
