//! The arithmetic helpers R's nmath sources reach through `nmath.h`: `fmax2`/`fmin2`
//! (`src/nmath/fmax2.c`, `fmin2.c`) and `R_pow`/`R_pow_di` (`src/main/arithmetic.c`).
//!
//! `fmax2`/`fmin2` differ from `f64::max`/`f64::min` exactly where it matters to a port: a NaN
//! on either side comes back as NaN (`x + y`), where Rust's would return the other operand.

/// `fmax2(x, y)`: the larger of the two, NaN if either is NaN.
pub(crate) fn fmax2(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    if x < y {
        y
    } else {
        x
    }
}

/// `fmin2(x, y)`: the smaller of the two, NaN if either is NaN.
pub(crate) fn fmin2(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    if x < y {
        x
    } else {
        y
    }
}

/// `R_pow(x, y)`: `pow` with R's handling of the non-finite corners.
pub(crate) fn r_pow(x: f64, y: f64) -> f64 {
    if x == 1.0 || y == 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        if y > 0.0 {
            return 0.0;
        } else if y < 0.0 {
            return f64::INFINITY;
        } else {
            return y; // NA or NaN
        }
    }
    if x.is_finite() && y.is_finite() {
        if y == 2.0 {
            return x * x;
        }
        return x.powf(y);
    }
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    if !x.is_finite() {
        if x > 0.0 {
            // Inf ^ y
            return if y < 0.0 { 0.0 } else { f64::INFINITY };
        } else {
            // (-Inf) ^ y
            if y.is_finite() && y == y.floor() {
                // (-Inf) ^ n
                return if y < 0.0 {
                    0.0
                } else if y % 2.0 != 0.0 {
                    x
                } else {
                    -x
                };
            }
        }
    }
    if !y.is_finite() && x >= 0.0 {
        if y > 0.0 {
            // y == +Inf
            return if x >= 1.0 { f64::INFINITY } else { 0.0 };
        } else {
            // y == -Inf
            return if x < 1.0 { f64::INFINITY } else { 0.0 };
        }
    }
    f64::NAN
}

/// `R_pow_di(x, n)`: `x` to an integer power by repeated squaring, as R computes it (the
/// rounding of the result differs from `powf`, and the ports that use it are matched to R).
pub(crate) fn r_pow_di(x: f64, n: i32) -> f64 {
    let mut x = x;
    let mut n = n;
    let mut xn = 1.0;
    if x.is_nan() {
        return x;
    }
    if n != 0 {
        if !x.is_finite() {
            return r_pow(x, n as f64);
        }
        let is_neg = n < 0;
        if is_neg {
            n = -n;
        }
        loop {
            if n & 1 != 0 {
                xn *= x;
            }
            n >>= 1;
            if n != 0 {
                x *= x;
            } else {
                break;
            }
        }
        if is_neg {
            xn = 1.0 / xn;
        }
    }
    xn
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmax2_and_fmin2_propagate_nan_and_pick_the_right_side() {
        assert!(fmax2(1.0, f64::NAN).is_nan());
        assert!(fmin2(f64::NAN, 1.0).is_nan());
        assert_eq!(fmax2(1.0, 2.0), 2.0);
        assert_eq!(fmin2(1.0, 2.0), 1.0);
    }

    #[test]
    fn pow_di_matches_repeated_multiplication_and_handles_signs() {
        assert_eq!(r_pow_di(2.0, 10), 1024.0);
        assert_eq!(r_pow_di(2.0, -2), 0.25);
        assert_eq!(r_pow_di(-3.0, 3), -27.0);
        assert_eq!(r_pow_di(5.0, 0), 1.0);
        assert_eq!(r_pow_di(f64::INFINITY, 2), f64::INFINITY);
        assert_eq!(r_pow_di(f64::NEG_INFINITY, 3), f64::NEG_INFINITY);
        assert!(r_pow_di(f64::NAN, 2).is_nan());
    }

    #[test]
    fn pow_corners_follow_r() {
        assert_eq!(r_pow(0.0, -1.0), f64::INFINITY);
        assert_eq!(r_pow(f64::NAN, 0.0), 1.0);
        assert_eq!(r_pow(1.0, f64::NAN), 1.0);
        assert_eq!(r_pow(f64::INFINITY, -1.0), 0.0);
        assert_eq!(r_pow(0.5, f64::INFINITY), 0.0);
        assert_eq!(r_pow(2.0, f64::NEG_INFINITY), 0.0);
        assert_eq!(r_pow(3.0, 2.0), 9.0);
    }
}
