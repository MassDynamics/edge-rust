//! Port of R's `src/nmath/pbeta.c` (`pbeta_raw`, `pbeta`) over `toms708::bratio`.
//!
//! Returns the distribution function of the beta distribution, i.e. the incomplete beta
//! ratio `I_x(a, b)`. As from R 2.3.0 a wrapper for TOMS708; as from R 2.6.0 `log_p` is
//! partially improved over `log(p..)` inside `bratio`.

use super::consts::M_LN2;
use super::dpq::{r_dt_0, r_dt_1};
use super::toms708::bratio;

/// `pbeta_raw` from `pbeta.c`: `pbeta` without the NaN and negative-parameter checks, so it
/// can be called from `qbeta`, with the limit cases `a == 0`, `b == 0`, `a == Inf`,
/// `b == Inf` handled as point masses.
pub(crate) fn pbeta_raw(x: f64, a: f64, b: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x >= 1.0 {
        // may happen when called from qbeta()
        return r_dt_1(lower_tail, log_p);
    }
    // treat limit cases correctly here:
    if a == 0.0 || b == 0.0 || !a.is_finite() || !b.is_finite() {
        // NB:  0 <= x < 1 :
        if a == 0.0 && b == 0.0 {
            // point mass 1/2 at each of {0,1} :
            return if log_p { -M_LN2 } else { 0.5 };
        }
        if a == 0.0 || a / b == 0.0 {
            // point mass 1 at 0 ==> P(X <= x) = 1, all x >= 0
            return r_dt_1(lower_tail, log_p);
        }
        if b == 0.0 || b / a == 0.0 {
            // point mass 1 at 1 ==> P(X <= x) = 0, all x < 1
            return r_dt_0(lower_tail, log_p);
        }
        // else, remaining case:  a = b = Inf : point mass 1 at 1/2
        return if x < 0.5 {
            r_dt_0(lower_tail, log_p)
        } else {
            r_dt_1(lower_tail, log_p)
        };
    }
    if x <= 0.0 {
        return r_dt_0(lower_tail, log_p);
    }

    // Now:  0 < a < Inf;  0 < b < Inf  and  0 < x < 1
    let x1 = 0.5 - x + 0.5;
    let (mut w, mut wc) = (0.0, 0.0);
    let mut ierr = 0;
    bratio(a, b, x, x1, &mut w, &mut wc, &mut ierr, log_p); // -> ./toms708.c
                                                            // ierr in {10,14} <==> bgrat() error code ierr-10 in 1:4; the C warns on the others.
    if lower_tail {
        w
    } else {
        wc
    }
}

/// `pbeta` from `pbeta.c`: the beta distribution function `P[X <= x]` for `X ~ Beta(a, b)`
/// (`lower_tail`), or `P[X > x]`, optionally as a log probability. `a == 0` and `b == 0`
/// are allowed and treated as one- or two-point masses; a negative `a` or `b` is `NaN`.
pub fn pbeta(x: f64, a: f64, b: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x.is_nan() || a.is_nan() || b.is_nan() {
        return x + a + b;
    }

    if a < 0.0 || b < 0.0 {
        return f64::NAN;
    }
    // allowing a==0 and b==0  <==> treat as one- or two-point mass

    pbeta_raw(x, a, b, lower_tail, log_p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_rel(got: f64, want: f64, rel: f64, what: &str) {
        assert!(
            (got - want).abs() <= rel * want.abs(),
            "{what}: got {got:e}, want {want:e}"
        );
    }

    /// `log_p` agrees with the log of the plain probability where both are representable.
    /// The points cover bpser, bup/bgrat, bfrac and basym branches of `bratio`.
    #[test]
    fn log_p_matches_log_of_probability_at_moderate_points() {
        let points = [
            (0.3, 2.0, 3.0),
            (0.7, 0.5, 0.5),
            (0.5, 10.0, 20.0),
            (0.85, 30.0, 5.0),
            (0.002, 0.05, 40.0),
            (0.45, 60.0, 70.0),
            (0.3, 200.0, 500.0),
            (0.6, 1500.0, 1000.0),
        ];
        for (x, a, b) in points {
            for lower in [true, false] {
                let plain = pbeta(x, a, b, lower, false);
                let logged = pbeta(x, a, b, lower, true);
                // Keep both tails moderate so ln(plain) is itself accurate to ~1e-14.
                assert!(plain > 0.02 && plain < 0.98, "({x}, {a}, {b}): p = {plain}");
                assert_rel(
                    logged,
                    plain.ln(),
                    1e-13,
                    &format!("pbeta({x}, {a}, {b}, lower={lower}, log)"),
                );
            }
        }
    }

    /// Deep in the lower tail the plain probability underflows to 0, so its log is `-Inf`;
    /// the `log_p` path stays finite and negative.
    #[test]
    fn log_p_is_finite_where_the_probability_underflows() {
        let (x, a, b) = (1e-10, 50.0, 50.0);
        assert_eq!(pbeta(x, a, b, true, false), 0.0);
        let logged = pbeta(x, a, b, true, true);
        assert!(logged.is_finite() && logged < -1000.0, "log p = {logged}");
        // And the upper tail of the same point is essentially 1, i.e. log 1 = 0 from below.
        let upper = pbeta(x, a, b, false, true);
        assert!(upper <= 0.0 && upper > -1e-300, "log(1 - p) = {upper}");
    }

    /// The limit cases `pbeta.c` handles before calling `bratio`.
    #[test]
    fn corners_follow_pbeta_c() {
        let (inf, nan) = (f64::INFINITY, f64::NAN);

        // x >= 1 and x <= 0
        assert_eq!(pbeta(1.0, 2.0, 3.0, true, false), 1.0);
        assert_eq!(pbeta(1.0, 2.0, 3.0, false, false), 0.0);
        assert_eq!(pbeta(1.0, 2.0, 3.0, true, true), 0.0);
        assert_eq!(pbeta(1.0, 2.0, 3.0, false, true), f64::NEG_INFINITY);
        assert_eq!(pbeta(0.0, 2.0, 3.0, true, false), 0.0);
        assert_eq!(pbeta(0.0, 2.0, 3.0, false, false), 1.0);
        assert_eq!(pbeta(-0.5, 2.0, 3.0, true, true), f64::NEG_INFINITY);
        assert_eq!(pbeta(1.5, 2.0, 3.0, true, false), 1.0);

        // a = b = 0: point mass 1/2 at each of {0, 1}
        assert_eq!(pbeta(0.0, 0.0, 0.0, true, false), 0.5);
        assert_eq!(pbeta(0.3, 0.0, 0.0, false, false), 0.5);
        assert_eq!(pbeta(0.3, 0.0, 0.0, true, true), -M_LN2);
        assert_eq!(pbeta(1.0, 0.0, 0.0, true, false), 1.0);

        // a = 0 (or a/b = 0): point mass at 0; b = 0 (or b/a = 0): point mass at 1
        assert_eq!(pbeta(0.0, 0.0, 1.0, true, false), 1.0);
        assert_eq!(pbeta(0.3, 0.0, 1.0, false, false), 0.0);
        assert_eq!(pbeta(0.3, 2.0, inf, true, false), 1.0);
        assert_eq!(pbeta(0.3, 1.0, 0.0, true, false), 0.0);
        assert_eq!(pbeta(0.3, 1.0, 0.0, false, true), 0.0);
        assert_eq!(pbeta(0.3, inf, 2.0, true, false), 0.0);
        assert_eq!(pbeta(0.999, inf, 2.0, true, false), 0.0);

        // a = b = Inf: point mass at 1/2
        assert_eq!(pbeta(0.3, inf, inf, true, false), 0.0);
        assert_eq!(pbeta(0.5, inf, inf, true, false), 1.0);
        assert_eq!(pbeta(0.7, inf, inf, false, false), 0.0);

        // NaN in, NaN out; negative parameters are NaN
        assert!(pbeta(nan, 1.0, 1.0, true, false).is_nan());
        assert!(pbeta(0.5, nan, 1.0, true, false).is_nan());
        assert!(pbeta(0.5, 1.0, nan, true, false).is_nan());
        assert!(pbeta(0.5, -1.0, 1.0, true, false).is_nan());
        assert!(pbeta(0.5, 1.0, -1.0, true, false).is_nan());
    }
}
