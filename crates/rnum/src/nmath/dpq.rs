//! Port of R's `src/nmath/dpq.h` — the `p`/`q` half of its density/probability/quantile
//! helpers — plus `nmath.h`'s `R_forceint` (which `dpq.h` used to hold).
//!
//! The C macros read `lower_tail` and `log_p` out of the enclosing function's scope. Here
//! they are ordinary functions and the flags are passed explicitly, always last, so a call
//! site still lines up with the C one argument for argument.
//!
//! The three macros that `return` from their caller — `R_Q_P01_check`,
//! `R_Q_P01_boundaries`, `R_P_bounds_01` — stay macros, because the early return is the
//! whole point of them.
//!
//! Only the helpers with a caller in this crate are ported; an unused one is a finding.
//!
//! `dpq.h`'s density-only section (`give_log`, `R_D_fexp`, `R_D_rtxp`, `R_D_negInonint`,
//! `R_D_nonint_check`) is not ported: no `d*` function is in scope for this crate.
//!
//! R returns `NaN` from `ML_WARN_return_NAN` after raising an R-level warning. There is no
//! warning channel here, so the ports return `f64::NAN` and nothing else; matching R's
//! return value is the spec (see the crate `README.md`).

use super::consts::M_LN2;

/// `R_D__0` — 0 on the `log_p` scale. (One underscore here: `r_d__0` is not snake case.)
pub(crate) fn r_d_0(log_p: bool) -> f64 {
    if log_p {
        f64::NEG_INFINITY
    } else {
        0.0
    }
}

/// `R_D__1` — 1 on the `log_p` scale.
pub(crate) fn r_d_1(log_p: bool) -> f64 {
    if log_p {
        0.0
    } else {
        1.0
    }
}

/// `R_DT_0` — 0 on the `lower_tail`/`log_p` scale.
pub(crate) fn r_dt_0(lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail {
        r_d_0(log_p)
    } else {
        r_d_1(log_p)
    }
}

/// `R_DT_1` — 1 on the `lower_tail`/`log_p` scale.
pub(crate) fn r_dt_1(lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail {
        r_d_1(log_p)
    } else {
        r_d_0(log_p)
    }
}

/// `R_D_half` — 1/2, lower or upper tail alike.
pub(crate) fn r_d_half(log_p: bool) -> f64 {
    if log_p {
        -M_LN2
    } else {
        0.5
    }
}

/// `R_D_Lval` — `p`. The `0.5 - p + 0.5` is R's, to perhaps gain a bit of accuracy.
pub(crate) fn r_d_lval(p: f64, lower_tail: bool) -> f64 {
    if lower_tail {
        p
    } else {
        0.5 - p + 0.5
    }
}

/// `R_D_Cval` — `1 - p`.
pub(crate) fn r_d_cval(p: f64, lower_tail: bool) -> f64 {
    if lower_tail {
        0.5 - p + 0.5
    } else {
        p
    }
}

/// `R_D_qIv` — `p` in `qF(p, ..)`.
pub(crate) fn r_d_qiv(p: f64, log_p: bool) -> f64 {
    if log_p {
        p.exp()
    } else {
        p
    }
}

/// `R_D_exp` — `exp(x)`.
pub(crate) fn r_d_exp(x: f64, log_p: bool) -> f64 {
    if log_p {
        x
    } else {
        x.exp()
    }
}

/// `R_D_log` — `log(p)`.
pub(crate) fn r_d_log(p: f64, log_p: bool) -> f64 {
    if log_p {
        p
    } else {
        p.ln()
    }
}

/// `R_Log1_Exp` — `log(1 - exp(x))`, in a more stable form than `log1p(-exp(x))`.
pub(crate) fn r_log1_exp(x: f64) -> f64 {
    if x > -M_LN2 {
        (-x.exp_m1()).ln()
    } else {
        (-x.exp()).ln_1p()
    }
}

/// `R_D_LExp` — `log(1 - exp(x))`, more stable still than `log1p(-R_D_qIv(x))`.
pub(crate) fn r_d_lexp(x: f64, log_p: bool) -> f64 {
    if log_p {
        r_log1_exp(x)
    } else {
        (-x).ln_1p()
    }
}

/// `R_DT_qIv` — `p` in `qF`.
pub(crate) fn r_dt_qiv(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if log_p {
        if lower_tail {
            p.exp()
        } else {
            -p.exp_m1()
        }
    } else {
        r_d_lval(p, lower_tail)
    }
}

/// `R_DT_CIv` — `1 - p` in `qF`.
pub(crate) fn r_dt_civ(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if log_p {
        if lower_tail {
            -p.exp_m1()
        } else {
            p.exp()
        }
    } else {
        r_d_cval(p, lower_tail)
    }
}

/// `R_DT_log` — `log(p)` in `qF`.
pub(crate) fn r_dt_log(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail {
        r_d_log(p, log_p)
    } else {
        r_d_lexp(p, log_p)
    }
}

/// `R_DT_Clog` — `log(1 - p)` in `qF`.
pub(crate) fn r_dt_clog(p: f64, lower_tail: bool, log_p: bool) -> f64 {
    if lower_tail {
        r_d_lexp(p, log_p)
    } else {
        r_d_log(p, log_p)
    }
}

/// `R_forceint` — `nearbyint`, i.e. round half to even under the default rounding mode.
/// `nmath.h` holds this one; the comment there records that it came out of `dpq.h`.
pub(crate) fn r_forceint(x: f64) -> f64 {
    x.round_ties_even()
}

/// `R_Q_P01_check(p)` — reject a `p` outside the unit interval (or `(-Inf, 0]` when
/// `log_p`) by returning `NaN` from the calling function.
macro_rules! r_q_p01_check {
    ($p:expr, $log_p:expr) => {
        // Not `(0.0..=1.0).contains(&p)`, which clippy suggests: that is true-for-neither
        // on a NaN `p`, so negating it would reject NaN here, where C falls through.
        #[allow(clippy::manual_range_contains)]
        if ($log_p && $p > 0.0) || (!$log_p && ($p < 0.0 || $p > 1.0)) {
            return f64::NAN;
        }
    };
}

/// `R_Q_P01_boundaries(p, _LEFT_, _RIGHT_)` — check `p`, then take the boundaries of a
/// `q*()` function exactly, returning from the calling function.
macro_rules! r_q_p01_boundaries {
    ($p:expr, $left:expr, $right:expr, $lower_tail:expr, $log_p:expr) => {
        if $log_p {
            if $p > 0.0 {
                return f64::NAN;
            }
            if $p == 0.0 {
                // upper bound
                return if $lower_tail { $right } else { $left };
            }
            if $p == f64::NEG_INFINITY {
                return if $lower_tail { $left } else { $right };
            }
        } else {
            // See the NaN note on `r_q_p01_check`.
            #[allow(clippy::manual_range_contains)]
            if $p < 0.0 || $p > 1.0 {
                return f64::NAN;
            }
            if $p == 0.0 {
                return if $lower_tail { $left } else { $right };
            }
            if $p == 1.0 {
                return if $lower_tail { $right } else { $left };
            }
        }
    };
}

/// `R_P_bounds_01(x, x_min, x_max)` — return 0 or 1 from the calling function for an `x`
/// at or beyond the support.
macro_rules! r_p_bounds_01 {
    ($x:expr, $x_min:expr, $x_max:expr, $lower_tail:expr, $log_p:expr) => {
        if $x <= $x_min {
            return $crate::nmath::dpq::r_dt_0($lower_tail, $log_p);
        }
        if $x >= $x_max {
            return $crate::nmath::dpq::r_dt_1($lower_tail, $log_p);
        }
    };
}

pub(crate) use {r_p_bounds_01, r_q_p01_boundaries, r_q_p01_check};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_and_one_follow_the_tail_and_log_flags() {
        assert_eq!(r_dt_0(true, false), 0.0);
        assert_eq!(r_dt_0(false, false), 1.0);
        assert_eq!(r_dt_0(true, true), f64::NEG_INFINITY);
        assert_eq!(r_dt_1(false, true), f64::NEG_INFINITY);
        assert_eq!(r_dt_1(true, true), 0.0);
    }

    /// `R_Log1_Exp` exists to keep `log(1 - exp(x))` accurate on both sides of `-log(2)`,
    /// so each reference below is analytic rather than one of the two naive forms.
    #[test]
    fn log1_exp_is_accurate_either_side_of_its_branch() {
        // log(1 - exp(log q)) = log(1 - q); 0.75 takes the upper branch, 0.25 the lower,
        // and 0.5 sits on the split.
        for q in [0.25f64, 0.5, 0.75] {
            let (got, want) = (r_log1_exp(q.ln()), (1.0 - q).ln());
            assert!(
                (got - want).abs() <= 1e-15 * want.abs(),
                "q = {q}: {got:e} vs {want:e}"
            );
        }
        // Near 0: log(1 - exp(x)) = log(-x) + x/2 + O(x^2). This is where log1p(-exp(x))
        // has already lost the answer to cancellation.
        let x = -1e-12;
        let (got, want) = (r_log1_exp(x), (-x).ln() + x / 2.0);
        assert!(
            (got - want).abs() <= 1e-15 * want.abs(),
            "x = {x:e}: {got:e} vs {want:e}"
        );
        // Far out: log(1 - exp(x)) = -exp(x) + O(exp(2x)).
        let x = -700.0f64;
        let (got, want) = (r_log1_exp(x), -x.exp());
        assert!(
            (got - want).abs() <= 1e-15 * want.abs(),
            "x = {x:e}: {got:e} vs {want:e}"
        );
    }

    #[test]
    fn q_p01_boundaries_returns_the_ends_exactly() {
        fn q(p: f64, lower_tail: bool, log_p: bool) -> f64 {
            r_q_p01_boundaries!(p, f64::NEG_INFINITY, f64::INFINITY, lower_tail, log_p);
            0.0
        }
        assert_eq!(q(0.0, true, false), f64::NEG_INFINITY);
        assert_eq!(q(1.0, true, false), f64::INFINITY);
        assert_eq!(q(0.0, false, false), f64::INFINITY);
        assert_eq!(q(0.0, true, true), f64::INFINITY);
        assert_eq!(q(f64::NEG_INFINITY, true, true), f64::NEG_INFINITY);
        assert!(q(1.5, true, false).is_nan());
        assert!(q(0.5, true, true).is_nan());
        assert_eq!(q(0.5, true, false), 0.0);
        // A NaN `p` fails every comparison, so it falls through to the caller's own
        // handling rather than being caught here — as it does in C.
        assert_eq!(q(f64::NAN, true, false), 0.0);
        assert_eq!(q(f64::NAN, true, true), 0.0);
    }

    #[test]
    fn forceint_rounds_half_to_even() {
        assert_eq!(r_forceint(2.5), 2.0);
        assert_eq!(r_forceint(3.5), 4.0);
        assert_eq!(r_forceint(-0.5), -0.0);
        assert_eq!(r_forceint(2.0), 2.0);
    }
}
