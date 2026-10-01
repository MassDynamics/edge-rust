//! Port of R's `src/nmath/dnbinom.c` (`dnbinom_mu`), R 4.5.0. DESeq2's C++ `fitBeta` and its
//! R `logLike` both evaluate the negative binomial density in the `(size, mu)`
//! parametrisation through this function.

use super::dpq::{r_d_0, r_d_1, r_d_exp, r_forceint};
use super::f::dbinom_raw;
use super::gamma::{dpois_raw, lgamma1p};

/// `dnbinom_mu` in `src/nmath/dnbinom.c`: the negative binomial density with `size` and mean
/// `mu`, or its log when `give_log`. Non-integer `x` (beyond R's `1e-9` relative slack) gives
/// density 0, as R does (R also warns). `mu < 0` or `size < 0` gives NaN.
pub fn dnbinom_mu(x: f64, size: f64, mu: f64, give_log: bool) -> f64 {
    if x.is_nan() || size.is_nan() || mu.is_nan() {
        return x + size + mu;
    }
    if mu < 0.0 || size < 0.0 {
        return f64::NAN;
    }
    if (x - r_forceint(x)).abs() > 1e-9 * x.abs().max(1.0) {
        return r_d_0(give_log);
    }
    if x < 0.0 || !x.is_finite() {
        return r_d_0(give_log);
    }
    if x == 0.0 && size == 0.0 {
        return r_d_1(give_log);
    }
    let x = r_forceint(x);
    if !size.is_finite() {
        return dpois_raw(x, mu, give_log);
    }
    if x == 0.0 {
        return r_d_exp(
            size * (if size < mu {
                (size / (size + mu)).ln()
            } else {
                (-mu / (size + mu)).ln_1p()
            }),
            give_log,
        );
    }
    if x < 1e-10 * size {
        let p = if size < mu {
            (size / (1.0 + size / mu)).ln()
        } else {
            (mu / (1.0 + mu / size)).ln()
        };
        r_d_exp(
            x * p - mu - lgamma1p(x) + (x * (x - 1.0) / (2.0 * size)).ln_1p(),
            give_log,
        )
    } else {
        let p = if give_log {
            if x < size {
                (-x / (size + x)).ln_1p()
            } else {
                (size / (size + x)).ln()
            }
        } else {
            size / (size + x)
        };
        let ans = dbinom_raw(
            size,
            x + size,
            size / (size + mu),
            mu / (size + mu),
            give_log,
        );
        if give_log {
            p + ans
        } else {
            p * ans
        }
    }
}

#[cfg(test)]
mod tests {
    use super::dnbinom_mu;

    #[test]
    fn matches_closed_forms() {
        // size = 1 is geometric: P(x) = (1/(1+mu)) (mu/(1+mu))^x.
        let mu: f64 = 2.5;
        for x in 0..6 {
            let want = (1.0 / (1.0 + mu)) * (mu / (1.0 + mu)).powi(x);
            let got = dnbinom_mu(x as f64, 1.0, mu, false);
            assert!((got - want).abs() <= 1e-15 * want, "{x}: {got} vs {want}");
            let lg = dnbinom_mu(x as f64, 1.0, mu, true);
            assert!((lg - want.ln()).abs() <= 1e-14 * want.ln().abs().max(1.0));
        }
        assert_eq!(dnbinom_mu(1.5, 1.0, 1.0, false), 0.0);
        assert!(dnbinom_mu(1.0, -1.0, 1.0, false).is_nan());
    }
}
