//! Port of R's `src/nmath/pf.c` (`pf`), `qf.c` (`qf`) and `df.c` (`df`), all from R 4.5.0,
//! plus the `dbinom_raw` (and its `pow1p`) of `dbinom.c` that `df` is written in terms of.
//! Proved against `scalar/pf.csv` and `scalar/qf.csv` in `tests/scalar_f.rs`; `df` has no
//! golden and is pinned by the unit tests at the bottom of this file against its closed
//! forms (`m = 2` is `(1 + 2x/n)^(-(n+2)/2)`, `m = n = 1` is `1/(pi sqrt(x) (1 + x))`, and
//! the general beta-function form for the rest).

use super::consts::{M_LN2, M_LN_2PI};
use super::dpq::{r_d_0, r_d_1, r_d_exp, r_dt_0, r_dt_1, r_p_bounds_01, r_q_p01_boundaries};
use super::gamma::{bd0, dgamma, stirlerr};
use super::pbeta::pbeta;
use super::pgamma::{pchisq, qchisq};
use super::qbeta::qbeta;

/// `pf` in `src/nmath/pf.c`: the distribution function of the F distribution with `df1`
/// and `df2` degrees of freedom, `P[F <= x]`, or the upper tail, or either on the log scale.
pub fn pf(x: f64, df1: f64, df2: f64, lower_tail: bool, log_p: bool) -> f64 {
    if x.is_nan() || df1.is_nan() || df2.is_nan() {
        return x + df2 + df1;
    }
    if df1 <= 0.0 || df2 <= 0.0 {
        return f64::NAN;
    }

    r_p_bounds_01!(x, 0.0, f64::INFINITY, lower_tail, log_p);

    // move to pchisq for very large values - was 'df1 > 4e5' in 2.0.x,
    // now only needed for df1 = Inf or df2 = Inf {since pbeta(0,*)=0} :
    if df2 == f64::INFINITY {
        if df1 == f64::INFINITY {
            if x < 1.0 {
                return r_dt_0(lower_tail, log_p);
            }
            if x == 1.0 {
                return if log_p { -M_LN2 } else { 0.5 };
            }
            if x > 1.0 {
                return r_dt_1(lower_tail, log_p);
            }
        }

        return pchisq(x * df1, df1, lower_tail, log_p);
    }

    if df1 == f64::INFINITY {
        // was "fudge" 'df1 > 4e5' in 2.0.x
        return pchisq(df2 / x, df2, !lower_tail, log_p);
    }

    // Avoid squeezing pbeta's first parameter against 1 :
    let x = if df1 * x > df2 {
        pbeta(
            df2 / (df2 + df1 * x),
            df2 / 2.0,
            df1 / 2.0,
            !lower_tail,
            log_p,
        )
    } else {
        pbeta(
            df1 * x / (df2 + df1 * x),
            df1 / 2.0,
            df2 / 2.0,
            lower_tail,
            log_p,
        )
    };

    // ML_VALID(x) ? x : ML_NAN
    if x.is_nan() {
        f64::NAN
    } else {
        x
    }
}

/// `qf` in `src/nmath/qf.c`: the quantile function of the F distribution with `df1` and
/// `df2` degrees of freedom.
pub fn qf(p: f64, df1: f64, df2: f64, lower_tail: bool, log_p: bool) -> f64 {
    if p.is_nan() || df1.is_nan() || df2.is_nan() {
        return p + df1 + df2;
    }
    if df1 <= 0.0 || df2 <= 0.0 {
        return f64::NAN;
    }

    r_q_p01_boundaries!(p, 0.0, f64::INFINITY, lower_tail, log_p);

    // fudge the extreme DF cases -- qbeta doesn't do this well.
    // But we still need to fudge the infinite ones.
    if df1 <= df2 && df2 > 4e5 {
        if !df1.is_finite() {
            // df1 == df2 == Inf :
            return 1.0;
        }
        // else value for df2 == Inf :
        return qchisq(p, df1, lower_tail, log_p) / df1;
    } else if df1 > 4e5 {
        // and so  df2 < df1 -- return value for df1 == Inf
        return df2 / qchisq(p, df2, !lower_tail, log_p);
    }

    // FIXME: (1/qb - 1) = (1 - qb)/qb; if we know qb ~= 1, should use other tail
    let p = (1.0 / qbeta(p, df2 / 2.0, df1 / 2.0, !lower_tail, log_p) - 1.0) * (df2 / df1);

    // ML_VALID(p) ? p : ML_NAN
    if p.is_nan() {
        f64::NAN
    } else {
        p
    }
}

/// `df` in `src/nmath/df.c`: the density function of the F distribution with `m` and `n`
/// degrees of freedom.
///
/// To evaluate it, write it as a Binomial probability with `p = x*m/(n+x*m)`.
/// For `m >= 2`, we use the simplest conversion.
/// For `m < 2`, `(m-2)/2 < 0` so the conversion will not work, and we must use a second
/// conversion. Note the division by `p`; this seems unavoidable for `m < 2`, since the F
/// density has a singularity as `x` (or `p`) `-> 0`.
pub fn df(x: f64, m: f64, n: f64, give_log: bool) -> f64 {
    if x.is_nan() || m.is_nan() || n.is_nan() {
        return x + m + n;
    }
    if m <= 0.0 || n <= 0.0 {
        return f64::NAN;
    }
    if x < 0.0 {
        return r_d_0(give_log);
    }
    if x == 0.0 {
        return if m > 2.0 {
            r_d_0(give_log)
        } else if m == 2.0 {
            r_d_1(give_log)
        } else {
            f64::INFINITY
        };
    }
    if !m.is_finite() && !n.is_finite() {
        // both +Inf
        if x == 1.0 {
            return f64::INFINITY;
        } else {
            return r_d_0(give_log);
        }
    }
    if !n.is_finite() {
        // must be +Inf by now
        return dgamma(x, m / 2.0, 2.0 / m, give_log);
    }
    if m > 1e14 {
        // includes +Inf: code below is inaccurate there
        let dens = dgamma(1.0 / x, n / 2.0, 2.0 / n, give_log);
        return if give_log {
            dens - 2.0 * x.ln()
        } else {
            dens / (x * x)
        };
    }

    let mut f = 1.0 / (n + x * m);
    let q = n * f;
    let p = x * m * f;

    let dens;
    if m >= 2.0 {
        f = m * q / 2.0;
        dens = dbinom_raw((m - 2.0) / 2.0, (m + n - 2.0) / 2.0, p, q, give_log);
    } else {
        f = m * m * q / (2.0 * p * (m + n));
        dens = dbinom_raw(m / 2.0, (m + n) / 2.0, p, q, give_log);
    }
    if give_log {
        f.ln() + dens
    } else {
        f * dens
    }
}

/// `pow1p` in `src/nmath/dbinom.c`: compute `(1+x)^y` accurately also for `|x| << 1`.
fn pow1p(x: f64, y: f64) -> f64 {
    if y.is_nan() {
        return if x == 0.0 { 1.0 } else { y }; // (0+1)^NaN := 1  by standards
    }
    if 0.0 <= y && y == y.trunc() && y <= 4.0 {
        match y as i32 {
            0 => return 1.0,
            1 => return x + 1.0,
            2 => return x * (x + 2.0) + 1.0,
            3 => return x * (x * (x + 3.0) + 3.0) + 1.0,
            4 => return x * (x * (x * (x + 4.0) + 6.0) + 4.0) + 1.0,
            _ => {}
        }
    }
    // naive algorithm in two cases: (1) when 1+x is exact (compiler should not over-optimize !),
    // and (2) when |x| > 1/2 and we have no better algorithm.
    if (x + 1.0) - 1.0 == x || x.abs() > 0.5 || x.is_nan() {
        (1.0 + x).powf(y)
    } else {
        // not perfect, e.g., for small |x|, non-huge y, use
        // binom expansion 1 + y*x + y(y-1)/2 x^2 + ..
        (y * x.ln_1p()).exp()
    }
}

/// `dbinom_raw` in `src/nmath/dbinom.c`: the binomial probability of `x` successes in `n`
/// trials with success probability `p` and failure probability `q`, without argument checks.
///
/// `p` and `q` are both passed because one may be represented more accurately than the
/// other (in particular, in `df`). `x` and `n` are not checked to be integers, nor
/// `0 <= p, q <= 1`; the caller does that where necessary.
pub(crate) fn dbinom_raw(x: f64, n: f64, p: f64, q: f64, give_log: bool) -> f64 {
    if p == 0.0 {
        return if x == 0.0 {
            r_d_1(give_log)
        } else {
            r_d_0(give_log)
        };
    }
    if q == 0.0 {
        return if x == n {
            r_d_1(give_log)
        } else {
            r_d_0(give_log)
        };
    }

    // NB: The smaller of p and q is the most accurate
    if x == 0.0 {
        if n == 0.0 {
            return r_d_1(give_log);
        }
        if p > q {
            return if give_log { n * q.ln() } else { q.powf(n) };
        } else {
            // 0 < p <= 1/2
            return if give_log {
                n * (-p).ln_1p()
            } else {
                pow1p(-p, n)
            };
        }
    }
    if x == n {
        // r = p^x = p^n  -- accurately
        if p > q {
            return if give_log {
                n * (-q).ln_1p()
            } else {
                pow1p(-q, n)
            };
        } else {
            return if give_log { n * p.ln() } else { p.powf(n) };
        }
    }
    if x < 0.0 || x > n {
        return r_d_0(give_log);
    }

    // n*p or n*q can underflow to zero if n and p or q are small.  This
    // used to occur in dbeta, and gives NaN as from R 2.3.0.
    let lc = stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0(x, n * p) - bd0(n - x, n * q);

    // f = (M_2PI*x*(n-x))/n; could overflow or underflow
    // Upto R 2.7.1:
    //  lf = log(M_2PI) + log(x) + log(n-x) - log(n);
    //  -- following is much better for  x << n :
    let lf = M_LN_2PI + x.ln() + (-x / n).ln_1p();

    r_d_exp(lc - 0.5 * lf, give_log)
}

#[cfg(test)]
mod tests {
    use super::super::consts::M_PI;
    use super::super::gamma::lbeta;
    use super::*;

    fn assert_rel(what: &str, got: f64, want: f64, tol: f64) {
        assert!(
            (got - want).abs() <= tol * want.abs(),
            "{what}: got {got:e}, want {want:e}, rel err {:e}",
            ((got - want) / want).abs()
        );
    }

    /// For a log-scale value: `log(w)` of a `w` near 1 is near 0, and a one-ulp change in
    /// `w` moves it by ~1e-16 absolute, which is a large *relative* error of a number that
    /// small. So a log density is held to `tol` relative to `max(1, |want|)`.
    fn assert_log(what: &str, got: f64, want: f64, tol: f64) {
        assert!(
            (got - want).abs() <= tol * want.abs().max(1.0),
            "{what}: got {got:e}, want {want:e}, abs err {:e}",
            (got - want).abs()
        );
    }

    /// The F density in its textbook form, for the general check below:
    /// `(m/n)^(m/2) x^(m/2 - 1) (1 + m x/n)^(-(m+n)/2) / B(m/2, n/2)`, done on the log scale.
    fn log_f_density(x: f64, m: f64, n: f64) -> f64 {
        (m / 2.0) * (m / n).ln() + (m / 2.0 - 1.0) * x.ln()
            - ((m + n) / 2.0) * (m * x / n).ln_1p()
            - lbeta(m / 2.0, n / 2.0)
    }

    const XS: [f64; 7] = [1e-3, 0.1, 0.5, 1.0, 2.5, 10.0, 200.0];

    /// m = 2: the density is exactly (1 + 2x/n)^(-(n+2)/2), and takes the `m >= 2` branch.
    #[test]
    fn df_m2_matches_its_closed_form() {
        for n in [0.5, 1.0, 3.0, 10.0, 100.0] {
            for &x in &XS {
                let want = (1.0 + 2.0 * x / n).powf(-(n + 2.0) / 2.0);
                assert_rel(
                    &format!("df({x}, 2, {n})"),
                    df(x, 2.0, n, false),
                    want,
                    1e-14,
                );
                assert_log(
                    &format!("df({x}, 2, {n}, log)"),
                    df(x, 2.0, n, true),
                    want.ln(),
                    1e-14,
                );
            }
        }
        // At x = 0 with m = 2 the density is exactly 1.
        assert_eq!(df(0.0, 2.0, 5.0, false), 1.0);
        assert_eq!(df(0.0, 2.0, 5.0, true), 0.0);
    }

    /// m = n = 1: the density is 1 / (pi sqrt(x) (1 + x)), and takes the `m < 2` branch.
    #[test]
    fn df_m1_n1_matches_its_closed_form() {
        for &x in &XS {
            let want = 1.0 / (M_PI * x.sqrt() * (1.0 + x));
            assert_rel(
                &format!("df({x}, 1, 1)"),
                df(x, 1.0, 1.0, false),
                want,
                1e-14,
            );
            assert_log(
                &format!("df({x}, 1, 1, log)"),
                df(x, 1.0, 1.0, true),
                want.ln(),
                1e-14,
            );
        }
        // The m < 2 singularity at 0.
        assert_eq!(df(0.0, 1.0, 1.0, false), f64::INFINITY);
    }

    /// General (m, n), both branches, against the beta-function form.
    #[test]
    fn df_matches_the_beta_function_form() {
        for (m, n) in [
            (1.0, 5.0),
            (1.5, 2.5),
            (3.0, 4.0),
            (4.0, 6.0),
            (7.0, 1.0),
            (20.0, 30.0),
        ] {
            for &x in &XS {
                let want_log = log_f_density(x, m, n);
                assert_rel(
                    &format!("df({x}, {m}, {n})"),
                    df(x, m, n, false),
                    want_log.exp(),
                    1e-14,
                );
                assert_log(
                    &format!("df({x}, {m}, {n}, log)"),
                    df(x, m, n, true),
                    want_log,
                    1e-14,
                );
            }
        }
    }

    #[test]
    fn df_edges_follow_the_c() {
        assert!(df(f64::NAN, 1.0, 1.0, false).is_nan());
        assert!(df(1.0, 0.0, 1.0, false).is_nan());
        assert!(df(1.0, 1.0, -1.0, false).is_nan());
        assert_eq!(df(-1.0, 3.0, 4.0, false), 0.0);
        assert_eq!(df(-1.0, 3.0, 4.0, true), f64::NEG_INFINITY);
        assert_eq!(df(0.0, 3.0, 4.0, false), 0.0);
        assert_eq!(df(1.0, f64::INFINITY, f64::INFINITY, false), f64::INFINITY);
        assert_eq!(df(2.0, f64::INFINITY, f64::INFINITY, false), 0.0);
        // n = Inf: m * F ~ chisq(m), i.e. F ~ gamma(m/2, scale 2/m).
        assert_rel(
            "df(1.5, 4, Inf)",
            df(1.5, 4.0, f64::INFINITY, false),
            dgamma(1.5, 2.0, 0.5, false),
            1e-15,
        );
        // m = Inf: 1/F ~ gamma(n/2, scale 2/n), Jacobian 1/x^2.
        assert_rel(
            "df(1.5, Inf, 4)",
            df(1.5, f64::INFINITY, 4.0, false),
            dgamma(1.0 / 1.5, 2.0, 0.5, false) / (1.5 * 1.5),
            1e-15,
        );
    }

    #[test]
    fn dbinom_raw_matches_small_binomials() {
        // C(5, 2) 0.3^2 0.7^3
        let want = 10.0 * 0.3f64.powi(2) * 0.7f64.powi(3);
        assert_rel(
            "dbinom_raw(2, 5, .3, .7)",
            dbinom_raw(2.0, 5.0, 0.3, 0.7, false),
            want,
            1e-14,
        );
        assert_rel(
            "dbinom_raw(2, 5, .3, .7, log)",
            dbinom_raw(2.0, 5.0, 0.3, 0.7, true),
            want.ln(),
            1e-14,
        );
        assert_eq!(dbinom_raw(0.0, 5.0, 0.0, 1.0, false), 1.0);
        assert_eq!(dbinom_raw(5.0, 5.0, 1.0, 0.0, false), 1.0);
        assert_eq!(dbinom_raw(0.0, 5.0, 0.3, 0.7, false), pow1p(-0.3, 5.0));
        assert_eq!(dbinom_raw(5.0, 5.0, 0.3, 0.7, false), 0.3f64.powf(5.0));
        assert_eq!(dbinom_raw(6.0, 5.0, 0.3, 0.7, false), 0.0);
    }

    #[test]
    fn pow1p_small_integer_powers_are_exact_polynomials() {
        let x = 1e-9;
        assert_eq!(pow1p(x, 0.0), 1.0);
        assert_eq!(pow1p(x, 1.0), x + 1.0);
        assert_eq!(pow1p(x, 2.0), x * (x + 2.0) + 1.0);
        assert_rel(
            "pow1p(1e-9, 10)",
            pow1p(x, 10.0),
            (10.0 * x.ln_1p()).exp(),
            0.0,
        );
        assert_rel("pow1p(0.75, 2.5)", pow1p(0.75, 2.5), 1.75f64.powf(2.5), 0.0);
        assert_eq!(pow1p(0.0, f64::NAN), 1.0);
        assert!(pow1p(0.5, f64::NAN).is_nan());
    }
}
