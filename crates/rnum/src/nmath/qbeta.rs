//! Port of R's `src/nmath/qbeta.c` (`qbeta`, `qbeta_raw`).
//!
//! Reference: Cran, Martin and Thomas (1977), Remark AS R19 and Algorithm AS 109, Applied
//! Statistics 26(1), 111-114; Remark AS R83 (v.39, 309-310) and the correction (v.40(1)
//! p.236) have been incorporated, as in R.
//!
//! The C's `goto maybe_swap` (a backward jump that redoes the tail swap and the initial
//! approximation) is the outer `'maybe_swap` loop; `goto L_Newton` is the `goto_newton`
//! flag that skips the AS 64 / AS 109 approximation; `goto L_converged` / `goto L_return`
//! are `break`s out of that loop with `goto_return` telling the two apart. The
//! `MAYBE_R_CheckUserInterrupt`, `ML_WARNING`, `MATHLIB_WARNING*` and `DEBUG_qbeta` lines
//! are dropped: they never touch the result.

use crate::nmath::arith::{fmax2, fmin2, r_pow_di};
use crate::nmath::consts::M_LN2;
use crate::nmath::dpq::{
    r_d_half, r_dt_0, r_dt_1, r_dt_civ, r_dt_clog, r_dt_log, r_dt_qiv, r_log1_exp,
};
use crate::nmath::gamma::lbeta;
use crate::nmath::pbeta::pbeta_raw;

/// `USE_LOG_X_CUTOFF` — based on some testing; had = -10.
const USE_LOG_X_CUTOFF: f64 = -5.0;

/// `n_NEWTON_FREE` — based on some testing; had = 10.
const N_NEWTON_FREE: i32 = 4;

/// `DBL_MIN_EXP` from `<float.h>`: `f64::MIN_EXP` is the same -1021.
const DBL_MIN_EXP: f64 = f64::MIN_EXP as f64;

/// `DBL_very_MIN = DBL_MIN / 4.` — CARE: assumes subnormal numbers, i.e., no underflow at
/// `DBL_MIN`.
const DBL_VERY_MIN: f64 = f64::MIN_POSITIVE / 4.0;

/// `DBL_log_v_MIN = M_LN2 * (DBL_MIN_EXP - 2)` = `log(DBL_very_MIN)`.
const DBL_LOG_V_MIN: f64 = M_LN2 * (DBL_MIN_EXP - 2.0);

/// `fpu`.
const FPU: f64 = 3e-308;

/// `acu_min`: minimal value for accuracy 'acu' which will depend on (a,p); `acu_min >= fpu`.
const ACU_MIN: f64 = 1e-300;

/// `p_lo`.
const P_LO: f64 = FPU;

/// `p_hi`.
const P_HI: f64 = 1.0 - 2.22e-16;

const CONST1: f64 = 2.30753;
const CONST2: f64 = 0.27061;
const CONST3: f64 = 0.99229;
const CONST4: f64 = 0.04481;

/// `log_eps_c = M_LN2 * (1. - DBL_MANT_DIG)` = `log(DBL_EPSILON)` = -36.04..
const LOG_EPS_C: f64 = M_LN2 * (1.0 - f64::MANTISSA_DIGITS as f64);

/// `qbeta(alpha, p, q, lower_tail, log_p)` from `src/nmath/qbeta.c`: the quantile function of
/// the Beta(p, q) distribution.
pub fn qbeta(alpha: f64, p: f64, q: f64, lower_tail: bool, log_p: bool) -> f64 {
    // test for admissibility of parameters
    if p.is_nan() || q.is_nan() || alpha.is_nan() {
        return p + q + alpha;
    }
    if p < 0.0 || q < 0.0 {
        return f64::NAN;
    }
    // allowing p==0 and q==0  <==> treat as one- or two-point mass

    let mut qbet = [0.0f64; 2]; // = { qbeta(), 1 - qbeta() }
    qbeta_raw(
        alpha,
        p,
        q,
        lower_tail,
        log_p,
        // log_q_cut ,      n_N
        USE_LOG_X_CUTOFF,
        N_NEWTON_FREE,
        &mut qbet,
    );
    qbet[0]
}

/// `return_q_0`.
fn return_q_0(give_log_q: bool, qb: &mut [f64; 2]) {
    if give_log_q {
        qb[0] = f64::NEG_INFINITY;
        qb[1] = 0.0;
    } else {
        qb[0] = 0.0;
        qb[1] = 1.0;
    }
}

/// `return_q_1`.
fn return_q_1(give_log_q: bool, qb: &mut [f64; 2]) {
    if give_log_q {
        qb[0] = 0.0;
        qb[1] = f64::NEG_INFINITY;
    } else {
        qb[0] = 1.0;
        qb[1] = 0.0;
    }
}

/// `return_q_half`.
fn return_q_half(give_log_q: bool, qb: &mut [f64; 2]) {
    if give_log_q {
        qb[0] = -M_LN2;
        qb[1] = -M_LN2;
    } else {
        qb[0] = 0.5;
        qb[1] = 0.5;
    }
}

/// `qbeta_raw` from `src/nmath/qbeta.c`. Returns both `qbeta()` and its "mirror"
/// `1 - qbeta()` in `qb`. Useful notably when `qbeta() ~= 1`.
///
/// `log_q_cut`: if `== Inf`: return `log(qbeta(..))`; otherwise, if finite: the bound for
/// switching to `log(x)`-scale; see `use_log_x`.
/// `n_n`: number of "unconstrained" Newton steps before switching to constrained.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qbeta_raw(
    alpha: f64,
    p: f64,
    q: f64,
    lower_tail: bool,
    log_p: bool,
    log_q_cut: f64,
    n_n: i32,
    qb: &mut [f64; 2],
) {
    let give_log_q = log_q_cut == f64::INFINITY;
    let mut use_log_x = give_log_q; // or u < log_q_cut  below
    let mut add_n_step = true;
    // `warned` only gates the non-convergence warning, which is dropped, so it is not kept.
    let (mut a, mut la, mut pp, mut qq);
    let (mut g, mut r, mut s, mut t, mut w);
    let mut y = -1.0f64;
    // The C leaves `u`, `xinbta`, `tx` uninitialised; every path assigns them before reading.
    let (mut u, mut xinbta, mut tx) = (0.0f64, 0.0f64, 0.0f64);

    // Assuming p >= 0, q >= 0  here ...

    // Deal with boundary cases here:
    if alpha == r_dt_0(lower_tail, log_p) {
        return_q_0(give_log_q, qb);
        return;
    }
    if alpha == r_dt_1(lower_tail, log_p) {
        return_q_1(give_log_q, qb);
        return;
    }

    // check alpha {*before* transformation which may lose all accuracy}:
    #[allow(clippy::manual_range_contains)]
    if (log_p && alpha > 0.0) || (!log_p && (alpha < 0.0 || alpha > 1.0)) {
        // alpha is outside
        qb[0] = f64::NAN;
        qb[1] = f64::NAN;
        return;
    }

    //  p==0, q==0, p = Inf, q = Inf  <==> treat as one- or two-point mass
    if p == 0.0 || q == 0.0 || !p.is_finite() || !q.is_finite() {
        // We know 0 < T(alpha) < 1 : pbeta() is constant and trivial in {0, 1/2, 1}
        if p == 0.0 && q == 0.0 {
            // point mass 1/2 at each of {0,1} :
            if alpha < r_d_half(log_p) {
                return_q_0(give_log_q, qb);
                return;
            }
            if alpha > r_d_half(log_p) {
                return_q_1(give_log_q, qb);
                return;
            }
            // else:  alpha == "1/2"
            return_q_half(give_log_q, qb);
            return;
        } else if p == 0.0 || p / q == 0.0 {
            // point mass 1 at 0 - "flipped around"
            return_q_0(give_log_q, qb);
            return;
        } else if q == 0.0 || q / p == 0.0 {
            // point mass 1 at 0 - "flipped around"
            return_q_1(give_log_q, qb);
            return;
        }
        // else:  p = q = Inf : point mass 1 at 1/2
        return_q_half(give_log_q, qb);
        return;
    }

    /* initialize */
    let p_ = r_dt_qiv(alpha, lower_tail, log_p); /* lower_tail prob (in any case) */
    // Conceptually,  0 < p_ < 1  (but can be 0 or 1 because of cancellation!)
    let logbeta = lbeta(p, q);

    let mut swap_tail = p_ > 0.5;

    let mut n_maybe_swaps = 0;
    // `u_n`: to be  log(xinbta) <==> xinbta = exp(u_n).  1 is impossible. Set to 1 on every
    // pass through `maybe_swap`, as its C declaration sits after the label.
    let mut u_n: f64;
    // true when the loop exits via `goto L_return`, skipping the `L_converged` block
    let mut goto_return = false;
    'maybe_swap: loop {
        // change tail; afterwards 0 < a <= 1/2
        if swap_tail {
            /* change tail, swap copies of {p,q}:  p <-> q :*/
            a = r_dt_civ(alpha, lower_tail, log_p); // = 1 - p_ , is < 1/2
                                                    /* la := log(a), but without numerical cancellation: */
            la = r_dt_clog(alpha, lower_tail, log_p);
            pp = q;
            qq = p;
        } else {
            a = p_;
            la = r_dt_log(alpha, lower_tail, log_p);
            pp = p;
            qq = q;
        }
        n_maybe_swaps += 1;

        /* calculate the initial approximation */

        /* Desired accuracy for Newton iterations (below) should depend on  (a,p)
        * This is from Remark .. on AS 109, adapted.
        * However, it's not clear if this is "optimal" for IEEE double prec.

        * acu = fmax2(acu_min, pow(10., -25. - 5./(pp * pp) - 1./(a * a)));

        * NEW: 'acu' accuracy NOT for squared adjustment, but simple;
        * ---- i.e.,  "new acu" = sqrt(old acu)
        */
        let acu = fmax2(ACU_MIN, 10f64.powf(-13.0 - 2.5 / (pp * pp) - 0.5 / (a * a)));
        // try to catch  "extreme left tail" early
        let u0 = (la + pp.ln() + logbeta) / pp; // = log(x_0)
        let mut rp = pp * (1.0 - qq) / (pp + 1.0);

        t = 0.2;
        // FIXME: Factor 0.2 is a bit arbitrary;  '1' is clearly much too much.
        let u0_maybe = M_LN2 * DBL_MIN_EXP < u0 && u0 < -0.01;
        /* 1. cannot allow exp(u0) = 0 ==> exp(u1) = exp(u0) = 0
         * 2. must: u0 < 0, but too close to 0 <==> x = exp(u0) = 0.99.. */

        u_n = 1.0;
        let mut goto_newton = false;
        if u0_maybe &&
            // qq <= 2 && // <--- "arbitrary"
            // u0 <  t*log_eps_c - log(fabs(rp)) &&
            u0 < (t * LOG_EPS_C - (pp * (1.0 - qq) * (2.0 - qq) / (2.0 * (pp + 2.0))).abs().ln()) / 2.0
        {
            // TODO: maybe jump here from below, when initial u "fails" ?
            // L_tail_u:
            // MM's one-step correction (cheaper than 1 Newton!)
            rp *= u0.exp(); // = rp*x0
            if rp > -1.0 {
                u = u0 - rp.ln_1p() / pp;
            } else {
                u = u0;
            }
            xinbta = u.exp();
            tx = xinbta;
            use_log_x = true; // or (u < log_q_cut)  ??
            goto_newton = true;
        }

        if !goto_newton {
            // y := y_\alpha in AS 64 := Hastings(1955) approximation of qnorm(1 - a) :
            r = (-2.0 * la).sqrt();
            y = r - (CONST1 + CONST2 * r) / (1.0 + (CONST3 + CONST4 * r) * r);
            if pp > 1.0 && qq > 1.0 {
                // use  Carter(1947), see AS 109, remark '5.'
                r = (y * y - 3.0) / 6.0;
                s = 1.0 / (pp + pp - 1.0);
                t = 1.0 / (qq + qq - 1.0);
                let h = 2.0 / (s + t);
                w = y * (h + r).sqrt() / h - (t - s) * (r + 5.0 / 6.0 - 2.0 / (3.0 * h));
                if w > 300.0 {
                    // exp(w+w) is huge or overflows
                    t = w + w + qq.ln() - pp.ln(); // = argument of log1pexp(.)
                    u = // log(xinbta) = - log1p(qq/pp * exp(w+w)) = -log(1 + exp(t))
                        if t <= 18.0 { -t.exp().ln_1p() } else { -t - (-t).exp() };
                    xinbta = u.exp();
                } else {
                    xinbta = pp / (pp + qq * (w + w).exp());
                    u = // log(xinbta)
                        -(qq / pp * (w + w).exp()).ln_1p();
                }
            } else {
                // use the original AS 64 proposal, Scheffé-Tukey (1944) and Wilson-Hilferty
                r = qq + qq;
                /* A slightly more stable version of  t := \chi^2_{alpha} of AS 64
                 * t = 1. / (9. * qq); t = r * R_pow_di(1. - t + y * sqrt(t), 3);  */
                t = 1.0 / (3.0 * qq.sqrt()); // = sqrt(t) of formula above
                t = r * r_pow_di(1.0 + t * (-t + y), 3); // = \chi^2_{alpha} of AS 64
                s = 4.0 * pp + r - 2.0; // 4p + 2q - 2 = numerator of new t' = s / t = s / chi^2
                if t == 0.0 || (t < 0.0 && s >= t) {
                    // cannot use chisq approx
                    // x0 = 1 - { (1-a)*q*B(p,q) } ^{1/q}    {AS 65}
                    // xinbta = 1. - exp((log(1-a)+ log(qq) + logbeta) / qq);
                    let l1ma = /* := log(1-a), directly from alpha (as 'la' above);
                                  though only seen very small improvements */
                        if swap_tail {
                            r_dt_log(alpha, lower_tail, log_p)
                        } else {
                            r_dt_clog(alpha, lower_tail, log_p)
                        };

                    let xx = (l1ma + qq.ln() + logbeta) / qq;
                    if xx <= 0.0 {
                        xinbta = -xx.exp_m1();
                        u = r_log1_exp(xx); // =  log(xinbta) = log(1 - exp(...A...))
                    } else {
                        // xx > 0 ==> 1 - e^xx < 0 .. is nonsense
                        // Try MM's one-step correction (or else u0)
                        let r_ = rp * u0.exp();
                        if r_ > -1.0 {
                            u = u0 - r_.ln_1p() / pp;
                        } else {
                            u = u0;
                        }
                        xinbta = u.exp();
                    }
                } else {
                    t = s / t;
                    if t <= 1.0 {
                        // cannot use chisq, either
                        u = u0;
                        xinbta = u.exp();
                    } else {
                        // (1+x0)/(1-x0) = t,  solved for x0 :
                        xinbta = 1.0 - 2.0 / (t + 1.0);
                        u = (-2.0 / (t + 1.0)).ln_1p();
                    }
                }
            }

            // Problem: If initial u is completely wrong, we make a wrong decision here
            if (swap_tail && u >= -(log_q_cut.exp())) || // ==> "swap back"
                (!swap_tail && u >= -((4.0 * log_q_cut).exp()) && pp / qq < 1000.0)
            // ==> "swap now"
            {
                // reverse swap (and typically use_log_x)
                swap_tail = !swap_tail;

                if swap_tail {
                    // "swap now" (much less easily)
                    a = r_dt_civ(alpha, lower_tail, log_p); // needed ?
                    la = r_dt_clog(alpha, lower_tail, log_p);
                    pp = q;
                    qq = p;
                } else {
                    // "swap back" :
                    a = p_;
                    la = r_dt_log(alpha, lower_tail, log_p);
                    pp = p;
                    qq = q;
                }
                // we could redo computations above, but this should be stable
                u = r_log1_exp(u);
                xinbta = u.exp();

                /* Careful: "swap now"  should not fail if
                   1) the above initial xinbta is "completely wrong"
                   2) The correction step can go outside (u_n > 0 ==>  e^u > 1 is illegal)
                   e.g., for  qbeta(0.2066, 0.143891, 0.05)
                */
            }

            if !use_log_x {
                use_log_x = u < log_q_cut; // <==> xinbta = e^u < exp(log_q_cut)
            }
            let bad_u = !u.is_finite();
            let bad_init = bad_u || xinbta > P_HI;

            tx = xinbta; // keeping "original initial x" (for now)

            if bad_u || u < log_q_cut {
                /* e.g.
                qbeta(0.21, .001, 0.05)
                try "left border" quickly, i.e.,
                try at smallest positive number: */
                w = pbeta_raw(DBL_VERY_MIN, pp, qq, true, log_p);
                if w > (if log_p { la } else { a }) {
                    // quantile is left of DBL_very_MIN: boundary "convergence"
                    if log_p || (w - a).abs() < (0.0 - a).abs() {
                        // DBL_very_MIN is better than 0
                        tx = DBL_VERY_MIN;
                        u_n = DBL_LOG_V_MIN; // = log(DBL_very_MIN)
                    } else {
                        tx = 0.0;
                        u_n = f64::NEG_INFINITY;
                    }
                    use_log_x = log_p;
                    add_n_step = false;
                    goto_return = true;
                    break 'maybe_swap;
                } else if u < DBL_LOG_V_MIN {
                    u = DBL_LOG_V_MIN; // = log(DBL_very_MIN)
                    xinbta = DBL_VERY_MIN;
                }
            }

            /* Sometimes the approximation is negative (and == 0 is also not "ok") */
            if bad_init && !(use_log_x && tx > 0.0) {
                if u == f64::NEG_INFINITY {
                    u = M_LN2 * DBL_MIN_EXP;
                    xinbta = f64::MIN_POSITIVE;
                } else {
                    xinbta = if xinbta > 1.1 {
                        // i.e. "way off"
                        0.5
                    } else if xinbta < P_LO {
                        // otherwise, keep the respective boundary:
                        u.exp()
                    } else {
                        P_HI
                    };
                    if bad_u {
                        u = xinbta.ln();
                    }
                    // otherwise: not changing "potentially better" u than the above
                }
            }
        }

        // L_Newton:
        /* --------------------------------------------------------------------

        * Solve for x by a modified Newton-Raphson method, using pbeta_raw()
        */
        r = 1.0 - pp;
        t = 1.0 - qq;
        let (mut wprev, mut prev, mut adj) = (0.0f64, 1.0f64, 1.0f64); // -Wall
        let mut converged = false;

        if use_log_x {
            // find  log(xinbta) -- work in  u := log(x) scale
            // if(bad_init && tx > 0) xinbta = tx;// may have been better

            for i_pb in 0..1000 {
                // using log_p == TRUE  unconditionally here
                /* FIXME: if exp(u) = xinbta underflows to 0,
                 *  want different formula pbeta_log(u, ..) */
                y = pbeta_raw(xinbta, pp, qq, /*lower_tail = */ true, true);

                /* w := Newton step size for   L(u) = log F(e^u)  =!= 0;   u := log(x)
                 *   =  (L(.) - la) / L'(.);  L'(u)= (F'(e^u) * e^u ) / F(e^u)
                 *   =  (L(.) - la)*F(.) / {F'(e^u) * e^u } =
                 *   =  (L(.) - la) * e^L(.) * e^{-log F'(e^u) - u}
                 *   =  ( y   - la) * e^{ y - u -log F'(e^u)}
                 and  -log F'(x)= -log f(x) = - -logbeta + (1-p) log(x) + (1-q) log(1-x)
                                            = logbeta + (1-p) u + (1-q) log(1-e^u)
                */
                w = if y == f64::NEG_INFINITY {
                    // y = -Inf  well possible: we are on log scale!
                    0.0
                } else {
                    (y - la) * (y - u + logbeta + r * u + t * r_log1_exp(u)).exp()
                };
                if !w.is_finite() {
                    // what we should do is ==> go back, get better starting value
                    if n_maybe_swaps <= 1 {
                        continue 'maybe_swap;
                    }
                    /* else  was 'break;' ...
                    but rather give up returning NaN directly as in "normal scale" Newton */
                    qb[0] = f64::NAN;
                    qb[1] = f64::NAN;
                    return;
                }
                if i_pb >= n_n && w * wprev <= 0.0 {
                    prev = fmax2(adj.abs(), FPU);
                }
                g = 1.0;
                for _i_inn in 0..1000 {
                    adj = g * w;
                    // safe guard (here, from the very beginning)
                    if adj.abs() < prev {
                        u_n = u - adj; // u_{n+1} = u_n - g*w
                        if u_n <= 0.0 {
                            // <==> 0 <  xinbta := e^u  <= 1
                            if prev <= acu || w.abs() <= acu {
                                converged = true; // goto L_converged
                                break;
                            }
                            // if (u_n != ML_NEGINF && u_n != 1)
                            break;
                        }
                    }
                    g /= 3.0;
                }
                if converged {
                    break;
                }
                // (cancellation in (u_n -u) => may differ from adj:
                let d = fmin2(adj.abs(), (u_n - u).abs());
                if d <= 4e-16 * (u_n + u).abs() {
                    break; // goto L_converged
                }
                u = u_n;
                xinbta = u.exp();
                wprev = w;
            } // for(i )
        } else {
            // "normal scale" Newton

            for i_pb in 0..1000 {
                y = pbeta_raw(xinbta, pp, qq, /*lower_tail = */ true, log_p);
                // delta{y} :   d_y = y - (log_p ? la : a);

                /* w := Newton step size  (F(.) - a) / F'(.)  or,
                 * --   log: (lF - la) / (F' / F) = exp(lF) * (lF - la) / F'
                 */
                w = if log_p {
                    (y - la) * (y + logbeta + r * xinbta.ln() + t * (-xinbta).ln_1p()).exp()
                } else {
                    (y - a) * (logbeta + r * xinbta.ln() + t * (-xinbta).ln_1p()).exp()
                };
                if !w.is_finite() {
                    // what we should do is ==> go back, get better starting value
                    if n_maybe_swaps <= 2 {
                        if !log_p && n_maybe_swaps == 2 {
                            use_log_x = true; // try now
                        }
                        if !log_p || n_maybe_swaps <= 1 {
                            continue 'maybe_swap;
                        }
                    }
                    /* else  was 'break;' ...
                    but rather give up returning NaN directly as in "normal scale" Newton */
                    qb[0] = f64::NAN;
                    qb[1] = f64::NAN;
                    return;
                }
                if i_pb >= n_n && w * wprev <= 0.0 {
                    prev = fmax2(adj.abs(), FPU);
                }
                g = 1.0;
                for _i_inn in 0..1000 {
                    adj = g * w;
                    // take full Newton steps at the beginning; only then safe guard:
                    if i_pb < n_n || adj.abs() < prev {
                        tx = xinbta - adj; // x_{n+1} = x_n - g*w
                        #[allow(clippy::manual_range_contains)]
                        if 0.0 <= tx && tx <= 1.0 {
                            if prev <= acu || w.abs() <= acu {
                                converged = true; // goto L_converged
                                break;
                            }
                            if tx != 0.0 && tx != 1.0 {
                                break;
                            }
                        }
                    }
                    g /= 3.0;
                }
                if converged {
                    break;
                }
                if (tx - xinbta).abs() <= 4e-16 * (tx + xinbta) {
                    // "<=" : (.) == 0
                    break; // goto L_converged
                }
                xinbta = tx;
                if tx == 0.0 {
                    // "we have lost"
                    break;
                }
                wprev = w;
            } // for( i_pb ..)
        } // end{else : normal scale Newton}

        /*-- NOT converged: Iteration count --*/
        // `warned = TRUE; ML_WARNING(ME_PRECISION, "qbeta");` — warning only, dropped.
        break 'maybe_swap;
    }

    if !goto_return {
        // L_converged:
        let log_ = log_p || use_log_x;
        if (log_ && y == f64::NEG_INFINITY) || (!log_ && y == 0.0) {
            // stuck at left, try if smallest positive number is "better"
            w = pbeta_raw(DBL_VERY_MIN, pp, qq, true, log_);
            if log_ || (w - a).abs() <= (y - a).abs() {
                tx = DBL_VERY_MIN;
                u_n = DBL_LOG_V_MIN; // = log(DBL_very_MIN)
            }
            add_n_step = false; // not trying to do better anymore
        }
        // The C's `else if(!warned && ...)` branch only raises the "not accurate"
        // MATHLIB_WARNING2 (its pbeta_raw(DBL_1__eps, ..) call has no side effect), so it
        // is dropped along with `warned` and the `DBL_1__eps` constant.
    }

    // L_return:
    // use  if (use_log_x) u_n else tx   {and u_n is on log scale}
    if give_log_q {
        // {currently not used from R's qbeta()} ==> use_log_x , too
        // (the "give_log_q=TRUE but use_log_x=FALSE -- please report!" warning is dropped)
        let r = r_log1_exp(u_n);
        if swap_tail {
            qb[0] = r;
            qb[1] = u_n;
        } else {
            qb[0] = u_n;
            qb[1] = r;
        }
    } else {
        if use_log_x {
            if add_n_step {
                /* add one last Newton step on original x scale, e.g., for
                qbeta(2^-98, 0.125, 2^-96) */
                if u_n != 1.0 {
                    // u_n has been computed above
                    xinbta = u_n.exp();
                }
                y = pbeta_raw(xinbta, pp, qq, /*lower_tail = */ true, log_p);
                w = if log_p {
                    (y - la) * (y + logbeta + r * xinbta.ln() + t * (-xinbta).ln_1p()).exp()
                } else {
                    (y - a) * (logbeta + r * xinbta.ln() + t * (-xinbta).ln_1p()).exp()
                };
                if w.is_finite() {
                    tx = xinbta - w;
                } else {
                    // Newton step w  cannot be used
                    tx = xinbta;
                }
            } else {
                if swap_tail {
                    qb[0] = -u_n.exp_m1();
                    qb[1] = u_n.exp();
                } else {
                    qb[0] = u_n.exp();
                    qb[1] = -u_n.exp_m1();
                }
                return;
            }
        }
        if swap_tail {
            qb[0] = 1.0 - tx;
            qb[1] = tx;
        } else {
            qb[0] = tx;
            qb[1] = 1.0 - tx;
        }
    }
}
