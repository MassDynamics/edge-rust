//! Port of R's `src/nmath/toms708.c`: `bratio` and its helpers (Didonato & Morris
//! incomplete beta, ACM TOMS 708, with R Core's `log_p` additions). See
//! `tests/scalar_pbeta.rs`.
//!
//! The C file is ported function for function, same names, same argument order, same
//! floating-point evaluation order; the Fortran-derived `goto`s become loops, `match`es on
//! a label enum, or early returns. The file's own private definitions are kept as such:
//! its `min`/`max` macros, its `R_Log1_Exp` redefinition over `rexpm1`, and `logspace_add`
//! (which `bgrat` takes from `pgamma.c`).
//!
//! Functions appear bottom-up (the small helpers first, `bratio` last) so the file could be
//! built after each chunk was written; the C file's order is top-down.

// R writes several constants to more digits than a double holds (e.g. `psi`'s `dx0`), and
// they are transcribed verbatim so they diff against the C.
#![allow(clippy::excessive_precision)]
// The polynomial and recurrence loops index arrays with the C's offsets (`c[i - 1]`,
// `d[nm1 - i]`, `p1[i]` next to `q1[i - 1]`); rewriting them as iterators would hide the
// correspondence with the Fortran-derived code.
#![allow(clippy::needless_range_loop)]
// The C's truncated literals for ln 2 (`exparg`) and pi/4 (`psi`) are kept verbatim.
#![allow(clippy::approx_constant)]
// `x < lo || x > hi` is kept as written so NaN takes the same branch as in C.
#![allow(clippy::manual_range_contains)]
// `t = x * t` and friends keep the C's operand order on the right-hand side.
#![allow(clippy::assign_op_pattern)]
// do-while loops are `loop { ...; if !(cond) { break } }` with the C condition negated
// verbatim, so a NaN ends the loop exactly as it does in C.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use super::arith::fmax2;
use super::consts::{M_LN2, M_LN_SQRT_2PI, M_SQRT_PI};
use super::dpq::{r_d_0, r_d_1, r_d_exp};

/// toms708.c's `min(a,b)` macro, `(a < b) ? a : b` (so a NaN on the left comes back as `b`).
fn min(a: f64, b: f64) -> f64 {
    if a < b {
        a
    } else {
        b
    }
}

/// toms708.c's `max(a,b)` macro, `(a > b) ? a : b`.
fn max(a: f64, b: f64) -> f64 {
    if a > b {
        a
    } else {
        b
    }
}

/// toms708.c `#undef`s `dpq.h`'s `R_Log1_Exp` and redefines it over its own `rexpm1`:
/// `(x > -M_LN2) ? log(-rexpm1(x)) : log1p(-exp(x))`.
fn r_log1_exp(x: f64) -> f64 {
    if x > -M_LN2 {
        (-rexpm1(x)).ln()
    } else {
        (-x.exp()).ln_1p()
    }
}

/// `logspace_add` from `pgamma.c`: `log(exp(logx) + exp(logy))` without leaving log space.
fn logspace_add(logx: f64, logy: f64) -> f64 {
    fmax2(logx, logy) + (-(logx - logy).abs()).exp().ln_1p()
}

/// `exparg(l)`: for `l = 0` the largest positive `W` for which `exp(W)` can be computed,
/// for `l != 0` the largest negative `W` for which `exp(W)` is nonzero. Only approximate
/// values are needed; `Rf_i1mach(16)` is `DBL_MAX_EXP` and `Rf_i1mach(15)` is `DBL_MIN_EXP`.
fn exparg(l: i32) -> f64 {
    const LNB: f64 = 0.69314718055995;
    let m = if l == 0 {
        f64::MAX_EXP
    } else {
        f64::MIN_EXP - 1
    };
    f64::from(m) * LNB * 0.99999
}

/// `esum(mu, x)`: evaluation of `exp(mu + x)`.
fn esum(mu: i32, x: f64, give_log: bool) -> f64 {
    if give_log {
        return x + f64::from(mu);
    }
    let w;
    if x > 0.0 {
        // L10:
        if mu > 0 {
            return f64::from(mu).exp() * x.exp();
        }
        w = f64::from(mu) + x;
        if w < 0.0 {
            return f64::from(mu).exp() * x.exp();
        }
    } else {
        // x <= 0
        if mu < 0 {
            return f64::from(mu).exp() * x.exp();
        }
        w = f64::from(mu) + x;
        if w > 0.0 {
            return f64::from(mu).exp() * x.exp();
        }
    }
    w.exp()
}

/// `rexpm1(x)`: evaluation of the function `exp(x) - 1`.
fn rexpm1(x: f64) -> f64 {
    const P1: f64 = 9.14041914819518e-10;
    const P2: f64 = 0.0238082361044469;
    const Q1: f64 = -0.499999999085958;
    const Q2: f64 = 0.107141568980644;
    const Q3: f64 = -0.0119041179760821;
    const Q4: f64 = 5.95130811860248e-4;

    if x.abs() <= 0.15 {
        x * (((P2 * x + P1) * x + 1.0) / ((((Q4 * x + Q3) * x + Q2) * x + Q1) * x + 1.0))
    } else {
        // |x| > 0.15 :
        let w = x.exp();
        if x > 0.0 {
            w * (0.5 - 1.0 / w + 0.5)
        } else {
            w - 0.5 - 0.5
        }
    }
}

/// `alnrel(a)`: evaluation of the function `ln(1 + a)`.
fn alnrel(a: f64) -> f64 {
    if a.abs() > 0.375 {
        return (1.0 + a).ln();
    }
    // else : |a| <= 0.375
    const P1: f64 = -1.29418923021993;
    const P2: f64 = 0.405303492862024;
    const P3: f64 = -0.0178874546012214;
    const Q1: f64 = -1.62752256355323;
    const Q2: f64 = 0.747811014037616;
    const Q3: f64 = -0.0845104217945565;
    let t = a / (a + 2.0);
    let t2 = t * t;
    let w = (((P3 * t2 + P2) * t2 + P1) * t2 + 1.0) / (((Q3 * t2 + Q2) * t2 + Q1) * t2 + 1.0);
    t * 2.0 * w
}

/// `rlog1(x)`: evaluation of the function `x - ln(1 + x)`.
fn rlog1(x: f64) -> f64 {
    const A: f64 = 0.0566749439387324;
    const B: f64 = 0.0456512608815524;
    const P0: f64 = 0.333333333333333;
    const P1: f64 = -0.224696413112536;
    const P2: f64 = 0.00620886815375787;
    const Q1: f64 = -1.27408923933623;
    const Q2: f64 = 0.354508718369557;

    if x < -0.39 || x > 0.57 {
        // direct evaluation
        let w = x + 0.5 + 0.5;
        return x - w.ln();
    }
    let (h, w1);
    if x < -0.18 {
        // L10:
        h = (x + 0.3) / 0.7;
        w1 = A - h * 0.3;
    } else if x > 0.18 {
        // L20:
        h = x * 0.75 - 0.25;
        w1 = B + h / 3.0;
    } else {
        // Argument Reduction
        h = x;
        w1 = 0.0;
    }

    // L30: Series Expansion
    let r = h / (h + 2.0);
    let t = r * r;
    let w = ((P2 * t + P1) * t + P0) / ((Q2 * t + Q1) * t + 1.0);
    t * 2.0 * (1.0 / (1.0 - r) - r * w) + w1
}

/// The rational-approximation coefficients `erf__` and `erfc1` share.
const ERF_C: f64 = 0.564189583547756;
const ERF_A: [f64; 5] = [
    7.7105849500132e-5,
    -0.00133733772997339,
    0.0323076579225834,
    0.0479137145607681,
    0.128379167095513,
];
const ERF_B: [f64; 3] = [0.00301048631703895, 0.0538971687740286, 0.375795757275549];
const ERF_P: [f64; 8] = [
    -1.36864857382717e-7,
    0.564195517478974,
    7.21175825088309,
    43.1622272220567,
    152.98928504694,
    339.320816734344,
    451.918953711873,
    300.459261020162,
];
const ERF_Q: [f64; 8] = [
    1.0,
    12.7827273196294,
    77.0001529352295,
    277.585444743988,
    638.980264465631,
    931.35409485061,
    790.950925327898,
    300.459260956983,
];
const ERF_R: [f64; 5] = [
    2.10144126479064,
    26.2370141675169,
    21.3688200555087,
    4.6580782871847,
    0.282094791773523,
];
const ERF_S: [f64; 4] = [
    94.153775055546,
    187.11481179959,
    99.0191814623914,
    18.0124575948747,
];

/// `erf__(x)`: evaluation of the real error function.
fn erf__(x: f64) -> f64 {
    let (a, b, p, q, r, s) = (ERF_A, ERF_B, ERF_P, ERF_Q, ERF_R, ERF_S);
    let ax = x.abs();
    if ax <= 0.5 {
        let t = x * x;
        let top = (((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4] + 1.0;
        let bot = ((b[0] * t + b[1]) * t + b[2]) * t + 1.0;
        return x * (top / bot);
    }

    // else:  |x| > 0.5
    if ax <= 4.0 {
        // |x| in (0.5, 4]
        let top = ((((((p[0] * ax + p[1]) * ax + p[2]) * ax + p[3]) * ax + p[4]) * ax + p[5]) * ax
            + p[6])
            * ax
            + p[7];
        let bot = ((((((q[0] * ax + q[1]) * ax + q[2]) * ax + q[3]) * ax + q[4]) * ax + q[5]) * ax
            + q[6])
            * ax
            + q[7];
        let rr = 0.5 - (-x * x).exp() * top / bot + 0.5;
        return if x < 0.0 { -rr } else { rr };
    }

    // else:  |x| > 4
    if ax >= 5.8 {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }

    // else:  4 < |x| < 5.8
    let x2 = x * x;
    let t = 1.0 / x2;
    let top = (((r[0] * t + r[1]) * t + r[2]) * t + r[3]) * t + r[4];
    let bot = (((s[0] * t + s[1]) * t + s[2]) * t + s[3]) * t + 1.0;
    let t = (ERF_C - top / (x2 * bot)) / ax;
    let rr = 0.5 - (-x2).exp() * t + 0.5;
    if x < 0.0 {
        -rr
    } else {
        rr
    }
}

/// `erfc1(ind, x)`: the complementary error function, `erfc(x)` for `ind = 0` and the
/// scaled `exp(x*x) * erfc(x)` otherwise (the only form used here).
fn erfc1(ind: i32, x: f64) -> f64 {
    let (a, b, p, q, r, s) = (ERF_A, ERF_B, ERF_P, ERF_Q, ERF_R, ERF_S);
    let mut ret_val;

    let ax = x.abs();
    // |X| <= 0.5
    if ax <= 0.5 {
        let t = x * x;
        let top = (((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4] + 1.0;
        let bot = ((b[0] * t + b[1]) * t + b[2]) * t + 1.0;
        ret_val = 0.5 - x * (top / bot) + 0.5;
        if ind != 0 {
            ret_val = t.exp() * ret_val;
        }
        return ret_val;
    }
    // else (L10:):  0.5 < |X| <= 4
    if ax <= 4.0 {
        let top = ((((((p[0] * ax + p[1]) * ax + p[2]) * ax + p[3]) * ax + p[4]) * ax + p[5]) * ax
            + p[6])
            * ax
            + p[7];
        let bot = ((((((q[0] * ax + q[1]) * ax + q[2]) * ax + q[3]) * ax + q[4]) * ax + q[5]) * ax
            + q[6])
            * ax
            + q[7];
        ret_val = top / bot;
    } else {
        // |X| > 4
        // L20:
        if x <= -5.6 {
            // L50: LIMIT VALUE FOR "LARGE" NEGATIVE X
            ret_val = 2.0;
            if ind != 0 {
                ret_val = (x * x).exp() * 2.0;
            }
            return ret_val;
        }
        if ind == 0 && (x > 100.0 || x * x > -exparg(1)) {
            // nowadays: -exparg(1) = 709.0825.. : above <===> |x| > 26.6286
            // Underflow to limit for large positive x   when ind = 0
            // L60:
            return 0.0;
        }

        // L30:  -5.6 < x < -4  or  4 < x <= 26.6286..
        let t = 1.0 / (x * x);
        let top = (((r[0] * t + r[1]) * t + r[2]) * t + r[3]) * t + r[4];
        let bot = (((s[0] * t + s[1]) * t + s[2]) * t + s[3]) * t + 1.0;
        ret_val = (ERF_C - t * top / bot) / ax;
    }

    // L40: FINAL ASSEMBLY
    if ind != 0 {
        if x < 0.0 {
            ret_val = (x * x).exp() * 2.0 - ret_val;
        }
    } else {
        // L41:  ind == 0 :
        let w = x * x;
        let t = w;
        let e = w - t;
        ret_val = (0.5 - e + 0.5) * (-t).exp() * ret_val;
        if x < 0.0 {
            ret_val = 2.0 - ret_val;
        }
    }
    ret_val
}

/// `gam1(a)`: computation of `1/gamma(a+1) - 1` for `-0.5 <= a <= 1.5`.
fn gam1(a: f64) -> f64 {
    let mut t = a;
    let d = a - 0.5;
    // t := if(a > 1/2)  a-1  else  a
    if d > 0.0 {
        t = d - 0.5;
    }
    if t < 0.0 {
        // L30:
        const R: [f64; 9] = [
            -0.422784335098468,
            -0.771330383816272,
            -0.244757765222226,
            0.118378989872749,
            9.30357293360349e-4,
            -0.0118290993445146,
            0.00223047661158249,
            2.66505979058923e-4,
            -1.32674909766242e-4,
        ];
        const S1: f64 = 0.273076135303957;
        const S2: f64 = 0.0559398236957378;

        let top = (((((((R[8] * t + R[7]) * t + R[6]) * t + R[5]) * t + R[4]) * t + R[3]) * t
            + R[2])
            * t
            + R[1])
            * t
            + R[0];
        let bot = (S2 * t + S1) * t + 1.0;
        let w = top / bot;
        if d > 0.0 {
            t * w / a
        } else {
            a * (w + 0.5 + 0.5)
        }
    } else if t == 0.0 {
        // L10: a in {0, 1}
        0.0
    } else {
        // t > 0;  L20:
        const P: [f64; 7] = [
            0.577215664901533,
            -0.409078193005776,
            -0.230975380857675,
            0.0597275330452234,
            0.0076696818164949,
            -0.00514889771323592,
            5.89597428611429e-4,
        ];
        const Q: [f64; 5] = [
            1.0,
            0.427569613095214,
            0.158451672430138,
            0.0261132021441447,
            0.00423244297896961,
        ];

        let top = (((((P[6] * t + P[5]) * t + P[4]) * t + P[3]) * t + P[2]) * t + P[1]) * t + P[0];
        let bot = (((Q[4] * t + Q[3]) * t + Q[2]) * t + Q[1]) * t + 1.0;
        let w = top / bot;
        if d > 0.0 {
            // L21:
            t / a * (w - 0.5 - 0.5)
        } else {
            a * w
        }
    }
}

/// `gamln1(a)`: evaluation of `ln(gamma(1 + a))` for `-0.2 <= a <= 1.25`.
fn gamln1(a: f64) -> f64 {
    if a < 0.6 {
        const P0: f64 = 0.577215664901533;
        const P1: f64 = 0.844203922187225;
        const P2: f64 = -0.168860593646662;
        const P3: f64 = -0.780427615533591;
        const P4: f64 = -0.402055799310489;
        const P5: f64 = -0.0673562214325671;
        const P6: f64 = -0.00271935708322958;
        const Q1: f64 = 2.88743195473681;
        const Q2: f64 = 3.12755088914843;
        const Q3: f64 = 1.56875193295039;
        const Q4: f64 = 0.361951990101499;
        const Q5: f64 = 0.0325038868253937;
        const Q6: f64 = 6.67465618796164e-4;
        let w = ((((((P6 * a + P5) * a + P4) * a + P3) * a + P2) * a + P1) * a + P0)
            / ((((((Q6 * a + Q5) * a + Q4) * a + Q3) * a + Q2) * a + Q1) * a + 1.0);
        -a * w
    } else {
        // 0.6 <= a <= 1.25
        const R0: f64 = 0.422784335098467;
        const R1: f64 = 0.848044614534529;
        const R2: f64 = 0.565221050691933;
        const R3: f64 = 0.156513060486551;
        const R4: f64 = 0.017050248402265;
        const R5: f64 = 4.97958207639485e-4;
        const S1: f64 = 1.24313399877507;
        const S2: f64 = 0.548042109832463;
        const S3: f64 = 0.10155218743983;
        const S4: f64 = 0.00713309612391;
        const S5: f64 = 1.16165475989616e-4;
        let x = a - 0.5 - 0.5;
        let w = (((((R5 * x + R4) * x + R3) * x + R2) * x + R1) * x + R0)
            / (((((S5 * x + S4) * x + S3) * x + S2) * x + S1) * x + 1.0);
        x * w
    }
}

/// `psi(x)`: the digamma function, by the rational Chebyshev approximations of Cody,
/// Strecok and Thacher (Math. Comp. 27, 1973). Returns 0 where it cannot be computed.
fn psi(x: f64) -> f64 {
    const PIOV4: f64 = 0.785398163397448; // == pi / 4
                                          // dx0 = zero of psi() to extended precision :
    const DX0: f64 = 1.461632144968362341262659542325721325;

    // COEFFICIENTS FOR RATIONAL APPROXIMATION OF  PSI(X) / (X - X0),  0.5 <= X <= 3.
    const P1: [f64; 7] = [
        0.0089538502298197,
        4.77762828042627,
        142.441585084029,
        1186.45200713425,
        3633.51846806499,
        4138.10161269013,
        1305.60269827897,
    ];
    const Q1: [f64; 6] = [
        44.8452573429826,
        520.752771467162,
        2210.0079924783,
        3641.27349079381,
        1908.310765963,
        6.91091682714533e-6,
    ];
    // COEFFICIENTS FOR RATIONAL APPROXIMATION OF  PSI(X) - LN(X) + 1 / (2*X),  X > 3.
    const P2: [f64; 4] = [
        -2.12940445131011,
        -7.01677227766759,
        -4.48616543918019,
        -0.648157123766197,
    ];
    const Q2: [f64; 4] = [
        32.2703493791143,
        89.2920700481861,
        54.6117738103215,
        7.77788548522962,
    ];

    let mut x = x;

    // XMAX1 = THE SMALLEST POSITIVE FLOATING POINT CONSTANT WITH ENTIRELY INT
    // REPRESENTATION. ALSO USED AS NEGATIVE OF LOWER BOUND ON ACCEPTABLE NEGATIVE
    // ARGUMENTS AND AS THE POSITIVE ARGUMENT BEYOND WHICH PSI MAY BE REPRESENTED AS LOG(X).
    let mut xmax1 = f64::from(i32::MAX);
    let d2 = 0.5 / (0.5 * f64::EPSILON); // = 1/DBL_EPSILON = 2^52
    if xmax1 > d2 {
        xmax1 = d2;
    }
    // XSMALL = ABSOLUTE ARGUMENT BELOW WHICH PI*COTAN(PI*X) MAY BE REPRESENTED BY 1/X.
    let xsmall = 1e-9;
    let mut aug = 0.0;
    if x < 0.5 {
        // X < 0.5,  USE REFLECTION FORMULA  PSI(1-X) = PSI(X) + PI * COTAN(PI*X)
        if x.abs() <= xsmall {
            if x == 0.0 {
                return 0.0; // L_err
            }
            // 0 < |X| <= XSMALL.  USE 1/X AS A SUBSTITUTE FOR  PI*COTAN(PI*X)
            aug = -1.0 / x;
        } else {
            // |x| > xsmall: REDUCTION OF ARGUMENT FOR COTAN
            // L100:
            let mut w = -x;
            let mut sgn = PIOV4;
            if w <= 0.0 {
                w = -w;
                sgn = -sgn;
            }
            // MAKE AN ERROR EXIT IF |X| >= XMAX1
            if w >= xmax1 {
                return 0.0; // L_err
            }
            let mut nq = w as i32;
            w -= f64::from(nq);
            nq = (w * 4.0) as i32;
            w = (w - f64::from(nq) * 0.25) * 4.0;
            // W IS NOW RELATED TO THE FRACTIONAL PART OF  4. * X.  ADJUST ARGUMENT TO
            // CORRESPOND TO VALUES IN FIRST QUADRANT AND DETERMINE SIGN
            let mut n = nq / 2;
            if n + n != nq {
                w = 1.0 - w;
            }
            let z = PIOV4 * w;
            let mut m = n / 2;
            if m + m != n {
                sgn = -sgn;
            }
            // DETERMINE FINAL VALUE FOR  -PI*COTAN(PI*X)
            n = (nq + 1) / 2;
            m = n / 2;
            m += m;
            if m == n {
                // CHECK FOR SINGULARITY
                if z == 0.0 {
                    return 0.0; // L_err
                }
                // USE COS/SIN AS A SUBSTITUTE FOR COTAN, AND SIN/COS AS A SUBSTITUTE FOR TAN
                aug = sgn * (z.cos() / z.sin() * 4.0);
            } else {
                // L140:
                aug = sgn * (z.sin() / z.cos() * 4.0);
            }
        }

        x = 1.0 - x;
    }
    // L200:
    if x <= 3.0 {
        // 0.5 <= X <= 3.
        let mut den = x;
        let mut upper = P1[0] * x;

        for i in 1..=5 {
            den = (den + Q1[i - 1]) * x;
            upper = (upper + P1[i]) * x;
        }

        den = (upper + P1[6]) / (den + Q1[5]);
        let xmx0 = x - DX0;
        return den * xmx0 + aug;
    }

    // IF X >= XMAX1, PSI = LN(X)
    if x < xmax1 {
        // 3. < X < XMAX1
        let w = 1.0 / (x * x);
        let mut den = w;
        let mut upper = P2[0] * w;

        for i in 1..=3 {
            den = (den + Q2[i - 1]) * w;
            upper = (upper + P2[i]) * w;
        }

        aug = upper / (den + Q2[3]) - 0.5 / x + aug;
    }
    aug + x.ln()
}

/// `betaln(a0, b0)`: the logarithm of the beta function, `ln(beta(a0, b0))`.
fn betaln(a0: f64, b0: f64) -> f64 {
    let mut a = min(a0, b0);
    let mut b = max(a0, b0);

    if a < 8.0 {
        if a < 1.0 {
            // A < 1
            return if b < 8.0 {
                gamln(a) + (gamln(b) - gamln(a + b))
            } else {
                gamln(a) + algdiv(a, b)
            };
        }
        // 1 <= A < 8
        let w;
        if a < 2.0 {
            if b <= 2.0 {
                return gamln(a) + gamln(b) - gsumln(a, b);
            }
            if b < 8.0 {
                w = 0.0; // goto L40
            } else {
                return gamln(a) + algdiv(a, b);
            }
        } else if b <= 1e3 {
            // L30:  REDUCTION OF A WHEN B <= 1000
            let n = (a - 1.0) as i32;
            let mut ww = 1.0;
            for _ in 1..=n {
                a -= 1.0;
                let h = a / b;
                ww *= h / (h + 1.0);
            }
            w = ww.ln();

            if b >= 8.0 {
                return w + gamln(a) + algdiv(a, b);
            }
            // else: fall through to L40
        } else {
            // L50:  reduction of A when  B > 1000
            let n = (a - 1.0) as i32;
            let mut w = 1.0;
            for _ in 1..=n {
                a -= 1.0;
                w *= a / (a / b + 1.0);
            }
            return w.ln() - f64::from(n) * b.ln() + (gamln(a) + algdiv(a, b));
        }
        // L40:  1 < A <= B < 8 :  reduction of B
        let n = (b - 1.0) as i32;
        let mut z = 1.0;
        for _ in 1..=n {
            b -= 1.0;
            z *= b / (a + b);
        }
        w + z.ln() + (gamln(a) + (gamln(b) - gsumln(a, b)))
    } else {
        // L60:  A >= 8
        const E: f64 = 0.918938533204673; // e == 0.5*LN(2*PI)
        let w = bcorr(a, b);
        let h = a / b;
        let u = -(a - 0.5) * (h / (h + 1.0)).ln();
        let v = b * alnrel(h);
        if u > v {
            b.ln() * -0.5 + E + w - v - u
        } else {
            b.ln() * -0.5 + E + w - u - v
        }
    }
}

/// `gsumln(a, b)`: evaluation of `ln(gamma(a + b))` for `1 <= a <= 2` and `1 <= b <= 2`.
fn gsumln(a: f64, b: f64) -> f64 {
    let x = a + b - 2.0; // in [0, 2]

    if x <= 0.25 {
        return gamln1(x + 1.0);
    }
    if x <= 1.25 {
        return gamln1(x) + alnrel(x);
    }
    // else x > 1.25 :
    gamln1(x - 1.0) + (x * (x + 1.0)).ln()
}

/// The Stirling-correction coefficients `bcorr`, `algdiv` and `gamln` share.
const C0: f64 = 0.0833333333333333;
const C1: f64 = -0.00277777777760991;
const C2: f64 = 7.9365066682539e-4;
const C3: f64 = -5.9520293135187e-4;
const C4: f64 = 8.37308034031215e-4;
const C5: f64 = -0.00165322962780713;

/// `bcorr(a0, b0)`: evaluation of `del(a0) + del(b0) - del(a0 + b0)` where
/// `ln(gamma(a)) = (a - 0.5)*ln(a) - a + 0.5*ln(2*pi) + del(a)`; assumes `a0, b0 >= 8`.
fn bcorr(a0: f64, b0: f64) -> f64 {
    let a = min(a0, b0);
    let b = max(a0, b0);

    let h = a / b;
    let c = h / (h + 1.0);
    let x = 1.0 / (h + 1.0);
    let x2 = x * x;

    // SET s<n> := (1 - x^n)/(1 - x)
    let s3 = x + x2 + 1.0;
    let s5 = x + x2 * s3 + 1.0;
    let s7 = x + x2 * s5 + 1.0;
    let s9 = x + x2 * s7 + 1.0;
    let s11 = x + x2 * s9 + 1.0;

    // SET W = DEL(B) - DEL(A + B)
    let mut t = 1.0 / b;
    t *= t; // t := 1 / b^2
    let mut w = ((((C5 * s11 * t + C4 * s9) * t + C3 * s7) * t + C2 * s5) * t + C1 * s3) * t + C0;
    w *= c / b;

    // COMPUTE  DEL(A) + W
    let mut t = 1.0 / a;
    t *= t; // t:= 1 / a^2
    (((((C5 * t + C4) * t + C3) * t + C2) * t + C1) * t + C0) / a + w
}

/// `algdiv(a, b)`: computation of `ln(gamma(b)/gamma(a+b))` when `b >= 8`.
fn algdiv(a: f64, b: f64) -> f64 {
    let (h, c, x, d);
    if a > b {
        h = b / a;
        c = 1.0 / (h + 1.0);
        x = h / (h + 1.0);
        d = a + (b - 0.5);
    } else {
        h = a / b;
        c = h / (h + 1.0);
        x = 1.0 / (h + 1.0);
        d = b + (a - 0.5);
    }

    // Set s<n> = (1 - x^n)/(1 - x) :
    let x2 = x * x;
    let s3 = x + x2 + 1.0;
    let s5 = x + x2 * s3 + 1.0;
    let s7 = x + x2 * s5 + 1.0;
    let s9 = x + x2 * s7 + 1.0;
    let s11 = x + x2 * s9 + 1.0;

    // w := Del(b) - Del(a + b)
    let t = 1.0 / (b * b);
    let mut w = ((((C5 * s11 * t + C4 * s9) * t + C3 * s7) * t + C2 * s5) * t + C1 * s3) * t + C0;
    w *= c / b;

    // COMBINE THE RESULTS
    let u = d * alnrel(a / b);
    let v = a * (b.ln() - 1.0);
    if u > v {
        w - v - u
    } else {
        w - u - v
    }
}

/// `gamln(a)`: evaluation of `ln(gamma(a))` for positive `a`.
fn gamln(a: f64) -> f64 {
    const D: f64 = 0.418938533204673; // d == 0.5*(LN(2*PI) - 1)

    if a <= 0.8 {
        gamln1(a) - a.ln() // ln(G(a+1)) - ln(a) == ln(G(a+1)/a) = ln(G(a))
    } else if a <= 2.25 {
        gamln1(a - 0.5 - 0.5)
    } else if a < 10.0 {
        let n = (a - 1.25) as i32;
        let mut t = a;
        let mut w = 1.0;
        for _ in 1..=n {
            t -= 1.0;
            w *= t;
        }
        gamln1(t - 1.0) + w.ln()
    } else {
        // a >= 10
        let t = 1.0 / (a * a);
        let w = (((((C5 * t + C4) * t + C3) * t + C2) * t + C1) * t + C0) / a;
        D + w + (a - 0.5) * (a.ln() - 1.0)
    }
}

/// `fpser(a, b, x, eps)`: evaluation of `I_x(a,b)` for `b < min(eps, eps*a)` and `x <= 0.5`.
fn fpser(a: f64, b: f64, x: f64, eps: f64, log_p: bool) -> f64 {
    let mut ans;
    // SET  ans := x^a :
    if log_p {
        ans = a * x.ln();
    } else if a > eps * 0.001 {
        let t = a * x.ln();
        if t < exparg(1) {
            // exp(t) would underflow
            return 0.0;
        }
        ans = t.exp();
    } else {
        ans = 1.0;
    }

    // NOTE THAT 1/B(A,B) = B
    if log_p {
        ans += b.ln() - a.ln();
    } else {
        ans *= b / a;
    }

    let tol = eps / a;
    let mut an = a + 1.0;
    let mut t = x;
    let mut s = t / an;
    loop {
        an += 1.0;
        t = x * t;
        let c = t / an;
        s += c;
        if !(c.abs() > tol) {
            break;
        }
    }

    if log_p {
        ans += (a * s).ln_1p();
    } else {
        ans *= a * s + 1.0;
    }
    ans
}

/// `apser(a, b, x, eps)`: the incomplete beta ratio `I_{1-x}(b,a)` for
/// `a <= min(eps, eps*b)`, `b*x <= 1` and `x <= 0.5`, i.e. `a` very small.
fn apser(a: f64, b: f64, x: f64, eps: f64) -> f64 {
    const G: f64 = 0.577215664901533;

    let bx = b * x;
    let mut t = x - bx;
    let c = if b * eps <= 0.02 {
        x.ln() + psi(b) + G + t
    } else {
        // b > 2e13 : psi(b) ~= log(b)
        bx.ln() + G + t
    };

    let tol = eps * 5.0 * c.abs();
    let mut j = 1.0;
    let mut s = 0.0;
    loop {
        j += 1.0;
        t *= x - bx / j;
        let aj = t / j;
        s += aj;
        if !(aj.abs() > tol) {
            break;
        }
    }

    -a * (c + s)
}

/// `bpser(a, b, x, eps)`: power series expansion for `I_x(a,b)` when `b <= 1` or
/// `b*x <= 0.7` (and, if `log_p`, also when `b < 40 & lambda > 650`).
fn bpser(a: f64, b: f64, x: f64, eps: f64, log_p: bool) -> f64 {
    if x == 0.0 {
        return r_d_0(log_p);
    }
    // compute the factor  x^a/(a*Beta(a,b))
    let mut ans;
    let a0 = min(a, b);
    if a0 >= 1.0 {
        // 1 <= a0 <= b0
        let z = a * x.ln() - betaln(a, b);
        ans = if log_p { z - a.ln() } else { z.exp() / a };
    } else {
        let mut b0 = max(a, b);
        if b0 < 8.0 {
            if b0 <= 1.0 {
                // a0 < 1  and  a0 <= b0 <= 1
                if log_p {
                    ans = a * x.ln();
                } else {
                    ans = x.powf(a);
                    if ans == 0.0 {
                        // once underflow, always underflow ..
                        return ans;
                    }
                }
                let apb = a + b;
                let z = if apb > 1.0 {
                    let u = a + b - 1.0;
                    (gam1(u) + 1.0) / apb
                } else {
                    gam1(apb) + 1.0
                };
                let c = (gam1(a) + 1.0) * (gam1(b) + 1.0) / z;

                if log_p {
                    // FIXME ? -- improve quite a bit for c ~= 1
                    ans += (c * (b / apb)).ln();
                } else {
                    ans *= c * (b / apb);
                }
            } else {
                // a0 < 1 < b0 < 8
                let mut u = gamln1(a0);
                let m = (b0 - 1.0) as i32;
                if m >= 1 {
                    let mut c = 1.0;
                    for _ in 1..=m {
                        b0 -= 1.0;
                        c *= b0 / (a0 + b0);
                    }
                    u += c.ln();
                }

                let z = a * x.ln() - u;
                b0 -= 1.0; // => b0 in (0, 7)
                let apb = a0 + b0;
                let t = if apb > 1.0 {
                    u = a0 + b0 - 1.0;
                    (gam1(u) + 1.0) / apb
                } else {
                    gam1(apb) + 1.0
                };

                ans = if log_p {
                    // FIXME? potential for improving log(t)
                    z + (a0 / a).ln() + gam1(b0).ln_1p() - t.ln()
                } else {
                    z.exp() * (a0 / a) * (gam1(b0) + 1.0) / t
                };
            }
        } else {
            // a0 < 1 < 8 <= b0
            let u = gamln1(a0) + algdiv(a0, b0);
            let z = a * x.ln() - u;

            ans = if log_p {
                z + (a0 / a).ln()
            } else {
                a0 / a * z.exp()
            };
        }
    }
    if ans == r_d_0(log_p) || (!log_p && a <= eps * 0.1) {
        return ans;
    }

    // COMPUTE THE SERIES
    let tol = eps / a;
    let mut n = 0.0;
    let mut sum = 0.0;
    let mut w;
    let mut c = 1.0;
    loop {
        // sum is alternating as long as n < b (<==> 1 - b/n < 0)
        n += 1.0;
        c *= (0.5 - b / n + 0.5) * x;
        w = c / (a + n);
        sum += w;
        if !(n < 1e7 && w.abs() > tol) {
            break;
        }
    }
    // (C warns here when the series did not converge in 1e7 terms; no warning channel.)
    if log_p {
        if a * sum > -1.0 {
            ans += (a * sum).ln_1p();
        } else {
            ans = f64::NEG_INFINITY;
        }
    } else if a * sum > -1.0 {
        ans *= a * sum + 1.0;
    } else {
        // underflow to
        ans = 0.0;
    }
    ans
}

/// `bup(a, b, x, y, n, eps)`: evaluation of `I_x(a,b) - I_x(a+n,b)` where `n` is a positive
/// integer.
#[allow(clippy::too_many_arguments)]
fn bup(a: f64, b: f64, x: f64, y: f64, n: i32, eps: f64, give_log: bool) -> f64 {
    // Obtain the scaling factor exp(-mu) and exp(mu)*(x^a * y^b / beta(a,b))/a
    let apb = a + b;
    let ap1 = a + 1.0;
    let (mu, mut d);
    if n > 1 && a >= 1.0 && apb >= ap1 * 1.1 {
        let mut m = exparg(1).abs() as i32;
        let k = exparg(0) as i32;
        if m > k {
            m = k;
        }
        mu = m;
        d = (-f64::from(mu)).exp(); // = exp(-709) = 1.216780751..e-308  nowadays
    } else {
        mu = 0;
        d = 1.0;
    }

    // L10:
    let mut ret_val = if give_log {
        brcmp1(mu, a, b, x, y, true) - a.ln()
    } else {
        brcmp1(mu, a, b, x, y, false) / a
    };
    if n == 1 || (give_log && ret_val == f64::NEG_INFINITY) || (!give_log && ret_val == 0.0) {
        return ret_val;
    }

    let nm1 = n - 1;
    let mut w = d;

    // LET K BE THE INDEX OF THE MAXIMUM TERM
    let mut k = 0;
    if b > 1.0 {
        if y > 1e-4 {
            let r = (b - 1.0) * x / y - a;
            if r >= 1.0 {
                k = if r < f64::from(nm1) { r as i32 } else { nm1 };
            }
        } else {
            k = nm1;
        }

        // ADD THE INCREASING TERMS OF THE SERIES - if k > 0
        // L30:
        for i in 0..k {
            let l = f64::from(i);
            d *= (apb + l) / (ap1 + l) * x;
            w += d;
        }
    }

    // L40:     ADD THE REMAINING TERMS OF THE SERIES
    for i in k..nm1 {
        let l = f64::from(i);
        d *= (apb + l) / (ap1 + l) * x;
        w += d;
        if d <= eps * w {
            // relativ convergence (eps)
            break;
        }
    }

    // L50: TERMINATE THE PROCEDURE
    if give_log {
        ret_val += w.ln();
    } else {
        ret_val *= w;
    }
    ret_val
}

/// `bfrac(a, b, x, y, lambda, eps)`: continued fraction expansion for `I_x(a,b)` when
/// `a, b > 1`; assumes `lambda = (a + b)*y - b`.
#[allow(clippy::too_many_arguments)]
fn bfrac(a: f64, b: f64, x: f64, y: f64, lambda: f64, eps: f64, log_p: bool) -> f64 {
    if !lambda.is_finite() {
        return f64::NAN; // TODO: can return 0 or 1 (?)
    }
    let brc = brcomp(a, b, x, y, log_p);
    if brc.is_nan() {
        // e.g. from   L <- 1e308; pnbinom(L, L, mu = 5)
        return f64::NAN; // TODO: could we know better?
    }
    if !log_p && brc == 0.0 {
        // brcomp(a,b,x,y) underflowed to 0.
        return 0.0;
    }

    let c = lambda + 1.0;
    let c0 = b / a;
    let c1 = 1.0 / a + 1.0;
    let yp1 = y + 1.0;

    let mut n = 0.0;
    let mut p = 1.0;
    let mut s = a + 1.0;
    let mut an = 0.0;
    let mut bn = 1.0;
    let mut anp1 = 1.0;
    let mut bnp1 = c / c1;
    let mut r = c1 / c;

    // CONTINUED FRACTION CALCULATION
    loop {
        n += 1.0;
        let mut t = n / a;
        let w = n * (b - n) * x;
        let mut e = a / s;
        let alpha = p * (p + c0) * e * e * (w * x);
        e = (t + 1.0) / (c1 + t + t);
        let beta = n + w / s + e * (c + n * yp1);
        p = t + 1.0;
        s += 2.0;

        // update an, bn, anp1, and bnp1
        t = alpha * an + beta * anp1;
        an = anp1;
        anp1 = t;
        t = alpha * bn + beta * bnp1;
        bn = bnp1;
        bnp1 = t;

        let r0 = r;
        r = anp1 / bnp1;
        if (r - r0).abs() <= eps * r {
            break;
        }

        // rescale an, bn, anp1, and bnp1
        an /= bnp1;
        bn /= bnp1;
        anp1 = r;
        bnp1 = 1.0;
        if !(n < 10000.0) {
            // arbitrary; had '1' --> infinite loop for  lambda = Inf
            break;
        }
    }
    // (C warns here when 10000 terms did not converge; no warning channel.)
    if log_p {
        brc + r.ln()
    } else {
        brc * r
    }
}

/// `1/sqrt(2*pi)` as `brcomp` and `brcmp1` write it (R has `M_1_SQRT_2PI`).
const CONST__: f64 = 0.398942280401433;

/// `brcomp(a, b, x, y)`: evaluation of `x^a * y^b / Beta(a,b)`.
fn brcomp(a: f64, b: f64, x: f64, y: f64, log_p: bool) -> f64 {
    if x == 0.0 || y == 0.0 {
        return r_d_0(log_p);
    }
    let a0 = min(a, b);
    if a0 < 8.0 {
        let (lnx, lny);
        if x <= 0.375 {
            lnx = x.ln();
            lny = alnrel(-x);
        } else if y > 0.375 {
            lnx = x.ln();
            lny = y.ln();
        } else {
            lnx = alnrel(-y);
            lny = y.ln();
        }

        let mut z = a * lnx + b * lny;
        if a0 >= 1.0 {
            z -= betaln(a, b);
            return r_d_exp(z, log_p);
        }

        // PROCEDURE FOR a < 1 OR b < 1
        let mut b0 = max(a, b);
        if b0 >= 8.0 {
            // L80:
            let u = gamln1(a0) + algdiv(a0, b0);
            return if log_p {
                a0.ln() + (z - u)
            } else {
                a0 * (z - u).exp()
            };
        }
        // else :
        if b0 <= 1.0 {
            // algorithm for max(a,b) = b0 <= 1
            let e_z = r_d_exp(z, log_p);

            if !log_p && e_z == 0.0 {
                // exp() underflow
                return 0.0;
            }

            let apb = a + b;
            if apb > 1.0 {
                let u = a + b - 1.0;
                z = (gam1(u) + 1.0) / apb;
            } else {
                z = gam1(apb) + 1.0;
            }

            let c = (gam1(a) + 1.0) * (gam1(b) + 1.0) / z;
            // FIXME? log(a0*c)= log(a0)+ log(c) and that is improvable
            return if log_p {
                e_z + (a0 * c).ln() - (a0 / b0).ln_1p()
            } else {
                e_z * (a0 * c) / (a0 / b0 + 1.0)
            };
        }

        // else :  ALGORITHM FOR 1 < b0 < 8
        let mut u = gamln1(a0);
        let n = (b0 - 1.0) as i32;
        if n >= 1 {
            let mut c = 1.0;
            for _ in 1..=n {
                b0 -= 1.0;
                c *= b0 / (a0 + b0);
            }
            u = c.ln() + u;
        }
        z -= u;
        b0 -= 1.0;
        let apb = a0 + b0;
        let t = if apb > 1.0 {
            u = a0 + b0 - 1.0;
            (gam1(u) + 1.0) / apb
        } else {
            gam1(apb) + 1.0
        };

        if log_p {
            a0.ln() + z + gam1(b0).ln_1p() - t.ln()
        } else {
            a0 * z.exp() * (gam1(b0) + 1.0) / t
        }
    } else {
        // PROCEDURE FOR A >= 8 AND B >= 8
        let (h, x0, y0, lambda);
        if a <= b {
            h = a / b;
            x0 = h / (h + 1.0);
            y0 = 1.0 / (h + 1.0);
            lambda = a - (a + b) * x;
        } else {
            h = b / a;
            x0 = 1.0 / (h + 1.0);
            y0 = h / (h + 1.0);
            lambda = (a + b) * y - b;
        }

        let mut e = -lambda / a;
        let u = if e.abs() > 0.6 {
            e - (x / x0).ln()
        } else {
            rlog1(e)
        };

        e = lambda / b;
        let v = if e.abs() <= 0.6 {
            rlog1(e)
        } else {
            e - (y / y0).ln()
        };

        let z = if log_p {
            -(a * u + b * v)
        } else {
            (-(a * u + b * v)).exp()
        };

        if log_p {
            -M_LN_SQRT_2PI + 0.5 * (b * x0).ln() + z - bcorr(a, b)
        } else {
            CONST__ * (b * x0).sqrt() * z * (-bcorr(a, b)).exp()
        }
    }
}

/// `brcmp1(mu, a, b, x, y)`: evaluation of `exp(mu) * x^a * y^b / beta(a,b)`. Called only
/// from `bup`.
fn brcmp1(mu: i32, a: f64, b: f64, x: f64, y: f64, give_log: bool) -> f64 {
    let a0 = min(a, b);
    if a0 < 8.0 {
        let (lnx, lny);
        if x <= 0.375 {
            lnx = x.ln();
            lny = alnrel(-x);
        } else if y > 0.375 {
            // L11:
            lnx = x.ln();
            lny = y.ln();
        } else {
            lnx = alnrel(-y);
            lny = y.ln();
        }

        // L20:
        let mut z = a * lnx + b * lny;
        if a0 >= 1.0 {
            z -= betaln(a, b);
            return esum(mu, z, give_log);
        }
        // else :  PROCEDURE FOR A < 1 OR B < 1
        // L30:
        let mut b0 = max(a, b);
        if b0 >= 8.0 {
            // L80:  ALGORITHM FOR b0 >= 8
            let u = gamln1(a0) + algdiv(a0, b0);
            return if give_log {
                a0.ln() + esum(mu, z - u, true)
            } else {
                a0 * esum(mu, z - u, false)
            };
        } else if b0 <= 1.0 {
            // a0 < 1, b0 <= 1
            let ans = esum(mu, z, give_log);
            if ans == (if give_log { f64::NEG_INFINITY } else { 0.0 }) {
                return ans;
            }

            let apb = a + b;
            if apb > 1.0 {
                // L40:
                let u = a + b - 1.0;
                z = (gam1(u) + 1.0) / apb;
            } else {
                z = gam1(apb) + 1.0;
            }
            // L50:
            let c = if give_log {
                gam1(a).ln_1p() + gam1(b).ln_1p() - z.ln()
            } else {
                (gam1(a) + 1.0) * (gam1(b) + 1.0) / z
            };
            return if give_log {
                ans + a0.ln() + c - (a0 / b0).ln_1p()
            } else {
                ans * (a0 * c) / (a0 / b0 + 1.0)
            };
        }
        // else:  algorithm for  a0 < 1 < b0 < 8
        // L60:
        let mut u = gamln1(a0);
        let n = (b0 - 1.0) as i32;
        if n >= 1 {
            let mut c = 1.0;
            for _ in 1..=n {
                b0 -= 1.0;
                c *= b0 / (a0 + b0);
                // L61:
            }
            u += c.ln(); // TODO?: log(c) = log( prod(...) ) =  sum( log(...) )
        }
        // L70:
        z -= u;
        b0 -= 1.0;
        let apb = a0 + b0;
        let t = if apb > 1.0 {
            // L71:
            (gam1(apb - 1.0) + 1.0) / apb
        } else {
            gam1(apb) + 1.0
        };
        // L72:
        if give_log {
            a0.ln() + esum(mu, z, true) + gam1(b0).ln_1p() - t.ln() // TODO? log(t) = log1p(..)
        } else {
            a0 * esum(mu, z, false) * (gam1(b0) + 1.0) / t
        }
    } else {
        // PROCEDURE FOR A >= 8 AND B >= 8
        // L100:
        let (h, x0, y0, lambda);
        if a > b {
            // L101:
            h = b / a;
            x0 = 1.0 / (h + 1.0); // => lx0 := log(x0) = 0 - log1p(h)
            y0 = h / (h + 1.0);
            lambda = (a + b) * y - b;
        } else {
            h = a / b;
            x0 = h / (h + 1.0); // => lx0 := log(x0) = - log1p(1/h)
            y0 = 1.0 / (h + 1.0);
            lambda = a - (a + b) * x;
        }
        let lx0 = -(b / a).ln_1p(); // in both cases

        // L110:
        let mut e = -lambda / a;
        let u = if e.abs() > 0.6 {
            // L111:
            e - (x / x0).ln()
        } else {
            rlog1(e)
        };

        // L120:
        e = lambda / b;
        let v = if e.abs() > 0.6 {
            // L121:
            e - (y / y0).ln()
        } else {
            rlog1(e)
        };

        // L130:
        let z = esum(mu, -(a * u + b * v), give_log);
        if give_log {
            CONST__.ln() + (b.ln() + lx0) / 2.0 + z - bcorr(a, b)
        } else {
            CONST__ * (b * x0).sqrt() * z * (-bcorr(a, b)).exp()
        }
    }
}

/// `bgrat(a, b, x, y, w, eps, ierr, log_w)`: asymptotic expansion for `I_x(a,b)` when `a`
/// is larger than `b`; computes `w := w + I_x(a,b)`, assuming `a >= 15` and `b <= 1`.
/// If `log_w`, `w` itself is in log space on entry and exit:
/// `w := log(exp(w) + I_x(a,b)) = logspace_add(w, log(I_x(a,b)))`.
#[allow(clippy::too_many_arguments)]
fn bgrat(a: f64, b: f64, x: f64, y: f64, w: &mut f64, eps: f64, ierr: &mut i32, log_w: bool) {
    const N_TERMS_BGRAT: usize = 30;
    let mut c = [0.0f64; N_TERMS_BGRAT];
    let mut d = [0.0f64; N_TERMS_BGRAT];
    let bm1 = b - 0.5 - 0.5;
    // nu = a + (b-1)/2 =: T, in (9.1) of Didonato & Morris(1992), p.362
    let nu = a + bm1 * 0.5;
    let lnx = if y > 0.375 { x.ln() } else { alnrel(-y) };
    let z = -nu * lnx; // z =: u in (9.1) of D.&M.(1992)

    if b * z == 0.0 {
        // should not happen, but does, e.g., for  pbeta(1e-320, 1e-5, 0.5)  i.e.,
        // _subnormal_ x. (C warns: "b*z == 0 underflow, hence inaccurate pbeta()".)
        // L_Error:    THE EXPANSION CANNOT BE COMPUTED
        *ierr = 1;
        return;
    }

    // COMPUTATION OF THE EXPANSION
    // r = exp(-z) * z^b / gamma(b) ;  gam1(b) = 1/gamma(b+1) - 1 , b in [-1/2, 3/2].
    // exp(a*lnx) underflows for large (a * lnx); e.g. large a ==> using log_r := log(r):
    // log(r)=log(b) + log1p(gam1(b)) + b * log(z) + (a * lnx) + (bm1 * 0.5 * lnx),
    let log_r = b.ln() + gam1(b).ln_1p() + b * z.ln() + nu * lnx;
    // u is 'factored out' from the expansion {and multiplied back, at the end}:
    // algdiv(b,a) = log(gamma(a)/gamma(a+b))
    let log_u = log_r - (algdiv(b, a) + b * nu.ln());
    // =: M  in (9.2) of {reference above}
    let u = log_u.exp();

    if log_u == f64::NEG_INFINITY {
        // L_Error:    THE EXPANSION CANNOT BE COMPUTED
        *ierr = 2;
        return;
    }

    let u_0 = u == 0.0; // underflow --> do work with log(u) == log_u !
                        // l := *w/u .. but with care: such that it also works when u underflows to 0:
    let l = if log_w {
        if *w == f64::NEG_INFINITY {
            0.0
        } else {
            (*w - log_u).exp()
        }
    } else if *w == 0.0 {
        0.0
    } else {
        (w.ln() - log_u).exp()
    };

    let q_r = grat_r(b, z, log_r, eps); // = q/r of former grat1(b,z, r, &p, &q)
    let v = 0.25 / (nu * nu);
    let t2 = lnx * 0.25 * lnx;
    let mut j = q_r;
    let mut sum = j;
    let mut t = 1.0;
    let mut cn = 1.0;
    let mut n2 = 0.0;
    for n in 1..=N_TERMS_BGRAT {
        let bp2n = b + n2;
        j = (bp2n * (bp2n + 1.0) * j + (z + bp2n + 1.0) * t) * v;
        n2 += 2.0;
        t *= t2;
        cn /= n2 * (n2 + 1.0);
        let nm1 = n - 1;
        c[nm1] = cn;
        let mut s = 0.0;
        if n > 1 {
            let mut coef = b - n as f64;
            for i in 1..=nm1 {
                s += coef * c[i - 1] * d[nm1 - i];
                coef += b;
            }
        }
        d[nm1] = bm1 * cn + s / n as f64;
        let dj = d[nm1] * j;
        sum += dj;
        if sum <= 0.0 {
            // should not happen
            // L_Error:    THE EXPANSION CANNOT BE COMPUTED
            *ierr = 3;
            return;
        }
        if dj.abs() <= eps * (sum + l) {
            *ierr = 0;
            break;
        } else if n == N_TERMS_BGRAT {
            // never? ; please notify R-core if seen (C warns here)
            *ierr = 4;
        }
    }

    // ADD THE RESULTS TO W
    if log_w {
        // *w is in log space already:
        *w = logspace_add(*w, log_u + sum.ln());
    } else {
        *w += if u_0 {
            (log_u + sum.ln()).exp()
        } else {
            u * sum
        };
    }
}

/// `grat_r(a, x, log_r, eps)`: scaled complement of the incomplete gamma ratio,
/// `Q(a,x) / r` with `r = e^(-x) * x^a / Gamma(a) == exp(log_r)`; assumes `a <= 1`.
/// Called only from `bgrat`.
fn grat_r(a: f64, x: f64, log_r: f64, eps: f64) -> f64 {
    if a * x == 0.0 {
        // L130:
        if x <= a {
            // L100:
            (-log_r).exp()
        } else {
            // L110:
            0.0
        }
    } else if a == 0.5 {
        // e.g. when called from pt()
        // L120:
        if x < 0.25 {
            let p = erf__(x.sqrt());
            (0.5 - p + 0.5) * (-log_r).exp()
        } else {
            // 2013-02-27: improvement for "large" x: direct computation of q/r:
            let sx = x.sqrt();
            erfc1(1, sx) / sx * M_SQRT_PI
        }
    } else if x < 1.1 {
        // L10:  Taylor series for  P(a,x)/x^a
        let mut an = 3.0;
        let mut c = x;
        let mut sum = x / (a + 3.0);
        let tol = eps * 0.1 / (a + 1.0);
        loop {
            an += 1.0;
            c *= -(x / an);
            let t = c / (a + an);
            sum += t;
            if !(t.abs() > tol) {
                break;
            }
        }

        let j = a * x * ((sum / 6.0 - 0.5 / (a + 2.0)) * x + 1.0 / (a + 1.0));
        let z = a * x.ln();
        let h = gam1(a);
        let g = h + 1.0;

        if (x >= 0.25 && (a < x / 2.59)) || (z > -0.13394) {
            // L40:
            let l = rexpm1(z);
            let q = ((l + 0.5 + 0.5) * j - l) * g - h;
            if q <= 0.0 {
                // L110:
                0.0
            } else {
                q * (-log_r).exp()
            }
        } else {
            let p = z.exp() * g * (0.5 - j + 0.5);
            // q/r =
            (0.5 - p + 0.5) * (-log_r).exp()
        }
    } else {
        // L50: ----  (x >= 1.1)  ---- Continued Fraction Expansion
        let mut a2n_1 = 1.0;
        let mut a2n = 1.0;
        let mut b2n_1 = x;
        let mut b2n = x + (1.0 - a);
        let mut c = 1.0;
        let mut an0;
        loop {
            a2n_1 = x * a2n + c * a2n_1;
            b2n_1 = x * b2n + c * b2n_1;
            let am0 = a2n_1 / b2n_1;
            c += 1.0;
            let c_a = c - a;
            a2n = a2n_1 + c_a * a2n;
            b2n = b2n_1 + c_a * b2n;
            an0 = a2n / b2n;
            if !((an0 - am0).abs() >= eps * an0) {
                break;
            }
        }
        // q/r = (r * an0)/r =
        an0
    }
}

/// `basym(a, b, lambda, eps)`: asymptotic expansion for `I_x(a,b)` for large `a` and `b`;
/// `lambda = (a + b)*y - b` is assumed nonnegative and `a, b >= 15`.
fn basym(a: f64, b: f64, lambda: f64, eps: f64, log_p: bool) -> f64 {
    // NUM IS THE MAXIMUM VALUE THAT N CAN TAKE IN THE DO LOOP ENDING AT STATEMENT 50.
    // IT IS REQUIRED THAT NUM BE EVEN. THE ARRAYS A0, B0, C, D HAVE DIMENSION NUM + 1.
    const NUM_IT: usize = 20;
    const E0: f64 = 1.12837916709551; // e0 == 2/sqrt(pi)
    const E1: f64 = 0.353553390593274; // e1 == 2^(-3/2)
    const LN_E0: f64 = 0.120782237635245; // == ln(e0)

    let mut a0 = [0.0f64; NUM_IT + 1];
    let mut b0 = [0.0f64; NUM_IT + 1];
    let mut c = [0.0f64; NUM_IT + 1];
    let mut d = [0.0f64; NUM_IT + 1];

    let f = a * rlog1(-lambda / a) + b * rlog1(lambda / b);
    let t;
    if log_p {
        t = -f;
    } else {
        t = (-f).exp();
        if t == 0.0 {
            return 0.0; // once underflow, always underflow ..
        }
    }
    let z0 = f.sqrt();
    let z = z0 / E1 * 0.5;
    let z2 = f + f;

    let (h, r0, r1, w0);
    if a < b {
        h = a / b;
        r0 = 1.0 / (h + 1.0);
        r1 = (b - a) / b;
        w0 = 1.0 / (a * (h + 1.0)).sqrt();
    } else {
        h = b / a;
        r0 = 1.0 / (h + 1.0);
        r1 = (b - a) / a;
        w0 = 1.0 / (b * (h + 1.0)).sqrt();
    }

    a0[0] = r1 * 0.66666666666666663;
    c[0] = a0[0] * -0.5;
    d[0] = -c[0];
    let mut j0 = 0.5 / E0 * erfc1(1, z0);
    let mut j1 = E1;
    let mut sum = j0 + d[0] * w0 * j1;

    let mut s = 1.0;
    let h2 = h * h;
    let mut hn = 1.0;
    let mut w = w0;
    let mut znm1 = z;
    let mut zn = z2;
    for n in (2..=NUM_IT).step_by(2) {
        hn *= h2;
        a0[n - 1] = r0 * 2.0 * (h * hn + 1.0) / (n as f64 + 2.0);
        let np1 = n + 1;
        s += hn;
        a0[np1 - 1] = r1 * 2.0 * s / (n as f64 + 3.0);

        for i in n..=np1 {
            let r = (i as f64 + 1.0) * -0.5;
            b0[0] = r * a0[0];
            for m in 2..=i {
                let mut bsum = 0.0;
                for j in 1..m {
                    let mmj = m - j;
                    bsum += (j as f64 * r - mmj as f64) * a0[j - 1] * b0[mmj - 1];
                }
                b0[m - 1] = r * a0[m - 1] + bsum / m as f64;
            }
            c[i - 1] = b0[i - 1] / (i as f64 + 1.0);

            let mut dsum = 0.0;
            for j in 1..i {
                dsum += d[i - j - 1] * c[j - 1];
            }
            d[i - 1] = -(dsum + c[i - 1]);
        }

        j0 = E1 * znm1 + (n as f64 - 1.0) * j0;
        j1 = E1 * zn + n as f64 * j1;
        znm1 = z2 * znm1;
        zn = z2 * zn;
        w *= w0;
        let t0 = d[n - 1] * w * j0;
        w *= w0;
        let t1 = d[np1 - 1] * w * j1;
        sum += t0 + t1;
        if t0.abs() + t1.abs() <= eps * sum {
            break;
        }
    }

    if log_p {
        LN_E0 + t - bcorr(a, b) + sum.ln()
    } else {
        let u = (-bcorr(a, b)).exp();
        E0 * t * u * sum
    }
}

/// The `goto` targets of `bratio`'s second half, so the C's jumps read as `match` arms.
enum Label {
    /// `L_w_bpser` (was L100): `w := bpser(a0, b0, x0)`, `w1 := 1 - w`.
    WBpser,
    /// `L_w1_bpser` (was L110): `w1 := bpser(b0, a0, y0)`, `w := 1 - w1`.
    W1Bpser,
    /// `L_bfrac`: `w := bfrac(...)`, `w1 := 1 - w`.
    Bfrac,
    /// `L140`: `b0 := fractional_part(b0)`, then `bup` + (`bpser` | `bup` + `bgrat`).
    L140,
    /// `L_end_from_w`: `w` is set on the natural scale; derive `w1` (and log both).
    EndFromW,
    /// `L_end_from_w1`: `w1` is set on the natural scale; derive `w` (and log both).
    EndFromW1,
    /// `L_end_from_w1_log`: `w1` is already `log(w1)`; derive `w`.
    EndFromW1Log,
    /// `L_end`: swap `w` and `w1` back if the arguments were swapped, and return.
    End,
}

/// `bratio(a, b, x, y, w, w1, ierr, log_p)` from `toms708.c`: evaluation of the incomplete
/// beta function `I_x(a,b)`, assuming `a, b >= 0`, `x <= 1` and `y = 1 - x`. Sets
/// `w = I_x(a,b)` and `w1 = 1 - I_x(a,b)` (both on the log scale when `log_p`), and
/// `ierr` to 0 or one of:
///
/// ```text
/// ierr = 1  if a or b is negative
/// ierr = 2  if a = b = 0
/// ierr = 3  if x < 0 or x > 1
/// ierr = 4  if y < 0 or y > 1
/// ierr = 5  if x + y != 1
/// ierr = 6  if x = a = 0
/// ierr = 7  if y = b = 0
/// ierr = 9  NaN in a, b, x, or y
/// ierr = 11..14  bgrat() error code 1..4
/// ```
///
/// Written by Alfred H. Morris, Jr. (NSWC, revised Nov 1991); `log_p` added by R Core.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bratio(
    a: f64,
    b: f64,
    x: f64,
    y: f64,
    w: &mut f64,
    w1: &mut f64,
    ierr: &mut i32,
    log_p: bool,
) {
    // eps is a machine dependent constant: the smallest floating point number for which
    // 1. + eps > 1.  NOTE: for almost all purposes it is replaced by 1e-15 below.
    let mut eps = 2.0 * (0.5 * f64::EPSILON); // == 2 * Rf_d1mach(3) == DBL_EPSILON

    *w = r_d_0(log_p);
    *w1 = r_d_0(log_p);

    // safeguard, preventing infinite loops further down
    if x.is_nan() || y.is_nan() || a.is_nan() || b.is_nan() {
        *ierr = 9;
        return;
    }
    if a < 0.0 || b < 0.0 {
        *ierr = 1;
        return;
    }
    if a == 0.0 && b == 0.0 {
        *ierr = 2;
        return;
    }
    if x < 0.0 || x > 1.0 {
        *ierr = 3;
        return;
    }
    if y < 0.0 || y > 1.0 {
        *ierr = 4;
        return;
    }

    // check that  'y == 1 - x' :
    let z = x + y - 0.5 - 0.5;
    if z.abs() > eps * 3.0 {
        *ierr = 5;
        return;
    }

    *ierr = 0;
    if x == 0.0 {
        // L200:
        if a == 0.0 {
            *ierr = 6;
            return;
        }
        // L201:
        *w = r_d_0(log_p);
        *w1 = r_d_1(log_p);
        return;
    }
    if y == 0.0 {
        // L210:
        if b == 0.0 {
            *ierr = 7;
            return;
        }
        // L211:
        *w = r_d_1(log_p);
        *w1 = r_d_0(log_p);
        return;
    }
    if a == 0.0 {
        // L211:
        *w = r_d_1(log_p);
        *w1 = r_d_0(log_p);
        return;
    }
    if b == 0.0 {
        // L201:
        *w = r_d_0(log_p);
        *w1 = r_d_1(log_p);
        return;
    }

    eps = max(eps, 1e-15); // = 1e-15 (for IEEE 754)
    let a_lt_b = a < b;
    if (if a_lt_b { b } else { a }) < eps * 0.001 {
        // procedure for a and b < 0.001 * eps = 1e-18
        // L230:  -- result *independent* of x (!)
        // *w  = a/(a+b)  and  w1 = b/(a+b) :
        if log_p {
            if a_lt_b {
                *w = (-a / (a + b)).ln_1p(); // notably if a << b
                *w1 = (a / (a + b)).ln();
            } else {
                // b <= a
                *w = (b / (a + b)).ln();
                *w1 = (-b / (a + b)).ln_1p();
            }
        } else {
            *w = b / (a + b);
            *w1 = a / (a + b);
        }
        return;
    }

    let (mut a0, mut b0, x0, y0);
    let mut lambda = 0.0;
    let mut n: i32 = 0;
    let mut ierr1 = 0;
    let do_swap;
    let mut label;

    if min(a, b) <= 1.0 {
        // ------------------------ a <= 1  or  b <= 1 ----
        do_swap = x > 0.5;
        if do_swap {
            a0 = b;
            x0 = y;
            b0 = a;
            y0 = x;
        } else {
            a0 = a;
            x0 = x;
            b0 = b;
            y0 = y;
        }
        // now have  x0 <= 1/2 <= y0  (still  x0+y0 == 1)

        label = 'block: {
            if b0 < min(eps, eps * a0) {
                // L80:
                *w = fpser(a0, b0, x0, eps, log_p);
                *w1 = if log_p {
                    r_log1_exp(*w)
                } else {
                    0.5 - *w + 0.5
                };
                break 'block Label::End;
            }

            if a0 < min(eps, eps * b0) && b0 * x0 <= 1.0 {
                // L90:
                *w1 = apser(a0, b0, x0, eps);
                break 'block Label::EndFromW1;
            }

            let mut did_bup = false;
            // `goto L131` from the `b0 > 15` case skips the `bup` step.
            let mut skip_bup = false;
            if max(a0, b0) > 1.0 {
                // L20:  min(a,b) <= 1 < max(a,b)
                if b0 <= 1.0 {
                    break 'block Label::WBpser;
                }

                if x0 >= 0.29 {
                    // was 0.3, PR#13786
                    break 'block Label::W1Bpser;
                }

                if x0 < 0.1 && (x0 * b0).powf(a0) <= 0.7 {
                    break 'block Label::WBpser;
                }

                if b0 > 15.0 {
                    *w1 = 0.0;
                    skip_bup = true; // goto L131
                }
            } else {
                // a, b <= 1
                if a0 >= min(0.2, b0) {
                    break 'block Label::WBpser;
                }

                if x0.powf(a0) <= 0.9 {
                    break 'block Label::WBpser;
                }

                if x0 >= 0.3 {
                    break 'block Label::W1Bpser;
                }
            }
            if !skip_bup {
                n = 20; // goto L130;
                *w1 = bup(b0, a0, y0, x0, n, eps, false);
                did_bup = true;
                b0 += f64::from(n);
            }
            // L131:
            bgrat(b0, a0, y0, x0, w1, 15.0 * eps, &mut ierr1, false);
            if *w1 == 0.0 || (0.0 < *w1 && *w1 < f64::MIN_POSITIVE) {
                // w1=0 or very close: "almost surely" from underflow, try more: [2013-03-04]
                // FIXME: it is even better to do this in bgrat *directly* at least for
                // the case !did_bup, i.e., where *w1 = (0 or -Inf) on entry
                if did_bup {
                    // re-do that part on log scale:
                    *w1 = bup(b0 - f64::from(n), a0, y0, x0, n, eps, true);
                } else {
                    *w1 = f64::NEG_INFINITY; // = 0 on log-scale
                }
                bgrat(b0, a0, y0, x0, w1, 15.0 * eps, &mut ierr1, true);
                if ierr1 != 0 {
                    *ierr = 10 + ierr1;
                }
                break 'block Label::EndFromW1Log;
            }
            // else
            if ierr1 != 0 {
                *ierr = 10 + ierr1;
            }
            // (C warns here if *w1 < 0.)
            Label::EndFromW1
        };
    } else {
        // L30: -------------------- both  a, b > 1  {a0 > 1  &  b0 > 1} ---

        // lambda := a y - b x  =  (a + b)y - b  =  a - (a+b)x    {using x + y == 1},
        // ------ using the numerically best version :
        lambda = if (a + b).is_finite() {
            if a > b {
                (a + b) * y - b
            } else {
                a - (a + b) * x
            }
        } else {
            a * y - b * x
        };
        do_swap = lambda < 0.0;
        if do_swap {
            lambda = -lambda;
            a0 = b;
            x0 = y;
            b0 = a;
            y0 = x;
        } else {
            a0 = a;
            x0 = x;
            b0 = b;
            y0 = y;
        }

        label = 'block: {
            if b0 < 40.0 {
                if b0 * x0 <= 0.7 || (log_p && lambda > 650.0) {
                    // << added 2010-03; svn r51327
                    break 'block Label::WBpser;
                } else {
                    break 'block Label::L140;
                }
            } else if a0 > b0 {
                // ----  a0 > b0 >= 40  ----
                if b0 <= 100.0 || lambda > b0 * 0.03 {
                    break 'block Label::Bfrac;
                }
            } else if a0 <= 100.0 {
                // a0 <= 100; a0 <= b0 >= 40;
                break 'block Label::Bfrac;
            } else if lambda > a0 * 0.03 {
                // b0 >= a0 > 100; lambda > a0 * 0.03
                break 'block Label::Bfrac;
            }

            // else if none of the above    L180:
            *w = basym(a0, b0, lambda, eps * 100.0, log_p);
            *w1 = if log_p {
                r_log1_exp(*w)
            } else {
                0.5 - *w + 0.5
            };
            Label::End
        };
    }

    // EVALUATION OF THE APPROPRIATE ALGORITHM
    loop {
        match label {
            Label::WBpser => {
                // was L100
                *w = bpser(a0, b0, x0, eps, log_p);
                *w1 = if log_p {
                    r_log1_exp(*w)
                } else {
                    0.5 - *w + 0.5
                };
                label = Label::End;
            }
            Label::W1Bpser => {
                // was L110
                *w1 = bpser(b0, a0, y0, eps, log_p);
                *w = if log_p {
                    r_log1_exp(*w1)
                } else {
                    0.5 - *w1 + 0.5
                };
                label = Label::End;
            }
            Label::Bfrac => {
                *w = bfrac(a0, b0, x0, y0, lambda, eps * 15.0, log_p);
                *w1 = if log_p {
                    r_log1_exp(*w)
                } else {
                    0.5 - *w + 0.5
                };
                label = Label::End;
            }
            Label::L140 => {
                // b0 := fractional_part( b0 )  in (0, 1]
                n = b0 as i32;
                b0 -= f64::from(n);
                if b0 == 0.0 {
                    n -= 1;
                    b0 = 1.0;
                }

                *w = bup(b0, a0, y0, x0, n, eps, false);

                if *w < f64::MIN_POSITIVE && log_p {
                    // do not believe it; try bpser() :
                    // revert:
                    b0 += f64::from(n);
                    // which is only valid if b0 <= 1 || b0*x0 <= 0.7
                    label = Label::WBpser;
                    continue;
                }
                // else :
                if x0 <= 0.7 {
                    // log_p :  TODO:  w = bup(.) + bpser(.)  -- not so easy to use log-scale
                    *w += bpser(a0, b0, x0, eps, /* log_p = */ false);
                    label = Label::EndFromW;
                    continue;
                }
                // L150:
                if a0 <= 15.0 {
                    n = 20;
                    *w += bup(a0, b0, x0, y0, n, eps, false);
                    a0 += f64::from(n);
                }
                bgrat(a0, b0, x0, y0, w, 15.0 * eps, &mut ierr1, false);
                if ierr1 != 0 {
                    *ierr = 10 + ierr1;
                }
                label = Label::EndFromW;
            }
            Label::EndFromW => {
                if log_p {
                    *w1 = (-*w).ln_1p();
                    *w = w.ln();
                } else {
                    *w1 = 0.5 - *w + 0.5;
                }
                label = Label::End;
            }
            Label::EndFromW1 => {
                if log_p {
                    *w = (-*w1).ln_1p();
                    *w1 = w1.ln();
                } else {
                    *w = 0.5 - *w1 + 0.5;
                }
                label = Label::End;
            }
            Label::EndFromW1Log => {
                // *w1 = log(w1) already; w = 1 - w1  ==> log(w) = log(1 - w1) = log(1 - exp(*w1))
                if log_p {
                    *w = r_log1_exp(*w1);
                } else {
                    *w = -w1.exp_m1(); // 1 - exp(*w1)
                    *w1 = w1.exp();
                }
                label = Label::End;
            }
            Label::End => {
                if do_swap {
                    std::mem::swap(w, w1);
                }
                return;
            }
        }
    }
}
