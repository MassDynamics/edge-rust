//! R 4.5's `as.character()` of a double, as it runs on x86-64: `coerceToString` sets
//! `R_print.digits` to 15 and calls `StringFromReal` (`src/main/printutils.c`), which is
//! `formatReal` (`src/main/format.c`) on the one value followed by `EncodeRealDrop0`.
//!
//! `formatReal` counts significant digits with `scientific()`, which scales the value by a power
//! of ten in x87 long double ([`Ld`]) and rounds it with `nearbyintl`. A correctly rounded
//! 15-digit shortcut disagrees with R on near-ties at the 15th digit (`-3.161245995276595` is
//! `"-3.1612459952766"` in R), so the scaling is emulated. R then prints fixed notation when it
//! is no wider than scientific (`scipen = 0`), with `sprintf`'s digits, and drops trailing zeros:
//! `1e5` is `"1e+05"`, `110000` is `"110000"` and `1234567890123456` is printed in full.
//! Checked against the image's R 4.5.0 (`tests/r_as_character.rs`).

use crate::ldouble::Ld;

const TBL: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];
const KP_MAX: i32 = 22;
const DIGITS: i32 = 15; // DBL_DIG

/// `scientific()` at 15 digits for `r > 0`: `(nsig, kpower, roundingwidens)`.
fn scientific(r: f64) -> (i32, i32, bool) {
    let mut kp = r.log10().floor() as i32 - DIGITS + 1;
    let x = Ld::from_f64(r);
    let mut r_prec = if kp.abs() <= KP_MAX {
        if kp > 0 {
            x.div(Ld::from_f64(TBL[kp as usize]))
        } else if kp < 0 {
            x.mul(Ld::from_f64(TBL[(-kp) as usize]))
        } else {
            x
        }
    } else {
        x.div(Ld::pow10(kp))
    };
    if r_prec.lt_pos(Ld::from_f64(TBL[(DIGITS - 1) as usize])) {
        r_prec = r_prec.mul(Ld::from_u64(10));
        kp -= 1;
    }
    let mut alpha = r_prec.nearbyint();
    let mut nsig = DIGITS;
    for _ in 0..DIGITS {
        if alpha % 10 != 0 {
            break;
        }
        alpha /= 10;
        nsig -= 1;
    }
    if nsig == 0 {
        nsig = 1;
        kp += 1;
    }
    let kpower = kp + DIGITS - 1;
    // Fixed notation can round up into one more integer digit than scientific shows.
    let rgt = (DIGITS - kpower).clamp(0, KP_MAX);
    let fuzz = 0.5 / TBL[rgt as usize];
    let widens = kpower > 0 && kpower <= KP_MAX && r < TBL[kpower as usize] - fuzz;
    (nsig, kpower, widens)
}

/// R's `as.character(x)` for a double. NaN is `"NaN"`: callers map NA themselves.
pub fn r_as_character(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Inf".into() } else { "-Inf".into() };
    }
    if x == 0.0 {
        return "0".into(); // -0 too
    }
    let neg = (x < 0.0) as i32;
    let (nsig, kpower, widens) = scientific(x.abs());
    let left = kpower + 1 - widens as i32;
    let right = (nsig - left).max(0);
    let w_fixed = neg + left.max(1) + right + (right != 0) as i32;
    let d = nsig - 1;
    let e = if left > 100 || left <= -99 { 2 } else { 1 };
    let w_sci = neg + (d > 0) as i32 + d + 4 + e;
    let s = if w_fixed <= w_sci {
        format!("{:.*}", right as usize, x)
    } else {
        let s = format!("{:.*e}", d as usize, x);
        let (mant, exp) = s.split_once('e').unwrap();
        let exp: i32 = exp.parse().unwrap();
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mant}e{sign}{:02}", exp.abs())
    };
    drop_trailing_zeros(&s)
}

/// `EncodeRealDrop0`: drop trailing zeros after the decimal point, and the point when nothing
/// is left after it, keeping any exponent.
fn drop_trailing_zeros(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let (mant, exp) = s.split_at(s.find('e').unwrap_or(s.len()));
    format!("{}{exp}", mant.trim_end_matches('0').trim_end_matches('.'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r_picks_scientific_when_narrower() {
        let cases = [
            (1e5, "1e+05"),
            (110000.0, "110000"),
            (1e-4, "1e-04"),
            (0.00012, "0.00012"),
            (1e15, "1e+15"),
            (1234567890123456.0, "1234567890123456"),
            (99999.99999999999, "1e+05"),
            (0.1 + 0.2, "0.3"),
            (-0.0, "0"),
            (-1e5, "-1e+05"),
            (100000.5, "100000.5"),
            (1e-100, "1e-100"),
            (5e-324, "4.94065645841247e-324"),
            (-3.161245995276595, "-3.1612459952766"),
        ];
        for (x, want) in cases {
            assert_eq!(r_as_character(x), want, "{x:e}");
        }
    }
}
