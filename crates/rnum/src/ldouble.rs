//! R's `long double` accumulators as they run on x86-64, where `LDOUBLE` is the x87 80-bit
//! extended type (64-bit significand, round to nearest even). `sum()` (`rsum`) and `mean()`
//! (`real_mean`) in R's `src/main/summary.c` accumulate in it; on arm64 it is plain `double`,
//! which is what [`crate::linalg::mean`] follows. Only finite, normal-range values are handled
//! (no overflow or subnormal results), which is all these sums see.

/// A finite x87 extended value `(-1)^neg * m * 2^e`, `m` normalised (top bit set) or zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ld {
    neg: bool,
    m: u64,
    e: i32,
}

impl Ld {
    pub const ZERO: Ld = Ld {
        neg: false,
        m: 0,
        e: 0,
    };

    pub fn from_f64(x: f64) -> Ld {
        if x == 0.0 {
            return Ld {
                neg: x.is_sign_negative(),
                ..Ld::ZERO
            };
        }
        let b = x.to_bits();
        let neg = b >> 63 != 0;
        let ex = ((b >> 52) & 0x7ff) as i32;
        let fr = b & ((1u64 << 52) - 1);
        let (m, e) = if ex == 0 {
            (fr, -1074)
        } else {
            (fr | (1u64 << 52), ex - 1075)
        };
        let z = m.leading_zeros() as i32;
        Ld {
            neg,
            m: m << z,
            e: e - z,
        }
    }

    pub fn from_u64(n: u64) -> Ld {
        if n == 0 {
            return Ld::ZERO;
        }
        let z = n.leading_zeros() as i32;
        Ld {
            neg: false,
            m: n << z,
            e: -z,
        }
    }

    /// `(double) s`: round the 64-bit significand to 53 bits, to nearest even.
    pub fn to_f64(self) -> f64 {
        if self.m == 0 {
            return if self.neg { -0.0 } else { 0.0 };
        }
        let mut m53 = self.m >> 11;
        let rem = self.m & 0x7ff;
        if rem > 0x400 || (rem == 0x400 && m53 & 1 == 1) {
            m53 += 1;
        }
        let v = m53 as f64 * 2f64.powi(self.e + 11);
        if self.neg {
            -v
        } else {
            v
        }
    }

    pub fn neg(self) -> Ld {
        Ld {
            neg: !self.neg,
            ..self
        }
    }

    /// Round the value `n * 2^e` (`n` nonzero) to a 64-bit significand.
    fn round(neg: bool, n: u128, e: i32) -> Ld {
        let p = 127 - n.leading_zeros() as i32; // position of the leading bit
        if p <= 63 {
            let m = (n as u64) << (63 - p);
            return Ld {
                neg,
                m,
                e: e - (63 - p),
            };
        }
        let sh = (p - 63) as u32;
        let mut m = (n >> sh) as u64;
        let rem = n & ((1u128 << sh) - 1);
        let half = 1u128 << (sh - 1);
        let mut e = e + sh as i32;
        if rem > half || (rem == half && m & 1 == 1) {
            m = m.wrapping_add(1);
            if m == 0 {
                m = 1u64 << 63;
                e += 1;
            }
        }
        Ld { neg, m, e }
    }

    pub fn add(self, o: Ld) -> Ld {
        if o.m == 0 {
            return if self.m == 0 {
                Ld {
                    neg: self.neg && o.neg,
                    ..Ld::ZERO
                }
            } else {
                self
            };
        }
        if self.m == 0 {
            return o;
        }
        let (a, b) = if (self.e, self.m) >= (o.e, o.m) {
            (self, o)
        } else {
            (o, self)
        };
        let big = (a.m as u128) << 62;
        let d = (a.e - b.e) as u32;
        let small = if d >= 126 {
            1 // sticky only
        } else {
            let s = (b.m as u128) << 62;
            let v = s >> d;
            v | ((v << d != s) as u128)
        };
        let e = a.e - 62;
        if a.neg == b.neg {
            Ld::round(a.neg, big + small, e)
        } else {
            let n = big - small;
            if n == 0 {
                return Ld::ZERO;
            }
            Ld::round(a.neg, n, e)
        }
    }

    pub fn sub(self, o: Ld) -> Ld {
        self.add(o.neg())
    }

    pub fn div(self, o: Ld) -> Ld {
        if self.m == 0 {
            return Ld {
                neg: self.neg != o.neg,
                ..Ld::ZERO
            };
        }
        let num = (self.m as u128) << 64;
        let q = num / o.m as u128;
        let r = num % o.m as u128;
        let neg = self.neg != o.neg;
        // Fold the next quotient bit and the sticky remainder into a 2-bit tail.
        let (n, e) = if q >> 64 != 0 {
            ((q << 1) | (r != 0) as u128, self.e - o.e - 65)
        } else {
            let r2 = r << 1;
            let bit = (r2 >= o.m as u128) as u128;
            let rest = r2 - bit * o.m as u128;
            ((q << 2) | (bit << 1) | (rest != 0) as u128, self.e - o.e - 66)
        };
        Ld::round(neg, n, e)
    }
}

/// R's `sum()` of doubles (`rsum`, no NA removal).
pub fn sum(x: &[f64]) -> f64 {
    let mut s = Ld::ZERO;
    for &v in x {
        s = s.add(Ld::from_f64(v));
    }
    s.to_f64()
}

/// One row of R's `rowMeans()` (no NA removal): the extended sum divided by `n` in extended
/// precision, rounded once.
pub fn row_mean(x: &[f64]) -> f64 {
    let mut s = Ld::ZERO;
    for &v in x {
        s = s.add(Ld::from_f64(v));
    }
    s.div(Ld::from_u64(x.len() as u64)).to_f64()
}

/// R's `mean()` of doubles (`real_mean`): an extended sum divided by `n`, then one correction
/// pass. Non-finite inputs fall back to plain `double` arithmetic.
pub fn mean(x: &[f64]) -> f64 {
    if x.iter().any(|v| !v.is_finite()) {
        return crate::linalg::mean(x);
    }
    let n = Ld::from_u64(x.len() as u64);
    let mut s = Ld::ZERO;
    for &v in x {
        s = s.add(Ld::from_f64(v));
    }
    s = s.div(n);
    let mut t = Ld::ZERO;
    for &v in x {
        t = t.add(Ld::from_f64(v).sub(s));
    }
    s.add(t.div(n)).to_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extended_sums_keep_bits_double_loses() {
        // 1 + 2^-60 is representable in 64 bits but not 53.
        let a = Ld::from_f64(1.0).add(Ld::from_f64(2f64.powi(-60)));
        assert_eq!(a.sub(Ld::from_f64(1.0)).to_f64(), 2f64.powi(-60));
        assert_eq!(sum(&[1e16, 1.0, -1e16]), 1.0);
        assert_eq!(mean(&[1.0, 2.0, 4.0]), 7.0 / 3.0);
        assert_eq!(Ld::from_f64(1.0).div(Ld::from_u64(3)).to_f64(), 1.0 / 3.0);
    }
}
