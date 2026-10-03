//! R's `long double` accumulators as they run on x86-64, where `LDOUBLE` is the x87 80-bit
//! extended type (64-bit significand, round to nearest even). `sum()` (`rsum`) and `mean()`
//! (`real_mean`) in R's `src/main/summary.c` accumulate in it (on arm64 it is plain `double`).
//! [`crate::linalg::mean`] routes finite input here. Only finite, normal-range values are handled
//! (no overflow or subnormal results), which is all these sums see.

/// A finite x87 extended value `(-1)^neg * m * 2^e`, `m` normalised (top bit set) or zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ld {
    neg: bool,
    m: u64,
    e: i32,
}

#[allow(clippy::should_implement_trait)]
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
            (
                (q << 2) | (bit << 1) | (rest != 0) as u128,
                self.e - o.e - 66,
            )
        };
        Ld::round(neg, n, e)
    }

    pub fn mul(self, o: Ld) -> Ld {
        let neg = self.neg != o.neg;
        if self.m == 0 || o.m == 0 {
            return Ld { neg, ..Ld::ZERO };
        }
        Ld::round(neg, self.m as u128 * o.m as u128, self.e + o.e)
    }

    /// `self < o` for non-negative values.
    pub fn lt_pos(self, o: Ld) -> bool {
        if self.m == 0 || o.m == 0 {
            return self.m < o.m;
        }
        (self.e, self.m) < (o.e, o.m)
    }

    /// `nearbyintl` of a non-negative value below 2^64 (ties to even).
    pub fn nearbyint(self) -> u64 {
        if self.m == 0 {
            return 0;
        }
        if self.e >= 0 {
            return self.m << self.e;
        }
        let sh = (-self.e) as u32;
        if sh > 64 {
            return 0; // below 0.5
        }
        let m = self.m as u128;
        let int = m >> sh;
        let rem = m & ((1u128 << sh) - 1);
        let half = 1u128 << (sh - 1);
        (int + (rem > half || (rem == half && int & 1 == 1)) as u128) as u64
    }

    /// `powl(10, k)` as glibc returns it on x86-64, taken as 10^k correctly rounded to 64 bits
    /// (exact for 0 <= k <= 27).
    pub fn pow10(k: i32) -> Ld {
        let j = k.unsigned_abs();
        let mut five = vec![1u64]; // 5^j, little-endian 64-bit limbs
        for _ in 0..j {
            let mut carry = 0u128;
            for l in five.iter_mut() {
                let v = *l as u128 * 5 + carry;
                *l = v as u64;
                carry = v >> 64;
            }
            if carry != 0 {
                five.push(carry as u64);
            }
        }
        let bits = big_bits(&five);
        if k >= 0 {
            // 5^j * 2^j: the top 66 bits, with everything below folded into a sticky bit.
            if bits <= 127 {
                return Ld::round(false, big_low_u128(&five), k);
            }
            let sh = bits - 66;
            let sticky = big_any_below(&five, sh);
            return Ld::round(
                false,
                big_shr_u128(&five, sh) | sticky as u128,
                k + sh as i32,
            );
        }
        // 1 / (5^j * 2^j): 66 quotient bits of 2^s / 5^j by binary long division.
        let s = bits + 65;
        let mut rem: Vec<u64> = vec![0; five.len() + 1];
        let mut q = 0u128;
        for i in (0..=s).rev() {
            big_shl1(&mut rem);
            if i == s {
                rem[0] |= 1;
            }
            let bit = !big_lt(&rem, &five);
            if bit {
                big_sub(&mut rem, &five);
            }
            q = (q << 1) | bit as u128;
        }
        let sticky = rem.iter().any(|&l| l != 0);
        Ld::round(false, q | sticky as u128, k - s as i32)
    }
}

fn big_bits(a: &[u64]) -> u32 {
    let top = a.iter().rposition(|&l| l != 0).unwrap_or(0);
    top as u32 * 64 + (64 - a[top].leading_zeros())
}

fn big_low_u128(a: &[u64]) -> u128 {
    a[0] as u128 | (*a.get(1).unwrap_or(&0) as u128) << 64
}

fn big_bit(a: &[u64], i: u32) -> bool {
    a.get((i / 64) as usize)
        .is_some_and(|l| (l >> (i % 64)) & 1 == 1)
}

/// `a >> sh`, where the result fits 128 bits.
fn big_shr_u128(a: &[u64], sh: u32) -> u128 {
    (0..128).fold(0u128, |acc, i| acc | ((big_bit(a, sh + i) as u128) << i))
}

/// Whether any of the low `sh` bits of `a` is set.
fn big_any_below(a: &[u64], sh: u32) -> bool {
    (0..sh).any(|i| big_bit(a, i))
}

fn big_shl1(a: &mut [u64]) {
    let mut carry = 0u64;
    for l in a.iter_mut() {
        let next = *l >> 63;
        *l = (*l << 1) | carry;
        carry = next;
    }
}

/// `a < b`; `a` may have more limbs than `b`.
fn big_lt(a: &[u64], b: &[u64]) -> bool {
    for i in (0..a.len().max(b.len())).rev() {
        let (x, y) = (*a.get(i).unwrap_or(&0), *b.get(i).unwrap_or(&0));
        if x != y {
            return x < y;
        }
    }
    false
}

/// `a -= b`, with `a >= b`.
fn big_sub(a: &mut [u64], b: &[u64]) {
    let mut borrow = false;
    for (i, l) in a.iter_mut().enumerate() {
        let (v, b1) = l.overflowing_sub(*b.get(i).unwrap_or(&0));
        let (v, b2) = v.overflowing_sub(borrow as u64);
        *l = v;
        borrow = b1 || b2;
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
