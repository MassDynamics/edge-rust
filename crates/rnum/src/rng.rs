//! R's default random number generators, ported from R 4.5.0.
//!
//! Only the default kinds are ported: `RNGkind("Mersenne-Twister", "Inversion", "Rejection")`.
//! Sources:
//! - `src/main/RNG.c`: `RNG_Init` (the `set.seed` scrambling), `FixupSeeds`, `MT_sgenrand`,
//!   `MT_genrand`, `fixup`, `unif_rand`.
//! - `src/nmath/snorm.c`: `norm_rand`, `INVERSION` branch (the `BIG = 2^27` trick).
//! - `src/nmath/sexp.c`: `exp_rand`.
//! - `src/nmath/rgamma.c`: `rgamma` (GS for `a < 1`, GD for `a >= 1`).
//! - `src/nmath/rnorm.c`, `src/nmath/rchisq.c`.
//!
//! The state lives in an [`RRng`] value instead of R's globals, so a caller writes
//! `let mut rng = RRng::set_seed(2);` where R code says `set.seed(2)`, and every draw
//! afterwards consumes the stream in the same order R does. DESeq2's
//! `estimateDispersionsPriorVar` (m - p <= 3) is the consumer; see `deseq2-core`.

use crate::nmath::qnorm;

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;
const TEMPERING_MASK_B: u32 = 0x9d2c_5680;
const TEMPERING_MASK_C: u32 = 0xefc6_0000;
/// `i2_32m1` (RNG.c): 1 / (2^32 - 1).
const I2_32M1: f64 = 2.328306437080797e-10;

/// R's Mersenne-Twister state: `dummy[0]` is `mti`, `dummy[1..=624]` is `mt`.
#[derive(Clone, Debug)]
pub struct RRng {
    mti: usize,
    mt: [u32; N],
}

impl RRng {
    /// `set.seed(seed)` with the default kinds: `RNG_Init(MERSENNE_TWISTER, seed)`.
    ///
    /// The seed is scrambled by 50 rounds of the LCG `69069 * seed + 1`, then the 625 seed
    /// words (`mti` first, for historical consistency) are filled by further rounds, and
    /// `FixupSeeds(.., initial = 1)` sets `mti = 624`.
    pub fn set_seed(seed: i32) -> Self {
        let mut s = seed as u32;
        for _ in 0..50 {
            s = s.wrapping_mul(69069).wrapping_add(1);
        }
        let mut words = [0u32; N + 1];
        for w in words.iter_mut() {
            s = s.wrapping_mul(69069).wrapping_add(1);
            *w = s;
        }
        let mut mt = [0u32; N];
        mt.copy_from_slice(&words[1..]);
        // FixupSeeds(initial = 1): I1 = 624. The all-zero check cannot trigger from an LCG fill.
        RRng { mti: N, mt }
    }

    /// `MT_genrand` (RNG.c): one tempered 32-bit word scaled to [0, 1).
    fn mt_genrand(&mut self) -> f64 {
        const MAG01: [u32; 2] = [0, MATRIX_A];
        if self.mti >= N {
            // mti == N + 1 (never seeded) cannot occur: construction always seeds.
            let mt = &mut self.mt;
            let mut kk = 0;
            while kk < N - M {
                let y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
                mt[kk] = mt[kk + M] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
                kk += 1;
            }
            while kk < N - 1 {
                let y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
                mt[kk] = mt[kk + M - N] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
                kk += 1;
            }
            let y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
            mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & TEMPERING_MASK_B;
        y ^= (y << 15) & TEMPERING_MASK_C;
        y ^= y >> 18;
        (y as f64) * 2.3283064365386963e-10
    }

    /// `unif_rand` (RNG.c) for Mersenne-Twister: `fixup(MT_genrand())`, never 0 or 1.
    pub fn unif_rand(&mut self) -> f64 {
        let x = self.mt_genrand();
        if x <= 0.0 {
            return 0.5 * I2_32M1;
        }
        if (1.0 - x) <= 0.0 {
            return 1.0 - 0.5 * I2_32M1;
        }
        x
    }

    /// `norm_rand` (snorm.c), `INVERSION`: two uniforms combined at 2^27 resolution, then
    /// `qnorm`.
    pub fn norm_rand(&mut self) -> f64 {
        const BIG: f64 = 134_217_728.0;
        let u1 = self.unif_rand();
        let u1 = ((BIG * u1) as i32) as f64 + self.unif_rand();
        qnorm(u1 / BIG, 0.0, 1.0, true, false)
    }

    /// `exp_rand` (sexp.c): Ahrens and Dieter (1972) standard exponential.
    pub fn exp_rand(&mut self) -> f64 {
        const Q: [f64; 16] = [
            0.6931471805599453,
            0.9333736875190459,
            0.9888777961838675,
            0.9984959252914960,
            0.9998292811061389,
            0.9999833164100727,
            0.9999985691438767,
            0.9999998906925558,
            0.9999999924734159,
            0.9999999995283275,
            0.9999999999728814,
            0.9999999999985598,
            0.9999999999999289,
            0.9999999999999968,
            0.9999999999999999,
            1.0000000000000000,
        ];
        let mut a = 0.0;
        let mut u = self.unif_rand();
        while u <= 0.0 || u >= 1.0 {
            u = self.unif_rand();
        }
        loop {
            u += u;
            if u > 1.0 {
                break;
            }
            a += Q[0];
        }
        u -= 1.0;
        if u <= Q[0] {
            return a + u;
        }
        let mut i = 0;
        let mut umin = self.unif_rand();
        loop {
            let ustar = self.unif_rand();
            if umin > ustar {
                umin = ustar;
            }
            i += 1;
            if u <= Q[i] {
                break;
            }
        }
        a + umin * Q[0]
    }

    /// `rnorm(mu, sigma)` (rnorm.c). `sigma == 0` returns `mu` without consuming a draw.
    pub fn rnorm(&mut self, mu: f64, sigma: f64) -> f64 {
        if mu.is_nan() || !sigma.is_finite() || sigma < 0.0 {
            return f64::NAN;
        }
        if sigma == 0.0 || !mu.is_finite() {
            mu
        } else {
            mu + sigma * self.norm_rand()
        }
    }

    /// `rchisq(df)` (rchisq.c): `rgamma(df / 2, 2)`.
    pub fn rchisq(&mut self, df: f64) -> f64 {
        if !df.is_finite() || df < 0.0 {
            return f64::NAN;
        }
        self.rgamma(df / 2.0, 2.0)
    }

    /// `rgamma(a, scale)` (rgamma.c). R caches the `a`-dependent constants in statics; they
    /// are recomputed here on every call, which gives the same values.
    pub fn rgamma(&mut self, a: f64, scale: f64) -> f64 {
        const SQRT32: f64 = 5.656854;
        const EXP_M1: f64 = 0.367_879_441_171_442_33;
        const Q1: f64 = 0.04166669;
        const Q2: f64 = 0.02083148;
        const Q3: f64 = 0.00801191;
        const Q4: f64 = 0.00144121;
        const Q5: f64 = -7.388e-5;
        const Q6: f64 = 2.4511e-4;
        const Q7: f64 = 2.424e-4;
        const A1: f64 = 0.3333333;
        const A2: f64 = -0.250003;
        const A3: f64 = 0.2000062;
        const A4: f64 = -0.1662921;
        const A5: f64 = 0.1423657;
        const A6: f64 = -0.1367177;
        const A7: f64 = 0.1233795;

        if a.is_nan() || scale.is_nan() {
            return f64::NAN;
        }
        if a <= 0.0 || scale <= 0.0 {
            if scale == 0.0 || a == 0.0 {
                return 0.0;
            }
            return f64::NAN;
        }
        if !a.is_finite() || !scale.is_finite() {
            return f64::INFINITY;
        }

        if a < 1.0 {
            // GS algorithm for parameters a < 1.
            let e = 1.0 + EXP_M1 * a;
            let x;
            loop {
                let p = e * self.unif_rand();
                if p >= 1.0 {
                    let xx = -((e - p) / a).ln();
                    if self.exp_rand() >= (1.0 - a) * xx.ln() {
                        x = xx;
                        break;
                    }
                } else {
                    let xx = (p.ln() / a).exp();
                    if self.exp_rand() >= xx {
                        x = xx;
                        break;
                    }
                }
            }
            return scale * x;
        }

        // GD algorithm, a >= 1.
        let s2 = a - 0.5;
        let s = s2.sqrt();
        let d = SQRT32 - s * 12.0;

        let mut t = self.norm_rand();
        let mut x = s + 0.5 * t;
        let ret_val = x * x;
        if t >= 0.0 {
            return scale * ret_val;
        }

        let mut u = self.unif_rand();
        if d * u <= t * t * t {
            return scale * ret_val;
        }

        let r = 1.0 / a;
        let q0 = ((((((Q7 * r + Q6) * r + Q5) * r + Q4) * r + Q3) * r + Q2) * r + Q1) * r;
        let (b, si, c) = if a <= 3.686 {
            (0.463 + s + 0.178 * s2, 1.235, 0.195 / s - 0.079 + 0.16 * s)
        } else if a <= 13.022 {
            (1.654 + 0.0076 * s2, 1.68 / s + 0.275, 0.062 / s + 0.024)
        } else {
            (1.77, 0.75, 0.1515 / s)
        };

        let qfun = |t: f64, v: f64| -> f64 {
            if v.abs() <= 0.25 {
                q0 + 0.5
                    * t
                    * t
                    * ((((((A7 * v + A6) * v + A5) * v + A4) * v + A3) * v + A2) * v + A1)
                    * v
            } else {
                q0 - s * t + 0.25 * t * t + (s2 + s2) * (1.0 + v).ln()
            }
        };

        if x > 0.0 {
            let v = t / (s + s);
            let q = qfun(t, v);
            if (1.0 - u).ln() <= q {
                return scale * ret_val;
            }
        }

        loop {
            let e = self.exp_rand();
            u = self.unif_rand();
            u = u + u - 1.0;
            if u < 0.0 {
                t = b - si * e;
            } else {
                t = b + si * e;
            }
            if t >= -0.71874483771719 {
                let v = t / (s + s);
                let q = qfun(t, v);
                if q > 0.0 {
                    let w = q.exp_m1();
                    if c * u.abs() <= w * (e - 0.5 * t * t).exp() {
                        break;
                    }
                }
            }
        }
        x = s + 0.5 * t;
        scale * x * x
    }
}
