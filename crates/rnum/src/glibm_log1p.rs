//! glibc 2.34 `log1p` (`sysdeps/ieee754/dbl-64/s_log1p.c`, fdlibm-derived). x86_64 has no FMA
//! ifunc variant for it, so the C is ported as written with no fused operations. Companion to
//! [`crate::glibm`]'s `exp` / `ln`; checked bit for bit against the reference R's `log1p`
//! (`tests/glibm_golden.rs`).

/// glibc `log1p` (`s_log1p.c`, fdlibm-derived; x86_64 has no FMA variant, so no fusing).
pub fn log1p(x: f64) -> f64 {
    const LN2_HI: f64 = 6.93147180369123816490e-01;
    const LN2_LO: f64 = 1.90821492927058770002e-10;
    const LP: [f64; 8] = [
        0.0,
        6.666666666666735130e-01,
        3.999999999940941908e-01,
        2.857142874366239149e-01,
        2.222219843214978396e-01,
        1.818357216161805012e-01,
        1.531383769920937332e-01,
        1.479819860511658591e-01,
    ];
    let high = |v: f64| (v.to_bits() >> 32) as u32 as i32;
    let set_high =
        |v: f64, h: i32| f64::from_bits(((h as u32 as u64) << 32) | (v.to_bits() & 0xffff_ffff));
    let hx = high(x);
    let ax = hx & 0x7fffffff;
    let mut k: i32 = 1;
    let mut f = 0.0;
    let mut hu: i32 = 0;
    let mut c = 0.0;
    if hx < 0x3FDA827A {
        if ax >= 0x3ff00000 {
            return if x == -1.0 {
                f64::NEG_INFINITY
            } else {
                f64::NAN
            };
        }
        if ax < 0x3e200000 {
            if ax < 0x3c900000 {
                return x;
            }
            return x - x * x * 0.5;
        }
        if hx > 0 || hx <= 0xbfd2bec3u32 as i32 {
            k = 0;
            f = x;
            hu = 1;
        }
    } else if hx >= 0x7ff00000 {
        return x + x;
    }
    if k != 0 {
        let mut u;
        if hx < 0x43400000 {
            u = 1.0 + x;
            hu = high(u);
            k = (hu >> 20) - 1023;
            c = if k > 0 { 1.0 - (u - x) } else { x - (u - 1.0) };
            c /= u;
        } else {
            u = x;
            hu = high(u);
            k = (hu >> 20) - 1023;
            c = 0.0;
        }
        hu &= 0x000fffff;
        if hu < 0x6a09e {
            u = set_high(u, hu | 0x3ff00000);
        } else {
            k += 1;
            u = set_high(u, hu | 0x3fe00000);
            hu = (0x00100000 - hu) >> 2;
        }
        f = u - 1.0;
    }
    let hfsq = 0.5 * f * f;
    let kf = k as f64;
    if hu == 0 {
        if f == 0.0 {
            if k == 0 {
                return 0.0;
            }
            c += kf * LN2_LO;
            return kf * LN2_HI + c;
        }
        let r = hfsq * (1.0 - 0.66666666666666666 * f);
        if k == 0 {
            return f - r;
        }
        return kf * LN2_HI - ((r - (kf * LN2_LO + c)) - f);
    }
    let s = f / (2.0 + f);
    let z = s * s;
    let r1 = z * LP[1];
    let z2 = z * z;
    let r2 = LP[2] + z * LP[3];
    let z4 = z2 * z2;
    let r3 = LP[4] + z * LP[5];
    let z6 = z4 * z2;
    let r4 = LP[6] + z * LP[7];
    let r = r1 + z2 * r2 + z4 * r3 + z6 * r4;
    if k == 0 {
        f - (hfsq - s * (hfsq + r))
    } else {
        kf * LN2_HI - ((hfsq - (s * (hfsq + r) + (kf * LN2_LO + c))) - f)
    }
}
