//! glibc 2.34 `lgamma` for positive arguments (`sysdeps/ieee754/dbl-64/e_lgamma_r.c`).
//!
//! edgeR's C (`compute_apl.c`) calls the C library's `lgamma`, not R's `lgammafn`, so matching
//! the reference image bit for bit needs glibc's fdlibm-derived algorithm. glibc builds this file
//! without FMA; its internal `__ieee754_log` is the FMA-dispatched log ported in [`crate::glibm`].
//! Negative arguments are not ported and fall back to [`crate::nmath::lgammafn`].

#![allow(clippy::excessive_precision, clippy::unreadable_literal)]

use crate::glibm::ln;

const A: [f64; 12] = [
    7.72156649015328655494e-02,
    3.22467033424113591611e-01,
    6.73523010531292681824e-02,
    2.05808084325167332806e-02,
    7.38555086081402883957e-03,
    2.89051383673415629091e-03,
    1.19270763183362067845e-03,
    5.10069792153511336608e-04,
    2.20862790713908385557e-04,
    1.08011567247583939954e-04,
    2.52144565451257326939e-05,
    4.48640949618915160150e-05,
];
const TC: f64 = 1.46163214496836224576e+00;
const TF: f64 = -1.21486290535849611461e-01;
const TT: f64 = -3.63867699703950536541e-18;
const T: [f64; 15] = [
    4.83836122723810047042e-01,
    -1.47587722994593911752e-01,
    6.46249402391333854778e-02,
    -3.27885410759859649565e-02,
    1.79706750811820387126e-02,
    -1.03142241298341437450e-02,
    6.10053870246291332635e-03,
    -3.68452016781138256760e-03,
    2.25964780900612472250e-03,
    -1.40346469989232843813e-03,
    8.81081882437654011382e-04,
    -5.38595305356740546715e-04,
    3.15632070903625950361e-04,
    -3.12754168375120860518e-04,
    3.35529192635519073543e-04,
];
const U: [f64; 6] = [
    -7.72156649015328655494e-02,
    6.32827064025093366517e-01,
    1.45492250137234768737e+00,
    9.77717527963372745603e-01,
    2.28963728064692451092e-01,
    1.33810918536787660377e-02,
];
const V: [f64; 6] = [
    0.0,
    2.45597793713041134822e+00,
    2.12848976379893395361e+00,
    7.69285150456672783825e-01,
    1.04222645593369134254e-01,
    3.21709242282423911810e-03,
];
const S: [f64; 7] = [
    -7.72156649015328655494e-02,
    2.14982415960608852501e-01,
    3.25778796408930981787e-01,
    1.46350472652464452805e-01,
    2.66422703033638609560e-02,
    1.84028451407337715652e-03,
    3.19475326584100867617e-05,
];
const R: [f64; 7] = [
    0.0,
    1.39200533467621045958e+00,
    7.21935547567138069525e-01,
    1.71933865632803078993e-01,
    1.86459191715652901344e-02,
    7.77942496381893596434e-04,
    7.32668430744625636189e-06,
];
const W: [f64; 7] = [
    4.18938533204672725052e-01,
    8.33333333333329678849e-02,
    -2.77777777728775536470e-03,
    7.93650558643019558500e-04,
    -5.95187557450339963135e-04,
    8.36339918996282139126e-04,
    -1.63092934096575273989e-03,
];

/// `lgamma(x)` as glibc 2.34 computes it, for `x > 0`.
pub fn lgamma(x: f64) -> f64 {
    let bits = x.to_bits();
    let hx = (bits >> 32) as u32 as i32;
    let lx = bits as u32;
    let ix = hx & 0x7fff_ffff;
    if ix >= 0x7ff0_0000 {
        return x * x;
    }
    if hx < 0 || (ix as u32 | lx) == 0 {
        return crate::nmath::lgammafn(x);
    }
    if ix < 0x3b90_0000 {
        return -ln(x);
    }
    let r;
    if ((ix - 0x3ff0_0000) as u32 | lx) == 0 || ((ix - 0x4000_0000) as u32 | lx) == 0 {
        r = 0.0;
    } else if ix < 0x4000_0000 {
        let (mut rr, y, i);
        if ix <= 0x3fec_cccc {
            rr = -ln(x);
            if ix >= 0x3FE7_6944 {
                y = 1.0 - x;
                i = 0;
            } else if ix >= 0x3FCD_A661 {
                y = x - (TC - 1.0);
                i = 1;
            } else {
                y = x;
                i = 2;
            }
        } else {
            rr = 0.0;
            if ix >= 0x3FFB_B4C3 {
                y = 2.0 - x;
                i = 0;
            } else if ix >= 0x3FF3_B4C4 {
                y = x - TC;
                i = 1;
            } else {
                y = x - 1.0;
                i = 2;
            }
        }
        match i {
            0 => {
                let z = y * y;
                let p1 = A[0] + z * (A[2] + z * (A[4] + z * (A[6] + z * (A[8] + z * A[10]))));
                let p2 = z * (A[1] + z * (A[3] + z * (A[5] + z * (A[7] + z * (A[9] + z * A[11])))));
                let p = y * p1 + p2;
                rr += p - 0.5 * y;
            }
            1 => {
                let z = y * y;
                let w = z * y;
                let p1 = T[0] + w * (T[3] + w * (T[6] + w * (T[9] + w * T[12])));
                let p2 = T[1] + w * (T[4] + w * (T[7] + w * (T[10] + w * T[13])));
                let p3 = T[2] + w * (T[5] + w * (T[8] + w * (T[11] + w * T[14])));
                let p = z * p1 - (TT - w * (p2 + y * p3));
                rr += TF + p;
            }
            _ => {
                let p1 = y * (U[0] + y * (U[1] + y * (U[2] + y * (U[3] + y * (U[4] + y * U[5])))));
                let p2 = 1.0 + y * (V[1] + y * (V[2] + y * (V[3] + y * (V[4] + y * V[5]))));
                rr += -0.5 * y + p1 / p2;
            }
        }
        r = rr;
    } else if ix < 0x4020_0000 {
        let i = x as i32;
        let y = x - f64::from(i);
        let p =
            y * (S[0] + y * (S[1] + y * (S[2] + y * (S[3] + y * (S[4] + y * (S[5] + y * S[6]))))));
        let q = 1.0 + y * (R[1] + y * (R[2] + y * (R[3] + y * (R[4] + y * (R[5] + y * R[6])))));
        let mut rr = 0.5 * y + p / q;
        let mut z = 1.0;
        if i >= 3 {
            for k in (2..i).rev() {
                z *= y + f64::from(k);
            }
            rr += ln(z);
        }
        r = rr;
    } else if ix < 0x4390_0000 {
        let t = ln(x);
        let z = 1.0 / x;
        let y = z * z;
        let w = W[0] + z * (W[1] + y * (W[2] + y * (W[3] + y * (W[4] + y * (W[5] + y * W[6])))));
        r = (x - 0.5) * (t - 1.0) + w;
    } else {
        r = x * (ln(x) - 1.0);
    }
    r
}
