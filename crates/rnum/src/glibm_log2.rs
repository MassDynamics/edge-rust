//! glibc 2.34 `log2` (`sysdeps/ieee754/dbl-64/e_log2.c`). x86-64 glibc 2.34 has no FMA variant
//! of `log2`, so this follows the plain SSE2 build: no fused operations, the `tab2` split for
//! `r`, and C's left-to-right evaluation order. R's `log2` reaches it through `logbase`.
//! Negative, zero, subnormal and non-finite inputs fall back to [`f64::log2`]. Verified
//! bit-identical to the image's `log2` on 3e6 arguments.

#![allow(clippy::unreadable_literal)]

const fn d(bits: u64) -> f64 {
    f64::from_bits(bits)
}

const OFF: u64 = 0x3fe6000000000000;

const INV_LN2_HI: f64 = d(0x3ff7154765200000);
const INV_LN2_LO: f64 = d(0x3de705fc2eefa200);
const A: [f64; 6] = [
    d(0xbfe71547652b8339),
    d(0x3fdec709dc3a04be),
    d(0xbfd7154764702ffb),
    d(0x3fd2776c50034c48),
    d(0xbfcec7b328ea92bc),
    d(0x3fca6225e117f92e),
];
const B: [f64; 10] = [
    d(0xbfe71547652b82fe),
    d(0x3fdec709dc3a03f7),
    d(0xbfd71547652b7c3f),
    d(0x3fd2776c50f05be4),
    d(0xbfcec709dd768fe5),
    d(0x3fca61761ec4e736),
    d(0xbfc7153fbc64a79b),
    d(0x3fc484d154f01b4a),
    d(0xbfc289e4a72c383c),
    d(0x3fc0b32f285aee66),
];
/// `__log2_data`: (invc, logc, chi, clo) bits.
#[rustfmt::skip]
static LOG2_TAB: [(u64, u64, u64, u64); 64] = [
    (0x3ff724286bb1acf8, 0xbfe1095feecdb000, 0x3fe6200012b90a8e, 0x3c8904ab0644b605),
    (0x3ff6e1f766d2cca1, 0xbfe08494bd76d000, 0x3fe66000045734a6, 0x3c61ff9bea62f7a9),
    (0x3ff6a13d0e30d48a, 0xbfe00143aee8f800, 0x3fe69fffc325f2c5, 0x3c827ecfcb3c90ba),
    (0x3ff661ec32d06c85, 0xbfdefec5360b4000, 0x3fe6e00038b95a04, 0x3c88ff8856739326),
    (0x3ff623fa951198f8, 0xbfddfdd91ab7e000, 0x3fe71fffe09994e3, 0x3c8afd40275f82b1),
    (0x3ff5e75ba4cf026c, 0xbfdcffae0cc79000, 0x3fe7600015590e10, 0xbc72fd75b4238341),
    (0x3ff5ac055a214fb8, 0xbfdc043811fda000, 0x3fe7a00012655bd5, 0x3c7808e67c242b76),
    (0x3ff571ed0f166e1e, 0xbfdb0b67323ae000, 0x3fe7e0003259e9a6, 0xbc6208e426f622b7),
    (0x3ff53909590bf835, 0xbfda152f5a2db000, 0x3fe81fffedb4b2d2, 0xbc8402461ea5c92f),
    (0x3ff5014fed61addd, 0xbfd9217f5af86000, 0x3fe860002dfafcc3, 0x3c6df7f4a2f29a1f),
    (0x3ff4cab88e487bd0, 0xbfd8304db0719000, 0x3fe89ffff78c6b50, 0xbc8e0453094995fd),
    (0x3ff49539b4334fee, 0xbfd74189f9a9e000, 0x3fe8e00039671566, 0xbc8a04f3bec77b45),
    (0x3ff460cbdfafd569, 0xbfd6552bb5199000, 0x3fe91fffe2bf1745, 0xbc77fa34400e203c),
    (0x3ff42d664ee4b953, 0xbfd56b23a29b1000, 0x3fe95fffcc5c9fd1, 0xbc76ff8005a0695d),
    (0x3ff3fb01111dd8a6, 0xbfd483650f5fa000, 0x3fe9a0003bba4767, 0x3c70f8c4c4ec7e03),
    (0x3ff3c995b70c5836, 0xbfd39de937f6a000, 0x3fe9dfffe7b92da5, 0x3c8e7fd9478c4602),
    (0x3ff3991c4ab6fd4a, 0xbfd2baa1538d6000, 0x3fea1fffd72efdaf, 0xbc6a0c554dcdae7e),
    (0x3ff3698e0ce099b5, 0xbfd1d98340ca4000, 0x3fea5fffde04ff95, 0x3c867da98ce9b26b),
    (0x3ff33ae48213e7b2, 0xbfd0fa853a40e000, 0x3fea9fffca5e8d2b, 0xbc8284c9b54c13de),
    (0x3ff30d191985bdb1, 0xbfd01d9c32e73000, 0x3feadfffddad03ea, 0x3c5812c8ea602e3c),
    (0x3ff2e025cab271d7, 0xbfce857da2fa6000, 0x3feb1ffff10d3d4d, 0xbc8efaddad27789c),
    (0x3ff2b404cf13cd82, 0xbfccd3c8633d8000, 0x3feb5fffce21165a, 0x3c53cb1719c61237),
    (0x3ff288b02c7ccb50, 0xbfcb26034c14a000, 0x3feb9fffd950e674, 0x3c73f7d94194ce00),
    (0x3ff25e2263944de5, 0xbfc97c1c2f4fe000, 0x3febe000139ca8af, 0x3c750ac4215d9bc0),
    (0x3ff234563d8615b1, 0xbfc7d6023f800000, 0x3fec20005b46df99, 0x3c6beea653e9c1c9),
    (0x3ff20b46e33eaf38, 0xbfc633a71a05e000, 0x3fec600040b9f7ae, 0xbc7c079f274a70d6),
    (0x3ff1e2eefdcda3dd, 0xbfc494f5e9570000, 0x3feca0006255fd8a, 0xbc7a0b4076e84c1f),
    (0x3ff1bb4a580b3930, 0xbfc2f9e424e0a000, 0x3fecdfffd94c095d, 0x3c88f933f99ab5d7),
    (0x3ff19453847f2200, 0xbfc162595afdc000, 0x3fed1ffff975d6cf, 0xbc582c08665fe1be),
    (0x3ff16e06c0d5d73c, 0xbfbf9c9a75bd8000, 0x3fed5fffa2561c93, 0xbc7b04289bd295f3),
    (0x3ff1485f47b7e4c2, 0xbfbc7b575bf9c000, 0x3fed9fff9d228b0c, 0x3c870251340fa236),
    (0x3ff12358ad0085d1, 0xbfb960c60ff48000, 0x3fede00065bc7e16, 0xbc75011e16a4d80c),
    (0x3ff0fef00f532227, 0xbfb64ce247b60000, 0x3fee200002f64791, 0x3c89802f09ef62e0),
    (0x3ff0db2077d03a8f, 0xbfb33f78b2014000, 0x3fee600057d7a6d8, 0xbc7e0b75580cf7fa),
    (0x3ff0b7e6d65980d9, 0xbfb0387d1a42c000, 0x3feea00027edc00c, 0xbc8c848309459811),
    (0x3ff0953efe7b408d, 0xbfaa6f9208b50000, 0x3feee0006cf5cb7c, 0xbc8f8027951576f4),
    (0x3ff07325cac53b83, 0xbfa47a954f770000, 0x3fef2000782b7dcc, 0xbc8f81d97274538f),
    (0x3ff05197e40d1b5c, 0xbf9d23a8c50c0000, 0x3fef6000260c450a, 0xbc4071002727ffdc),
    (0x3ff03091c1208ea2, 0xbf916a2629780000, 0x3fef9fffe88cd533, 0xbc581bdce1fda8b0),
    (0x3ff0101025b37e21, 0xbf7720f8d8e80000, 0x3fefdfffd50f8689, 0x3c87f91acb918e6e),
    (0x3fefc07ef9caa76b, 0x3f86fe53b1500000, 0x3ff0200004292367, 0x3c9b7ff365324681),
    (0x3fef4465d3f6f184, 0x3fa11ccce10f8000, 0x3ff05fffe3e3d668, 0x3c86fa08ddae957b),
    (0x3feecc079f84107f, 0x3fac4dfc8c8b8000, 0x3ff0a0000a85a757, 0xbc57e2de80d3fb91),
    (0x3fee573a99975ae8, 0x3fb3aa321e574000, 0x3ff0e0001a5f3fcc, 0xbc91823305c5f014),
    (0x3fede5d6f0bd3de6, 0x3fb918a0d08b8000, 0x3ff11ffff8afbaf5, 0xbc8bfabb6680bac2),
    (0x3fed77b681ff38b3, 0x3fbe72e9da044000, 0x3ff15fffe54d91ad, 0xbc9d7f121737e7ef),
    (0x3fed0cb5724de943, 0x3fc1dcd2507f6000, 0x3ff1a00011ac36e1, 0x3c9c000a0516f5ff),
    (0x3feca4b2dc0e7563, 0x3fc476ab03dea000, 0x3ff1e00019c84248, 0xbc9082fbe4da5da0),
    (0x3fec3f8ee8d6cb51, 0x3fc7074377e22000, 0x3ff220000ffe5e6e, 0xbc88fdd04c9cfb43),
    (0x3febdd2b4f020c4c, 0x3fc98ede8ba94000, 0x3ff26000269fd891, 0x3c8cfe2a7994d182),
    (0x3feb7d6c006015ca, 0x3fcc0db86ad2e000, 0x3ff2a00029a6e6da, 0xbc700273715e8bc5),
    (0x3feb20366e2e338f, 0x3fce840aafcee000, 0x3ff2dfffe0293e39, 0x3c9b7c39dab2a6f9),
    (0x3feac57026295039, 0x3fd0790ab4678000, 0x3ff31ffff7dcf082, 0x3c7df1336edc5254),
    (0x3fea6d01bc2731dd, 0x3fd1ac056801c000, 0x3ff35ffff05a8b60, 0xbc9e03564ccd31eb),
    (0x3fea16d3bc3ff18b, 0x3fd2db11d4fee000, 0x3ff3a0002e0eaecc, 0x3c75f0e74bd3a477),
    (0x3fe9c2d14967fead, 0x3fd406464ec58000, 0x3ff3e000043bb236, 0x3c9c7dcb149d8833),
    (0x3fe970e4f47c9902, 0x3fd52dbe093af000, 0x3ff4200002d187ff, 0x3c7e08afcf2d3d28),
    (0x3fe920fb3982bcf2, 0x3fd651902050d000, 0x3ff460000d387cb1, 0x3c820837856599a6),
    (0x3fe8d30187f759f1, 0x3fd771d2cdeaf000, 0x3ff4a00004569f89, 0xbc89fa5c904fbcd2),
    (0x3fe886e5ebb9f66d, 0x3fd88e9c857d9000, 0x3ff4e000043543f3, 0xbc781125ed175329),
    (0x3fe83c97b658b994, 0x3fd9a80155e16000, 0x3ff51fffcc027f0f, 0x3c9883d8847754dc),
    (0x3fe7f405ffc61022, 0x3fdabe186ed3d000, 0x3ff55ffffd87b36f, 0xbc8709e731d02807),
    (0x3fe7ad22181415ca, 0x3fdbd0f2aea0e000, 0x3ff59ffff21df7ba, 0x3c87f79f68727b02),
    (0x3fe767dcf99eff8c, 0x3fdce0a43dbf4000, 0x3ff5dfffebfc3481, 0xbc9180902e30e93e),
];

fn hi32(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & (u64::MAX << 32))
}

/// `log2(x)` as glibc 2.34 computes it on x86-64.
pub fn log2(x: f64) -> f64 {
    let ix = x.to_bits();
    let top = (ix >> 48) as u32;
    let lo_b = (1.0 - d(0x3fa5b51000000000)).to_bits();
    let hi_b = (1.0 + d(0x3fa6ab2000000000)).to_bits();
    if ix.wrapping_sub(lo_b) < hi_b - lo_b {
        if ix == 1f64.to_bits() {
            return 0.0;
        }
        let r = x - 1.0;
        let rhi = hi32(r);
        let rlo = r - rhi;
        let hi = rhi * INV_LN2_HI;
        let mut lo = rlo * INV_LN2_HI + r * INV_LN2_LO;
        let r2 = r * r;
        let r4 = r2 * r2;
        let p = r2 * (B[0] + r * B[1]);
        let mut y = hi + p;
        lo += hi - y + p;
        lo += r4
            * (B[2]
                + r * B[3]
                + r2 * (B[4] + r * B[5])
                + r4 * (B[6] + r * B[7] + r2 * (B[8] + r * B[9])));
        y += lo;
        return y;
    }
    if top.wrapping_sub(0x0010) >= 0x7ff0 - 0x0010 {
        return x.log2();
    }
    let tmp = ix.wrapping_sub(OFF);
    let i = ((tmp >> 46) % 64) as usize;
    let k = (tmp as i64) >> 52;
    let iz = ix.wrapping_sub(tmp & (0xfffu64 << 52));
    let (invc, logc, chi, clo) = LOG2_TAB[i];
    let (invc, logc) = (d(invc), d(logc));
    let z = f64::from_bits(iz);
    let kd = k as f64;
    let r = (z - d(chi) - d(clo)) * invc;
    let rhi = hi32(r);
    let rlo = r - rhi;
    let t1 = rhi * INV_LN2_HI;
    let t2 = rlo * INV_LN2_HI + r * INV_LN2_LO;
    let t3 = kd + logc;
    let hi = t3 + t1;
    let lo = t3 - hi + t1 + t2;
    let r2 = r * r;
    let r4 = r2 * r2;
    let p = A[0] + r * A[1] + r2 * (A[2] + r * A[3]) + r4 * (A[4] + r * A[5]);
    lo + r2 * p + hi
}
