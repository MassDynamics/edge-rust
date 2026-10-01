// Env-gated bit checks of the glibc ports against sample dumps from the docker image
// (md-flexi-r45-local). Each test is a no-op unless its env var points at the dump.
fn rd(p: &str) -> Vec<f64> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
#[test]
fn glibm_vs_r() {
    let Ok(dir) = std::env::var("GLIBM_SAMPLES") else {
        return;
    };
    let e = rd(&format!("{dir}/exp.bin"));
    let m = e.len() / 2;
    let bad = (0..m)
        .filter(|&i| rnum::glibm::exp(e[i]).to_bits() != e[m + i].to_bits())
        .count();
    let badsys = (0..m)
        .filter(|&i| e[i].exp().to_bits() != e[m + i].to_bits())
        .count();
    let l = rd(&format!("{dir}/log.bin"));
    let k = l.len() / 4;
    let badl = (0..k)
        .filter(|&i| rnum::glibm::ln(l[i]).to_bits() != l[k + i].to_bits())
        .count()
        + (0..k)
            .filter(|&i| rnum::glibm::ln(l[2 * k + i]).to_bits() != l[3 * k + i].to_bits())
            .count();
    for i in 0..2 * k {
        let (x, r) = if i < k {
            (l[i], l[k + i])
        } else {
            (l[2 * k + i - k], l[3 * k + i - k])
        };
        let g = rnum::glibm::ln(x);
        if g.to_bits() != r.to_bits() {
            println!(
                "x={:e} bits {:016x} got {:016x} want {:016x}",
                x,
                x.to_bits(),
                g.to_bits(),
                r.to_bits()
            );
        }
    }
    println!(
        "exp mismatches {bad} (system {badsys}) of {m}; log mismatches {badl} of {}",
        2 * k
    );
    assert_eq!(bad + badl, 0);
}

#[test]
fn ldouble_vs_r() {
    let Ok(dir) = std::env::var("GLIBM_SAMPLES") else {
        return;
    };
    let Ok(b) = std::fs::read(format!("{dir}/sums.bin")) else {
        return;
    };
    let v: Vec<f64> = b
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let (mut i, mut k, mut bs, mut bm, mut dm) = (0, 0, 0, 0, 0);
    while i < v.len() {
        let n = v[i] as usize;
        let x = &v[i + 1..i + 1 + n];
        let (s, m) = (v[i + 1 + n], v[i + 2 + n]);
        if rnum::ldouble::sum(x).to_bits() != s.to_bits() {
            bs += 1;
        }
        if rnum::ldouble::mean(x).to_bits() != m.to_bits() {
            bm += 1;
        }
        if rnum::linalg::mean(x).to_bits() != m.to_bits() {
            dm += 1;
        }
        i += n + 3;
        k += 1;
    }
    println!("{k} vectors: sum mismatches {bs}, mean mismatches {bm} (double mean {dm})");
    assert_eq!(bs + bm, 0);
}

#[test]
fn lgamma_vs_c() {
    let Ok(dir) = std::env::var("LGAMMA_SAMPLES") else {
        return;
    };
    let x = rd(&format!("{dir}/in.bin"));
    let y = rd(&format!("{dir}/out.bin"));
    let bad: Vec<usize> = (0..x.len())
        .filter(|&i| rnum::glibm_lgamma::lgamma(x[i]).to_bits() != y[i].to_bits())
        .collect();
    let badr = (0..x.len())
        .filter(|&i| rnum::nmath::lgammafn(x[i]).to_bits() != y[i].to_bits())
        .count();
    for &i in bad.iter().take(10) {
        println!(
            "x={:e} got {:e} want {:e}",
            x[i],
            rnum::glibm_lgamma::lgamma(x[i]),
            y[i]
        );
    }
    println!(
        "lgamma mismatches {} (lgammafn {badr}) of {}",
        bad.len(),
        x.len()
    );
    assert!(bad.is_empty());
}

#[test]
fn pow_vs_c() {
    let Ok(f) = std::env::var("POW_SAMPLES") else {
        return;
    };
    let v = rd(&f);
    let n = v.len() / 2;
    let bad = (0..n)
        .filter(|&i| rnum::glibm_pow::pow(2.0, v[2 * i]).to_bits() != v[2 * i + 1].to_bits())
        .count();
    let bads = (0..n)
        .filter(|&i| 2f64.powf(v[2 * i]).to_bits() != v[2 * i + 1].to_bits())
        .count();
    println!("pow mismatches {bad} (system {bads}) of {n}");
    assert_eq!(bad, 0);
}

#[test]
fn log2_vs_c() {
    let Ok(f) = std::env::var("LOG2_SAMPLES") else {
        return;
    };
    let v = rd(&f);
    let n = v.len() / 2;
    let bad = (0..n)
        .filter(|&i| rnum::glibm_log2::log2(v[2 * i]).to_bits() != v[2 * i + 1].to_bits())
        .count();
    let bads = (0..n)
        .filter(|&i| v[2 * i].log2().to_bits() != v[2 * i + 1].to_bits())
        .count();
    println!("log2 mismatches {bad} (system {bads}) of {n}");
    assert_eq!(bad, 0);
}
