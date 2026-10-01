// Temporary: compare rnum::glibm with R's exp/log samples.
fn rd(p: &str) -> Vec<f64> { std::fs::read(p).unwrap().chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect() }
#[test]
fn glibm_vs_r() {
    let Ok(dir) = std::env::var("GLIBM_SAMPLES") else { return };
    let e = rd(&format!("{dir}/exp.bin")); let m = e.len() / 2;
    let bad = (0..m).filter(|&i| rnum::glibm::exp(e[i]).to_bits() != e[m + i].to_bits()).count();
    let badsys = (0..m).filter(|&i| e[i].exp().to_bits() != e[m + i].to_bits()).count();
    let l = rd(&format!("{dir}/log.bin")); let k = l.len() / 4;
    let badl = (0..k).filter(|&i| rnum::glibm::ln(l[i]).to_bits() != l[k + i].to_bits()).count()
        + (0..k).filter(|&i| rnum::glibm::ln(l[2 * k + i]).to_bits() != l[3 * k + i].to_bits()).count();
    for i in 0..2*k { let (x,r) = if i<k {(l[i],l[k+i])} else {(l[2*k+i-k],l[3*k+i-k])}; let g=rnum::glibm::ln(x); if g.to_bits()!=r.to_bits() { println!("x={:e} bits {:016x} got {:016x} want {:016x}", x, x.to_bits(), g.to_bits(), r.to_bits()); } }
    println!("exp mismatches {bad} (system {badsys}) of {m}; log mismatches {badl} of {}", 2 * k);
    assert_eq!(bad + badl, 0);
}
