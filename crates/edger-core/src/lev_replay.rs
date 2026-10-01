//! Temporary: replay R's instrumented Levenberg trace op by op (EDGER_LEV_TRACE=<dir>).
use crate::glm::*;
use crate::lapack::{dpotrf_upper, dpotrs_upper};
use std::collections::BTreeMap;

fn rd(path: &std::path::Path) -> Vec<f64> {
    let b = std::fs::read(path).unwrap();
    b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}
fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}
fn rel(a: f64, b: f64) -> f64 {
    if same(a, b) { 0.0 } else { (a - b).abs() / b.abs().max(1e-300) }
}

#[test]
fn replay() {
    let Ok(dir) = std::env::var("EDGER_LEV_TRACE") else { return };
    let dir = std::path::PathBuf::from(dir);
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let only: Option<String> = std::env::var("EDGER_LEV_CALL").ok();
    for (tag, m) in meta.as_object().unwrap() {
        if let Some(o) = &only { if o != tag { continue; } }
        let ng = m["ngenes"].as_u64().unwrap() as usize;
        let n = m["nlib"].as_u64().unwrap() as usize;
        let p = m["p"].as_u64().unwrap() as usize;
        let ld = |s: &str| rd(&dir.join(format!("{tag}_{s}.bin")));
        let (y, off, disp, x, start, coef, mu, dev, tr) =
            (ld("y"), ld("off"), ld("disp"), ld("x"), ld("start"), ld("coef"), ld("mu"), ld("dev"), ld("trace"));
        // R matrices are column-major ng x n.
        let at = |v: &[f64], g: usize, j: usize| v[g + ng * j];
        let offrow: Vec<f64> = (0..n).map(|j| at(&off, 0, j)).collect();
        for g in 0..ng { for j in 0..n { assert!(same(at(&off, g, j), offrow[j]), "offset not a shared row"); } }
        let dg: Vec<f64> = (0..ng).map(|g| at(&disp, g, 0)).collect();
        // whole-fit comparison from R's start
        let yg: Vec<f64> = (0..ng).flat_map(|g| (0..n).map(move |j| (g, j))).map(|(g, j)| at(&y, g, j)).collect();
        let sg: Vec<f64> = (0..ng).flat_map(|g| (0..p).map(move |c| (g, c))).map(|(g, c)| at(&start, g, c)).collect();
        let fit = levenberg_given(&yg, n, &x, p, &offrow, &dg, &sg);
        let (mut nc, mut nd, mut wc, mut wd) = (0, 0, 0.0f64, 0.0f64);
        for g in 0..ng {
            let mut bad = false;
            for c in 0..p { let r = rel(fit.coefficients[g * p + c], at(&coef, g, c)); if r > 0.0 { bad = true; wc = wc.max(r); } }
            if bad { nc += 1; }
            let r = rel(fit.deviance[g], dev[g]); if r > 0.0 { nd += 1; wd = wd.max(r); }
            let _ = &mu;
        }
        // null start comparison
        let mut nstart = 0; let mut wstart = 0.0f64;
        if m["null_start"].as_bool().unwrap() {
            let s = null_start(&yg, n, &x, p, &offrow, &dg);
            for g in 0..ng { let mut bad = false; for c in 0..p { let r = rel(s[g * p + c], at(&start, g, c)); if r > 0.0 { bad = true; wstart = wstart.max(r); } } if bad { nstart += 1; } }
        }
        // op-by-op replay
        let rl = 8 + 3 * p + 2 * p * p + n;
        let mut ops: BTreeMap<&str, (usize, f64)> = BTreeMap::new();
        let mut first: BTreeMap<usize, &str> = BTreeMap::new();
        let mut bump = |ops: &mut BTreeMap<&'static str, (usize, f64)>, k: &'static str, r: f64| {
            let e = ops.entry(k).or_insert((0, 0.0)); if r > 0.0 { e.0 += 1; e.1 = e.1.max(r); }
        };
        let mut obt = vec![0.0; p]; let mut omu = vec![0.0; n]; let mut lastg = usize::MAX;
        let mut i = 0;
        let mut xtwx = vec![0.0; p * p]; let mut buf = vec![0.0; n];
        while i < tr.len() {
            let r = &tr[i..i + rl]; i += rl;
            let g = r[0] as usize; let iter = r[1] as usize;
            let (lam, dev0, ndev, acc) = (r[3], r[5], r[6], r[7] != 0.0);
            let nbt = &r[8..8 + p]; let db = &r[8 + p..8 + 2 * p]; let dl = &r[8 + 2 * p..8 + 3 * p];
            let rx = &r[8 + 3 * p..8 + 3 * p + p * p]; let rc = &r[8 + 3 * p + p * p..8 + 3 * p + 2 * p * p];
            let nmu = &r[8 + 3 * p + 2 * p * p..rl];
            let row: Vec<f64> = (0..n).map(|j| at(&y, g, j)).collect();
            let mut note = |k: &'static str, rr: f64, ops: &mut BTreeMap<&'static str, (usize, f64)>| {
                bump(ops, k, rr); if rr > 0.0 { first.entry(g).or_insert(k); }
            };
            if iter == 0 {
                lastg = g;
                obt.copy_from_slice(nbt); omu.copy_from_slice(nmu);
                autofill_pub(nbt, &offrow, &x, &mut buf);
                let mut w = 0.0f64; for j in 0..n { w = w.max(rel(buf[j], nmu[j])); } note("start_mu", w, &mut ops);
                let mut d = 0.0; for j in 0..n { d += unit_nb_deviance(row[j], nmu[j], dg[g]); } note("start_dev", rel(d, dev0), &mut ops);
                continue;
            }
            assert_eq!(g, lastg);
            // xtwx and dl from the accepted omu
            let zw: Vec<f64> = (0..n).map(|j| { let c = omu[j]; c / (1.0 + c * dg[g]) }).collect();
            let drv: Vec<f64> = (0..n).map(|j| { let c = omu[j]; (row[j] - c) / (1.0 + c * dg[g]) }).collect();
            compute_xtwx(n, p, &x, &zw, &mut xtwx);
            let mut w = 0.0f64; for c1 in 0..p { for c2 in 0..=c1 { w = w.max(rel(xtwx[c1 * p + c2], rx[c1 * p + c2])); } } note("xtwx", w, &mut ops);
            let mut w = 0.0f64; for c in 0..p { let mut s = 0.0; for j in 0..n { s += drv[j] * x[c * n + j]; } w = w.max(rel(s, dl[c])); } note("dl", w, &mut ops);
            let mut ch = vec![0.0; p * p];
            for c1 in 0..p { for c2 in 0..=c1 { ch[c1 * p + c2] = rx[c1 * p + c2]; } ch[c1 * p + c1] += lam; }
            assert!(dpotrf_upper(&mut ch, p));
            let mut w = 0.0f64; for c1 in 0..p { for c2 in 0..=c1 { w = w.max(rel(ch[c1 * p + c2], rc[c1 * p + c2])); } } note("chol", w, &mut ops);
            let mut b = dl.to_vec(); dpotrs_upper(rc, p, &mut b);
            let mut w = 0.0f64; for c in 0..p { w = w.max(rel(b[c], db[c])); } note("potrs", w, &mut ops);
            let mut w = 0.0f64; for c in 0..p { w = w.max(rel(obt[c] + db[c], nbt[c])); } note("nbt", w, &mut ops);
            autofill_pub(nbt, &offrow, &x, &mut buf);
            let mut w = 0.0f64; for j in 0..n { w = w.max(rel(buf[j], nmu[j])); } note("autofill", w, &mut ops);
            let mut d = 0.0; for j in 0..n { d += unit_nb_deviance(row[j], nmu[j], dg[g]); } note("ndev", rel(d, ndev), &mut ops);
            if acc { obt.copy_from_slice(nbt); omu.copy_from_slice(nmu); }
        }
        let mut firsts: BTreeMap<&str, usize> = BTreeMap::new();
        for v in first.values() { *firsts.entry(v).or_default() += 1; }
        println!("{tag} p={p} fit: coef_mismatch_genes={nc} worst={wc:.2e} dev_mismatch={nd} worst={wd:.2e} start_mismatch_genes={nstart} worst={wstart:.2e}");
        println!("   ops: {:?}", ops.iter().filter(|(_, v)| v.0 > 0).collect::<Vec<_>>());
        println!("   first divergent op per gene: {firsts:?}");
        if let Some((g, k)) = first.iter().next() { println!("   e.g. gene {g}: {k}"); }
    }
}
