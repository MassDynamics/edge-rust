//! Stage gate for `calcNormFactors`: `edger_norm` (lib size, norm factor) and
//! `edger_norm_ref_column` on every run that dumps them, from the golden kept counts.

mod common;
use common::*;

use edger_core::norm::calc_norm_factors;

#[test]
fn norm_factors_on_every_edger_run() {
    let mut n = 0;
    for run in edger_runs() {
        if !has_csv(&run, "edger_norm") {
            continue;
        }
        let fx = fixture(&run);
        let want = read_csv(&run, "edger_norm");
        let mut lib = vec![0.0; fx.nlib];
        for r in fx.counts.chunks(fx.nlib) {
            for (l, v) in lib.iter_mut().zip(r) {
                *l += v;
            }
        }
        assert_close(&run, &lib, &want.f64s("lib_size"), 0.0);
        let j = reference_json(&run);
        let method = j["scalars"]["edger_norm_method"]
            .as_str()
            .unwrap()
            .to_string();
        let got = calc_norm_factors(&fx.counts, fx.nlib, &lib, &method).unwrap();
        let m = max_rel(&run, &got.norm_factors, &want.f64s("norm_factor"));
        eprintln!("{run} ({method}): norm max rel {m:e}");
        assert!(m <= 1e-12, "{run}: norm factors max rel {m:e}");
        let rc = &j["scalars"]["edger_norm_ref_column"];
        match got.ref_column {
            Some(r) => assert_eq!(rc.as_f64(), Some((r + 1) as f64), "{run}: ref column"),
            None => assert!(!rc.is_number(), "{run}: ref column {rc}"),
        }
        n += 1;
    }
    assert_eq!(
        n,
        per_tier(26, 6),
        "runs dump edger_norm (the full corpus has 26, the small tier 6)"
    );
}
