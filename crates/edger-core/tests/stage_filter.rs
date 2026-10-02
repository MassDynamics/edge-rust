//! Stage gate for `filterByExpr`: `edger_filter` (n above cutoff, total, keep) and the
//! `edger_filter_min_sample_size` / `edger_filter_cpm_cutoff` scalars, on every run that dumps them.

mod common;
use common::*;

use edger_core::filter::filter_by_expr;

#[test]
fn filter_by_expr_on_every_edger_run() {
    let mut n = 0;
    for run in edger_runs() {
        if !has_csv(&run, "edger_filter") {
            continue;
        }
        let fx = fixture(&run);
        let want = read_csv(&run, "edger_filter");
        assert_eq!(want.strings("id"), fx.all_ids, "{run}: gene order");
        let got = filter_by_expr(&fx.all_counts, fx.nlib, &fx.design, fx.p).unwrap();
        assert_eq!(got.keep, want.bools("keep"), "{run}: keep");
        assert_close(&run, &got.n_above_cutoff, &want.f64s("n_above_cutoff"), 0.0);
        assert_close(&run, &got.total, &want.f64s("total"), 0.0);
        let s = |k: &str| scalar(&run, k);
        assert_close(
            &run,
            &[got.min_sample_size],
            &[s("edger_filter_min_sample_size")],
            1e-14,
        );
        assert_close(
            &run,
            &[got.cpm_cutoff],
            &[s("edger_filter_cpm_cutoff")],
            1e-14,
        );
        n += 1;
    }
    assert!(n >= 24, "only {n} runs dump edger_filter");
}
