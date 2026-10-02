//! Stage gate for the NB GLM kernels (`glmFit.default`: oneway and Levenberg branches,
//! `compute_unit_nb_deviance`): `edger_disp_glmfit005`, the deviance of `glmFit(sely, design,
//! dispersion = 0.05, prior.count = 0)` that `estimateDisp` starts from.

mod common;
use common::*;

use edger_core::glm::glm_fit;

#[test]
fn glmfit_at_dispersion_005_on_every_edger_run() {
    let mut n = 0;
    for run in edger_runs() {
        if !has_csv(&run, "edger_disp_glmfit005") {
            continue;
        }
        let fx = fixture(&run);
        let sel: Vec<bool> = fx
            .counts
            .chunks(fx.nlib)
            .map(|r| r.iter().sum::<f64>() >= 5.0)
            .collect();
        let y = rows_where(&fx.counts, fx.nlib, &sel);
        let ng = y.len() / fx.nlib;
        let fit = glm_fit(
            &y,
            fx.nlib,
            &fx.design,
            fx.p,
            &fx.offset,
            &vec![0.05; ng],
            None,
        )
        .unwrap();
        let want = read_csv(&run, "edger_disp_glmfit005").f64s("deviance");
        let m = max_rel_floor(&run, &fit.deviance, &want, 1.0);
        let mr = max_rel(&run, &fit.deviance, &want);
        if std::env::var("EDGER_DEBUG").is_ok() {
            let mut v: Vec<(f64, usize)> = (0..ng)
                .map(|g| ((fit.deviance[g] - want[g]).abs() / want[g].abs(), g))
                .collect();
            v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            for &(r, g) in v.iter().take(5) {
                eprintln!(
                    "  gene {g} rel {r:e} got {} want {} coef {:?}",
                    fit.deviance[g],
                    want[g],
                    &fit.coefficients[g * fx.p..(g + 1) * fx.p]
                );
            }
        }
        eprintln!("{run}: glmfit005 deviance max rel (floor 1) {m:e}, plain {mr:e}");
        assert!(m <= 1e-8, "{run}: deviance max rel {m:e}");
        n += 1;
    }
    assert_eq!(n, 25, "runs (the corpus has 25)");
}
