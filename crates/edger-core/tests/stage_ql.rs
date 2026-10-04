//! Stage gate for `glmQLFit` (`legacy = FALSE`): `edger_ql_fit`, `edger_ql_coefficients` and the
//! `edger_ql_*` scalars, from the golden kept counts, offsets, AveLogCPM and trended dispersion.

mod common;
use common::*;

use edger_core::ql::glm_ql_fit;

#[test]
fn glm_ql_fit_on_every_edger_run() {
    let mut n = 0;
    for run in edger_runs() {
        if !has_csv(&run, "edger_ql_fit") {
            continue;
        }
        let fx = fixture(&run);
        let dt = read_csv(&run, "edger_disp");
        let ave = dt.f64s("ave_logcpm_common");
        let trended = dt.f64s("trended");
        let q = glm_ql_fit(
            &fx.counts, fx.nlib, &fx.design, fx.p, &fx.offset, &ave, &trended,
        )
        .unwrap();
        let t = read_csv(&run, "edger_ql_fit");
        let ng = ave.len();
        let mut worst = Vec::new();
        let mut check = |what: &str, got: &[f64], want: &[f64], floor: f64| {
            let m = max_rel_floor(&format!("{run} {what}"), got, want, floor);
            worst.push(format!("{what} {m:.1e}"));
            let tol = 1e-8;
            assert!(m <= tol, "{run}: {what} max rel {m:e}");
        };
        let s = |k: &str| scalar(&run, k);
        assert_eq!(q.top_n as f64, s("edger_ql_top_n"), "{run}: top_n");
        check(
            "disp_uncapped",
            &[q.dispersion_uncapped],
            &[s("edger_ql_dispersion_uncapped")],
            0.0,
        );
        check("disp", &[q.dispersion], &[s("edger_ql_dispersion")], 0.0);
        check(
            "aqd",
            &[q.ave_ql_dispersion],
            &[s("edger_ql_ave_ql_dispersion")],
            0.0,
        );
        check(
            "deviance_first",
            &q.deviance_first,
            &t.f64s("deviance_first"),
            1.0,
        );
        check("deviance", &q.deviance, &t.f64s("deviance"), 1.0);
        check(
            "df_residual",
            &vec![q.df_residual; ng],
            &t.f64s("df_residual"),
            0.0,
        );
        check("s2", &q.s2, &t.f64s("s2"), 1.0);
        // The adjusted df of a gene fitted to zero counts sits at ~1e-18 and inherits the
        // deviance chaos; its scale is the residual df, so floor 1 like the deviances.
        check(
            "df_adj",
            &q.df_residual_adj,
            &t.f64s("df_residual_adj"),
            1.0,
        );
        check(
            "deviance_adj",
            &q.deviance_adj,
            &t.f64s("deviance_adj"),
            1.0,
        );
        check("s2_prior", &q.s2_prior, &t.f64s("s2_prior"), 0.0);
        check("s2_post", &q.s2_post, &t.f64s("s2_post"), 0.0);
        check("df_prior", &q.df_prior, &t.f64s("df_prior"), 0.0);
        check("fdist_scale", &q.fdist_scale, &t.f64s("fdist_scale"), 0.0);
        check(
            "fdist_df2",
            &vec![q.fdist_df2; ng],
            &t.f64s("fdist_df2"),
            0.0,
        );
        let shr = q
            .fdist_df2_shrunk
            .clone()
            .unwrap_or_else(|| vec![f64::NAN; ng]);
        check("fdist_df2_shrunk", &shr, &t.f64s("fdist_df2_shrunk"), 0.0);
        let coef: Vec<f64> = read_csv(&run, "edger_ql_coefficients")
            .matrix()
            .into_iter()
            .flatten()
            .collect();
        // Coefficients of groups with zero counts run off towards -1e8 and are path-dependent in
        // their last digits; compare relative to max(|c|, 1).
        check("coefficients", &q.coefficients, &coef, 1.0);
        eprintln!("{run}: {}", worst.join(", "));
        n += 1;
    }
    assert_eq!(
        n,
        per_tier(25, 5),
        "runs (the full corpus has 25, the small tier 5)"
    );
}
