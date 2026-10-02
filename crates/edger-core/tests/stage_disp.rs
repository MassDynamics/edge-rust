//! Stage gate for `estimateDisp`: `edger_disp_sel`, `edger_disp_l0` (APL grid), `edger_disp_m0`,
//! `edger_disp_prior_df_inputs`, `edger_disp` (AveLogCPM, trended, tagwise) and the common,
//! overall, span, prior df / prior n and squeezeVar scalars, from the golden kept counts and
//! effective library sizes.

mod common;
use common::*;

use edger_core::disp::estimate_disp;

fn flat(m: Vec<Vec<f64>>) -> Vec<f64> {
    m.into_iter().flatten().collect()
}

#[test]
fn estimate_disp_on_every_edger_run() {
    let mut n = 0;
    for run in edger_runs() {
        if !has_csv(&run, "edger_disp") {
            continue;
        }
        let fx = fixture(&run);
        let d = estimate_disp(
            &fx.counts,
            fx.nlib,
            &fx.design,
            fx.p,
            &fx.lib_eff,
            &fx.offset,
        )
        .unwrap();
        assert_eq!(
            d.sel,
            read_csv(&run, "edger_disp_sel").bools("sel"),
            "{run}: sel"
        );
        let mut worst = Vec::new();
        let mut check = |what: &str, got: &[f64], want: &[f64], floor: f64| {
            let m = max_rel_floor(&format!("{run} {what}"), got, want, floor);
            worst.push(format!("{what} {m:.1e}"));
            // Given R's full-precision inputs, l0 is bit-identical to R. The fixture's effective
            // library sizes are 15 significant digits (data.table fwrite), and that 4e-15
            // perturbation moves the Levenberg stopping point enough to shift a high-dispersion
            // APL by up to 1.01e-8 (airway_all_ctlfactor gene 2928, grid 0).
            let tol = if what == "l0" { 2e-8 } else { 1e-8 };
            assert!(m <= tol, "{run}: {what} max rel {m:e}");
        };
        // APL values are log-likelihood sums; their scale is the sum, so near-zero cells use
        // a floor of 1 like the deviances.
        check(
            "l0",
            &d.l0,
            &flat(read_csv(&run, "edger_disp_l0").matrix()),
            1.0,
        );
        check(
            "m0",
            &d.m0,
            &flat(read_csv(&run, "edger_disp_m0").matrix()),
            1.0,
        );
        check(
            "glmfit005",
            &d.glmfit005_deviance,
            &read_csv(&run, "edger_disp_glmfit005").f64s("deviance"),
            1.0,
        );
        let pi = read_csv(&run, "edger_disp_prior_df_inputs");
        check(
            "prior deviance",
            &d.prior_deviance,
            &pi.f64s("deviance"),
            1.0,
        );
        check(
            "prior df",
            &d.prior_df_residual,
            &pi.f64s("df_residual"),
            0.0,
        );
        check("prior s2", &d.prior_s2, &pi.f64s("s2"), 1.0);
        let dt = read_csv(&run, "edger_disp");
        check(
            "AveLogCPM",
            &d.ave_logcpm,
            &dt.f64s("ave_logcpm_common"),
            0.0,
        );
        check("trended", &d.trended, &dt.f64s("trended"), 0.0);
        check("tagwise", &d.tagwise, &dt.f64s("tagwise"), 0.0);
        let s = |k: &str| scalar(&run, k);
        check("common", &[d.common], &[s("edger_disp_common")], 0.0);
        check(
            "overall",
            &[d.overall_log2],
            &[s("edger_disp_overall_log2")],
            0.0,
        );
        check("span", &[d.span], &[s("edger_disp_span")], 0.0);
        check("prior_df", &[d.prior_df], &[s("edger_disp_prior_df")], 0.0);
        check("prior_n", &[d.prior_n], &[s("edger_disp_prior_n")], 0.0);
        let j = reference_json(&run);
        assert_eq!(
            Some(d.squeezevar_legacy),
            j["scalars"]["edger_disp_squeezevar_legacy"].as_bool(),
            "{run}: legacy"
        );
        let vp: Vec<f64> = j["scalars"]["edger_disp_squeezevar_var_prior"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_f64().unwrap()).collect())
            .unwrap_or_else(|| {
                vec![j["scalars"]["edger_disp_squeezevar_var_prior"]
                    .as_f64()
                    .unwrap()]
            });
        check("var_prior", &d.squeezevar_var_prior, &vp, 0.0);
        eprintln!("{run}: {}", worst.join(", "));
        n += 1;
    }
    assert_eq!(n, 25, "runs (the corpus has 25)");
}
