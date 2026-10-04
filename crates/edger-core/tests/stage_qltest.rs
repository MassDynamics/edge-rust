//! Stage gate for `glmQLFTest`: `edger_test_omnibus` and `edger_test_pair<i>` (null deviance, LR,
//! df, F, df.total, p-value, FDR, logFC), from the golden inputs through our own QL fit.

mod common;
use common::*;

use edger_core::ql::glm_ql_fit;
use edger_core::qltest::glm_ql_ftest;
use rnum::linalg::p_adjust_bh;
use rnum::nmath::pf;

#[test]
fn glm_ql_ftest_on_every_edger_run() {
    let mut n = 0;
    for run in edger_runs() {
        if !has_csv(&run, "edger_test_omnibus") {
            continue;
        }
        let fx = fixture(&run);
        let dt = read_csv(&run, "edger_disp");
        let ave = dt.f64s("ave_logcpm_common");
        let q = glm_ql_fit(
            &fx.counts,
            fx.nlib,
            &fx.design,
            fx.p,
            &fx.offset,
            &ave,
            &dt.f64s("trended"),
        )
        .unwrap();
        // Contrasts: the golden omnibus matrix, then one column per comparison built from the
        // design column names (`<cc><encoded level>`).
        let oc = read_csv(&run, "edger_contrast_omnibus");
        let design_cols = read_csv(&run, "edger_design").header[1..].to_vec();
        assert_eq!(oc.strings("id"), design_cols, "{run}: contrast rows");
        let ncon = oc.header.len() - 1;
        let om = oc.matrix();
        let mut omni = vec![0.0; fx.p * ncon];
        for (l, r) in om.iter().enumerate() {
            for k in 0..ncon {
                omni[k * fx.p + l] = r[k];
            }
        }
        let levels: Vec<String> = reference_json(&run)["scalars"]["condition_levels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        let cc = oc.header[1]
            .rsplit(" - ")
            .next()
            .unwrap()
            .strip_suffix(levels[0].as_str())
            .unwrap()
            .to_string();
        let mut tests = vec![("omnibus".to_string(), omni, ncon)];
        let cmp = read_csv(&run, "input_comparisons");
        for (i, (le, re)) in cmp
            .strings("encoded_left")
            .iter()
            .zip(cmp.strings("encoded_right"))
            .enumerate()
        {
            let mut c = vec![0.0; fx.p];
            c[design_cols
                .iter()
                .position(|d| *d == format!("{cc}{le}"))
                .unwrap()] = 1.0;
            c[design_cols
                .iter()
                .position(|d| *d == format!("{cc}{re}"))
                .unwrap()] = -1.0;
            tests.push((format!("pair{}", i + 1), c, 1));
        }
        let mut worst = Vec::new();
        for (label, con, nc) in tests {
            let t = glm_ql_ftest(
                &fx.counts, fx.nlib, &fx.design, fx.p, &fx.offset, &q, &con, nc,
            )
            .unwrap();
            let g = read_csv(&run, &format!("edger_test_{label}"));
            let mut check = |what: &str, got: &[f64], want: &[f64], floor: f64, tol: f64| {
                let m = max_rel_floor(&format!("{run} {label} {what}"), got, want, floor);
                worst.push(format!("{label}.{what} {m:.1e}"));
                assert!(m <= tol, "{run}: {label} {what} max rel {m:e}");
            };
            let ng = t.f.len();
            check(
                "deviance_null",
                &t.deviance_null,
                &g.f64s("deviance_null"),
                1.0,
                1e-8,
            );
            check("LR", &t.lr, &g.f64s("LR"), 1.0, 1e-8);
            check(
                "df_test",
                &vec![t.df_test as f64; ng],
                &g.f64s("df_test"),
                0.0,
                0.0,
            );
            check("F", &t.f, &g.f64s("F"), 1.0, 1e-8);
            check("df_total", &t.df_total, &g.f64s("df_total"), 0.0, 1e-8);
            for k in 0..t.ncon {
                let col = if t.ncon == 1 {
                    "logFC".to_string()
                } else {
                    format!("logFC_{}", k + 1)
                };
                let got: Vec<f64> = (0..ng).map(|i| t.logfc[i * t.ncon + k]).collect();
                check(&col, &got, &g.f64s(&col), 1.0, 1e-8);
            }
            // p-values are gated through the F band: near F = 0 (p ~ 1) and at large F (tiny p)
            // the F noise is amplified (measured: 1.5e-8 at p = 0.99996, 1.6e-6 at large F).
            let (wf, wd) = (g.f64s("F"), g.f64s("df_total"));
            let r = t.df_test as f64;
            let (plo, phi) = band(&wf, &wd, |_, f, d| pf(f, r, d, false, false));
            let (m, nb) = assert_in_band(
                &format!("{run} {label} PValue"),
                &t.pvalue,
                &g.f64s("PValue"),
                &plo,
                &phi,
            );
            worst.push(format!("{label}.PValue {m:.1e} ({nb} in band)"));
            let (m, nb) = assert_in_band(
                &format!("{run} {label} FDR"),
                &t.fdr,
                &g.f64s("FDR"),
                &p_adjust_bh(&plo),
                &p_adjust_bh(&phi),
            );
            worst.push(format!("{label}.FDR {m:.1e} ({nb} in band)"));
        }
        eprintln!("{run}: {}", worst.join(", "));
        n += 1;
    }
    assert_eq!(
        n,
        per_tier(28, 5),
        "runs (the full corpus has 28, the small tier 5)"
    );
}
