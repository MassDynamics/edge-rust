//! Temporary: per-stage bit-mismatch counts against the goldens (EDGER_BITS=1).
mod common;
use common::*;
use edger_core::disp::estimate_disp;
use edger_core::glm::glm_fit;
use edger_core::norm::calc_norm_factors;
use edger_core::ql::glm_ql_fit;

fn fb(run: &str, name: &str, col: &str) -> Vec<f64> {
    let dir = std::env::var("EDGER_FULL").unwrap();
    let p = format!("{dir}/{run}/{name}__{col}.bin");
    std::fs::read(&p)
        .unwrap_or_else(|_| panic!("{p}"))
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
fn fm(run: &str, name: &str, ncol: usize) -> Vec<f64> {
    let cols: Vec<Vec<f64>> = (1..=ncol).map(|j| fb(run, name, &format!("V{j}"))).collect();
    (0..cols[0].len()).flat_map(|i| cols.iter().map(move |c| c[i])).collect()
}

fn cmp(got: &[f64], want: &[f64]) -> String {
    let mut n = 0;
    let mut w: f64 = 0.0;
    for (a, b) in got.iter().zip(want) {
        if a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()) {
            n += 1;
            w = w.max((a - b).abs() / b.abs().max(1e-300));
        }
    }
    format!("{n}/{} {w:.1e}", got.len())
}

#[test]
fn bits() {
    if std::env::var("EDGER_BITS").is_err() {
        return;
    }
    for run in edger_runs() {
        if !has_csv(&run, "edger_ql_fit") {
            continue;
        }
        if let Ok(f) = std::env::var("EDGER_RUN") {
            if !run.contains(&f) {
                continue;
            }
        }
        let dir = std::env::var("EDGER_FULL").unwrap();
        if !std::path::Path::new(&format!("{dir}/{run}")).exists() {
            continue;
        }
        let mut fx = fixture(&run);
        let libf = fb(&run, "edger_norm", "lib_size");
        let nff = fb(&run, "edger_norm", "norm_factor");
        fx.lib_eff = libf.iter().zip(&nff).map(|(a, b)| a * b).collect();
        fx.offset = fx.lib_eff.iter().map(|&v| rnum::glibm::ln(v)).collect();
        let mut out = vec![];
        let want = read_csv(&run, "edger_norm");
        let mut lib = vec![0.0; fx.nlib];
        for r in fx.counts.chunks(fx.nlib) {
            for (l, v) in lib.iter_mut().zip(r) {
                *l += v;
            }
        }
        let j = reference_json(&run);
        let method = j["scalars"]["edger_norm_method"].as_str().unwrap().to_string();
        let nf = calc_norm_factors(&fx.counts, fx.nlib, &lib, &method).unwrap();
        out.push(format!("nf {}", cmp(&nf.norm_factors, &nff)));
        let sel: Vec<bool> = fx.counts.chunks(fx.nlib).map(|r| r.iter().sum::<f64>() >= 5.0).collect();
        let y = rows_where(&fx.counts, fx.nlib, &sel);
        let ng = y.len() / fx.nlib;
        let fit = glm_fit(&y, fx.nlib, &fx.design, fx.p, &fx.offset, &vec![0.05; ng], None);
        out.push(format!("glm005 {}", cmp(&fit.deviance, &fb(&run, "edger_disp_glmfit005", "deviance"))));
        let d = estimate_disp(&fx.counts, fx.nlib, &fx.design, fx.p, &fx.lib_eff, &fx.offset).unwrap();
        let fl = |m: Vec<Vec<f64>>| m.into_iter().flatten().collect::<Vec<f64>>();
        out.push(format!("l0 {}", cmp(&d.l0, &fm(&run, "edger_disp_l0", 21))));
        out.push(format!("m0 {}", cmp(&d.m0, &fm(&run, "edger_disp_m0", 21))));
        let dt = read_csv(&run, "edger_disp");
        out.push(format!("alc {}", cmp(&d.ave_logcpm, &fb(&run, "edger_disp", "ave_logcpm_common"))));
        out.push(format!("trended {}", cmp(&d.trended, &fb(&run, "edger_disp", "trended"))));
        out.push(format!("common {}", cmp(&[d.common], &fb(&run, "scalar", "edger_disp_common"))));
        let ave = fb(&run, "edger_disp", "ave_logcpm_common");
        let q = glm_ql_fit(&fx.counts, fx.nlib, &fx.design, fx.p, &fx.offset, &ave, &fb(&run, "edger_disp", "trended")).unwrap();
        let t = read_csv(&run, "edger_ql_fit");
        out.push(format!("qdisp {}", cmp(&[q.dispersion], &fb(&run, "scalar", "edger_ql_dispersion"))));
        out.push(format!("aqd {}", cmp(&[q.ave_ql_dispersion], &fb(&run, "scalar", "edger_ql_ave_ql_dispersion"))));
        out.push(format!("dev1 {}", cmp(&q.deviance_first, &fb(&run, "edger_ql_fit", "deviance_first"))));
        out.push(format!("dev {}", cmp(&q.deviance, &fb(&run, "edger_ql_fit", "deviance"))));
        out.push(format!("s2 {}", cmp(&q.s2, &fb(&run, "edger_ql_fit", "s2"))));
        out.push(format!("dfadj {}", cmp(&q.df_residual_adj, &fb(&run, "edger_ql_fit", "df_residual_adj"))));
        out.push(format!("s2prior {}", cmp(&q.s2_prior, &fb(&run, "edger_ql_fit", "s2_prior"))));
        out.push(format!("dfprior {}", cmp(&q.df_prior, &fb(&run, "edger_ql_fit", "df_prior"))));
        out.push(format!("s2post {}", cmp(&q.s2_post, &fb(&run, "edger_ql_fit", "s2_post"))));
        println!("{run}\n   {}", out.join(" | "));
    }
}
