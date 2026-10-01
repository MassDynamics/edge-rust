//! Gate for `rnum::locfit` (degree 0, ev = "tree"): reproduce `edger_disp_m0`, the trended
//! log-likelihood surface edgeR's `WLEB` builds with `locfitByCol(l0, AveLogCPM, span, 0)`, from
//! `edger_disp_l0` on every edgeR run that reaches estimateDisp.

mod common;
use common::*;

use rnum::locfit::{locfit, LocfitOptions};
use rnum::quad::choose_lowess_span;

#[test]
fn m0_from_l0_on_every_edger_run() {
    let mut n = 0;
    for run in edger_runs() {
        if !has_csv(&run, "edger_disp_l0") {
            continue;
        }
        let l0 = read_csv(&run, "edger_disp_l0").matrix();
        let m0 = read_csv(&run, "edger_disp_m0").matrix();
        let sel = read_csv(&run, "edger_disp_sel").bools("sel");
        let ave = read_csv(&run, "edger_disp").f64s("ave_logcpm_common");
        let x: Vec<f64> = ave
            .iter()
            .zip(&sel)
            .filter(|(_, &s)| s)
            .map(|(&a, _)| a)
            .collect();
        assert_eq!(x.len(), l0.len());
        // WLEB's default span; reference.json keeps 15 significant digits.
        let span = choose_lowess_span(x.len(), 50.0, 0.3, 1.0 / 3.0);
        assert_close(&run, &[span], &[scalar(&run, "edger_disp_span")], 1e-14);
        let opts = LocfitOptions {
            alpha: span,
            deg: 0,
            ..Default::default()
        };
        let mut worst: f64 = 0.0;
        for j in 0..l0[0].len() {
            let y: Vec<f64> = l0.iter().map(|r| r[j]).collect();
            let want: Vec<f64> = m0.iter().map(|r| r[j]).collect();
            let got = locfit(&x, &y, None, &opts).unwrap().fitted();
            worst = worst.max(max_rel(&format!("{run} m0[,{j}]"), &got, &want));
        }
        eprintln!("{run}: m0 max rel {worst:e}");
        assert!(worst <= 1e-8, "{run}: m0 max rel {worst:e}");
        n += 1;
    }
    assert!(n >= 20, "only {n} runs reached estimateDisp");
}
