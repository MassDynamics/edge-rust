//! `eBayes` + `topTable` + `decideTests` against the `matrix/<case>/ebayes_rob*_trend*_*`
//! goldens, for all four robust/trend combinations, at the manifest tolerances: t/F/s2 rel
//! 1e-8, p rel 1e-8 with a 1e-12 floor, lods rel 1e-8, df/s2 prior rel 1e-6, logFC and CI abs
//! 1e-10, decideTests exact. The topTable goldens were written with `sort.by = "none"`, so
//! rows are matched by gene name rather than by position.

mod common;

use common::matrix::{matrix_dir, Matrix, MATRIX_CASES};
use common::{Report, Tol};
use rnum::contrasts::contrasts_fit;
use rnum::ebayes::{ebayes, EBayes, EBayesOptions};
use rnum::fit::{lm_fit_matrix, MArrayLm};
use rnum::toptable::{decide_tests_separate, top_table_f, top_table_t};

const ABS_1E10: Tol = Tol {
    rel: 0.0,
    abs_floor: 1e-10,
};
const REL_1E8: Tol = Tol {
    rel: 1e-8,
    abs_floor: 0.0,
};
const P_TOL: Tol = Tol {
    rel: 1e-8,
    abs_floor: 1e-12,
};
const REL_1E6: Tol = Tol {
    rel: 1e-6,
    abs_floor: 0.0,
};
const EXACT: Tol = Tol {
    rel: 0.0,
    abs_floor: 0.0,
};

fn load_case(dir: &std::path::Path) -> (Matrix, MArrayLm) {
    let exprs = Matrix::read(dir, "input_log2.csv");
    let design = Matrix::read(dir, "design.csv");
    let contrasts = Matrix::read(dir, "contrasts.csv");
    let fit = lm_fit_matrix(
        &exprs.data,
        exprs.nrow,
        exprs.ncol,
        &design.data,
        design.ncol,
    )
    .expect("lmFit");
    let cfit = contrasts_fit(&fit, &contrasts.data, contrasts.ncol).expect("contrasts.fit");
    (exprs, cfit)
}

fn check_matrix(report: &mut Report, tol: Tol, what: &str, got: &[f64], want: &Matrix) {
    assert_eq!(got.len(), want.data.len(), "{what}: dimension mismatch");
    for j in 0..want.ncol {
        for i in 0..want.nrow {
            let g = got[j * want.nrow + i];
            let w = want.get(i, j);
            if g.is_nan() && w.is_nan() {
                continue;
            }
            report.check(
                tol,
                format!("{what}[{},{}]", want.row_names[i], want.col_names[j]),
                g,
                w,
            );
        }
    }
}

fn check_vec(report: &mut Report, tol: Tol, what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if g.is_nan() && w.is_nan() {
            continue;
        }
        report.check(tol, format!("{what}[{}]", i + 1), g, w);
    }
}

fn index_by_name(m: &Matrix) -> std::collections::HashMap<&str, usize> {
    m.row_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect()
}

fn recycle(v: &[f64], n: usize) -> Vec<f64> {
    if v.len() == 1 {
        vec![v[0]; n]
    } else {
        v.to_vec()
    }
}

fn check_ebayes(
    report: &mut Report,
    dir: &std::path::Path,
    prefix: &str,
    cfit: &MArrayLm,
    eb: &EBayes,
) {
    let n = cfit.ngenes;
    let t = Matrix::read(dir, &format!("{prefix}_t.csv"));
    check_matrix(report, REL_1E8, "t", &eb.t, &t);
    let p = Matrix::read(dir, &format!("{prefix}_p.csv"));
    check_matrix(report, P_TOL, "p.value", &eb.p_value, &p);
    let lods = Matrix::read(dir, &format!("{prefix}_lods.csv"));
    check_matrix(report, REL_1E8, "lods", &eb.lods, &lods);
    let s = Matrix::read(dir, &format!("{prefix}_scalars.csv"));
    check_vec(report, REL_1E8, "s2.post", &eb.s2_post, s.col("s2_post"));
    check_vec(report, REL_1E6, "df.total", &eb.df_total, s.col("df_total"));
    check_vec(report, REL_1E8, "F", eb.f.as_ref().expect("F"), s.col("F"));
    check_vec(
        report,
        P_TOL,
        "F.p.value",
        eb.f_p_value.as_ref().expect("F.p"),
        s.col("F_p"),
    );
    check_vec(
        report,
        REL_1E6,
        "s2.prior",
        &recycle(&eb.s2_prior, n),
        s.col("s2_prior"),
    );
    check_vec(
        report,
        REL_1E6,
        "df.prior",
        &recycle(&eb.df_prior, n),
        s.col("df_prior"),
    );
}

fn check_toptables(
    report: &mut Report,
    dir: &std::path::Path,
    prefix: &str,
    exprs: &Matrix,
    cfit: &MArrayLm,
    eb: &EBayes,
) {
    for k in 0..cfit.ncoef {
        let want = Matrix::read(dir, &format!("{prefix}_toptable_{}.csv", k + 1));
        let rows = top_table_t(cfit, eb, k, 0.95).expect("topTable");
        assert_eq!(rows.len(), want.nrow);
        let by_name = index_by_name(&want);
        for row in &rows {
            let name = &exprs.row_names[row.gene];
            let i = by_name[name.as_str()];
            let tag = format!("toptable_{}[{name}]", k + 1);
            report.check(ABS_1E10, format!("{tag}.logFC"), row.log_fc, want.get(i, 0));
            report.check(ABS_1E10, format!("{tag}.CI.L"), row.ci_l, want.get(i, 1));
            report.check(ABS_1E10, format!("{tag}.CI.R"), row.ci_r, want.get(i, 2));
            report.check(
                REL_1E8,
                format!("{tag}.AveExpr"),
                row.ave_expr,
                want.get(i, 3),
            );
            report.check(REL_1E8, format!("{tag}.t"), row.t, want.get(i, 4));
            report.check(P_TOL, format!("{tag}.P.Value"), row.p_value, want.get(i, 5));
            report.check(
                P_TOL,
                format!("{tag}.adj.P.Val"),
                row.adj_p_value,
                want.get(i, 6),
            );
            report.check(REL_1E8, format!("{tag}.B"), row.b, want.get(i, 7));
        }
    }
    let want = Matrix::read(dir, &format!("{prefix}_toptable_F.csv"));
    let rows = top_table_f(cfit, eb).expect("topTable F");
    assert_eq!(rows.len(), want.nrow);
    let nc = cfit.ncoef;
    assert_eq!(want.ncol, nc + 4);
    let by_name = index_by_name(&want);
    for row in &rows {
        let name = &exprs.row_names[row.gene];
        let i = by_name[name.as_str()];
        let tag = format!("toptable_F[{name}]");
        for j in 0..nc {
            report.check(
                ABS_1E10,
                format!("{tag}.coef{}", j + 1),
                row.coefs[j],
                want.get(i, j),
            );
        }
        report.check(
            REL_1E8,
            format!("{tag}.AveExpr"),
            row.ave_expr,
            want.get(i, nc),
        );
        report.check(REL_1E8, format!("{tag}.F"), row.f, want.get(i, nc + 1));
        report.check(
            P_TOL,
            format!("{tag}.P.Value"),
            row.p_value,
            want.get(i, nc + 2),
        );
        report.check(
            P_TOL,
            format!("{tag}.adj.P.Val"),
            row.adj_p_value,
            want.get(i, nc + 3),
        );
    }
}

#[test]
fn ebayes_matches_golden() {
    for case in MATRIX_CASES {
        let Some(dir) = matrix_dir("ebayes_matches_golden", case) else {
            return;
        };
        let (exprs, cfit) = load_case(&dir);
        for robust in [false, true] {
            for trend in [false, true] {
                let prefix = format!(
                    "ebayes_rob{}_trend{}",
                    if robust { "TRUE" } else { "FALSE" },
                    if trend { "TRUE" } else { "FALSE" }
                );
                let mut report = Report::new(&format!("{case}/{prefix}"));
                let opts = EBayesOptions {
                    robust,
                    trend,
                    ..EBayesOptions::default()
                };
                let eb = ebayes(&cfit, &opts).expect("eBayes");
                check_ebayes(&mut report, &dir, &prefix, &cfit, &eb);
                check_toptables(&mut report, &dir, &prefix, &exprs, &cfit, &eb);
                let want = Matrix::read(&dir, &format!("{prefix}_decidetests.csv"));
                let got = decide_tests_separate(&cfit, &eb, 0.05);
                check_matrix(&mut report, EXACT, "decideTests", &got, &want);
                report.finish();
            }
        }
    }
}
