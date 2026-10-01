//! `lmFit` + `contrasts.fit` against the `matrix/<case>/` goldens: coefficients,
//! stdev.unscaled, sigma, df.residual, Amean and cov.coefficients before and after the
//! contrasts, at the manifest tolerances (logFC abs 1e-10, everything else rel 1e-8).

mod common;

use common::matrix::{matrix_dir, Matrix, MATRIX_CASES};
use common::{Report, Tol};
use rnum::contrasts::contrasts_fit;
use rnum::fit::{lm_fit_matrix, MArrayLm};

const ABS_1E10: Tol = Tol {
    rel: 0.0,
    abs_floor: 1e-10,
};
const REL_1E8: Tol = Tol {
    rel: 1e-8,
    abs_floor: 0.0,
};

pub fn load_case(dir: &std::path::Path) -> (Matrix, Matrix, Matrix, MArrayLm, MArrayLm) {
    let exprs = Matrix::read(dir, "input_log2.csv");
    let design = Matrix::read(dir, "design.csv");
    let contrasts = Matrix::read(dir, "contrasts.csv");
    assert_eq!(design.nrow, exprs.ncol);
    assert_eq!(contrasts.nrow, design.ncol);
    assert_eq!(contrasts.row_names, design.col_names);
    let fit = lm_fit_matrix(
        &exprs.data,
        exprs.nrow,
        exprs.ncol,
        &design.data,
        design.ncol,
    )
    .expect("lmFit");
    let cfit = contrasts_fit(&fit, &contrasts.data, contrasts.ncol).expect("contrasts.fit");
    (exprs, design, contrasts, fit, cfit)
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

fn check_vec(
    report: &mut Report,
    tol: Tol,
    what: &str,
    got: &[f64],
    want: &[f64],
    names: &[String],
) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if g.is_nan() && w.is_nan() {
            continue;
        }
        report.check(tol, format!("{what}[{}]", names[i]), g, w);
    }
}

#[test]
fn lmfit_matches_golden() {
    for case in MATRIX_CASES {
        let Some(dir) = matrix_dir("lmfit_matches_golden", case) else {
            return;
        };
        let (_exprs, design, _contrasts, fit, _cfit) = load_case(&dir);
        let mut report = Report::new(&format!("{case}/lmfit"));

        let coef = Matrix::read(&dir, "lmfit_coefficients.csv");
        assert_eq!(coef.col_names, design.col_names);
        check_matrix(
            &mut report,
            ABS_1E10,
            "coefficients",
            &fit.coefficients,
            &coef,
        );
        let sd = Matrix::read(&dir, "lmfit_stdev_unscaled.csv");
        check_matrix(
            &mut report,
            REL_1E8,
            "stdev.unscaled",
            &fit.stdev_unscaled,
            &sd,
        );
        let scalars = Matrix::read(&dir, "lmfit_scalars.csv");
        check_vec(
            &mut report,
            REL_1E8,
            "sigma",
            &fit.sigma,
            scalars.col("sigma"),
            &scalars.row_names,
        );
        check_vec(
            &mut report,
            REL_1E8,
            "df.residual",
            &fit.df_residual,
            scalars.col("df_residual"),
            &scalars.row_names,
        );
        check_vec(
            &mut report,
            REL_1E8,
            "Amean",
            &fit.amean,
            scalars.col("Amean"),
            &scalars.row_names,
        );
        let cov = Matrix::read(&dir, "lmfit_cov_coefficients.csv");
        assert_eq!(cov.nrow, fit.rank, "{case}: rank");
        check_matrix(
            &mut report,
            REL_1E8,
            "cov.coefficients",
            &fit.cov_coefficients,
            &cov,
        );
        report.finish();
    }
}

#[test]
fn contrasts_fit_matches_golden() {
    for case in MATRIX_CASES {
        let Some(dir) = matrix_dir("contrasts_fit_matches_golden", case) else {
            return;
        };
        let (_exprs, _design, contrasts, _fit, cfit) = load_case(&dir);
        let mut report = Report::new(&format!("{case}/contrasts"));

        let coef = Matrix::read(&dir, "contrasts_coefficients.csv");
        assert_eq!(coef.col_names, contrasts.col_names);
        check_matrix(
            &mut report,
            ABS_1E10,
            "coefficients",
            &cfit.coefficients,
            &coef,
        );
        let sd = Matrix::read(&dir, "contrasts_stdev_unscaled.csv");
        check_matrix(
            &mut report,
            REL_1E8,
            "stdev.unscaled",
            &cfit.stdev_unscaled,
            &sd,
        );
        let cov = Matrix::read(&dir, "contrasts_cov_coefficients.csv");
        check_matrix(
            &mut report,
            REL_1E8,
            "cov.coefficients",
            &cfit.cov_coefficients,
            &cov,
        );
        report.finish();
    }
}
