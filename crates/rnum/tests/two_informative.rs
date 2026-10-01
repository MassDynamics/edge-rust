//! `fitFDistUnequalDF1` with exactly two informative variances (round-3 review must-fix 1).
//!
//! R sets `prior.weights <- NULL` after `PriorWeights` was already computed. When PriorWeights is
//! TRUE (an NA variance or a df1 below 0.01, which a df-0 feature supplies), `w * NULL` is
//! `numeric(0)`: the scale is NaN and df2 is whatever `optimize()` returns for a constant
//! objective. The port reproduces that; the end-to-end pins are the goldens
//! `edge_two_informative_*`, where every moderated statistic of the pair is NaN.

use rnum::ebayes::fit_f_dist_unequal_df1;

// R 4.5.0 / limma 3.66.0: 2 * m / (1 - m), m = optimize(function(p) 0, c(1/2, 0.9998))$minimum.
const R_DF2_CONSTANT_OBJECTIVE: f64 = 8134.844780382662;

#[test]
fn two_informative_with_prior_weights_is_nan_as_in_r() {
    // R: fitFDistUnequalDF1(c(0.8,1.7,0,0), c(3,5,4,4), robust=TRUE,
    //                       prior.weights=c(0.2,0.9,1,1)) -> scale NaN, df2 8134.845.
    let x = [0.8, 1.7, 0.0, 0.0];
    let df1 = [3.0, 5.0, 4.0, 4.0];
    let fit =
        fit_f_dist_unequal_df1(&x, &df1, None, None, true, Some(&[0.2, 0.9, 1.0, 1.0])).unwrap();
    assert_eq!(fit.scale.len(), 1);
    assert!(fit.scale[0].is_nan());
    assert_eq!(fit.df2.to_bits(), R_DF2_CONSTANT_OBJECTIVE.to_bits());
    assert!(fit.df2_outlier.is_none() && fit.df2_shrunk.is_none());
}

#[test]
fn two_informative_from_na_and_df0_is_nan_as_in_r() {
    // R: fitFDistUnequalDF1(c(0.8,1.7,NA,0.5), c(3,5,4,0.001)) -> scale NaN, df2 8134.845.
    let fit = fit_f_dist_unequal_df1(
        &[0.8, 1.7, f64::NAN, 0.5],
        &[3.0, 5.0, 4.0, 0.001],
        None,
        None,
        false,
        None,
    )
    .unwrap();
    assert!(fit.scale[0].is_nan());
    assert_eq!(fit.df2.to_bits(), R_DF2_CONSTANT_OBJECTIVE.to_bits());
}

#[test]
fn two_informative_without_prior_weights_fits_unweighted() {
    // No NA and every df1 >= 0.01, so PriorWeights is FALSE and R fits the two values:
    // fitFDistUnequalDF1(c(0.8,1.7,0), c(3,5,4), robust=TRUE).
    let fit =
        fit_f_dist_unequal_df1(&[0.8, 1.7, 0.0], &[3.0, 5.0, 4.0], None, None, true, None).unwrap();
    let rel = |got: f64, want: f64| ((got - want) / want).abs();
    assert!(
        rel(fit.scale[0], 9.62625969525079e-05) < 1e-8,
        "{:?}",
        fit.scale
    );
    assert!(rel(fit.df2, 2.000423313604798) < 1e-8, "{}", fit.df2);
}

#[test]
fn fewer_than_two_informative_is_nan_as_in_r() {
    let fit = fit_f_dist_unequal_df1(&[0.8, 0.0, 0.0], &[3.0], None, None, false, None).unwrap();
    assert!(fit.scale[0].is_nan() && fit.df2.is_nan());
}

/// Both length checks say "covariate", as fitFDistUnequalDF1.R:16 and :22 do.
#[test]
fn prior_weights_length_message_matches_r() {
    let x = [1.0, 2.0, 3.0];
    let df1 = [2.0, 3.0, 4.0];
    let Err(err) = fit_f_dist_unequal_df1(&x, &df1, None, None, false, Some(&[1.0, 1.0])) else {
        panic!("length mismatch must fail");
    };
    assert!(
        err.to_string()
            .contains("x and covariate are different lengths"),
        "{err}"
    );
}
