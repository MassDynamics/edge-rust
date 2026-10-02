//! F5 and SE-3 at the public `glm_ql_fit` boundary.
use edger_core::ql::glm_ql_fit;

fn counts(ng: usize) -> Vec<f64> {
    (0..ng)
        .flat_map(|g| (0..6).map(move |j| (20 + (g * 7 + j * 13) % 50) as f64))
        .collect()
}

fn offset(y: &[f64], nlib: usize) -> Vec<f64> {
    let mut lib = vec![0.0; nlib];
    for r in y.chunks(nlib) {
        for (l, v) in lib.iter_mut().zip(r) {
            *l += v;
        }
    }
    lib.iter().map(|l| l.ln()).collect()
}

const X: [f64; 12] = [1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 1.]; // ~0 + cond, 6 x 2

// F5: glmQLFit's `if (max(dispersion) > 4)` fails on NA; f64::min(NaN, 4) gave 4.
#[test]
fn nan_trended_dispersion_is_an_error_not_a_cap() {
    let y = counts(50);
    let off = offset(&y, 6);
    let r = glm_ql_fit(&y, 6, &X, 2, &off, &[5.0; 50], &[f64::NAN; 50]);
    match r {
        Ok(q) => panic!("NaN dispersion accepted as {}", q.dispersion),
        Err(e) => assert!(e
            .to_string()
            .contains("missing value where TRUE/FALSE needed")),
    }
}

// SE-3: no residual df is an Err at the boundary, not a panic.
#[test]
fn no_residual_df_is_an_error_at_the_boundary() {
    let x = vec![1.0, 0.0, 0.0, 1.0]; // 2 x 2 identity
    let y: Vec<f64> = (0..20).flat_map(|g| [10.0 + g as f64, 30.0]).collect();
    let off = offset(&y, 2);
    let r = std::panic::catch_unwind(|| {
        glm_ql_fit(&y, 2, &x, 2, &off, &[5.0; 20], &[0.1; 20]).map(|q| q.df_residual)
    });
    assert!(matches!(r, Ok(Err(_))), "expected Err, got {r:?}");
}

// SE-3: no genes is an Err at the boundary, not a panic.
#[test]
fn no_genes_is_an_error_at_the_boundary() {
    let off = vec![0.0; 6];
    let r = std::panic::catch_unwind(|| {
        glm_ql_fit(&[], 6, &X, 2, &off, &[], &[]).map(|q| q.df_residual)
    });
    assert!(matches!(r, Ok(Err(_))), "expected Err, got {r:?}");
}
