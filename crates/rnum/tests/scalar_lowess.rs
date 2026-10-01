//! Golden tests for `rnum::lowess` against `scalar/lowess_*.csv` (R's
//! `lowess(x, y, f, iter = 3)` with the default `delta`), plus corpus-free checks on
//! `weighted_lowess` / `weighted_lowess_r` / `loess_fit`, which have no golden.
//!
//! Run with `--nocapture` to see each file's max relative error:
//!
//! ```text
//! cargo test -p limma-core --test scalar_lowess -- --nocapture
//! ```

mod common;

use common::*;
use rnum::lowess::{loess_fit, lowess, weighted_lowess, weighted_lowess_r};

/// One golden file: `x` (unsorted input), `y`, `lowess_x` (R's sorted `x`, bit-exact) and
/// `lowess_y` (the fit, at `LOWESS_Y`).
fn check_lowess_file(test: &str, file: &str, f: f64) {
    let Some(dir) = scalar_dir(test) else {
        return;
    };
    let table = Table::read(&dir, file);
    let x: Vec<f64> = table.rows().map(|r| r.get("x")).collect();
    let y: Vec<f64> = table.rows().map(|r| r.get("y")).collect();
    let (lx, ly) = lowess(&x, &y, f, 3, None);
    assert_eq!(lx.len(), x.len());
    assert_eq!(ly.len(), x.len());

    let mut report = Report::new(file);
    for (k, row) in table.rows().enumerate() {
        report.check(
            EXACT,
            format_args!("line {} lowess_x", row.line),
            lx[k],
            row.get("lowess_x"),
        );
        report.check(
            LOWESS_Y,
            format_args!("line {} lowess_y", row.line),
            ly[k],
            row.get("lowess_y"),
        );
    }
    report.finish();
}

#[test]
fn lowess_n50_f0p667() {
    // `f0p667` is `format(2/3, digits = 3)`: the generator passes `f = 2/3` exactly.
    check_lowess_file("lowess_n50_f0p667", "lowess_n50_f0p667.csv", 2.0 / 3.0);
}

#[test]
fn lowess_n200_f0p3() {
    check_lowess_file("lowess_n200_f0p3", "lowess_n200_f0p3.csv", 0.3);
}

#[test]
fn lowess_n500_f0p5() {
    check_lowess_file("lowess_n500_f0p5", "lowess_n500_f0p5.csv", 0.5);
}

#[test]
fn lowess_n3000_f0p5() {
    check_lowess_file("lowess_n3000_f0p5", "lowess_n3000_f0p5.csv", 0.5);
}

/// `loessFit(y, x, span, iterations)` with no weights is `lowess(x, y, f=span,
/// iter=iterations-1)` put back in the original order: check it against the same goldens,
/// with a non-finite `y` and a non-finite `x` dropped and returned as `NaN`.
#[test]
fn loess_fit_unweighted_matches_lowess_golden_in_original_order() {
    let test = "loess_fit_unweighted_matches_lowess_golden_in_original_order";
    let Some(dir) = scalar_dir(test) else {
        return;
    };
    let file = "lowess_n200_f0p3.csv";
    let table = Table::read(&dir, file);
    let x: Vec<f64> = table.rows().map(|r| r.get("x")).collect();
    let y: Vec<f64> = table.rows().map(|r| r.get("y")).collect();
    let n = x.len();

    // Whole table: fitted[i] is lowess_y at x[i]'s rank in the stable order.
    let fitted = loess_fit(&y, &x, None, 0.3, 4);
    let mut o: Vec<usize> = (0..n).collect();
    o.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
    let mut report = Report::new(file);
    for (k, row) in table.rows().enumerate() {
        report.check(
            LOWESS_Y,
            format_args!("line {} loessFit fitted", row.line),
            fitted[o[k]],
            row.get("lowess_y"),
        );
    }
    report.finish();

    // Non-finite rows are dropped before the fit and come back NaN.
    let mut y2 = y.clone();
    let mut x2 = x.clone();
    y2[7] = f64::NAN;
    x2[42] = f64::INFINITY;
    let keep: Vec<usize> = (0..n).filter(|&i| i != 7 && i != 42).collect();
    let xk: Vec<f64> = keep.iter().map(|&i| x[i]).collect();
    let yk: Vec<f64> = keep.iter().map(|&i| y[i]).collect();
    let want = loess_fit(&yk, &xk, None, 0.3, 4);
    let got = loess_fit(&y2, &x2, None, 0.3, 4);
    assert!(got[7].is_nan() && got[42].is_nan());
    for (k, &i) in keep.iter().enumerate() {
        assert_eq!(got[i], want[k], "row {i}");
    }
}

/// A smooth, noise-free curve on distinct, unevenly spaced `x`, returned in a scrambled
/// order so the wrappers' sort/unsort is exercised.
fn smooth_data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            10.0 * t + 0.3 * (7.0 * t).sin()
        })
        .collect();
    // Deterministic scramble: rotate by a stride coprime with n.
    let stride = 7;
    x = (0..n).map(|k| x[(k * stride) % n]).collect();
    let y: Vec<f64> = x.iter().map(|&v| 2.0 + 0.5 * v + (0.4 * v).sin()).collect();
    (x, y)
}

/// (a) With all prior weights equal, `delta = 0` (so every point is a seed on both sides)
/// and `n * span` an integer, `weighted_lowess_r` and `lowess` select the same `n * span`
/// nearest points and solve the same weighted line, differing only in the C's cut-offs at
/// the edges of the tricube (`r <= 0.999h` vs `r < h`, `r <= 0.001h` -> 1) and in the
/// robustness cut-off (`r <= 0.999 cmad` vs `r < cmad`); those move the fit by well under
/// 1e-6 (measured 1.8e-12 here for both settings). `weighted_lowess` counts total fits
/// where `lowess` counts robustness steps, hence `iterations = iter + 1`.
#[test]
fn weighted_lowess_r_with_equal_weights_agrees_with_lowess() {
    let n = 100;
    let (x, y) = smooth_data(n);
    let w = vec![1.0; n];
    for (iterations, iter) in [(1usize, 0usize), (4, 3)] {
        let (_lx, ly) = lowess(&x, &y, 0.5, iter, Some(0.0));
        let fitted = weighted_lowess_r(&x, &y, &w, 0.5, iterations, Some(0.0), 200);
        let mut o: Vec<usize> = (0..n).collect();
        o.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
        let mut max_diff: f64 = 0.0;
        for (k, &i) in o.iter().enumerate() {
            max_diff = max_diff.max((fitted[i] - ly[k]).abs());
        }
        println!(
            "iterations {iterations} vs iter {iter}: max |weightedLowess - lowess| = {max_diff:e}"
        );
        assert!(
            max_diff < 1e-6,
            "iterations {iterations}: max diff {max_diff:e}"
        );
    }
}

/// (b) Sanity on the C `weighted_lowess` itself: constant `y` gives a constant fit and a
/// linear `y` is reproduced within 1e-10 through all four robustness passes.
///
/// Robustness weights stay at 1 only when the residuals are exactly zero: the C's early
/// exit is `cmad <= 1e-7 * resid_scale`, and for a non-zero constant or linear `y` both
/// sides are roundoff-sized numbers of the same magnitude, so it reweights from roundoff
/// ratios (verified: `y = 3.25` leaves weights in `[0.56, 1]`). `y == 0` gives exact zero
/// residuals, `cmad == 0`, and the break, so that is where all-ones is asserted.
#[test]
fn weighted_lowess_reproduces_constant_and_linear_y_with_unit_robust_weights() {
    let n = 60;
    let x: Vec<f64> = (0..n)
        .map(|i| 1.0 + 0.1 * i as f64 + 0.01 * (i % 3) as f64)
        .collect();
    let w: Vec<f64> = (0..n).map(|i| 0.5 + (i % 4) as f64).collect();

    let y_const = vec![3.25; n];
    let (fit, rob) = weighted_lowess(&x, &y_const, &w, 0.3, 4, 0.0);
    for (i, &f) in fit.iter().enumerate() {
        assert!((f - 3.25).abs() < 1e-10, "constant y at {i}: {f}");
    }
    assert!(
        rob.iter().all(|&r| (0.0..=1.0).contains(&r)),
        "robust weights {rob:?}"
    );

    let y_zero = vec![0.0; n];
    let (fit, rob) = weighted_lowess(&x, &y_zero, &w, 0.3, 4, 0.0);
    assert!(fit.iter().all(|&f| f == 0.0), "zero y fit {fit:?}");
    assert!(rob.iter().all(|&r| r == 1.0), "robust weights {rob:?}");

    let y_lin: Vec<f64> = x.iter().map(|&v| -1.5 + 2.0 * v).collect();
    for delta in [0.0, 0.25] {
        let (fit, rob) = weighted_lowess(&x, &y_lin, &w, 0.3, 4, delta);
        for (i, (&f, &want)) in fit.iter().zip(&y_lin).enumerate() {
            assert!(
                (f - want).abs() < 1e-10,
                "linear y, delta {delta}, at {i}: {f} vs {want}"
            );
        }
        assert!(
            rob.iter().all(|&r| (0.0..=1.0).contains(&r)),
            "robust weights {rob:?}"
        );
    }

    // The R wrapper returns the same fit in the original order.
    let (xs, _ys) = smooth_data(n);
    let y_lin2: Vec<f64> = xs.iter().map(|&v| 0.7 - 0.3 * v).collect();
    let fitted = weighted_lowess_r(&xs, &y_lin2, &w, 0.3, 4, None, 200);
    for (i, (&f, &want)) in fitted.iter().zip(&y_lin2).enumerate() {
        assert!(
            (f - want).abs() < 1e-10,
            "wrapper linear y at {i}: {f} vs {want}"
        );
    }
}

/// `loessFit` with weights: equal weights fall back to the `lowess` path (so it matches the
/// unweighted call bit for bit), and unequal weights go through `weightedLowess`, which
/// still reproduces a line.
#[test]
fn loess_fit_weighted_paths() {
    let n = 150;
    let (x, y) = smooth_data(n);
    let unweighted = loess_fit(&y, &x, None, 0.3, 4);
    let equal = loess_fit(&y, &x, Some(&vec![2.0; n]), 0.3, 4);
    assert_eq!(unweighted, equal);

    let w: Vec<f64> = (0..n).map(|i| 1.0 + (i % 5) as f64).collect();
    let y_lin: Vec<f64> = x.iter().map(|&v| 4.0 - 0.25 * v).collect();
    let fitted = loess_fit(&y_lin, &x, Some(&w), 0.3, 4);
    for (i, (&f, &want)) in fitted.iter().zip(&y_lin).enumerate() {
        assert!((f - want).abs() < 1e-10, "at {i}: {f} vs {want}");
    }

    // Too few weighted observations: the weighted line.
    let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
    let ys = [1.0, 3.0, 5.0, 7.0, 9.0];
    let ws = [1.0, 2.0, 1.0, 3.0, 1.0];
    let fitted = loess_fit(&ys, &xs, Some(&ws), 0.3, 4);
    for (f, want) in fitted.iter().zip(&ys) {
        assert!((f - want).abs() < 1e-12);
    }
}
