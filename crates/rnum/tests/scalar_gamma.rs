mod common;
use common::*;

use rnum::nmath::gamma::{digamma, lgammafn, trigamma, trigamma_inverse};

/// `scalar/gamma_family.csv`: `lgamma`, `digamma`, `trigamma` at each `x`.
#[test]
fn gamma_family_matches_r() {
    let Some(dir) = scalar_dir("gamma_family") else {
        return;
    };
    let table = Table::read(&dir, "gamma_family.csv");
    let mut report = Report::new("gamma_family.csv");
    for row in table.rows() {
        let x = row.get("x");
        report.check(
            PQ,
            format_args!("line {} lgamma({x})", row.line),
            lgammafn(x),
            row.get("lgamma"),
        );
        report.check(
            PQ,
            format_args!("line {} digamma({x})", row.line),
            digamma(x),
            row.get("digamma"),
        );
        report.check(
            PQ,
            format_args!("line {} trigamma({x})", row.line),
            trigamma(x),
            row.get("trigamma"),
        );
    }
    report.finish();
}

/// `scalar/trigamma_inverse.csv`: limma's `trigammaInverse` at each `x`.
#[test]
fn trigamma_inverse_matches_limma() {
    let Some(dir) = scalar_dir("trigamma_inverse") else {
        return;
    };
    let table = Table::read(&dir, "trigamma_inverse.csv");
    let mut report = Report::new("trigamma_inverse.csv");
    for row in table.rows() {
        let x = row.get("x");
        report.check(
            PQ,
            format_args!("line {} trigamma_inverse({x})", row.line),
            trigamma_inverse(x),
            row.get("trigamma_inverse"),
        );
    }
    report.finish();
}
