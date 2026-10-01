//! `pbeta` against `scalar/pbeta.csv`: `lower = pbeta(x, a, b, TRUE, FALSE)` and
//! `upper = pbeta(x, a, b, FALSE, FALSE)`, every row, at the `PQ` tolerance.

mod common;

use common::*;
use rnum::nmath::pbeta;

#[test]
fn pbeta_matches_the_golden_table() {
    let test = "pbeta_matches_the_golden_table";
    let Some(dir) = scalar_dir(test) else {
        return;
    };
    let table = Table::read(&dir, "pbeta.csv");
    let mut report = Report::new("pbeta.csv");
    for row in table.rows() {
        let (x, a, b) = (row.get("x"), row.get("a"), row.get("b"));
        report.check(
            PQ,
            format_args!("line {} lower: pbeta({x}, {a}, {b})", row.line),
            pbeta(x, a, b, true, false),
            row.get("lower"),
        );
        report.check(
            PQ,
            format_args!("line {} upper: pbeta({x}, {a}, {b})", row.line),
            pbeta(x, a, b, false, false),
            row.get("upper"),
        );
    }
    report.finish();
}
