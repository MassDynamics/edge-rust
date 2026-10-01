//! `qbeta` against `scalar/qbeta.csv`: `q = qbeta(p, a, b)` (lower tail, not log), every
//! row, at the `PQ` tolerance.

mod common;

use common::*;
use rnum::nmath::qbeta;

#[test]
fn qbeta_matches_the_golden_table() {
    let test = "qbeta_matches_the_golden_table";
    let Some(dir) = scalar_dir(test) else {
        return;
    };
    let table = Table::read(&dir, "qbeta.csv");
    let mut report = Report::new("qbeta.csv");
    for row in table.rows() {
        let (p_printed, a, b) = (row.get("p"), row.get("a"), row.get("b"));
        // The generator's p grid ends in `1 - 10^seq(-1, -15)`, so its last point is
        // 1 - 1e-15, but `fwrite` prints to 15 significant digits and rounds that to `1`.
        // The golden was computed for 1 - 1e-15 (qbeta(1, a, b) would be exactly 1), so a
        // row carrying p == 1 is that grid point and gets the value R actually used. The
        // grid repeats per (a, b) pair, so this applies to every such row.
        let p = if p_printed == 1.0 {
            1.0 - 1e-15
        } else {
            p_printed
        };
        report.check(
            PQ,
            format_args!("line {} q: qbeta({p}, {a}, {b})", row.line),
            qbeta(p, a, b, true, false),
            row.get("q"),
        );
    }
    report.finish();
}
