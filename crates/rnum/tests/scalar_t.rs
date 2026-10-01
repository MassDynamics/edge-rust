//! `pt`/`qt` against `scalar/pt.csv` and `scalar/qt.csv`. Run with `--nocapture` to see each
//! file's max relative error, or the skip.

mod common;
use common::*;

use rnum::nmath::{pt, qt};

/// `pt.csv`: `lower` = pt(x, df), `upper` = lower.tail=FALSE, `log_lower`/`log_upper` the
/// same two with log.p=TRUE.
#[test]
fn pt_matches_golden() {
    let Some(dir) = scalar_dir("pt_matches_golden") else {
        return;
    };
    let table = Table::read(&dir, "pt.csv");
    let mut report = Report::new("pt.csv");
    for row in table.rows() {
        let x = row.get("x");
        let df = row.get("df");
        for (column, lower_tail, log_p) in [
            ("lower", true, false),
            ("upper", false, false),
            ("log_lower", true, true),
            ("log_upper", false, true),
        ] {
            report.check(
                PQ,
                format_args!("line {} {column} (x = {x}, df = {df})", row.line),
                pt(x, df, lower_tail, log_p),
                row.get(column),
            );
        }
    }
    report.finish();
}

/// `qt.csv`: `q` = qt(p, df), `q_upper` = lower.tail=FALSE.
#[test]
fn qt_matches_golden() {
    let Some(dir) = scalar_dir("qt_matches_golden") else {
        return;
    };
    let table = Table::read(&dir, "qt.csv");
    let mut report = Report::new("qt.csv");
    for row in table.rows() {
        let mut p = row.get("p");
        let df = row.get("df");
        // The last row of each df block is the generator's `1 - 1e-15` (the last of
        // `1 - 10^seq(-1, -15)` in `scalar_goldens.R`), which `fwrite` prints as `1` at 15
        // significant digits while its `q` column is the finite qt(1 - 1e-15, df). The
        // input, not the output, is what the printing lost, so restore it rather than
        // compare qt(1) = Inf against a finite golden.
        if p == 1.0 {
            p = 1.0 - 1e-15;
        }
        for (column, lower_tail) in [("q", true), ("q_upper", false)] {
            report.check(
                PQ,
                format_args!("line {} {column} (p = {p:e}, df = {df})", row.line),
                qt(p, df, lower_tail, false),
                row.get(column),
            );
        }
    }
    report.finish();
}
