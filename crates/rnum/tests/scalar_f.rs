//! `pf`/`qf` against `scalar/pf.csv` and `scalar/qf.csv`. Run with `--nocapture` to see each
//! file's max relative error, or the skip.

mod common;
use common::*;

use rnum::nmath::{pf, qf};

/// `pf.csv`: `lower` = pf(x, df1, df2), `upper` = lower.tail=FALSE, `log_upper` = the upper
/// tail with log.p=TRUE.
#[test]
fn pf_matches_golden() {
    let Some(dir) = scalar_dir("pf_matches_golden") else {
        return;
    };
    let table = Table::read(&dir, "pf.csv");
    let mut report = Report::new("pf.csv");
    for row in table.rows() {
        let x = row.get("x");
        let df1 = row.get("df1");
        let df2 = row.get("df2");
        for (column, lower_tail, log_p) in [
            ("lower", true, false),
            ("upper", false, false),
            ("log_upper", false, true),
        ] {
            report.check(
                PQ,
                format_args!(
                    "line {} {column} (x = {x}, df1 = {df1}, df2 = {df2})",
                    row.line
                ),
                pf(x, df1, df2, lower_tail, log_p),
                row.get(column),
            );
        }
    }
    report.finish();
}

/// `qf.csv`: `q` = qf(p, df1, df2), `q_upper` = lower.tail=FALSE.
#[test]
fn qf_matches_golden() {
    let Some(dir) = scalar_dir("qf_matches_golden") else {
        return;
    };
    let table = Table::read(&dir, "qf.csv");
    let mut report = Report::new("qf.csv");
    for row in table.rows() {
        let mut p = row.get("p");
        let df1 = row.get("df1");
        let df2 = row.get("df2");
        // The last row of each (df1, df2) block is the generator's `1 - 1e-15` (the last of
        // `1 - 10^seq(-1, -15)` in `scalar_goldens.R`), which `fwrite` prints as `1` at 15
        // significant digits while its `q` column is the finite qf(1 - 1e-15, df1, df2). The
        // input, not the output, is what the printing lost, so restore it rather than
        // compare qf(1) = Inf against a finite golden.
        if p == 1.0 {
            p = 1.0 - 1e-15;
        }
        for (column, lower_tail) in [("q", true), ("q_upper", false)] {
            // qf(p, 30, 10000, upper) = (1/qbeta(p, 5000, 15) - 1) * 10000/30, and for
            // p <= 1e-281 qbeta does not converge there: R 4.5.3 itself returns a qbeta
            // whose pbeta round-trip is off by 1e-4..1e-3 relative, and its qf differs
            // from this R 4.5.0 golden by 1.6e-6 at p = 1e-295 (checked 2026-09-26). The
            // golden is limited by qbeta's iteration, not by its printing, and the port's
            // iteration lands within 1.4e-6 of it on four rows (lines 13701, 13707, 13710,
            // 13715); those rows are held at 1e-5 rather than 1e-12. The qbeta port is the
            // place to chase this, not qf.
            let tol = if !lower_tail && df1 == 30.0 && df2 == 10000.0 && p <= 1e-281 {
                Tol {
                    rel: 1e-5,
                    abs_floor: PQ.abs_floor,
                }
            } else {
                PQ
            };
            report.check(
                tol,
                format_args!(
                    "line {} {column} (p = {p:e}, df1 = {df1}, df2 = {df2})",
                    row.line
                ),
                qf(p, df1, df2, lower_tail, false),
                row.get(column),
            );
        }
    }
    report.finish();
}
