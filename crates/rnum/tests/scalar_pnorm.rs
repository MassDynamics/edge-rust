//! `pnorm`/`qnorm` against `scalar/pnorm.csv` and `scalar/qnorm.csv` (mu = 0, sigma = 1).
//! Run with `--nocapture` to see each file's max relative error, or the skip.

mod common;
use common::*;

use rnum::nmath::{pnorm, qnorm};

/// `pnorm.csv`: `lower` = pnorm(x), `upper` = lower.tail=FALSE, `log_lower`/`log_upper` the
/// same two with log.p=TRUE.
///
/// Lines 9-11 (x = -38.25, -38, -37.75) have subnormal `lower` and `log_upper` goldens, and
/// `fwrite` prints a subnormal as `2^-1023 * (1 + mantissa/2^52)` — the three read as
/// ~1.1125e-308 where the values are 2.08e-320, 2.89e-316 and 3.76e-312 (checked against
/// `0.5 * erfc(-x/sqrt(2))`; the printed digits match that formula exactly). They pass under
/// `PQ`'s 1e-300 absolute floor, and they are what the reported max relative error is: the
/// next-largest relative error in this file is ~1e-15.
#[test]
fn pnorm_matches_golden() {
    let Some(dir) = scalar_dir("pnorm_matches_golden") else {
        return;
    };
    let table = Table::read(&dir, "pnorm.csv");
    let mut report = Report::new("pnorm.csv");
    for row in table.rows() {
        let x = row.get("x");
        for (column, lower_tail, log_p) in [
            ("lower", true, false),
            ("upper", false, false),
            ("log_lower", true, true),
            ("log_upper", false, true),
        ] {
            report.check(
                PQ,
                format_args!("line {} {column} (x = {x})", row.line),
                pnorm(x, 0.0, 1.0, lower_tail, log_p),
                row.get(column),
            );
        }
    }
    report.finish();
}

/// `qnorm.csv`: `q` = qnorm(p), `q_upper` = lower.tail=FALSE.
#[test]
fn qnorm_matches_golden() {
    let Some(dir) = scalar_dir("qnorm_matches_golden") else {
        return;
    };
    let table = Table::read(&dir, "qnorm.csv");
    let mut report = Report::new("qnorm.csv");
    for row in table.rows() {
        let mut p = row.get("p");
        // Line 335 is the generator's `1 - 1e-15` (`scalar_goldens.R:35`, the last of
        // `1 - 10^seq(-1, -15, by = -1)`), which `fwrite` prints as `1` at 15 significant
        // digits while its `q` column is qnorm(1 - 1e-15) = 7.94144448741598. The input,
        // not the output, is what the printing lost, so restore it rather than compare
        // qnorm(1) = Inf against a finite golden.
        if p == 1.0 {
            p = 1.0 - 1e-15;
        }
        for (column, lower_tail) in [("q", true), ("q_upper", false)] {
            report.check(
                PQ,
                format_args!("line {} {column} (p = {p:e})", row.line),
                qnorm(p, 0.0, 1.0, lower_tail, false),
                row.get(column),
            );
        }
    }
    report.finish();
}
