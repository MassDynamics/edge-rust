//! `pgamma`/`qgamma`/`pchisq`/`qchisq` against `scalar/pgamma.csv`, `scalar/qgamma.csv`,
//! `scalar/pchisq.csv` and `scalar/qchisq.csv` (scale = 1 for the gamma pair, as the
//! generator's default). Run with `--nocapture` to see each file's max relative error, or
//! the skip.

mod common;
use common::*;

use rnum::nmath::{pchisq, pgamma, qchisq, qgamma};

/// The generator's `p` grid ends in `1 - 10^seq(-1, -15)` (`scalar_goldens.R:35`), and
/// `fwrite` prints its last value, `1 - 1e-15`, as `1` at 15 significant digits while the
/// `q` columns on that row hold the quantile of `1 - 1e-15`. The input, not the output, is
/// what the printing lost, so restore it rather than compare `q(1) = Inf` against a finite
/// golden. One such row per shape/df in each `q*.csv`.
fn restore_p(p: f64) -> f64 {
    if p == 1.0 {
        1.0 - 1e-15
    } else {
        p
    }
}

/// `pgamma.csv`: `lower` = pgamma(x, shape), `upper` = lower.tail=FALSE.
#[test]
fn pgamma_matches_golden() {
    let Some(dir) = scalar_dir("pgamma_matches_golden") else {
        return;
    };
    let table = Table::read(&dir, "pgamma.csv");
    let mut report = Report::new("pgamma.csv");
    for row in table.rows() {
        let x = row.get("x");
        let shape = row.get("shape");
        for (column, lower_tail) in [("lower", true), ("upper", false)] {
            report.check(
                PQ,
                format_args!("line {} {column} (x = {x}, shape = {shape})", row.line),
                pgamma(x, shape, 1.0, lower_tail, false),
                row.get(column),
            );
        }
    }
    report.finish();
}

/// `qgamma.csv`: `q` = qgamma(p, shape).
#[test]
fn qgamma_matches_golden() {
    let Some(dir) = scalar_dir("qgamma_matches_golden") else {
        return;
    };
    let table = Table::read(&dir, "qgamma.csv");
    let mut report = Report::new("qgamma.csv");
    for row in table.rows() {
        let p = restore_p(row.get("p"));
        let shape = row.get("shape");
        report.check(
            PQ,
            format_args!("line {} q (p = {p:e}, shape = {shape})", row.line),
            qgamma(p, shape, 1.0, true, false),
            row.get("q"),
        );
    }
    report.finish();
}

/// `pchisq.csv`: `lower` = pchisq(x, df), `upper` = lower.tail=FALSE.
#[test]
fn pchisq_matches_golden() {
    let Some(dir) = scalar_dir("pchisq_matches_golden") else {
        return;
    };
    let table = Table::read(&dir, "pchisq.csv");
    let mut report = Report::new("pchisq.csv");
    for row in table.rows() {
        let x = row.get("x");
        let df = row.get("df");
        for (column, lower_tail) in [("lower", true), ("upper", false)] {
            report.check(
                PQ,
                format_args!("line {} {column} (x = {x}, df = {df})", row.line),
                pchisq(x, df, lower_tail, false),
                row.get(column),
            );
        }
    }
    report.finish();
}

/// `qchisq.csv`: `q` = qchisq(p, df), `q_upper` = lower.tail=FALSE.
#[test]
fn qchisq_matches_golden() {
    let Some(dir) = scalar_dir("qchisq_matches_golden") else {
        return;
    };
    let table = Table::read(&dir, "qchisq.csv");
    let mut report = Report::new("qchisq.csv");
    for row in table.rows() {
        let p = restore_p(row.get("p"));
        let df = row.get("df");
        for (column, lower_tail) in [("q", true), ("q_upper", false)] {
            report.check(
                PQ,
                format_args!("line {} {column} (p = {p:e}, df = {df})", row.line),
                qchisq(p, df, lower_tail, false),
                row.get(column),
            );
        }
    }
    report.finish();
}
