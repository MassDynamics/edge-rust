#![allow(dead_code)]
//! Shared harness for the scalar golden tests: corpus lookup, the CSV reader, the comparator
//! and the per-file error report. Each port has its own `tests/scalar_<port>.rs` that does
//! `mod common;` and builds on this. The harness's own checks live in `scalar_goldens.rs`.
//!
//! The scalar golden tables ARE the spec for `nmath` and `lowess` (see the repo `README.md`).
//! This file holds the harness those tests are built on — corpus lookup, the CSV reader, the
//! comparator and the per-file error report — plus the checks on the harness itself. Each
//! port's card adds its own `#[test]` here.
//!
//! The tests read the small corpus tier committed at `tests/golden/corpus/` (all of `scalar/`
//! and `matrix/`), or the full corpus when `MD_GOLDEN_CORPUS_DIR` points at it; see
//! [`corpus_root`]. Nothing skips. Run with `--nocapture` to see each file's max relative error:
//!
//! ```text
//! cargo test -p limma-core --test scalar_goldens -- --nocapture
//! ```
//!
//! The goldens are written by `MDFlexiComparisons/data-raw/golden-corpus/scalar_goldens.R`
//! with `data.table::fwrite`, which prints a double to 15 significant digits. A golden
//! therefore sits up to ~1e-15 relatively away from the double R actually computed; that is
//! the floor on what any port can score here, and the reason every tolerance below is
//! relative.

pub mod matrix;

use std::fmt;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::{env, fs};

/// Tolerance for one comparison: `|got - want| <= max(abs_floor, rel * |want|)`.
#[derive(Clone, Copy)]
pub struct Tol {
    pub rel: f64,
    pub abs_floor: f64,
}

/// Distribution and gamma-family values: every `p*`/`q*` column, `lgamma`/`digamma`/
/// `trigamma`, and `trigamma_inverse`.
pub const PQ: Tol = Tol {
    rel: 1e-12,
    abs_floor: 1e-300,
};

/// `lowess_y`.
pub const LOWESS_Y: Tol = Tol {
    rel: 1e-10,
    abs_floor: 0.0,
};

/// `lowess_x`, which is R's sorted copy of the `x` column rather than a computed value: both
/// sides come from the same 15-digit decimal, so they must agree bit for bit.
pub const EXACT: Tol = Tol {
    rel: 0.0,
    abs_floor: 0.0,
};

pub const CORPUS_ENV: &str = "MD_GOLDEN_CORPUS_DIR";

/// Failures listed in a report's panic message before it falls back to a count.
pub const MAX_REPORTED_FAILURES: usize = 10;

/// The in-scope `scalar/*.csv` tables and the columns each one carries, in the order
/// [`in_scope_files`] returns them. The `fitfdist_*` tables are deliberately absent: those
/// goldens belong to the `fitFDist` cards.
pub const SCHEMA: [(&str, &[&str]); 18] = [
    ("gamma_family.csv", &["x", "lgamma", "digamma", "trigamma"]),
    ("lowess_n200_f0p3.csv", &["x", "y", "lowess_x", "lowess_y"]),
    ("lowess_n3000_f0p5.csv", &["x", "y", "lowess_x", "lowess_y"]),
    ("lowess_n500_f0p5.csv", &["x", "y", "lowess_x", "lowess_y"]),
    ("lowess_n50_f0p667.csv", &["x", "y", "lowess_x", "lowess_y"]),
    ("pbeta.csv", &["x", "a", "b", "lower", "upper"]),
    ("pchisq.csv", &["x", "df", "lower", "upper"]),
    (
        "pf.csv",
        &["x", "df1", "df2", "lower", "upper", "log_upper"],
    ),
    ("pgamma.csv", &["x", "shape", "lower", "upper"]),
    (
        "pnorm.csv",
        &["x", "lower", "upper", "log_lower", "log_upper"],
    ),
    (
        "pt.csv",
        &["x", "df", "lower", "upper", "log_lower", "log_upper"],
    ),
    ("qbeta.csv", &["p", "a", "b", "q"]),
    ("qchisq.csv", &["p", "df", "q", "q_upper"]),
    ("qf.csv", &["p", "df1", "df2", "q", "q_upper"]),
    ("qgamma.csv", &["p", "shape", "q"]),
    ("qnorm.csv", &["p", "q", "q_upper"]),
    ("qt.csv", &["p", "df", "q", "q_upper"]),
    ("trigamma_inverse.csv", &["x", "trigamma_inverse"]),
];

/// The corpus root: `MD_GOLDEN_CORPUS_DIR` when set (the full corpus), otherwise the small tier
/// committed at `tests/golden/corpus/`. A set variable that is not a directory panics rather
/// than falling back, so a mistyped path cannot quietly shrink the test run.
pub fn corpus_root() -> PathBuf {
    match env::var_os(CORPUS_ENV).filter(|value| !value.is_empty()) {
        Some(root) => {
            let root = PathBuf::from(root);
            assert!(
                root.is_dir(),
                "{CORPUS_ENV}={} is not a directory",
                root.display()
            );
            root
        }
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/corpus"),
    }
}

/// The corpus `scalar/` directory. Always `Some`; the `Option` keeps the callers' early return.
pub fn scalar_dir(_test: &str) -> Option<PathBuf> {
    let root = corpus_root();
    let dir = root.join("scalar");
    assert!(dir.is_dir(), "{} has no scalar/ directory", root.display());
    Some(dir)
}

/// Every in-scope table in `scalar/`: all `*.csv` bar the `fitfdist_*` ones, sorted by name.
pub fn in_scope_files(dir: &Path) -> Vec<String> {
    let entries =
        fs::read_dir(dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
    let mut files: Vec<String> = entries
        .map(|entry| {
            let entry = entry.unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
            entry.file_name().to_string_lossy().into_owned()
        })
        .filter(|name| name.ends_with(".csv") && !name.starts_with("fitfdist_"))
        .collect();
    files.sort();
    files
}

/// One `scalar/*.csv` in memory: a header row of column names, and rows of doubles.
///
/// Every cell in these tables is numeric — `Inf`, `-Inf` and exact zeros included. The
/// generator writes missings as `NA`, which has no `f64` parse and so panics here; the corpus
/// is held to producing none of them by
/// [`every_in_scope_scalar_table_reads_with_the_documented_columns`].
pub struct Table {
    pub file: String,
    pub headers: Vec<String>,
    pub rows: Vec<Vec<f64>>,
}

impl Table {
    pub fn read(dir: &Path, file: &str) -> Table {
        let path = dir.join(file);
        let mut reader = csv::Reader::from_path(&path)
            .unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()));
        let headers: Vec<String> = reader
            .headers()
            .unwrap_or_else(|e| panic!("{file}: cannot read the header row: {e}"))
            .iter()
            .map(str::to_owned)
            .collect();
        let rows = reader
            .records()
            .enumerate()
            .map(|(index, record)| {
                // Line 1 is the header, so row `index` is on line `index + 2`.
                let line = index + 2;
                let record =
                    record.unwrap_or_else(|e| panic!("{file} line {line}: unreadable: {e}"));
                assert_eq!(
                    record.len(),
                    headers.len(),
                    "{file} line {line}: {} fields for {} columns",
                    record.len(),
                    headers.len()
                );
                record
                    .iter()
                    .zip(&headers)
                    .map(|(field, column)| {
                        field.parse::<f64>().unwrap_or_else(|e| {
                            panic!(
                                "{file} line {line}: `{column}` = {field:?} is not a double: {e}"
                            )
                        })
                    })
                    .collect()
            })
            .collect();
        Table {
            file: file.to_owned(),
            headers,
            rows,
        }
    }

    pub fn rows(&self) -> impl Iterator<Item = Row<'_>> {
        self.rows.iter().enumerate().map(|(index, values)| Row {
            file: &self.file,
            line: index + 2,
            headers: &self.headers,
            values,
        })
    }
}

/// One row of a [`Table`], read by column name so a test lines up with the R call it mirrors.
pub struct Row<'a> {
    pub file: &'a str,
    /// The row's line in the CSV, header counted, so a failure can be opened at that line.
    pub line: usize,
    pub headers: &'a [String],
    pub values: &'a [f64],
}

impl Row<'_> {
    pub fn get(&self, column: &str) -> f64 {
        let index = self
            .headers
            .iter()
            .position(|header| header == column)
            .unwrap_or_else(|| {
                panic!(
                    "{}: no `{column}` column (has {:?})",
                    self.file, self.headers
                )
            });
        self.values[index]
    }
}

/// One file's comparisons: the largest relative error, which each card reports per file, and
/// every value that missed its tolerance.
#[derive(Default)]
pub struct Report {
    pub file: String,
    pub checked: usize,
    pub max_rel: f64,
    /// What [`Report::check`] was told the `max_rel` value was, for tracing it back.
    pub worst: String,
    /// Goldens that are `±Inf` or `NaN`: matched in kind and sign rather than by tolerance,
    /// and carrying no relative error.
    pub non_finite: usize,
    /// Goldens that are exactly zero: an error against zero has no relative form either, so
    /// the largest absolute one is reported on its own.
    pub zeros: usize,
    pub max_abs_at_zero: f64,
    pub failed: usize,
    pub failures: Vec<String>,
}

impl Report {
    pub fn new(file: &str) -> Report {
        Report {
            file: file.to_owned(),
            ..Report::default()
        }
    }

    /// Compare one value against its golden. `what` names the value and the row it came from
    /// — `format_args!` is enough — and is only formatted when it is worth keeping.
    pub fn check(&mut self, tol: Tol, what: impl fmt::Display, got: f64, want: f64) {
        self.checked += 1;

        if !want.is_finite() {
            self.non_finite += 1;
            // Kind and sign, exactly: `got == want` settles ±Inf, and NaN equals nothing.
            if got != want && !(want.is_nan() && got.is_nan()) {
                self.fail(tol, what, got, want);
            }
            return;
        }

        let err = (got - want).abs();
        if want == 0.0 {
            self.zeros += 1;
            self.max_abs_at_zero = self.max_abs_at_zero.max(err);
        } else {
            let rel = err / want.abs();
            if rel > self.max_rel {
                self.max_rel = rel;
                self.worst = what.to_string();
            }
        }
        // Negated rather than `err > limit`, which clippy asks for, because a non-finite
        // `got` against a finite golden gives a NaN `err`: that compares false either way,
        // and it has to fail.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(err <= tol.abs_floor.max(tol.rel * want.abs())) {
            self.fail(tol, what, got, want);
        }
    }

    pub fn fail(&mut self, tol: Tol, what: impl fmt::Display, got: f64, want: f64) {
        self.failed += 1;
        if self.failures.len() < MAX_REPORTED_FAILURES {
            self.failures.push(format!(
                "{what}: got {got:e}, want {want:e} (tolerance rel {:e}, abs floor {:e})",
                tol.rel, tol.abs_floor
            ));
        }
    }

    /// Print this file's max relative error, then panic if any value missed its tolerance.
    pub fn finish(self) {
        let mut line = format!("{}: {} values checked", self.file, self.checked);
        if self.checked > self.non_finite + self.zeros {
            let _ = write!(line, ", max rel error {:e} at {}", self.max_rel, self.worst);
        }
        if self.non_finite > 0 {
            let _ = write!(
                line,
                ", {} non-finite goldens matched in kind and sign",
                self.non_finite
            );
        }
        if self.zeros > 0 {
            let _ = write!(
                line,
                ", {} zero goldens within {:e}",
                self.zeros, self.max_abs_at_zero
            );
        }
        println!("{line}");

        if self.failed > 0 {
            let hidden = self.failed - self.failures.len();
            let more = if hidden > 0 {
                format!("\n  ... and {hidden} more")
            } else {
                String::new()
            };
            panic!(
                "{}: {} of {} values missed tolerance:\n  {}{more}",
                self.file,
                self.failed,
                self.checked,
                self.failures.join("\n  ")
            );
        }
    }
}
