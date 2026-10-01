//! The harness's own checks (see `common/mod.rs`): the corpus schema, and the comparator and
//! report rules every `scalar_<port>.rs` relies on.
//!
//! Run with `--nocapture` to see which tests skipped and each file's max relative error:
//!
//! ```text
//! MD_GOLDEN_CORPUS_DIR=~/wd/md-limma-golden-corpus cargo test -p limma-core --tests -- --nocapture
//! ```

mod common;

use common::*;

/// Every in-scope table is present, parses as doubles, and still carries the columns the
/// ports are written against. This is what the reader is proved on before any port exists,
/// and it pins the corpus schema: a regenerated corpus that renamed, dropped or added a
/// table or column fails here rather than inside a port's test.
#[test]
fn every_in_scope_scalar_table_reads_with_the_documented_columns() {
    let test = "every_in_scope_scalar_table_reads_with_the_documented_columns";
    let Some(dir) = scalar_dir(test) else {
        return;
    };

    let documented: Vec<String> = SCHEMA.iter().map(|(file, _)| (*file).to_owned()).collect();
    assert_eq!(
        in_scope_files(&dir),
        documented,
        "the in-scope scalar/*.csv set has changed"
    );

    for (file, columns) in SCHEMA {
        let table = Table::read(&dir, file);
        assert_eq!(table.headers, columns, "{file}: columns");
        assert!(!table.rows.is_empty(), "{file}: no rows");
        // The generator promises no NA/NaN. `NA` cannot parse, so it would have panicked
        // above; `NaN` can, and the comparator would then read it as a value to match
        // exactly — so hold the corpus to the promise here.
        for row in table.rows() {
            for column in columns {
                assert!(
                    !row.get(column).is_nan(),
                    "{file} line {}: NaN in `{column}`",
                    row.line
                );
            }
        }
        println!("{file}: {} rows, columns {columns:?}", table.rows.len());
    }
}

/// The comparator's three jobs: a non-finite golden matched in kind and sign, a finite one
/// matched by relative error with an absolute floor under it, and the bit-exact case
/// `lowess_x` needs. Corpus-free, so these rules are checked even where the goldens are not
/// available.
#[test]
fn comparator_matches_non_finite_exactly_and_finite_by_rel_or_floor() {
    fn passes(tol: Tol, got: f64, want: f64) -> bool {
        let mut report = Report::new("comparator");
        report.check(tol, format_args!("got {got:e}, want {want:e}"), got, want);
        report.failed == 0
    }

    let (inf, nan) = (f64::INFINITY, f64::NAN);

    // Non-finite goldens: kind and sign, nothing else.
    assert!(passes(PQ, inf, inf));
    assert!(passes(PQ, -inf, -inf));
    assert!(passes(PQ, nan, nan));
    assert!(!passes(PQ, -inf, inf));
    assert!(!passes(PQ, f64::MAX, inf));
    assert!(!passes(PQ, nan, inf));
    // ... and a non-finite `got` never satisfies a finite golden.
    assert!(!passes(PQ, inf, 1.0));
    assert!(!passes(PQ, -inf, 1.0));
    assert!(!passes(PQ, nan, 1.0));

    // Finite goldens: relative error at 1e-12 for the p/q/gamma-family values.
    assert!(passes(PQ, 1.0 + 5e-13, 1.0));
    assert!(!passes(PQ, 1.0 + 2e-12, 1.0));
    assert!(passes(PQ, -1.0 - 5e-13, -1.0));
    assert!(!passes(PQ, -1.0 - 2e-12, -1.0));

    // The 1e-300 absolute floor under it, which is what makes the deep-tail rows (`p` from
    // 1e-300) and the exact zeros checkable at all.
    assert!(passes(PQ, 0.0, 5e-301));
    assert!(!passes(PQ, 0.0, 1e-299));
    assert!(passes(PQ, 0.0, 0.0));
    assert!(passes(PQ, 1e-320, 0.0));
    assert!(!passes(PQ, 1e-200, 0.0));

    // `lowess_y` is the looser 1e-10; `lowess_x` is exact.
    assert!(passes(LOWESS_Y, 1.0 + 5e-11, 1.0));
    assert!(!passes(LOWESS_Y, 1.0 + 5e-10, 1.0));
    assert!(passes(EXACT, 0.25, 0.25));
    assert!(!passes(EXACT, f64::from_bits(0.25f64.to_bits() + 1), 0.25));
}

/// The per-file report keeps the largest relative error and the value it came from — the
/// number each card records — and leaves the non-finite and zero goldens out of it, having no
/// relative error to contribute.
#[test]
fn report_keeps_the_largest_relative_error_and_the_value_it_came_from() {
    let mut report = Report::new("example.csv");
    report.check(PQ, "small", 1.0 + 1e-15, 1.0);
    report.check(PQ, "large", 2.0 + 4e-13, 2.0);
    report.check(PQ, "huge", 1e300 * (1.0 + 1e-14), 1e300);
    report.check(PQ, "upper_inf", f64::INFINITY, f64::INFINITY);
    report.check(PQ, "lower_zero", 0.0, 0.0);

    assert_eq!(report.worst, "large");
    // `2.0 + 4e-13` is not that decimal, so the recorded error is 2e-13 only to a few digits.
    assert!(
        (report.max_rel - 2e-13).abs() < 1e-3 * 2e-13,
        "max_rel = {:e}",
        report.max_rel
    );
    assert_eq!(report.checked, 5);
    assert_eq!(report.non_finite, 1);
    assert_eq!(report.zeros, 1);
    // Prints the line the cards quote, and passes because nothing missed tolerance.
    report.finish();
}
