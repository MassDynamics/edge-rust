//! Corpus-free regression tests for the review-r1 findings at the engine boundary. Each input is
//! small and deterministic; the expected messages are production's (MDFlexiComparisons
//! `.buildCountMatrixFromLongDT` / `.fitEdgeRModel`, edgeR 4.8.2, R's model.matrix and qr),
//! checked in the md-flexi-r45-local image.
use edger_core::pipeline::{
    max_abs_log2fc, run_edger, Comparison, Control, ControlKind, EdgerInput, EdgerOutput,
    PairResult,
};
use std::sync::mpsc;
use std::time::Duration;

const COND: [&str; 6] = ["A", "A", "A", "B", "B", "B"];

/// ng genes x nlib samples, gene-major; with 50 x 6 every gene passes filterByExpr.
fn synth_counts(ng: usize, nlib: usize) -> Vec<f64> {
    (0..ng)
        .flat_map(|g| (0..nlib).map(move |j| (20 + (g * 7 + j * 13) % 50) as f64))
        .collect()
}

fn input(counts: Vec<f64>, controls: Vec<Control>, cmp: (&str, &str)) -> EdgerInput {
    let nlib = COND.len();
    let ng = counts.len() / nlib;
    EdgerInput {
        gene_ids: (0..ng).map(|g| format!("g{g}")).collect(),
        sample_ids: (0..nlib).map(|j| format!("s{j}")).collect(),
        counts,
        condition_col: "condition".into(),
        condition: COND.iter().map(|s| s.to_string()).collect(),
        controls,
        comparisons: vec![Comparison {
            left: cmp.0.into(),
            right: cmp.1.into(),
            encoded_left: cmp.0.into(),
            encoded_right: cmp.1.into(),
        }],
        norm_method: "TMM".into(),
        entity_type: "gene".into(),
    }
}

fn numeric_control(values: [&str; 6]) -> Control {
    Control {
        name: "x".into(),
        kind: ControlKind::Numerical,
        values: values.map(String::from).to_vec(),
    }
}

/// Run the engine on a worker thread and fail (instead of hanging the suite) when it does not
/// return within `secs`. A timed-out worker keeps spinning until the test process exits.
fn run_bounded(inp: EdgerInput, secs: u64) -> Result<EdgerOutput, String> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(run_edger(&inp).map_err(|e| e.to_string()));
    });
    rx.recv_timeout(Duration::from_secs(secs))
        .unwrap_or_else(|_| panic!("engine did not return within {secs} s"))
}

fn expect_err(inp: EdgerInput, needle: &str) {
    match run_bounded(inp, 20) {
        Ok(_) => panic!("expected an error containing {needle:?}, got results"),
        Err(m) => assert!(m.contains(needle), "{m:?} does not contain {needle:?}"),
    }
}

#[test]
fn baseline_synth_input_runs() {
    let out = run_bounded(input(synth_counts(50, 6), vec![], ("B", "A")), 20).unwrap();
    assert!(out.kept.iter().all(|&k| k));
}

// SE-1 / F1: R stops in filterByExpr -> cpm (cpm.R:91).
#[test]
fn zero_library_sample_is_rejected_with_the_r_message() {
    let mut c = synth_counts(50, 6);
    for g in 0..50 {
        c[g * 6] = 0.0;
    }
    expect_err(
        input(c, vec![], ("B", "A")),
        "library sizes should be greater than zero",
    );
}

// SE-1: R's qr.qy in filterByExpr's hat() refuses the non-finite QR of a +-1e308 control.
#[test]
fn extreme_numeric_control_is_rejected_with_the_r_message() {
    let ctl = numeric_control(["1e308", "-1e308", "1", "2", "3", "1e308"]);
    expect_err(
        input(synth_counts(50, 6), vec![ctl], ("B", "A")),
        "NA/NaN/Inf in foreign function call (arg 1)",
    );
}

// An infinite numeric control: R's qr refuses it the same way.
#[test]
fn infinite_numeric_control_is_rejected_with_the_r_message() {
    let ctl = numeric_control(["Inf", "1", "2", "3", "4", "5"]);
    expect_err(
        input(synth_counts(50, 6), vec![ctl], ("B", "A")),
        "NA/NaN/Inf in foreign function call (arg 1)",
    );
}

// P2 at the engine: model.matrix drops the NA row, then filterByExpr stops.
#[test]
fn missing_numeric_control_is_rejected_with_the_r_message() {
    let ctl = numeric_control(["NaN", "1", "2", "3", "4", "5"]);
    expect_err(
        input(synth_counts(50, 6), vec![ctl], ("B", "A")),
        "nrow(design) disagrees with ncol(y)",
    );
}

// F2: model.matrix stops for a one-level factor.
#[test]
fn single_level_categorical_control_is_rejected() {
    let ctl = Control {
        name: "batch".into(),
        kind: ControlKind::Categorical,
        values: vec!["b1".into(); 6],
    };
    expect_err(
        input(synth_counts(50, 6), vec![ctl], ("B", "A")),
        "contrasts can be applied only to factors with 2 or more levels",
    );
}

// F3: makeContrasts / glmQLFTest stop("contrasts are all zero").
#[test]
fn comparison_of_a_level_with_itself_is_rejected() {
    expect_err(
        input(synth_counts(50, 6), vec![], ("A", "A")),
        "contrasts are all zero",
    );
}

// P3: values within 1e-6 of an integer are rounded, as production's as.integer(round(x)).
#[test]
fn near_integer_counts_are_rounded() {
    let c = synth_counts(50, 6);
    let drift: Vec<f64> = c
        .iter()
        .enumerate()
        .map(|(i, v)| v + if i % 2 == 0 { 1e-7 } else { -1e-7 })
        .collect();
    let a = run_bounded(input(c, vec![], ("B", "A")), 20).unwrap();
    let b = run_bounded(input(drift, vec![], ("B", "A")), 20).unwrap();
    assert_eq!(a.pairs[0].f, b.pairs[0].f);
    assert_eq!(a.pairs[0].log2fc, b.pairs[0].log2fc);
    assert_eq!(a.f, b.f);
}

// P3: non-integer counts are refused with production's message.
#[test]
fn non_integer_counts_are_rejected() {
    let mut c = synth_counts(50, 6);
    c[7] += 0.5;
    expect_err(
        input(c, vec![], ("B", "A")),
        "Non-integer values detected in the count column.",
    );
}

// P3: the CPM variant, every sample summing to ~1e6.
#[test]
fn cpm_like_counts_are_rejected_as_normalised() {
    let c = synth_counts(50, 6);
    let mut lib = [0.0; 6];
    for (i, v) in c.iter().enumerate() {
        lib[i % 6] += v;
    }
    let cpm: Vec<f64> = c
        .iter()
        .enumerate()
        .map(|(i, v)| v / lib[i % 6] * 1e6)
        .collect();
    expect_err(
        input(cpm, vec![], ("B", "A")),
        "Input data appears to be CPM/TPM-normalised",
    );
}

// P6: as.integer() of a count above .Machine$integer.max is NA, and DGEList refuses it.
#[test]
fn counts_above_int32_are_rejected() {
    let mut c = synth_counts(50, 6);
    c[3] = 3e9;
    expect_err(input(c, vec![], ("B", "A")), "NA counts not allowed");
}

// .Machine$integer.max itself is a valid count.
#[test]
fn int32_max_count_is_accepted() {
    let mut c = synth_counts(50, 6);
    c[3] = 2147483647.0;
    run_bounded(input(c, vec![], ("B", "A")), 20).unwrap();
}

// get_max_fc / which.max: first maximum on |Log2FC| ties, NaN skipped, all-NaN gives None.
#[test]
fn max_abs_log2fc_takes_the_first_maximum_and_skips_nan() {
    let pr = |v: Vec<f64>| PairResult {
        label: String::new(),
        log2fc: v,
        stat: vec![],
        se: vec![],
        ci_left: vec![],
        ci_right: vec![],
        f: vec![],
        pvalue: vec![],
        adj_pvalue: vec![],
    };
    let pairs = vec![
        pr(vec![-2.0, f64::NAN, f64::NAN]),
        pr(vec![2.0, 1.0, f64::NAN]),
    ];
    let (idx, val) = max_abs_log2fc(&pairs);
    assert_eq!(idx, vec![Some(0), Some(1), None]);
    assert_eq!(val[0], -2.0);
    assert_eq!(val[1], 1.0);
    assert!(val[2].is_nan());
}
