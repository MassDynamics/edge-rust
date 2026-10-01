//! Shared helpers for the edgeR golden tests: locate the count corpus, list the edgeR runs and
//! read the reference CSVs and `reference.json`.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

/// `MD_COUNT_CORPUS_DIR`, defaulting to `~/wd/md-count-golden-corpus`.
pub fn corpus_dir() -> PathBuf {
    match std::env::var("MD_COUNT_CORPUS_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(std::env::var("HOME").expect("HOME")).join("wd/md-count-golden-corpus"),
    }
}

pub fn ref_dir(run: &str) -> PathBuf {
    corpus_dir().join("reference").join(run)
}

/// Every `count_edger_*` and `edge_edger_*` run in the reference directory, sorted.
pub fn edger_runs() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(corpus_dir().join("reference"))
        .expect("corpus reference dir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("count_edger_") || n.starts_with("edge_edger_"))
        .collect();
    v.sort();
    assert!(!v.is_empty(), "no edgeR runs under {:?}", corpus_dir());
    v
}

pub fn reference_json(run: &str) -> serde_json::Value {
    let s = std::fs::read_to_string(ref_dir(run).join("reference.json")).expect("reference.json");
    serde_json::from_str(&s).expect("json")
}

pub fn manifest(run: &str) -> serde_json::Value {
    let p = corpus_dir().join("runs").join(run).join("manifest.json");
    serde_json::from_str(&std::fs::read_to_string(p).expect("manifest")).expect("json")
}

pub fn scalar(run: &str, name: &str) -> f64 {
    let j = reference_json(run);
    let v = &j["scalars"][name];
    v.as_f64()
        .or_else(|| v.as_array().and_then(|a| a[0].as_f64()))
        .unwrap_or_else(|| panic!("{run}: scalar {name} missing: {v}"))
}

/// Whether the run is an expected-error run (`manifest.status == "error"`).
pub fn is_error_run(run: &str) -> bool {
    manifest(run)["status"].as_str() == Some("error")
}

/// A CSV as header + string rows.
pub struct Table {
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl Table {
    pub fn col(&self, name: &str) -> usize {
        self.header
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("no column {name} in {:?}", self.header))
    }
    pub fn strings(&self, name: &str) -> Vec<String> {
        let j = self.col(name);
        self.rows.iter().map(|r| r[j].clone()).collect()
    }
    pub fn f64s(&self, name: &str) -> Vec<f64> {
        let j = self.col(name);
        self.rows.iter().map(|r| parse_num(&r[j])).collect()
    }
    pub fn bools(&self, name: &str) -> Vec<bool> {
        let j = self.col(name);
        self.rows.iter().map(|r| r[j] == "TRUE").collect()
    }
    /// All columns but the first, as a row-major numeric matrix.
    pub fn matrix(&self) -> Vec<Vec<f64>> {
        self.rows
            .iter()
            .map(|r| r[1..].iter().map(|s| parse_num(s)).collect())
            .collect()
    }
    pub fn index(&self) -> HashMap<String, usize> {
        self.rows.iter().enumerate().map(|(i, r)| (r[0].clone(), i)).collect()
    }
}

pub fn parse_num(s: &str) -> f64 {
    match s {
        "NA" | "" => f64::NAN,
        "Inf" => f64::INFINITY,
        "-Inf" => f64::NEG_INFINITY,
        _ => s.parse().unwrap_or_else(|_| panic!("not a number: {s:?}")),
    }
}

pub fn read_csv(run: &str, name: &str) -> Table {
    let p = ref_dir(run).join(format!("{name}.csv"));
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&p)
        .unwrap_or_else(|e| panic!("{p:?}: {e}"));
    let header = rdr.headers().unwrap().iter().map(|s| s.to_string()).collect();
    let rows = rdr
        .records()
        .map(|r| r.unwrap().iter().map(|s| s.to_string()).collect())
        .collect();
    Table { header, rows }
}

pub fn has_csv(run: &str, name: &str) -> bool {
    ref_dir(run).join(format!("{name}.csv")).exists()
}

/// Relative difference as `compare_vec` in `ref_common.R` computes it, with an exact NA/Inf
/// pattern check. Returns the max relative difference or panics with the first mismatch.
pub fn max_rel(what: &str, got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut m: f64 = 0.0;
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        if a.is_nan() || b.is_nan() {
            assert!(a.is_nan() && b.is_nan(), "{what}[{i}]: NA pattern, got {a} want {b}");
            continue;
        }
        if a.is_infinite() || b.is_infinite() {
            assert_eq!(a, b, "{what}[{i}]: Inf mismatch");
            continue;
        }
        if a != b {
            m = m.max((a - b).abs() / b.abs().max(1e-300));
        }
    }
    m
}

pub fn assert_close(what: &str, got: &[f64], want: &[f64], tol: f64) {
    let m = max_rel(what, got, want);
    assert!(m <= tol, "{what}: max rel {m:e} > {tol:e}");
}
